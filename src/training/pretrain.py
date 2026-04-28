"""Pretraining entry point for the DeepSleep language model.

Implements a full pretraining loop with:
* FSDP (Fully Sharded Data Parallel) for distributed training
* BF16 mixed precision
* Gradient accumulation and clipping
* Cosine LR schedule with linear warmup
* Checkpoint saving / resume
* wandb integration
* Evaluation on a held-out validation split

Usage::

    torchrun --nproc_per_node=8 src/training/pretrain.py \
        --config configs/pretrain/default.yaml
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf

# Ensure the project root is on the path so that ``src.*`` imports work.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.data.dataset.pretrain_dataset import PretrainDataset
from src.model.config import DeepSleepConfig
from src.training.callbacks import (
    CheckpointCallback,
    EvalCallback,
    WandbCallback,
)
from src.training.loss import compute_lm_loss
from src.training.schedulers import get_cosine_schedule_with_warmup
from src.utils.checkpoint import find_latest_checkpoint, load_checkpoint
from src.utils.distributed import (
    barrier,
    get_rank,
    get_world_size,
    is_main_process,
    print_rank0,
    setup_distributed,
)
from src.utils.fsdp_config import get_fsdp_config
from src.utils.logging import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> DictConfig:
    """Parse CLI arguments and return an OmegaConf config.

    Config values can be overridden from the command line::

        python pretrain.py --config default.yaml data.seq_length=4096

    Returns:
        A resolved ``DictConfig`` with all training hyperparameters.
    """
    import argparse

    parser = argparse.ArgumentParser(description="DeepSleep pretraining")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    known_args, unknown = parser.parse_known_args()

    # Start with default values.
    defaults: Dict[str, Any] = {
        # Data
        "data": {
            "data_dir": "",
            "seq_length": 8192,
            "seed": 42,
        },
        # Model
        "model": {
            "d_model": 2048,
            "n_layers": 24,
            "n_heads": 16,
            "n_kv_heads": 4,
            "vocab_size": 64000,
            "max_position_embeddings": 8192,
            "num_experts": 6,
            "num_routed_experts": 5,
            "num_shared_experts": 1,
            "top_k": 2,
            "aux_loss_coeff": 0.01,
            "z_loss_coeff": 0.001,
        },
        # Training
        "training": {
            "max_steps": 100_000,
            "batch_size": 2,
            "micro_batch_size": 1,
            "learning_rate": 3e-4,
            "weight_decay": 0.1,
            "beta1": 0.9,
            "beta2": 0.95,
            "grad_clip": 1.0,
            "warmup_steps": 2000,
            "min_lr_ratio": 0.1,
        },
        # Checkpointing
        "checkpoint": {
            "output_dir": "checkpoints/pretrain",
            "save_every": 5000,
            "resume_from": None,
            "keep_last_n": 3,
        },
        # Evaluation
        "eval": {
            "eval_every": 1000,
            "num_eval_batches": 20,
        },
        # Logging
        "logging": {
            "log_every": 10,
            "wandb_project": "deepsleep-pretrain",
            "wandb_name": None,
        },
        # FSDP
        "fsdp": {
            "sharding_strategy": "full_shard",
            "mixed_precision_dtype": "bf16",
            "offload_params": False,
        },
    }

    cfg = OmegaConf.create(defaults)

    if known_args.config is not None:
        file_cfg = OmegaConf.load(known_args.config)
        cfg = OmegaConf.merge(cfg, file_cfg)

    # Apply CLI overrides (e.g. training.max_steps=200000).
    cli_overrides = OmegaConf.from_cli(unknown)
    cfg = OmegaConf.merge(cfg, cli_overrides)
    OmegaConf.resolve(cfg)

    return cfg


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------


def build_model(cfg: DictConfig) -> nn.Module:
    """Construct the DeepSleep model from config.

    Attempts to import ``DeepSleepForCausalLM`` from the model package.
    If the full model is not yet implemented, falls back to a thin wrapper
    around ``nn.Linear`` for integration testing.

    Args:
        cfg: Resolved configuration with a ``model`` sub-config.

    Returns:
        An ``nn.Module`` ready for FSDP wrapping.
    """
    model_cfg = DeepSleepConfig(**OmegaConf.to_container(cfg.model))
    logger.info("Model config: %s", OmegaConf.to_yaml(cfg.model))

    try:
        from src.model.modeling_deepsleep import DeepSleepForCausalLM  # type: ignore[import-untyped]
        model = DeepSleepForCausalLM(model_cfg)
        logger.info("Loaded DeepSleepForCausalLM successfully.")
        return model
    except (ImportError, AttributeError):
        logger.warning(
            "DeepSleepForCausalLM not found.  Using a placeholder model for testing."
        )
        # Placeholder: simple causal LM head for integration testing.
        model = nn.Sequential(
            nn.Embedding(model_cfg.vocab_size, model_cfg.d_model),
            nn.TransformerEncoderLayer(
                d_model=model_cfg.d_model,
                nhead=model_cfg.n_heads,
                dim_feedforward=model_cfg.intermediate_size,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            nn.LayerNorm(model_cfg.d_model),
            nn.Linear(model_cfg.d_model, model_cfg.vocab_size, bias=False),
        )
        return model


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------


def pretrain_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate a list of dataset samples into a padded batch.

    Args:
        batch: List of dicts with ``input_ids`` and ``labels`` tensors.

    Returns:
        Batch dict with stacked tensors.
    """
    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {"input_ids": input_ids, "labels": labels}


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def train(cfg: DictConfig) -> None:
    """Execute the full pretraining loop.

    Args:
        cfg: Resolved training configuration.
    """
    setup_distributed()
    rank = get_rank()
    world_size = get_world_size()
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    print_rank0(f"[rank {rank}] Using device: {device}")

    # -- Build datasets --
    train_dataset = PretrainDataset(
        data_dir=cfg.data.data_dir,
        seq_length=cfg.data.seq_length,
        split="train",
        seed=cfg.data.seed,
        rank=rank,
        world_size=world_size,
    )
    eval_dataset = PretrainDataset(
        data_dir=cfg.data.data_dir,
        seq_length=cfg.data.seq_length,
        split="val",
        seed=cfg.data.seed,
        rank=rank,
        world_size=world_size,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=cfg.data.seed,
    ) if world_size > 1 else None

    micro_batch_size = cfg.training.micro_batch_size
    train_loader = DataLoader(
        train_dataset,
        batch_size=micro_batch_size,
        sampler=train_sampler,
        collate_fn=pretrain_collate_fn,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=micro_batch_size,
        collate_fn=pretrain_collate_fn,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )

    print_rank0(
        f"Train samples: {len(train_dataset)}, "
        f"Eval samples: {len(eval_dataset)}, "
        f"World size: {world_size}"
    )

    # -- Build model --
    model = build_model(cfg)
    model = model.to(device)

    # -- Wrap with FSDP --
    if world_size > 1 and torch.cuda.is_available():
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # type: ignore[import-untyped]

        fsdp_kwargs = get_fsdp_config(
            sharding_strategy=cfg.fsdp.sharding_strategy,
            mixed_precision_dtype=cfg.fsdp.mixed_precision_dtype,
            offload_params=cfg.fsdp.offload_params,
        )
        model = FSDP(model, **fsdp_kwargs)
        print_rank0("Model wrapped with FSDP.")

    # -- Optimizer --
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        betas=(cfg.training.beta1, cfg.training.beta2),
        weight_decay=cfg.training.weight_decay,
    )

    # -- LR Scheduler --
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.training.warmup_steps,
        num_training_steps=cfg.training.max_steps,
        min_lr_ratio=cfg.training.min_lr_ratio,
    )

    # -- Resume from checkpoint --
    start_step = 0
    resume_dir = cfg.checkpoint.resume_from
    if resume_dir is None or resume_dir == "auto":
        resume_dir = find_latest_checkpoint(cfg.checkpoint.output_dir)

    if resume_dir is not None:
        print_rank0(f"Resuming from checkpoint: {resume_dir}")
        metadata = load_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            checkpoint_dir=resume_dir,
            map_location=device,
        )
        start_step = metadata.get("step", 0) + 1
        print_rank0(f"Resumed at step {start_step}")

    # -- Callbacks --
    callbacks = []
    callbacks.append(
        CheckpointCallback(
            output_dir=cfg.checkpoint.output_dir,
            save_every=cfg.checkpoint.save_every,
            keep_last_n=cfg.checkpoint.keep_last_n,
        )
    )
    callbacks.append(
        EvalCallback(
            eval_dataloader=eval_loader,
            eval_every=cfg.eval.eval_every,
            num_eval_batches=cfg.eval.num_eval_batches,
        )
    )
    callbacks.append(
        WandbCallback(
            project=cfg.logging.wandb_project,
            name=cfg.logging.wandb_name,
            config=OmegaConf.to_container(cfg),
        )
    )

    # -- Training loop --
    gradient_accumulation_steps = cfg.training.batch_size // micro_batch_size
    model.train()

    data_iter = iter(train_loader)
    log_count = 0
    total_loss = 0.0
    tokens_seen = 0

    print_rank0(f"Starting training from step {start_step} to {cfg.training.max_steps}")

    for step in range(start_step, cfg.training.max_steps):
        # Reset gradient accumulators.
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0

        for _micro_step in range(gradient_accumulation_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                # Re-create iterator for next epoch.
                if train_sampler is not None:
                    train_sampler.set_epoch(step)
                data_iter = iter(train_loader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass.
            outputs = model(input_ids=input_ids, labels=labels)

            if hasattr(outputs, "loss") and outputs.loss is not None:
                loss = outputs.loss
            else:
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                loss = compute_lm_loss(logits, labels)

            # MoE auxiliary loss aggregation.
            moe_aux = 0.0
            if hasattr(outputs, "moe_aux_losses") and outputs.moe_aux_losses:
                for layer_losses in outputs.moe_aux_losses:
                    for _, val in layer_losses.items():
                        moe_aux = moe_aux + val.float().mean()
                loss = loss + moe_aux

            # Scale loss for gradient accumulation.
            scaled_loss = loss / gradient_accumulation_steps
            scaled_loss.backward()

            step_loss += loss.item()
            tokens_seen += input_ids.numel()

        # -- Gradient clipping --
        if cfg.training.grad_clip > 0:
            if world_size > 1 and hasattr(model, "clip_grad_norm_"):
                model.clip_grad_norm_(cfg.training.grad_clip)
            else:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.training.grad_clip,
                )

        # -- Optimizer step --
        optimizer.step()
        scheduler.step()

        # -- Logging --
        avg_step_loss = step_loss / gradient_accumulation_steps
        total_loss += avg_step_loss
        log_count += 1

        metrics = {
            "train/lm_loss": avg_step_loss,
            "train/moe_aux_loss": moe_aux if isinstance(moe_aux, float) else moe_aux.item(),
            "train/lr": scheduler.get_last_lr()[0],
            "train/tokens_seen": tokens_seen,
            "train/step": step,
        }

        # Compute gradient norm for logging.
        total_norm = _compute_grad_norm(model)
        metrics["train/grad_norm"] = total_norm

        if step % cfg.logging.log_every == 0:
            avg_loss = total_loss / log_count
            print_rank0(
                f"Step {step}/{cfg.training.max_steps} | "
                f"loss: {avg_step_loss:.6f} | "
                f"avg_loss: {avg_loss:.6f} | "
                f"lr: {scheduler.get_last_lr()[0]:.2e} | "
                f"grad_norm: {total_norm:.4f} | "
                f"tokens: {tokens_seen}"
            )
            total_loss = 0.0
            log_count = 0

        # -- Callbacks --
        for cb in callbacks:
            cb.on_step_end(
                step=step,
                metrics=metrics,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
            )

    print_rank0("Training complete.")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _compute_grad_norm(model: nn.Module, max_norm: float = float("inf")) -> float:
    """Compute the total gradient norm of the model.

    Handles both FSDP-wrapped and vanilla models.

    Args:
        model: The model whose gradient norm to compute.
        max_norm: Maximum value for clipping (unused in the computation).

    Returns:
        Total gradient norm as a float.
    """
    if hasattr(model, "unwrapped_model"):
        # FSDP model: use the unwrapped model to access parameters.
        params = model.unwrapped_model.parameters()
    else:
        params = model.parameters()

    norms = [p.grad.norm(p=2).item() for p in params if p.grad is not None]
    if not norms:
        return 0.0
    total = sum(n ** 2 for n in norms) ** 0.5
    return min(total, max_norm)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config = parse_args()
    train(config)
