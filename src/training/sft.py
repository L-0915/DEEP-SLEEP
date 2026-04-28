"""Supervised Fine-Tuning (SFT) entry point for the DeepSleep model.

Supports two modes:
1. **Full fine-tuning** -- all parameters are updated.
2. **LoRA fine-tuning** -- only low-rank adapter parameters are updated,
   using the ``peft`` library.  The adapter can be saved and merged later.

Usage::

    # Full fine-tuning
    python src/training/sft.py --config configs/sft/default.yaml

    # LoRA fine-tuning
    python src/training/sft.py --config configs/sft/lora.yaml

    # Single-GPU
    python src/training/sft.py --config configs/sft/default.yaml training.batch_size=4
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

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.data.dataset.sft_dataset import SFTDataset
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
    """Parse CLI arguments and return a resolved OmegaConf config.

    Returns:
        A resolved ``DictConfig`` with all SFT hyperparameters.
    """
    import argparse

    parser = argparse.ArgumentParser(description="DeepSleep SFT training")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    known_args, unknown = parser.parse_known_args()

    defaults: Dict[str, Any] = {
        # Data
        "data": {
            "data_path": "",
            "max_length": 4096,
            "tokenizer_path": "",
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
        # LoRA
        "lora": {
            "enabled": False,
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "lora_dropout": 0.05,
            "bias": "none",
        },
        # Training
        "training": {
            "num_epochs": 3,
            "batch_size": 4,
            "micro_batch_size": 1,
            "learning_rate": 2e-5,
            "weight_decay": 0.01,
            "beta1": 0.9,
            "beta2": 0.999,
            "grad_clip": 1.0,
            "warmup_ratio": 0.03,
            "max_steps": None,
        },
        # Checkpointing
        "checkpoint": {
            "output_dir": "checkpoints/sft",
            "save_every": 500,
            "resume_from": None,
            "keep_last_n": 3,
            "pretrain_checkpoint": None,
        },
        # Evaluation
        "eval": {
            "eval_every": 200,
            "num_eval_batches": 20,
            "eval_data_path": None,
        },
        # Logging
        "logging": {
            "log_every": 10,
            "wandb_project": "deepsleep-sft",
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

    cli_overrides = OmegaConf.from_cli(unknown)
    cfg = OmegaConf.merge(cfg, cli_overrides)
    OmegaConf.resolve(cfg)

    return cfg


# ---------------------------------------------------------------------------
# Model & tokenizer loading
# ---------------------------------------------------------------------------


def load_tokenizer(cfg: DictConfig) -> Any:
    """Load a HuggingFace tokenizer.

    Attempts to load from the path specified in ``cfg.data.tokenizer_path``.
    Falls back to a simple whitespace tokenizer for integration testing.

    Args:
        cfg: Resolved configuration.

    Returns:
        A tokenizer instance with ``pad_token_id`` set.
    """
    tokenizer_path = cfg.data.tokenizer_path
    if tokenizer_path and os.path.isdir(tokenizer_path):
        from transformers import AutoTokenizer  # type: ignore[import-untyped]

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info("Loaded tokenizer from %s", tokenizer_path)
        return tokenizer

    logger.warning("No tokenizer path provided. Using a mock tokenizer for testing.")
    # Create a minimal mock tokenizer for integration testing.
    from transformers import PreTrainedTokenizerFast  # type: ignore[import-untyped]

    mock = PreTrainedTokenizerFast(
        vocab={"<pad>": 0, "<s>": 1, "</s>": 2, "a": 3, "b": 4, "c": 5},
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
    )
    return mock


def build_model(cfg: DictConfig) -> nn.Module:
    """Construct the DeepSleep model from config.

    Args:
        cfg: Resolved configuration.

    Returns:
        An ``nn.Module`` instance.
    """
    model_cfg = DeepSleepConfig(**OmegaConf.to_container(cfg.model))

    try:
        from src.model.modeling_deepsleep import DeepSleepForCausalLM  # type: ignore[import-untyped]
        model = DeepSleepForCausalLM(model_cfg)
    except (ImportError, AttributeError):
        logger.warning(
            "DeepSleepForCausalLM not found. Using a placeholder model."
        )
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


def apply_lora(model: nn.Module, cfg: DictConfig) -> nn.Module:
    """Apply LoRA adapters to the model.

    Args:
        model: The base model.
        cfg: Resolved configuration with a ``lora`` sub-config.

    Returns:
        The model with LoRA adapters applied.
    """
    if not cfg.lora.enabled:
        return model

    try:
        from peft import LoraConfig, get_peft_model  # type: ignore[import-untyped]

        lora_cfg = LoraConfig(
            r=cfg.lora.r,
            lora_alpha=cfg.lora.lora_alpha,
            target_modules=cfg.lora.target_modules,
            lora_dropout=cfg.lora.lora_dropout,
            bias=cfg.lora.bias,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
        logger.info("LoRA adapters applied (r=%d, alpha=%d).", cfg.lora.r, cfg.lora.lora_alpha)
    except ImportError:
        raise ImportError(
            "LoRA requires the 'peft' library.  Install it with: pip install peft"
        )

    return model


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def train(cfg: DictConfig) -> None:
    """Execute the SFT training loop.

    Args:
        cfg: Resolved training configuration.
    """
    setup_distributed()
    rank = get_rank()
    world_size = get_world_size()
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    print_rank0(f"[rank {rank}] Using device: {device}")

    # -- Tokenizer --
    tokenizer = load_tokenizer(cfg)

    # -- Datasets --
    train_dataset = SFTDataset(
        data_path=cfg.data.data_path,
        tokenizer=tokenizer,
        max_length=cfg.data.max_length,
    )

    eval_path = cfg.eval.eval_data_path if cfg.eval.eval_data_path else cfg.data.data_path
    eval_dataset = SFTDataset(
        data_path=eval_path,
        tokenizer=tokenizer,
        max_length=cfg.data.max_length,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    ) if world_size > 1 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.micro_batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=cfg.training.micro_batch_size,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )

    print_rank0(
        f"Train samples: {len(train_dataset)}, "
        f"Eval samples: {len(eval_dataset)}"
    )

    # -- Model --
    model = build_model(cfg)

    # -- Load pretrain checkpoint if specified --
    pretrain_ckpt = cfg.checkpoint.pretrain_checkpoint
    if pretrain_ckpt is not None:
        print_rank0(f"Loading pretrain checkpoint: {pretrain_ckpt}")
        state_dict = torch.load(
            os.path.join(pretrain_ckpt, "model.pt"),
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(state_dict, strict=False)
        print_rank0("Pretrain checkpoint loaded.")

    model = model.to(device)

    # -- Apply LoRA --
    model = apply_lora(model, cfg)

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
    total_steps = cfg.training.max_steps
    if total_steps is None:
        steps_per_epoch = len(train_loader)
        total_steps = int(steps_per_epoch * cfg.training.num_epochs / cfg.training.batch_size)
    warmup_steps = int(total_steps * cfg.training.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        min_lr_ratio=0.1,
    )

    # -- Resume --
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

    # -- Callbacks --
    callbacks = [
        CheckpointCallback(
            output_dir=cfg.checkpoint.output_dir,
            save_every=cfg.checkpoint.save_every,
            keep_last_n=cfg.checkpoint.keep_last_n,
        ),
        EvalCallback(
            eval_dataloader=eval_loader,
            eval_every=cfg.eval.eval_every,
            num_eval_batches=cfg.eval.num_eval_batches,
        ),
        WandbCallback(
            project=cfg.logging.wandb_project,
            name=cfg.logging.wandb_name,
            config=OmegaConf.to_container(cfg),
        ),
    ]

    # -- Training loop --
    gradient_accumulation_steps = cfg.training.batch_size // cfg.training.micro_batch_size
    model.train()

    data_iter = iter(train_loader)
    total_loss = 0.0
    log_count = 0

    print_rank0(f"Starting SFT training from step {start_step} to {total_steps}")

    for step in range(start_step, total_steps):
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0

        for _micro_step in range(gradient_accumulation_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                if train_sampler is not None:
                    train_sampler.set_epoch(step)
                data_iter = iter(train_loader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, labels=labels)

            if hasattr(outputs, "loss") and outputs.loss is not None:
                loss = outputs.loss
            else:
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                loss = compute_lm_loss(logits, labels)

            scaled_loss = loss / gradient_accumulation_steps
            scaled_loss.backward()
            step_loss += loss.item()

        # Gradient clipping.
        if cfg.training.grad_clip > 0:
            if hasattr(model, "clip_grad_norm_"):
                model.clip_grad_norm_(cfg.training.grad_clip)
            else:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.training.grad_clip,
                )

        optimizer.step()
        scheduler.step()

        avg_step_loss = step_loss / gradient_accumulation_steps
        total_loss += avg_step_loss
        log_count += 1

        metrics = {
            "train/loss": avg_step_loss,
            "train/lr": scheduler.get_last_lr()[0],
            "train/step": step,
        }

        if step % cfg.logging.log_every == 0:
            avg = total_loss / max(log_count, 1)
            print_rank0(
                f"Step {step}/{total_steps} | "
                f"loss: {avg_step_loss:.6f} | "
                f"avg_loss: {avg:.6f} | "
                f"lr: {scheduler.get_last_lr()[0]:.2e}"
            )
            total_loss = 0.0
            log_count = 0

        for cb in callbacks:
            cb.on_step_end(
                step=step,
                metrics=metrics,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
            )

    # -- Final save --
    output_dir = cfg.checkpoint.output_dir
    if is_main_process():
        final_dir = os.path.join(output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(final_dir, "model.pt"))
        logger.info("Saved final model to %s", final_dir)

    print_rank0("SFT training complete.")


if __name__ == "__main__":
    config = parse_args()
    train(config)
