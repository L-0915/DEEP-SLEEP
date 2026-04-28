"""Direct Preference Optimization (DPO) entry point for the DeepSleep model.

Implements the DPO algorithm that aligns the model with human preferences
without training a separate reward model.  The training loop maintains a
frozen copy of the reference model (the SFT checkpoint) and optimizes the
policy model to increase the likelihood of chosen responses while decreasing
the likelihood of rejected responses.

DPO loss formula::

    L = -log(sigmoid(beta * (
        log pi_chosen - log pi_ref_chosen
        - log pi_rejected + log pi_ref_rejected
    )))

Usage::

    torchrun --nproc_per_node=4 src/training/dpo.py --config configs/dpo/default.yaml
"""

from __future__ import annotations

import copy
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

from src.data.dataset.dpo_dataset import DPODataCollator, DPODataset
from src.model.config import DeepSleepConfig
from src.training.callbacks import (
    CheckpointCallback,
    WandbCallback,
)
from src.training.loss import (
    compute_dpo_loss,
    get_batch_logps,
)
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
        A resolved ``DictConfig`` with all DPO hyperparameters.
    """
    import argparse

    parser = argparse.ArgumentParser(description="DeepSleep DPO training")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    known_args, unknown = parser.parse_known_args()

    defaults: Dict[str, Any] = {
        # Data
        "data": {
            "data_path": "",
            "max_length": 4096,
            "max_prompt_length": 1024,
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
        # DPO
        "dpo": {
            "beta": 0.1,
            "loss_type": "sigmoid",
        },
        # Training
        "training": {
            "num_epochs": 1,
            "batch_size": 4,
            "micro_batch_size": 1,
            "learning_rate": 5e-7,
            "weight_decay": 0.0,
            "beta1": 0.9,
            "beta2": 0.95,
            "grad_clip": 1.0,
            "warmup_ratio": 0.1,
            "max_steps": None,
        },
        # Checkpointing
        "checkpoint": {
            "output_dir": "checkpoints/dpo",
            "save_every": 500,
            "resume_from": None,
            "keep_last_n": 3,
            "sft_checkpoint": None,
        },
        # Logging
        "logging": {
            "log_every": 10,
            "wandb_project": "deepsleep-dpo",
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

    Args:
        cfg: Resolved configuration.

    Returns:
        A tokenizer instance.
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
    from transformers import PreTrainedTokenizerFast  # type: ignore[import-untyped]

    return PreTrainedTokenizerFast(
        vocab={"<pad>": 0, "<s>": 1, "</s>": 2, "a": 3, "b": 4, "c": 5},
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
    )


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


# ---------------------------------------------------------------------------
# Forward helper
# ---------------------------------------------------------------------------


def _model_forward(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Run a forward pass and return logits.

    Works with both FSDP-wrapped models and plain models.

    Args:
        model: The language model.
        input_ids: Input token IDs of shape ``(batch, seq_len)``.
        attention_mask: Attention mask of shape ``(batch, seq_len)``.

    Returns:
        Logits tensor of shape ``(batch, seq_len, vocab_size)``.
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    if hasattr(outputs, "logits"):
        return outputs.logits
    return outputs  # Fallback for placeholder models.


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def train(cfg: DictConfig) -> None:
    """Execute the DPO training loop.

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

    # -- Dataset --
    train_dataset = DPODataset(
        data_path=cfg.data.data_path,
        tokenizer=tokenizer,
        max_length=cfg.data.max_length,
        max_prompt_length=cfg.data.max_prompt_length,
    )
    collator = DPODataCollator(
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
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
        collate_fn=collator,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    print_rank0(f"DPO samples: {len(train_dataset)}, World size: {world_size}")

    # -- Policy model --
    policy_model = build_model(cfg)

    # Load SFT checkpoint into the policy model.
    sft_ckpt = cfg.checkpoint.sft_checkpoint
    if sft_ckpt is not None:
        print_rank0(f"Loading SFT checkpoint: {sft_ckpt}")
        model_path = os.path.join(sft_ckpt, "model.pt") if os.path.isdir(sft_ckpt) else sft_ckpt
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        policy_model.load_state_dict(state_dict, strict=False)
        print_rank0("SFT checkpoint loaded.")
    else:
        logger.warning("No SFT checkpoint specified. Training from scratch (not recommended for DPO).")

    policy_model = policy_model.to(device)

    # -- Reference model (frozen copy of the policy model) --
    ref_model = copy.deepcopy(policy_model)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    print_rank0("Reference model initialized (no gradient).")

    # -- Wrap policy model with FSDP --
    if world_size > 1 and torch.cuda.is_available():
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # type: ignore[import-untyped]

        fsdp_kwargs = get_fsdp_config(
            sharding_strategy=cfg.fsdp.sharding_strategy,
            mixed_precision_dtype=cfg.fsdp.mixed_precision_dtype,
            offload_params=cfg.fsdp.offload_params,
        )
        policy_model = FSDP(policy_model, **fsdp_kwargs)
        print_rank0("Policy model wrapped with FSDP.")

    # -- Optimizer (only policy model parameters) --
    optimizer = torch.optim.AdamW(
        policy_model.parameters(),
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
            model=policy_model,
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
        WandbCallback(
            project=cfg.logging.wandb_project,
            name=cfg.logging.wandb_name,
            config=OmegaConf.to_container(cfg),
        ),
    ]

    # -- DPO training loop --
    gradient_accumulation_steps = cfg.training.batch_size // cfg.training.micro_batch_size
    beta = cfg.dpo.beta
    policy_model.train()

    data_iter = iter(train_loader)
    total_loss = 0.0
    total_rewards = 0.0
    total_chosen_rewards = 0.0
    total_rejected_rewards = 0.0
    log_count = 0

    print_rank0(
        f"Starting DPO training from step {start_step} to {total_steps} (beta={beta})"
    )

    for step in range(start_step, total_steps):
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        step_chosen_reward = 0.0
        step_rejected_reward = 0.0

        for _micro_step in range(gradient_accumulation_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                if train_sampler is not None:
                    train_sampler.set_epoch(step)
                data_iter = iter(train_loader)
                batch = next(data_iter)

            chosen_ids = batch["chosen_input_ids"].to(device)
            chosen_mask = batch["chosen_attention_mask"].to(device)
            chosen_labels = batch["chosen_labels"].to(device)
            rejected_ids = batch["rejected_input_ids"].to(device)
            rejected_mask = batch["rejected_attention_mask"].to(device)
            rejected_labels = batch["rejected_labels"].to(device)

            # -- Policy log-probs --
            with torch.no_grad() if world_size > 1 else torch.enable_grad():
                policy_chosen_logits = _model_forward(policy_model, chosen_ids, chosen_mask)
                policy_chosen_logps = get_batch_logps(policy_chosen_logits, chosen_labels)

            # For FSDP, the no_grad context on the forward pass is not applied
            # because we need gradients through the policy model for the DPO loss.
            # The reference model is always in no_grad.
            policy_rejected_logits = _model_forward(policy_model, rejected_ids, rejected_mask)
            policy_rejected_logps = get_batch_logps(policy_rejected_logits, rejected_labels)

            # -- Reference log-probs (no gradient) --
            with torch.no_grad():
                ref_chosen_logits = _model_forward(ref_model, chosen_ids, chosen_mask)
                ref_chosen_logps = get_batch_logps(ref_chosen_logits, chosen_labels)
                ref_rejected_logits = _model_forward(ref_model, rejected_ids, rejected_mask)
                ref_rejected_logps = get_batch_logps(ref_rejected_logits, rejected_labels)

            # -- DPO loss --
            loss = compute_dpo_loss(
                policy_chosen_logps=policy_chosen_logps,
                policy_rejected_logps=policy_rejected_logps,
                reference_chosen_logps=ref_chosen_logps,
                reference_rejected_logps=ref_rejected_logps,
                beta=beta,
            )

            scaled_loss = loss / gradient_accumulation_steps
            scaled_loss.backward()

            step_loss += loss.item()

            # Compute reward margins for logging.
            with torch.no_grad():
                chosen_reward = beta * (
                    policy_chosen_logps - ref_chosen_logps
                ).mean().item()
                rejected_reward = beta * (
                    policy_rejected_logps - ref_rejected_logps
                ).mean().item()
                step_chosen_reward += chosen_reward
                step_rejected_reward += rejected_reward

        # Gradient clipping.
        if cfg.training.grad_clip > 0:
            if hasattr(policy_model, "clip_grad_norm_"):
                policy_model.clip_grad_norm_(cfg.training.grad_clip)
            else:
                torch.nn.utils.clip_grad_norm_(
                    policy_model.parameters(), cfg.training.grad_clip,
                )

        optimizer.step()
        scheduler.step()

        avg_step_loss = step_loss / gradient_accumulation_steps
        avg_chosen_reward = step_chosen_reward / gradient_accumulation_steps
        avg_rejected_reward = step_rejected_reward / gradient_accumulation_steps
        reward_margin = avg_chosen_reward - avg_rejected_reward

        total_loss += avg_step_loss
        total_rewards += reward_margin
        log_count += 1

        metrics = {
            "train/dpo_loss": avg_step_loss,
            "train/chosen_reward": avg_chosen_reward,
            "train/rejected_reward": avg_rejected_reward,
            "train/reward_margin": reward_margin,
            "train/lr": scheduler.get_last_lr()[0],
            "train/step": step,
        }

        if step % cfg.logging.log_every == 0:
            avg = total_loss / max(log_count, 1)
            print_rank0(
                f"Step {step}/{total_steps} | "
                f"dpo_loss: {avg_step_loss:.6f} | "
                f"avg_loss: {avg:.6f} | "
                f"reward_margin: {reward_margin:.4f} | "
                f"chosen: {avg_chosen_reward:.4f} | "
                f"rejected: {avg_rejected_reward:.4f} | "
                f"lr: {scheduler.get_last_lr()[0]:.2e}"
            )
            total_loss = 0.0
            total_rewards = 0.0
            log_count = 0

        for cb in callbacks:
            cb.on_step_end(
                step=step,
                metrics=metrics,
                model=policy_model,
                optimizer=optimizer,
                scheduler=scheduler,
            )

    # -- Final save --
    output_dir = cfg.checkpoint.output_dir
    if is_main_process():
        final_dir = os.path.join(output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)

        # Get the unwrapped model for saving.
        if hasattr(policy_model, "module"):
            save_model = policy_model.module
        else:
            save_model = policy_model
        torch.save(save_model.state_dict(), os.path.join(final_dir, "model.pt"))
        logger.info("Saved final DPO model to %s", final_dir)

    print_rank0("DPO training complete.")


if __name__ == "__main__":
    config = parse_args()
    train(config)
