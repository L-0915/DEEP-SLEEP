"""Checkpoint management for DeepSleep training.

Provides functions for saving, loading, and pruning model checkpoints.
Checkpoints store the full training state (model weights, optimizer state,
LR scheduler state, and metadata) so that training can be resumed from any
saved step.

Directory layout::

    output_dir/
        checkpoint-0001000/
            model.pt
            optimizer.pt
            scheduler.pt
            metadata.json
        checkpoint-0002000/
            ...
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from src.utils.distributed import barrier, get_rank, is_main_process
from src.utils.logging import setup_logger

logger = setup_logger(__name__)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    step: int,
    loss: float,
    output_dir: str,
    fsdp_state: Optional[Dict[str, Any]] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Save a full training checkpoint.

    The checkpoint directory is named ``checkpoint-<step padded to 10 digits>``.
    Only the main process writes to disk; other ranks synchronize via a
    barrier before returning.

    Args:
        model: The model to save.  If wrapped in FSDP, a consolidated
            ``state_dict`` is gathered before saving.
        optimizer: Optimizer whose state to persist.
        scheduler: LR scheduler whose state to persist.
        step: Current global training step.
        loss: Training loss at this step (scalar).
        output_dir: Root directory for checkpoints.
        fsdp_state: Optional FSDP state dict to save alongside the model.
        extra_metadata: Optional additional key-value pairs stored in
            ``metadata.json``.

    Returns:
        Absolute path to the checkpoint directory.
    """
    if is_main_process():
        ckpt_dir = os.path.join(output_dir, f"checkpoint-{step:010d}")
        os.makedirs(ckpt_dir, exist_ok=True)

        # -- Model state --
        model_path = os.path.join(ckpt_dir, "model.pt")
        torch.save(model.state_dict(), model_path)

        # -- Optimizer state --
        optimizer_path = os.path.join(ckpt_dir, "optimizer.pt")
        torch.save(optimizer.state_dict(), optimizer_path)

        # -- Scheduler state --
        scheduler_path = os.path.join(ckpt_dir, "scheduler.pt")
        torch.save(scheduler.state_dict(), scheduler_path)

        # -- FSDP state (optional) --
        if fsdp_state is not None:
            fsdp_path = os.path.join(ckpt_dir, "fsdp.pt")
            torch.save(fsdp_state, fsdp_path)

        # -- Metadata --
        metadata: Dict[str, Any] = {
            "step": step,
            "loss": loss,
        }
        if extra_metadata is not None:
            metadata.update(extra_metadata)
        metadata_path = os.path.join(ckpt_dir, "metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)

        logger.info("Saved checkpoint to %s (step=%d, loss=%.6f)", ckpt_dir, step, loss)

    barrier()
    return os.path.join(output_dir, f"checkpoint-{step:010d}")


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    checkpoint_dir: str = "",
    map_location: str = "cpu",
) -> Dict[str, Any]:
    """Load a training checkpoint.

    Args:
        model: Model whose weights should be restored.
        optimizer: Optimizer to restore state for.  Pass ``None`` to skip.
        scheduler: LR scheduler to restore state for.  Pass ``None`` to skip.
        checkpoint_dir: Path to the checkpoint directory (containing
            ``model.pt``, ``optimizer.pt``, ``scheduler.pt``,
            ``metadata.json``).
        map_location: Device mapping for ``torch.load``.

    Returns:
        Metadata dictionary parsed from ``metadata.json``.
    """
    metadata_path = os.path.join(checkpoint_dir, "metadata.json")
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(
            f"Metadata file not found at {metadata_path}.  "
            f"Is '{checkpoint_dir}' a valid checkpoint directory?"
        )

    with open(metadata_path, "r", encoding="utf-8") as fh:
        metadata = json.load(fh)

    # -- Model --
    model_path = os.path.join(checkpoint_dir, "model.pt")
    model_state = torch.load(model_path, map_location=map_location, weights_only=True)
    model.load_state_dict(model_state)
    logger.info("Loaded model weights from %s", model_path)

    # -- Optimizer --
    if optimizer is not None:
        optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt")
        if os.path.isfile(optimizer_path):
            optimizer_state = torch.load(
                optimizer_path, map_location=map_location, weights_only=True,
            )
            optimizer.load_state_dict(optimizer_state)
            logger.info("Loaded optimizer state from %s", optimizer_path)
        else:
            logger.warning("Optimizer checkpoint not found at %s -- skipping.", optimizer_path)

    # -- Scheduler --
    if scheduler is not None:
        scheduler_path = os.path.join(checkpoint_dir, "scheduler.pt")
        if os.path.isfile(scheduler_path):
            scheduler_state = torch.load(
                scheduler_path, map_location=map_location, weights_only=True,
            )
            scheduler.load_state_dict(scheduler_state)
            logger.info("Loaded scheduler state from %s", scheduler_path)
        else:
            logger.warning("Scheduler checkpoint not found at %s -- skipping.", scheduler_path)

    logger.info(
        "Resumed from checkpoint %s (step=%d, loss=%.6f)",
        checkpoint_dir,
        metadata.get("step", -1),
        metadata.get("loss", float("inf")),
    )
    return metadata


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find the most recent checkpoint subdirectory inside *checkpoint_dir*.

    Scans for directories matching the pattern ``checkpoint-NNNNNNNNNN`` and
    returns the one with the largest step number.

    Args:
        checkpoint_dir: Root directory that contains checkpoint subdirectories.

    Returns:
        Absolute path to the latest checkpoint directory, or ``None`` if no
        checkpoints are found.
    """
    if not os.path.isdir(checkpoint_dir):
        return None

    ckpt_pattern = re.compile(r"^checkpoint-(\d{10})$")
    candidates: list[tuple[int, str]] = []

    for entry in os.listdir(checkpoint_dir):
        full_path = os.path.join(checkpoint_dir, entry)
        if not os.path.isdir(full_path):
            continue
        match = ckpt_pattern.match(entry)
        if match:
            step = int(match.group(1))
            candidates.append((step, full_path))

    if not candidates:
        return None

    candidates.sort(key=lambda pair: pair[0])
    return candidates[-1][1]


def cleanup_old_checkpoints(
    checkpoint_dir: str,
    keep_last_n: int = 3,
) -> int:
    """Remove all but the *keep_last_n* most recent checkpoints.

    Args:
        checkpoint_dir: Root directory containing checkpoint subdirectories.
        keep_last_n: Number of checkpoints to retain.

    Returns:
        Number of checkpoint directories that were removed.
    """
    if not os.path.isdir(checkpoint_dir):
        return 0

    ckpt_pattern = re.compile(r"^checkpoint-(\d{10})$")
    candidates: list[tuple[int, str]] = []

    for entry in os.listdir(checkpoint_dir):
        full_path = os.path.join(checkpoint_dir, entry)
        if not os.path.isdir(full_path):
            continue
        match = ckpt_pattern.match(entry)
        if match:
            step = int(match.group(1))
            candidates.append((step, full_path))

    if len(candidates) <= keep_last_n:
        return 0

    candidates.sort(key=lambda pair: pair[0])
    to_remove = candidates[: len(candidates) - keep_last_n]

    removed = 0
    for _, path in to_remove:
        shutil.rmtree(path, ignore_errors=True)
        removed += 1
        logger.info("Removed old checkpoint: %s", path)

    return removed
