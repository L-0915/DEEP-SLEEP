"""Custom training callbacks for DeepSleep.

Provides callback classes that plug into the training loops of the pretrain,
SFT, and DPO scripts.  Each callback implements a well-defined interface
so that training logic stays decoupled from monitoring, checkpointing, and
logging concerns.

Callbacks are invoked at fixed intervals (measured in training steps) from
the main training loop.
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from src.utils.checkpoint import cleanup_old_checkpoints, save_checkpoint
from src.utils.distributed import barrier, get_rank, is_main_process
from src.utils.logging import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class TrainingCallback(ABC):
    """Abstract base class for training callbacks."""

    @abstractmethod
    def on_step_end(
        self,
        step: int,
        metrics: Dict[str, float],
        model: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Called at the end of every training step."""
        ...


# ---------------------------------------------------------------------------
# Checkpoint callback
# ---------------------------------------------------------------------------


class CheckpointCallback(TrainingCallback):
    """Save model checkpoints at regular intervals.

    Keeps only the *keep_last_n* most recent checkpoints and cleans up
    older ones automatically.

    Args:
        output_dir: Root directory for checkpoint storage.
        save_every: Number of steps between saves.
        keep_last_n: Number of checkpoints to retain.
    """

    def __init__(
        self,
        output_dir: str,
        save_every: int = 1000,
        keep_last_n: int = 3,
    ) -> None:
        self._output_dir = output_dir
        self._save_every = save_every
        self._keep_last_n = keep_last_n

    def on_step_end(
        self,
        step: int,
        metrics: Dict[str, float],
        model: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        if step == 0 or step % self._save_every != 0:
            return
        if model is None or optimizer is None or scheduler is None:
            return

        loss = metrics.get("loss", float("inf"))
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=step,
            loss=loss,
            output_dir=self._output_dir,
        )
        # Prune old checkpoints (only on main process).
        if is_main_process():
            cleanup_old_checkpoints(self._output_dir, keep_last_n=self._keep_last_n)


# ---------------------------------------------------------------------------
# Evaluation callback
# ---------------------------------------------------------------------------


class EvalCallback(TrainingCallback):
    """Run evaluation at regular intervals.

    Computes the validation loss over a configurable number of batches and
    optionally generates sample text.

    Args:
        eval_dataloader: Validation ``DataLoader``.
        eval_every: Number of training steps between evaluations.
        num_eval_batches: Maximum number of batches to evaluate.
            ``None`` means evaluate the full validation set.
        max_new_tokens: Number of tokens to generate for sample inspection.
    """

    def __init__(
        self,
        eval_dataloader: Any,
        eval_every: int = 500,
        num_eval_batches: Optional[int] = 20,
        max_new_tokens: int = 64,
    ) -> None:
        self._eval_dataloader = eval_dataloader
        self._eval_every = eval_every
        self._num_eval_batches = num_eval_batches
        self._max_new_tokens = max_new_tokens

    def on_step_end(
        self,
        step: int,
        metrics: Dict[str, float],
        model: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        if step == 0 or step % self._eval_every != 0:
            return
        if model is None:
            return

        model.eval()
        total_loss = 0.0
        count = 0

        with torch.no_grad():
            for i, batch in enumerate(self._eval_dataloader):
                if self._num_eval_batches is not None and i >= self._num_eval_batches:
                    break

                # Move batch to the same device as the model.
                device = next(model.parameters()).device
                input_ids = batch["input_ids"].to(device)
                labels = batch.get("labels", input_ids).to(device)

                outputs = model(input_ids=input_ids, labels=labels)
                total_loss += outputs.loss.item()
                count += 1

        avg_loss = total_loss / max(count, 1)
        metrics["eval_loss"] = avg_loss
        logger.info("Eval step %d: eval_loss=%.6f", step, avg_loss)

        model.train()


# ---------------------------------------------------------------------------
# Wandb callback
# ---------------------------------------------------------------------------


class WandbCallback(TrainingCallback):
    """Log training metrics to Weights & Biases.

    Logs the provided metrics dictionary on every step and periodically logs
    additional information such as gradient norms and MoE statistics.

    Args:
        project: Wandb project name.
        name: Wandb run name (optional).
        config: Configuration dict to attach to the run.
        log_every: Log every N steps (default 1).
    """

    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        log_every: int = 1,
    ) -> None:
        self._project = project
        self._name = name
        self._config = config
        self._log_every = log_every
        self._initialized = False
        self._step_offset = 0

    def _lazy_init(self) -> None:
        """Initialize wandb only once and only on the main process."""
        if self._initialized:
            return
        self._initialized = True

        if not is_main_process():
            return

        try:
            import wandb

            wandb.init(
                project=self._project,
                name=self._name,
                config=self._config,
            )
        except ImportError:
            logger.warning(
                "wandb is not installed.  Skipping wandb logging."
            )

    def on_step_end(
        self,
        step: int,
        metrics: Dict[str, float],
        **kwargs: Any,
    ) -> None:
        self._lazy_init()

        if not is_main_process():
            return
        if step % self._log_every != 0:
            return

        try:
            import wandb

            if wandb.run is None:
                return
            wandb.log(metrics, step=step + self._step_offset)
        except ImportError:
            pass

    def log_generation(
        self,
        step: int,
        prompt: str,
        generated_text: str,
    ) -> None:
        """Log a sample generation to wandb as a text table entry.

        Args:
            step: Current training step.
            prompt: The input prompt.
            generated_text: The model's completion.
        """
        if not is_main_process():
            return

        try:
            import wandb

            if wandb.run is None:
                return
            wandb.log(
                {
                    "generation/step": step,
                    "generation/prompt": prompt,
                    "generation/completion": generated_text,
                },
                step=step + self._step_offset,
            )
        except ImportError:
            pass
