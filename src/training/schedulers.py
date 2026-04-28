"""Custom learning-rate schedulers for DeepSleep training.

Provides cosine-decay schedulers commonly used in LLM pretraining and
fine-tuning.  These are lightweight alternatives to the HuggingFace
``get_cosine_schedule_with_warmup`` with a few project-specific defaults
(e.g. minimum LR floor).

Usage::

    from src.training.schedulers import get_cosine_schedule_with_warmup

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=500,
        num_training_steps=100_000,
    )
"""

from __future__ import annotations

import math
from typing import List

import torch
from torch.optim.lr_scheduler import LambdaLR


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
    num_cycles: float = 0.5,
) -> LambdaLR:
    """Create a cosine-annealing schedule with linear warmup.

    During the warmup phase (steps ``0 .. num_warmup_steps``) the LR is
    linearly increased from 0 to the optimizer's base LR.  After warmup the LR
    follows a cosine curve that decays to *min_lr_ratio* of the base LR.

    Args:
        optimizer: Wrapped optimizer.
        num_warmup_steps: Number of steps for the linear warmup phase.
        num_training_steps: Total number of training steps.
        min_lr_ratio: The minimum LR as a fraction of the peak LR.  A value
            of ``0.1`` means the LR decays to 10 % of its peak value.
        num_cycles: Number of cosine cycles in the decay phase.  ``0.5``
            corresponds to a single half-cosine (standard in LLM training).

    Returns:
        A ``LambdaLR`` scheduler instance.
    """

    def lr_lambda(current_step: int) -> float:
        # Warmup phase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # Cosine decay phase
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        # Scale the cosine output from [0, 1] -> [min_lr_ratio, 1.0]
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * num_cycles * progress))
        return max(min_lr_ratio, (1.0 - min_lr_ratio) * cosine_decay + min_lr_ratio)

    return LambdaLR(optimizer, lr_lambda)


def get_linear_warmup_cosine_decay(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0,
) -> LambdaLR:
    """Create a linear warmup followed by a cosine decay to zero (or min ratio).

    This is a simpler variant of ``get_cosine_schedule_with_warmup`` where the
    LR decays all the way to *min_lr_ratio* (default ``0.0``) rather than
    stopping at a floor.

    Args:
        optimizer: Wrapped optimizer.
        num_warmup_steps: Number of warmup steps.
        num_training_steps: Total number of training steps.
        min_lr_ratio: Fraction of base LR to decay to at the end.

    Returns:
        A ``LambdaLR`` scheduler instance.
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        # Cosine goes from 1 -> min_lr_ratio
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine_decay)

    return LambdaLR(optimizer, lr_lambda)
