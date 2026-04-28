"""Distributed training utilities for the DeepSleep project.

Provides thin wrappers around ``torch.distributed`` so that training scripts
can operate in both single-GPU and multi-GPU (DDP / FSDP) configurations
without sprinkling conditional logic throughout the codebase.

Usage::

    from src.utils.distributed import setup_distributed, get_rank, is_main_process

    setup_distributed()
    if is_main_process():
        print("This only appears on rank 0.")
"""

from __future__ import annotations

import os
import torch
import torch.distributed as dist
from typing import Optional


def setup_distributed() -> None:
    """Initialize the default process group.

    Respects the standard environment variables set by ``torchrun`` /
    ``torch.distributed.launch``:

    * ``RANK`` -- global rank of this process
    * ``WORLD_SIZE`` -- total number of processes
    * ``LOCAL_RANK`` -- local rank on the current node
    * ``MASTER_ADDR`` -- address of the rank-0 process
    * ``MASTER_PORT`` -- free port for communication

    If ``RANK`` is not set the function assumes single-GPU mode and returns
    without calling ``dist.init_process_group``.
    """
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if not dist.is_initialized() and world_size > 1:
        dist.init_process_group(backend="nccl")
        # Ensure every process uses the correct GPU.
        torch.cuda.set_device(local_rank)


def get_rank() -> int:
    """Return the global rank of the current process.

    Returns ``0`` when ``torch.distributed`` is not initialized (single-GPU).
    """
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    """Return the total number of processes.

    Returns ``1`` when ``torch.distributed`` is not initialized (single-GPU).
    """
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_local_rank() -> int:
    """Return the local rank on the current node.

    Returns ``0`` when ``torch.distributed`` is not initialized.
    """
    return int(os.environ.get("LOCAL_RANK", 0))


def is_main_process() -> bool:
    """Return ``True`` if this is the rank-0 process."""
    return get_rank() == 0


def barrier() -> None:
    """Synchronize all processes (no-op in single-GPU mode)."""
    if dist.is_initialized():
        dist.barrier()


def all_reduce(
    tensor: torch.Tensor,
    op: dist.ReduceOp = dist.ReduceOp.SUM,
) -> None:
    """Perform an in-place all-reduce on *tensor*.

    Args:
        tensor: Tensor to reduce in-place.
        op: Reduction operation (default ``SUM``).
    """
    if dist.is_initialized():
        dist.all_reduce(tensor, op=op)


def all_gather(tensor: torch.Tensor) -> torch.Tensor:
    """Gather *tensor* from all ranks into a list and concatenate.

    Args:
        tensor: Tensor of shape ``(n, ...)`` on the local rank.

    Returns:
        Concatenated tensor of shape ``(world_size * n, ...)``.
    """
    if not dist.is_initialized():
        return tensor
    gathered = [torch.zeros_like(tensor) for _ in range(get_world_size())]
    dist.all_gather(gathered, tensor)
    return torch.cat(gathered, dim=0)


def print_rank0(msg: str) -> None:
    """Print *msg* only on the main process (rank 0).

    Args:
        msg: Message to print.
    """
    if is_main_process():
        print(msg, flush=True)
