"""Pretraining dataset for DeepSleep.

Loads tokenized data from memory-mapped ``.bin`` files (``uint16`` format) and
produces fixed-length sequences suitable for next-token prediction.  Multiple
shard files within a directory are concatenated transparently.

Each sample is a dict with:
* ``input_ids``  -- ``torch.LongTensor`` of length ``seq_length``
* ``labels``     -- ``input_ids`` shifted left by one position

The dataset supports deterministic sharding across distributed ranks so that
each rank sees a disjoint subset of the data.
"""

from __future__ import annotations

import glob
import os
import struct
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class PretrainDataset(Dataset):
    """Memory-mapped pretraining dataset backed by ``.bin`` token files.

    The data directory may contain one or more shard files named
    ``*.bin``.  All shards are concatenated in lexicographic order to form a
    single contiguous stream of ``uint16`` token IDs.

    Args:
        data_dir: Directory containing ``*.bin`` files.
        seq_length: Length of each training sequence (context window).
        split: One of ``"train"`` or ``"val"``.  When ``"val"``, the last 5 %%
            of tokens (by position) are reserved for validation.
        seed: Random seed for shuffling (deterministic across runs).
        rank: Distributed rank for data-parallel sharding.  ``0`` for single-GPU.
        world_size: Total number of distributed workers.
    """

    def __init__(
        self,
        data_dir: str,
        seq_length: int = 8192,
        split: str = "train",
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got '{split}'")
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        self._seq_length = seq_length
        self._seed = seed
        self._rank = rank
        self._world_size = world_size

        # Discover and sort shard files.
        shard_pattern = os.path.join(data_dir, "*.bin")
        shard_paths = sorted(glob.glob(shard_pattern))
        if not shard_paths:
            raise FileNotFoundError(
                f"No .bin shard files found in {data_dir}"
            )

        # Memory-map each shard as a read-only numpy array of uint16.
        self._mmaps: List[np.ndarray] = []
        for path in shard_paths:
            mmap = np.memmap(path, dtype=np.uint16, mode="r")
            self._mmaps.append(mmap)

        # Compute cumulative lengths for fast indexing.
        self._shard_lengths = [len(m) for m in self._mmaps]
        self._total_tokens = sum(self._shard_lengths)

        if self._total_tokens < seq_length + 1:
            raise ValueError(
                f"Total tokens ({self._total_tokens}) is smaller than "
                f"seq_length + 1 ({seq_length + 1}).  Not enough data."
            )

        # Train / val split by token position.
        val_ratio = 0.05
        split_boundary = int(self._total_tokens * (1.0 - val_ratio))
        if split == "val":
            self._start = split_boundary
            self._end = self._total_tokens
        else:
            self._start = 0
            self._end = split_boundary

        # Number of complete sequences available in the split range.
        num_sequences = (self._end - self._start - 1) // seq_length

        # Shard sequences across distributed workers deterministically.
        rng = np.random.RandomState(seed)
        indices = rng.permutation(num_sequences)
        per_worker = len(indices) // world_size
        start_idx = rank * per_worker
        end_idx = (rank + 1) * per_worker if rank < world_size - 1 else len(indices)
        self._sequence_indices = sorted(indices[start_idx:end_idx])

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _read_tokens(self, start: int, length: int) -> np.ndarray:
        """Read *length* contiguous tokens starting at global position *start*.

        Handles cross-shard boundaries transparently.

        Args:
            start: Global token offset.
            length: Number of tokens to read.

        Returns:
            ``np.ndarray`` of ``uint16`` values of length *length*.
        """
        result = np.empty(length, dtype=np.uint16)
        filled = 0
        offset = start

        while filled < length:
            # Find the shard that contains *offset*.
            cumulative = 0
            shard_idx = 0
            for i, slen in enumerate(self._shard_lengths):
                if offset < cumulative + slen:
                    shard_idx = i
                    break
                cumulative += slen
            else:
                shard_idx = len(self._shard_lengths) - 1
                cumulative = sum(self._shard_lengths) - self._shard_lengths[-1]

            local_offset = offset - cumulative
            remaining_in_shard = self._shard_lengths[shard_idx] - local_offset
            to_read = min(length - filled, remaining_in_shard)

            result[filled: filled + to_read] = self._mmaps[shard_idx][
                local_offset: local_offset + to_read
            ]
            filled += to_read
            offset += to_read

        return result

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of training sequences assigned to this worker."""
        return len(self._sequence_indices)

    def __getitem__(self, idx: int) -> dict:
        """Return a single training sample.

        Args:
            idx: Local index within this worker's shard.

        Returns:
            Dict with ``input_ids`` and ``labels`` tensors of shape
            ``(seq_length,)``.
        """
        global_seq_idx = self._sequence_indices[idx]
        token_start = self._start + global_seq_idx * self._seq_length
        tokens = self._read_tokens(token_start, self._seq_length + 1)

        input_ids = torch.from_numpy(tokens[: self._seq_length].copy()).long()
        labels = torch.from_numpy(tokens[1: self._seq_length + 1].copy()).long()

        return {
            "input_ids": input_ids,
            "labels": labels,
        }
