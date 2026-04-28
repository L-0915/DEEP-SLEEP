"""Direct Preference Optimization (DPO) dataset for DeepSleep.

Loads preference pairs from JSONL files.  Each row contains a prompt together
with a chosen (preferred) and rejected response.  The dataset formats these
into three tokenized sequences (prompt, chosen, rejected) that are later
consumed by the DPO training loop.

Supported input format::

    {"prompt": "...", "chosen": "...", "rejected": "..."}

Alternatively, conversations can be provided under ``"prompt"``,
``"chosen"``, and ``"rejected"`` keys, each as a list of message dicts
compatible with ChatML formatting.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch.utils.data import Dataset

from src.utils.logging import setup_logger

logger = setup_logger(__name__)


class DPODataset(Dataset):
    """Dataset for Direct Preference Optimization.

    Args:
        data_path: Path to a JSONL file or directory of JSONL files.
            Each line must have ``"prompt"``, ``"chosen"``, and ``"rejected"``
            keys.
        tokenizer: A HuggingFace tokenizer instance.
        max_length: Maximum token length per sequence (prompt + response).
        max_prompt_length: Maximum token length for the prompt portion.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: Any,
        max_length: int = 4096,
        max_prompt_length: int = 1024,
    ) -> None:
        self._tokenizer = tokenizer
        self._max_length = max_length
        self._max_prompt_length = max_prompt_length

        self._pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        self._samples: List[Dict[str, Any]] = []
        self._load_data(data_path)

        if not self._samples:
            raise ValueError(f"No valid DPO samples found in {data_path}")

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_data(self, data_path: str) -> None:
        """Load preference pairs from JSONL file(s).

        Args:
            data_path: Path to a single JSONL file or a directory.
        """
        paths: List[str] = []
        if os.path.isdir(data_path):
            for fname in sorted(os.listdir(data_path)):
                if fname.endswith(".jsonl"):
                    paths.append(os.path.join(data_path, fname))
        elif os.path.isfile(data_path) and data_path.endswith(".jsonl"):
            paths.append(data_path)
        else:
            raise FileNotFoundError(f"No JSONL files found at {data_path}")

        for path in paths:
            with open(path, "r", encoding="utf-8") as fh:
                for line_num, line in enumerate(fh, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        logger.warning(
                            "Skipping invalid JSON on line %d of %s: %s",
                            line_num, path, exc,
                        )
                        continue

                    if not all(k in record for k in ("prompt", "chosen", "rejected")):
                        logger.warning(
                            "Skipping line %d of %s: missing required keys.",
                            line_num, path,
                        )
                        continue

                    self._samples.append(record)

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    def _format_prompt(self, prompt: Any) -> str:
        """Convert a prompt (string or message list) to a string.

        Args:
            prompt: Either a plain string or a list of message dicts.

        Returns:
            Formatted prompt string.
        """
        if isinstance(prompt, str):
            return prompt
        if isinstance(prompt, list):
            if hasattr(self._tokenizer, "apply_chat_template") and callable(
                self._tokenizer.apply_chat_template
            ):
                return self._tokenizer.apply_chat_template(
                    prompt, tokenize=False, add_generation_prompt=True,
                )
            # Fallback: join the last user message content.
            parts: List[str] = []
            for msg in prompt:
                parts.append(msg.get("content", ""))
            return "\n".join(parts)
        return str(prompt)

    def _format_response(self, response: Any) -> str:
        """Convert a response (string or message list) to a string.

        Args:
            response: Either a plain string or a list of message dicts.

        Returns:
            Formatted response string.
        """
        if isinstance(response, str):
            return response
        if isinstance(response, list):
            parts: List[str] = []
            for msg in response:
                parts.append(msg.get("content", ""))
            return "\n".join(parts)
        return str(response)

    def _tokenize_sequence(
        self,
        prompt_text: str,
        response_text: str,
    ) -> tuple[List[int], List[int]]:
        """Tokenize a prompt + response pair and split into prompt / response ids.

        The response tokens are also used to compute labels (loss only on the
        response portion).

        Args:
            prompt_text: Formatted prompt string.
            response_text: Formatted response string.

        Returns:
            Tuple of (all_input_ids, labels) where labels has -100 for prompt
            positions.
        """
        # Tokenize prompt (truncated).
        prompt_enc = self._tokenizer(
            prompt_text,
            truncation=True,
            max_length=self._max_prompt_length,
            add_special_tokens=False,
        )
        prompt_ids = prompt_enc["input_ids"]

        # Tokenize response (truncated to remaining budget).
        remaining = self._max_length - len(prompt_ids)
        response_enc = self._tokenizer(
            response_text,
            truncation=True,
            max_length=max(remaining, 1),
            add_special_tokens=False,
        )
        response_ids = response_enc["input_ids"]

        # Combine: prompt + response.
        all_ids = prompt_ids + response_ids

        # Labels: -100 for prompt, actual token ids for response.
        labels = [-100] * len(prompt_ids) + response_ids

        return all_ids, labels

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of preference pairs."""
        return len(self._samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return a single DPO sample with chosen and rejected sequences.

        Args:
            idx: Index into the sample list.

        Returns:
            Dict with:
            * ``prompt_input_ids`` -- tokenized prompt.
            * ``chosen_input_ids`` -- prompt + chosen response.
            * ``chosen_labels`` -- labels (loss only on response).
            * ``rejected_input_ids`` -- prompt + rejected response.
            * ``rejected_labels`` -- labels (loss only on response).
        """
        sample = self._samples[idx]
        prompt_text = self._format_prompt(sample["prompt"])
        chosen_text = self._format_response(sample["chosen"])
        rejected_text = self._format_response(sample["rejected"])

        # Prompt-only tokenization (shared prefix for chosen / rejected).
        prompt_enc = self._tokenizer(
            prompt_text,
            truncation=True,
            max_length=self._max_prompt_length,
            add_special_tokens=False,
        )
        prompt_ids = prompt_enc["input_ids"]

        # Chosen: prompt + chosen response.
        chosen_ids, chosen_labels = self._tokenize_sequence(prompt_text, chosen_text)

        # Rejected: prompt + rejected response.
        rejected_ids, rejected_labels = self._tokenize_sequence(prompt_text, rejected_text)

        return {
            "prompt_input_ids": torch.tensor(prompt_ids, dtype=torch.long),
            "chosen_input_ids": torch.tensor(chosen_ids, dtype=torch.long),
            "chosen_labels": torch.tensor(chosen_labels, dtype=torch.long),
            "rejected_input_ids": torch.tensor(rejected_ids, dtype=torch.long),
            "rejected_labels": torch.tensor(rejected_labels, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------


class DPODataCollator:
    """Custom collate function for DPO batch preparation.

    Pads sequences within each batch to the longest sequence in that batch
    (right-side padding) and constructs attention masks.

    Args:
        pad_token_id: Token ID used for padding (default 0).
        ignore_index: Label value for non-loss positions (default -100).
    """

    def __init__(
        self,
        pad_token_id: int = 0,
        ignore_index: int = -100,
    ) -> None:
        self._pad_token_id = pad_token_id
        self._ignore_index = ignore_index

    def _pad_batch(self, sequences: List[torch.Tensor]) -> tuple:
        """Pad a list of 1-D tensors to equal length.

        Args:
            sequences: List of tensors with potentially different lengths.

        Returns:
            Tuple of (padded_tensor, attention_mask).
        """
        max_len = max(s.size(0) for s in sequences)
        batch_size = len(sequences)

        padded = torch.full(
            (batch_size, max_len), self._pad_token_id, dtype=torch.long,
        )
        mask = torch.zeros(batch_size, max_len, dtype=torch.long)

        for i, seq in enumerate(sequences):
            seq_len = seq.size(0)
            padded[i, :seq_len] = seq
            mask[i, :seq_len] = 1

        return padded, mask

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate a list of DPO samples into a batch.

        Args:
            features: List of dicts as returned by ``DPODataset.__getitem__``.

        Returns:
            Batch dict with padded tensors and attention masks.
        """
        prompt_ids = [f["prompt_input_ids"] for f in features]
        chosen_ids = [f["chosen_input_ids"] for f in features]
        chosen_labels = [f["chosen_labels"] for f in features]
        rejected_ids = [f["rejected_input_ids"] for f in features]
        rejected_labels = [f["rejected_labels"] for f in features]

        prompt_padded, prompt_mask = self._pad_batch(prompt_ids)
        chosen_padded, chosen_mask = self._pad_batch(chosen_ids)
        rejected_padded, rejected_mask = self._pad_batch(rejected_ids)
        chosen_labels_padded, _ = self._pad_batch(chosen_labels)
        rejected_labels_padded, _ = self._pad_batch(rejected_labels)

        # Replace padding positions in labels with ignore_index.
        chosen_labels_padded[chosen_labels_padded == self._pad_token_id] = self._ignore_index
        rejected_labels_padded[
            rejected_labels_padded == self._pad_token_id
        ] = self._ignore_index

        return {
            "prompt_input_ids": prompt_padded,
            "prompt_attention_mask": prompt_mask,
            "chosen_input_ids": chosen_padded,
            "chosen_attention_mask": chosen_mask,
            "chosen_labels": chosen_labels_padded,
            "rejected_input_ids": rejected_padded,
            "rejected_attention_mask": rejected_mask,
            "rejected_labels": rejected_labels_padded,
        }
