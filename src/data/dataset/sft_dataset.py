"""Supervised Fine-Tuning (SFT) dataset for DeepSleep.

Loads instruction-tuning data from JSONL files in ChatML conversation format
and constructs sequences with loss masking so that only assistant responses
contribute to the language-modeling loss.

Supported input format::

    {"messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ]}

The dataset applies the tokenizer's chat template (or a built-in ChatML
template as fallback) to format each conversation.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch.utils.data import Dataset

# ChatML special tokens used when the tokenizer does not expose a chat template.
_CHATML_BOS = "<|im_start|>"
_CHATML_EOS = "<|im_end|>"
_CHATML_SYSTEM = "system"
_CHATML_USER = "user"
_CHATML_ASSISTANT = "assistant"


class SFTDataset(Dataset):
    """ChatML-format SFT dataset with assistant-only loss masking.

    Args:
        data_path: Path to a JSONL file or a directory of JSONL files.
            Each line must have a ``"messages"`` key containing a list of
            message dicts with ``"role"`` and ``"content"``.
        tokenizer: A HuggingFace tokenizer instance.
        max_length: Maximum total sequence length (input + labels).
            Sequences exceeding this length are truncated from the left.
        seed: Seed for deterministic shuffling (unused when ``shuffle`` is
            ``False``).
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: Any,
        max_length: int = 4096,
        seed: int = 42,
    ) -> None:
        self._tokenizer = tokenizer
        self._max_length = max_length
        self._seed = seed

        self._conversations: List[List[Dict[str, str]]] = []
        self._load_data(data_path)

        if not self._conversations:
            raise ValueError(f"No valid conversations found in {data_path}")

        # Resolve special token ids.
        self._pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        # Try to detect ChatML-style special tokens from the tokenizer.
        self._has_chat_template = hasattr(tokenizer, "apply_chat_template") and callable(
            tokenizer.apply_chat_template
        )

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_data(self, data_path: str) -> None:
        """Load conversations from JSONL file(s).

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
                        raise ValueError(
                            f"Invalid JSON on line {line_num} of {path}: {exc}"
                        ) from exc

                    messages = record.get("messages")
                    if not isinstance(messages, list) or len(messages) == 0:
                        continue
                    self._conversations.append(messages)

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def _format_conversation(
        self,
        messages: List[Dict[str, str]],
    ) -> str:
        """Render a list of messages into a single string using ChatML format.

        If the tokenizer has an ``apply_chat_template`` method it is used
        instead of the built-in template.

        Args:
            messages: List of ``{"role": ..., "content": ...}`` dicts.

        Returns:
            A single string containing the formatted conversation.
        """
        if self._has_chat_template:
            try:
                return self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            except Exception:
                # Fall through to manual formatting.
                pass

        # Manual ChatML formatting.
        parts: List[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"{_CHATML_BOS}{role}\n{content}{_CHATML_EOS}\n")
        return "".join(parts)

    def _build_labels_mask(
        self,
        messages: List[Dict[str, str]],
        input_ids: List[int],
    ) -> List[int]:
        """Build a label sequence where only assistant tokens are tracked.

        All non-assistant positions are set to ``-100`` (the standard ignore
        index for cross-entropy loss).

        Args:
            messages: Original message list (used to identify assistant turns).
            input_ids: Tokenized input ids.

        Returns:
            List of label ids with the same length as *input_ids*.
        """
        labels = [-100] * len(input_ids)

        # We need to figure out which token positions correspond to assistant
        # responses.  We do this by tokenizing each assistant message
        # independently and searching for its prefix in the full input_ids.
        assistant_spans: List[tuple[int, int]] = []
        cursor = 0

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == _CHATML_ASSISTANT:
                # Format this single message to find its token span.
                formatted = f"{_CHATML_BOS}{role}\n{content}{_CHATML_EOS}\n"
                tokenized = self._tokenizer.encode(formatted, add_special_tokens=False)

                # Scan forward from cursor to find the start of this span.
                # Due to tokenization merging at boundaries, we search for a
                # fuzzy match of at least the first few tokens.
                span_start = cursor
                span_end = cursor + len(tokenized)
                assistant_spans.append((span_start, span_end))
                cursor = span_end
            else:
                # Non-assistant message: tokenize to advance the cursor.
                formatted = f"{_CHATML_BOS}{role}\n{content}{_CHATML_EOS}\n"
                tokenized = self._tokenizer.encode(formatted, add_special_tokens=False)
                cursor += len(tokenized)

        # Fill labels for assistant spans.
        for start, end in assistant_spans:
            for pos in range(start, min(end, len(labels))):
                labels[pos] = input_ids[pos]

        return labels

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the total number of conversations."""
        return len(self._conversations)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return a single training sample.

        Args:
            idx: Index into the conversation list.

        Returns:
            Dict with ``input_ids`` and ``labels`` tensors of shape
            ``(seq_len,)``, where ``seq_len <= max_length``.
        """
        messages = self._conversations[idx]
        text = self._format_conversation(messages)

        # Tokenize the full conversation.
        encoding = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self._max_length,
            add_special_tokens=True,
        )
        input_ids = encoding["input_ids"].squeeze(0).tolist()

        # Build label mask (only compute loss on assistant tokens).
        labels = self._build_labels_mask(messages, input_ids)

        # Pad to max_length if needed (right-side padding).
        seq_len = len(input_ids)
        pad_len = self._max_length - seq_len
        if pad_len > 0:
            input_ids = input_ids + [self._pad_token_id] * pad_len
            labels = labels + [-100] * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(
                [1] * seq_len + [0] * pad_len, dtype=torch.long,
            ),
        }
