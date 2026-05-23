"""Streaming pretrain dataset for DeepSleep — bilingual Chinese + English.

Supports two data source modes:
  1. Single dataset: pass dataset_name="some/repo" (legacy CCI4.0-HQ compat)
  2. Bilingual mix (default): alternates SkyPile-150B (Chinese) + OpenWebText (English)
     with configurable zh_ratio (default 0.7 = 70% Chinese, 30% English)

Supports two sequence modes:
  - pack_sequences=True (default): packs multiple docs into fixed-length
    sequences with zero padding. Maximizes throughput and enables Flash
    Attention on every step. Cross-document positions are masked in labels.
  - pack_sequences=False: pads each doc to max_length with attention_mask.

Usage:
    from dataset.streaming_dataset import CCI4PretrainDataset

    # Bilingual mix (default)
    train_ds = CCI4PretrainDataset(tokenizer, max_length=2048, seed=42)

    # Single dataset (legacy)
    train_ds = CCI4PretrainDataset(tokenizer, max_length=2048,
                                   dataset_name="Skywork/SkyPile-150B")
"""

import random
import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset


class CCI4PretrainDataset(IterableDataset):
    """Streaming pretrain dataset with bilingual Chinese + English support.

    Tokenizes on-the-fly. Each __iter__ creates a fresh streaming
    iterator from HuggingFace so eval datasets can be iterated multiple
    times with deterministic output (fixed shuffle seed).
    """

    # Default bilingual sources
    ZH_DATASET = "Skywork/SkyPile-150B"
    EN_DATASET = "Skylion007/openwebtext"

    def __init__(
        self,
        tokenizer,
        max_length: int = 2048,
        dataset_name: str | None = None,
        seed: int = 42,
        num_samples: int | None = None,
        min_length: int = 50,
        text_field: str = "text",
        pack_sequences: bool = True,
        zh_ratio: float = 0.7,
        bilingual: bool = True,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset_name = dataset_name
        self.seed = seed
        self.num_samples = num_samples
        self.min_length = min_length
        self.text_field = text_field
        self.pack_sequences = pack_sequences
        self.zh_ratio = zh_ratio
        self.bilingual = bilingual and dataset_name is None

        self.bos_id = tokenizer.bos_token_id or tokenizer.pad_token_id
        self.eos_id = tokenizer.eos_token_id or tokenizer.pad_token_id
        self.pad_id = tokenizer.pad_token_id or 0

    def _create_stream(self, dataset_name: str = None):
        """Create a fresh streaming iterator from HuggingFace."""
        name = dataset_name or self.dataset_name or self.ZH_DATASET
        ds = load_dataset(
            name,
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
        ds = ds.shuffle(buffer_size=10_000, seed=self.seed)
        return ds

    def _create_bilingual_stream(self):
        """Interleave Chinese and English streams at document level."""
        rng = random.Random(self.seed)
        zh_stream = iter(self._create_stream(self.ZH_DATASET))
        en_stream = iter(self._create_stream(self.EN_DATASET))

        while True:
            is_zh = rng.random() < self.zh_ratio
            try:
                if is_zh:
                    item = next(zh_stream)
                else:
                    item = next(en_stream)
                yield item
            except StopIteration:
                # One stream exhausted, switch to the other
                try:
                    if is_zh:
                        yield next(en_stream)
                    else:
                        yield next(zh_stream)
                except StopIteration:
                    return

    def _tokenize_doc(self, text: str) -> list[int]:
        """Tokenize a single document with BOS/EOS."""
        tokens = self.tokenizer(
            text,
            add_special_tokens=False,
            max_length=self.max_length - 2,
            truncation=True,
        ).input_ids
        return [self.bos_id] + tokens + [self.eos_id]

    def __iter__(self):
        if self.bilingual:
            stream = self._create_bilingual_stream()
        else:
            stream = self._create_stream()

        if self.pack_sequences:
            yield from self._iter_packed(stream)
        else:
            yield from self._iter_padded(stream)

    # ------------------------------------------------------------------
    # Mode 1: Packing (default, industrial standard)
    # ------------------------------------------------------------------

    def _iter_packed(self, stream):
        """Pack multiple docs into fixed-length sequences (zero padding).

        Cross-document boundary positions are masked with label=-100 so
        the model never predicts the start of a new document from the end
        of the previous one.
        """
        yielded = 0
        buffer_ids: list[int] = []

        for item in stream:
            text = item.get(self.text_field, "")
            if not text or len(text.strip()) < self.min_length:
                continue

            tokens = self._tokenize_doc(text)
            buffer_ids.extend(tokens)

            while len(buffer_ids) >= self.max_length:
                chunk = buffer_ids[: self.max_length]
                buffer_ids = buffer_ids[self.max_length :]

                # Mask labels at document boundaries: any BOS after
                # position 0 marks a new document start.
                labels = list(chunk)
                for i in range(1, len(labels)):
                    if chunk[i] == self.bos_id:
                        labels[i] = -100

                yield {
                    "input_ids": torch.tensor(chunk, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                }
                yielded += 1
                if self.num_samples is not None and yielded >= self.num_samples:
                    return

    # ------------------------------------------------------------------
    # Mode 2: Padding (fallback, per-doc padding + attention_mask)
    # ------------------------------------------------------------------

    def _iter_padded(self, stream):
        """Pad each doc to max_length with attention_mask."""
        yielded = 0

        for item in stream:
            text = item.get(self.text_field, "")
            if not text or len(text.strip()) < self.min_length:
                continue

            tokens = self._tokenize_doc(text)
            seq_len = len(tokens)

            input_ids = tokens + [self.pad_id] * (self.max_length - seq_len)
            labels = tokens + [-100] * (self.max_length - seq_len)
            attention_mask = [1] * seq_len + [0] * (self.max_length - seq_len)

            yield {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }
            yielded += 1
            if self.num_samples is not None and yielded >= self.num_samples:
                return
