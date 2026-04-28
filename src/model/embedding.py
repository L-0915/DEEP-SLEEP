"""Embedding layer for the DeepSleep model.

Provides the token embedding layer used at the input of the transformer.
Positional encoding is handled separately by RoPE in the attention layer.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class DeepSleepEmbedding(nn.Module):
    """Token embedding layer for the DeepSleep model.

    Maps input token IDs to dense vectors.  Positional information is not
    handled here but rather by Rotary Position Embeddings (RoPE) applied
    inside the attention layer.

    Args:
        config: Model configuration with ``vocab_size`` and ``d_model`` attributes.
        padding_idx: Optional index for the padding token. Embeddings at this
            index are initialized to zero and remain fixed during training.
    """

    def __init__(self, config, padding_idx: Optional[int] = None) -> None:  # type: ignore[no-untyped-def]
        super().__init__()
        self.embed_tokens = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model,
            padding_idx=padding_idx,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Map token IDs to embedding vectors.

        Args:
            input_ids: Token ID tensor of shape ``(batch, seq_len)``.

        Returns:
            Embedding tensor of shape ``(batch, seq_len, d_model)``.
        """
        return self.embed_tokens(input_ids)
