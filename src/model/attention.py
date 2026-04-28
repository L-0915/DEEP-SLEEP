"""Multi-head attention with Grouped Query Attention (GQA) and Flash Attention.

Implements the attention mechanism used in DeepSleep, featuring:
- Grouped Query Attention (GQA) with configurable head counts
- Flash Attention 2 integration with fallback to manual SDPA
- Rotary Position Embedding (RoPE) integration
- KV-cache support for autoregressive generation
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import DeepSleepRotaryEmbedding, apply_rotary_pos_emb

logger = logging.getLogger(__name__)

# Check if flash-attn is available
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import pad_input, unpad_input

    _FLASH_ATTN_AVAILABLE = True
except ImportError:
    _FLASH_ATTN_AVAILABLE = False

# Optional cache for flash-attn backend availability (lazy evaluation)
_USE_FLASH_ATTN: Optional[bool] = None


def _can_use_flash_attn() -> bool:
    """Check whether Flash Attention 2 can be used at runtime.

    Returns:
        True if flash-attn is importable, False otherwise.
    """
    global _USE_FLASH_ATTN
    if _USE_FLASH_ATTN is None:
        _USE_FLASH_ATTN = _FLASH_ATTN_AVAILABLE
        if not _USE_FLASH_ATTN:
            logger.info(
                "flash-attn is not available. Falling back to "
                "torch.nn.functional.scaled_dot_product_attention."
            )
    return _USE_FLASH_ATTN


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand key/value tensors to match the number of query heads for GQA.

    This broadcasts the key/value heads so that each key/value head is
    repeated ``n_rep`` times along the head dimension.

    Args:
        hidden_states: Tensor of shape ``(batch, num_kv_heads, seq_len, head_dim)``.
        n_rep: Number of repetitions (n_heads // n_kv_heads).

    Returns:
        Expanded tensor of shape ``(batch, num_heads, seq_len, head_dim)``.
    """
    if n_rep == 1:
        return hidden_states

    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
    # Expand: (batch, num_kv_heads, seq_len, head_dim)
    # -> (batch, num_kv_heads, n_rep, seq_len, head_dim)
    # -> (batch, num_kv_heads * n_rep, seq_len, head_dim)
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, seq_len, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)


class DeepSleepAttention(nn.Module):
    """Grouped Query Attention (GQA) layer with Flash Attention support.

    Uses separate projections for Q, K, V with configurable head counts.
    Query heads are grouped with key/value heads for efficiency:
    ``num_key_value_groups = n_heads // n_kv_heads``.

    Args:
        config: Model configuration with attention-related attributes.
        layer_idx: Index of this attention layer (used for caching).
    """

    def __init__(self, config, layer_idx: int) -> None:  # type: ignore[no-untyped-def]
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.num_key_value_groups = config.num_key_value_groups
        self.head_dim = config.head_dim
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            self.d_model, self.n_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.d_model, self.n_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.d_model, self.n_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim, self.d_model, bias=False
        )

        # Rotary position embedding
        rope_type = "default"
        scaling_factor = None
        if config.rope_scaling is not None:
            rope_type = config.rope_scaling.get("type", "default")
            scaling_factor = config.rope_scaling.get("factor")

        self.rotary_emb = DeepSleepRotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
            scaling_factor=scaling_factor,
            rope_type=rope_type,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, ...]]]:
        """Compute grouped query attention.

        Args:
            hidden_states: Input tensor of shape ``(batch, seq_len, d_model)``.
            attention_mask: Optional attention mask. For causal attention, a
                lower-triangular boolean mask or a float mask where -inf
                represents masked positions.
            position_ids: Position indices of shape ``(batch, seq_len)``.
                Defaults to ``0..seq_len-1``.
            past_key_value: Tuple of (past_key, past_value) tensors from a
                previous forward pass, used for KV-caching.
            output_attentions: Whether to return attention weights.
            use_cache: Whether to return the updated KV cache.

        Returns:
            A tuple of:
            - ``attn_output``: Attention output of shape ``(batch, seq_len, d_model)``.
            - ``attn_weights``: Optional attention weights (None if not requested).
            - ``present_kv``: Optional updated KV cache (None if use_cache is False).
        """
        bsz, q_len, _ = hidden_states.shape

        # Project to Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape to (batch, num_heads, seq_len, head_dim)
        query_states = query_states.view(bsz, q_len, self.n_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply rotary position embeddings
        cos, sin = self.rotary_emb(value_states, position_ids=position_ids)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids=position_ids
        )

        # Handle KV-cache (supports both tuple and DynamicCache)
        cache_layer_idx = kwargs.get("cache_layer_idx", None)
        if past_key_value is not None:
            if hasattr(past_key_value, "update"):
                # DynamicCache (transformers 5.x) - use its API
                key_states, value_states = past_key_value.update(
                    key_states, value_states, cache_layer_idx
                )
                past_key_value_out = None  # Cache updated in-place
            else:
                # Old-style tuple (key, value)
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)
                past_key_value_out = (key_states, value_states) if use_cache else None
        else:
            past_key_value_out = (key_states, value_states) if use_cache else None

        # Repeat KV heads for GQA
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Apply attention
        attn_output, attn_weights = self._compute_attention(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            attention_mask=attention_mask,
            q_len=q_len,
            output_attentions=output_attentions,
        )

        # Reshape back to (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.d_model)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights, past_key_value_out

    def _compute_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        q_len: int,
        output_attentions: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute attention using Flash Attention or SDPA fallback.

        Args:
            query_states: Q tensor of shape ``(batch, n_heads, q_len, head_dim)``.
            key_states: K tensor of shape ``(batch, n_heads, kv_len, head_dim)``.
            value_states: V tensor of shape ``(batch, n_heads, kv_len, head_dim)``.
            attention_mask: Optional mask tensor.
            q_len: Query sequence length.
            output_attentions: Whether to return attention weights.

        Returns:
            Tuple of (attn_output, attn_weights).
        """
        use_flash = (
            self.config.use_flash_attention
            and _can_use_flash_attn()
            and not output_attentions
            and attention_mask is None
        )

        if use_flash:
            return self._flash_attention_forward(
                query_states, key_states, value_states, q_len
            )

        return self._sdpa_attention_forward(
            query_states, key_states, value_states, attention_mask, output_attentions
        )

    def _flash_attention_forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        q_len: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute attention using Flash Attention 2.

        Args:
            query_states: Q tensor of shape ``(batch, n_heads, q_len, head_dim)``.
            key_states: K tensor of shape ``(batch, n_heads, kv_len, head_dim)``.
            value_states: V tensor of shape ``(batch, n_heads, kv_len, head_dim)``.
            q_len: Query sequence length.

        Returns:
            Tuple of (attn_output, None) since flash-attn does not return weights.
        """
        # Flash attention expects (batch, seq_len, n_heads, head_dim)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        attn_output = flash_attn_func(
            query_states,
            key_states,
            value_states,
            causal=True,
        )

        # Reshape back to (batch, n_heads, seq_len, head_dim)
        attn_output = attn_output.transpose(1, 2)
        return attn_output, None

    def _sdpa_attention_forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        output_attentions: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute attention using PyTorch's scaled_dot_product_attention.

        Falls back to manual attention computation if attention_mask is
        provided in a format that SDPA cannot handle directly.

        Args:
            query_states: Q tensor of shape ``(batch, n_heads, q_len, head_dim)``.
            key_states: K tensor of shape ``(batch, n_heads, kv_len, head_dim)``.
            value_states: V tensor of shape ``(batch, n_heads, kv_len, head_dim)``.
            attention_mask: Optional mask tensor.
            output_attentions: Whether to return attention weights.

        Returns:
            Tuple of (attn_output, attn_weights).
        """
        L = query_states.shape[-2]

        # Scale queries
        query_states = query_states * self.scaling

        if attention_mask is not None:
            # When using a non-SDPA-compatible mask, compute attention manually
            if output_attentions:
                # Manual attention to retrieve weights
                attn_weights = torch.matmul(
                    query_states, key_states.transpose(2, 3)
                ) / math.sqrt(self.head_dim)

                if attention_mask.dim() == 2:
                    attn_weights = attn_weights + attention_mask[:, None, None, :]
                elif attention_mask.dim() == 4:
                    attn_weights = attn_weights + attention_mask

                attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
                    query_states.dtype
                )
                attn_output = torch.matmul(attn_weights, value_states)
                return attn_output, attn_weights
            else:
                # Use SDPA with compatible mask
                attn_output = F.scaled_dot_product_attention(
                    query_states,
                    key_states,
                    value_states,
                    attn_mask=attention_mask,
                    is_causal=attention_mask is None,
                )
                return attn_output, None
        else:
            # No mask: use SDPA with causal masking
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                is_causal=True,
            )

            if not output_attentions:
                return attn_output, None

            # Recompute attention weights if requested (SDPA does not return them)
            attn_weights = torch.matmul(
                query_states, key_states.transpose(2, 3)
            ) / math.sqrt(self.head_dim)
            attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
                query_states.dtype
            )
            return attn_output, attn_weights
