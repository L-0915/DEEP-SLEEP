"""Basic neural network layers for the DeepSleep model.

Provides RMSNorm, Rotary Position Embeddings (RoPE), and SwiGLU MLP layers.
All layer implementations follow the Qwen2.5 conventions with bias=False.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepSleepRMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Applies RMS normalization over the last dimension of the input tensor.
    Unlike LayerNorm, this does not center the activations (no subtracting mean).

    Args:
        hidden_size: Dimensionality of the input and output.
        eps: Small constant for numerical stability (default 1e-6).
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization.

        Args:
            x: Input tensor of shape ``(batch, seq_len, hidden_size)`` or
                ``(batch, seq_len, num_heads, head_dim)``.

        Returns:
            Normalized tensor of the same shape as input.
        """
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight.float() * x).to(x.dtype)


class DeepSleepRotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) implementation.

    Computes sinusoidal frequency tensors and applies rotary embeddings to
    query and key tensors.  Supports optional NTK-aware and linear scaling
    for extended context lengths.

    Args:
        dim: Rotary embedding dimension (typically head_dim).
        max_position_embeddings: Maximum sequence length for precomputation.
        base: Base frequency theta (default 10000.0).
        device: Device for precomputed tensors.
        scaling_factor: Optional scaling factor for extended context.
        rope_type: Type of RoPE scaling (``"default"``, ``"linear"``, or ``"dynamic"``).
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 8192,
        base: float = 10000.0,
        device: Optional[torch.device] = None,
        scaling_factor: Optional[float] = None,
        rope_type: str = "default",
    ) -> None:
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.device = device
        self.scaling_factor = scaling_factor
        self.rope_type = rope_type

        # Compute inverse frequencies
        inv_freq = self._compute_inv_freq(
            dim=dim,
            base=base,
            scaling_factor=scaling_factor,
            rope_type=rope_type,
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build cache
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=device,
            dtype=torch.float32,
        )

    def _compute_inv_freq(
        self,
        dim: int,
        base: float,
        scaling_factor: Optional[float],
        rope_type: str,
    ) -> torch.Tensor:
        """Compute inverse frequency tensor for RoPE.

        Supports three modes:
        - ``"default"``: Standard RoPE with no scaling.
        - ``"linear"``: Linearly scale all frequencies by ``1/scaling_factor``.
        - ``"dynamic"``: NTK-aware scaling that adjusts the base frequency.

        Args:
            dim: Rotary dimension.
            base: Base theta.
            scaling_factor: Scaling factor for extended context.
            rope_type: Type of scaling.

        Returns:
            Tensor of inverse frequencies of shape ``(dim // 2,)``.
        """
        if rope_type == "default" or scaling_factor is None:
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        elif rope_type == "linear":
            inv_freq = 1.0 / (
                (scaling_factor * base)
                ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
            )
        elif rope_type == "dynamic":
            # NTK-aware scaling: adjust the base frequency
            base = base * ((scaling_factor * dim / (dim - 2)) - (dim - 2) / dim) ** (
                dim / (dim - 2)
            )
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        else:
            raise ValueError(f"Unknown rope_type: {rope_type}")

        return inv_freq

    def _set_cos_sin_cache(
        self,
        seq_len: int,
        device: Optional[torch.device],
        dtype: torch.dtype,
    ) -> None:
        """Precompute cosine and sine caches for a given sequence length.

        Args:
            seq_len: Sequence length to precompute.
            device: Target device.
            dtype: Data type for the cached tensors.
        """
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=torch.float32)

        freqs = torch.outer(t, self.inv_freq)
        # Concatenate frequencies for complex representation: (seq_len, dim/2, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve precomputed cos/sin embeddings for the given positions.

        Args:
            x: Input tensor of shape ``(batch, num_heads, seq_len, head_dim)``.
                Used only to determine the device and dtype.
            position_ids: Position indices of shape ``(batch, seq_len)``.
                If ``None``, assumes positions 0..seq_len-1.

        Returns:
            Tuple of (cos, sin) tensors of shape
            ``(batch, 1, seq_len, head_dim)``.
        """
        seq_len = x.shape[2]

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=x.device, dtype=torch.long)
            position_ids = position_ids.unsqueeze(0).expand(x.shape[0], -1)

        # Gather cos/sin from cache
        cos = self.cos_cached[position_ids].unsqueeze(1)  # (batch, 1, seq_len, dim)
        sin = self.sin_cached[position_ids].unsqueeze(1)  # (batch, 1, seq_len, dim)

        return cos.to(x.dtype), sin.to(x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the second half of the last dimension to the front with negation.

    Given input ``[x0, x1, x2, x3, x4, x5]`` with dim=6,
    returns ``[-x3, -x4, -x5, x0, x1, x2]``.

    Args:
        x: Input tensor whose last dimension is even.

    Returns:
        Rotated tensor of the same shape.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key tensors.

    Args:
        q: Query tensor of shape ``(batch, num_heads, seq_len, head_dim)``.
        k: Key tensor of shape ``(batch, num_kv_heads, seq_len, head_dim)``.
        cos: Cosine tensor from RoPE of shape ``(batch, 1, seq_len, head_dim)``.
        sin: Sine tensor from RoPE of shape ``(batch, 1, seq_len, head_dim)``.
        position_ids: Unused, kept for API compatibility.
        unsqueeze_dim: Unused, kept for API compatibility.

    Returns:
        Tuple of rotated (q_rot, k_rot) tensors.
    """
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class DeepSleepMLP(nn.Module):
    """SwiGLU feed-forward network (Multi-Layer Perceptron).

    Implements the SwiGLU variant of the gated feed-forward network used in
    LLaMA and Qwen2 models.  The computation is:

        ``output = down_proj(silu(gate_proj(x)) * up_proj(x))``

    All linear projections are bias-free.

    Args:
        config: Model configuration object with ``hidden_size`` and
            ``intermediate_size`` attributes.
    """

    def __init__(self, config) -> None:  # type: ignore[no-untyped-def]
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size if hasattr(config, "hidden_size") else config.d_model,
            config.intermediate_size,
            bias=False,
        )
        self.up_proj = nn.Linear(
            config.hidden_size if hasattr(config, "hidden_size") else config.d_model,
            config.intermediate_size,
            bias=False,
        )
        self.down_proj = nn.Linear(
            config.intermediate_size,
            config.hidden_size if hasattr(config, "hidden_size") else config.d_model,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the SwiGLU MLP.

        Args:
            x: Input tensor of shape ``(batch, seq_len, d_model)``.

        Returns:
            Output tensor of shape ``(batch, seq_len, d_model)``.
        """
        # gate = silu(x @ gate_proj.T)
        gate = F.silu(self.gate_proj(x))
        # up = x @ up_proj.T
        up = self.up_proj(x)
        # output = (gate * up) @ down_proj.T
        return self.down_proj(gate * up)
