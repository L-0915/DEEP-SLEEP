"""DeepSleep: Medical Sleep Health Domain Language Model.

A Mixture-of-Experts (MoE) language model for sleep health, inspired by
Qwen2.5-MoE and MiniMind architectures. Supports flexible MoE configurations:
dense-only, all-MoE, alternating, with optional shared experts.

Architecture:
    DeepSleepForCausalLM
    ├── Embedding (vocab, d_model, tied with lm_head)
    ├── N DeepSleepBlocks (configurable MoE/dense per layer)
    │   ├── DeepSleepAttention (GQA + RoPE + Flash/SDPA)
    │   └── DeepSleepFeedForward or DeepSleepMoE
    │       └── Routed experts + optional shared experts, top-k routing
    ├── Final RMSNorm
    └── LM Head (tied, no bias)
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import MoeCausalLMOutputWithPast


# =============================================================================
# Configuration
# =============================================================================


class DeepSleepConfig(PretrainedConfig):
    """Configuration for DeepSleep model.

    Supports flexible MoE configurations:
    - use_moe=False: all dense layers
    - use_moe=True, moe_layers=None: alternating (odd layers are MoE)
    - use_moe=True, moe_layers=[0,1,...,N]: specific layers are MoE
    - num_shared_experts > 0: adds always-active shared experts
    """

    model_type = "deepsleep"

    def __init__(
        self,
        d_model: int = 768,
        n_layers: int = 8,
        n_heads: int = 8,
        n_kv_heads: int = 4,
        head_dim: int = 96,
        vocab_size: int = 7200,
        max_position_embeddings: int = 8192,
        hidden_act: str = "silu",
        rms_norm_eps: float = 1e-6,
        tie_word_embeddings: bool = True,
        dropout: float = 0.0,
        # MoE
        use_moe: bool = True,
        moe_layers: Optional[List[int]] = None,
        num_experts: int = 8,
        num_routed_experts: int = 8,
        num_shared_experts: int = 0,
        top_k: int = 2,
        aux_loss_coeff: float = 0.1,
        z_loss_coeff: float = 0.01,
        router_jitter_noise: float = 0.1,
        router_aux_loss_coef: float = 5e-4,
        # FFN sizes
        intermediate_size: Optional[int] = None,
        moe_intermediate_size: Optional[int] = None,
        # Attention
        use_flash_attention: bool = True,
        flash_attn: bool = True,
        rope_theta: float = 10000.0,
        rope_scaling: Optional[Dict[str, Any]] = None,
        inference_rope_scaling: bool = False,
        # Init
        initializer_range: float = 0.02,
        # Legacy aliases
        hidden_size: Optional[int] = None,
        num_hidden_layers: Optional[int] = None,
        num_attention_heads: Optional[int] = None,
        num_key_value_heads: Optional[int] = None,
        layer_pattern: Optional[str] = None,
        **kwargs,
    ):
        # Resolve legacy aliases
        if hidden_size is not None:
            d_model = hidden_size
        if num_hidden_layers is not None:
            n_layers = num_hidden_layers
        if num_attention_heads is not None:
            n_heads = num_attention_heads
        if num_key_value_heads is not None:
            n_kv_heads = num_key_value_heads

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.dropout = dropout

        # MoE
        self.use_moe = use_moe
        self.moe_layers = self._resolve_moe_layers(moe_layers, n_layers, use_moe, layer_pattern)
        self.num_experts = num_experts
        self.num_routed_experts = num_routed_experts
        self.num_shared_experts = num_shared_experts
        self.top_k = top_k
        self.aux_loss_coeff = aux_loss_coeff
        self.z_loss_coeff = z_loss_coeff
        self.router_jitter_noise = router_jitter_noise
        self.router_aux_loss_coef = router_aux_loss_coef

        # FFN
        self.intermediate_size = intermediate_size or _default_intermediate(d_model)
        self.moe_intermediate_size = moe_intermediate_size or _default_moe_intermediate(d_model)

        # Attention
        self.use_flash_attention = use_flash_attention
        self.flash_attn = flash_attn
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.inference_rope_scaling = inference_rope_scaling

        # Init
        self.initializer_range = initializer_range

    @staticmethod
    def _resolve_moe_layers(moe_layers, n_layers, use_moe, layer_pattern):
        if not use_moe:
            return []
        if moe_layers is not None:
            return list(moe_layers)
        if layer_pattern == "all_moe":
            return list(range(n_layers))
        if layer_pattern == "all_dense":
            return []
        # Default: all layers are MoE
        return list(range(n_layers))

    @classmethod
    def from_legacy(cls, config_dict: Dict[str, Any]) -> "DeepSleepConfig":
        """Create config from a legacy checkpoint config.json."""
        d = dict(config_dict)
        # Map old keys
        for old, new in [
            ("hidden_size", "d_model"),
            ("num_hidden_layers", "n_layers"),
            ("num_attention_heads", "n_heads"),
            ("num_key_value_heads", "n_kv_heads"),
        ]:
            if old in d and new not in d:
                d[new] = d.pop(old)
        # Map layer_pattern
        if "layer_pattern" in d:
            d["layer_pattern"] = d.pop("layer_pattern")
        return cls(**d)


def _default_intermediate(d_model: int) -> int:
    """Dense FFN intermediate size. ~2.67x d_model (LLaMA-style)."""
    return {512: 1408, 768: 2048, 1024: 2816}.get(d_model, int(d_model * 8 / 3 / 64) * 64)


def _default_moe_intermediate(d_model: int) -> int:
    """MoE per-expert intermediate size. ~1.58x d_model."""
    return {512: 832, 768: 1216, 1024: 1664}.get(d_model, int(d_model * 1.58 / 64) * 64)


# =============================================================================
# Building Blocks
# =============================================================================


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.weight.float() * self.norm(x.float())).type_as(x)


def precompute_freqs_cis(dim: int, end: int, rope_base: float = 10000.0, rope_scaling: Optional[Dict] = None):
    """Precompute RoPE cos/sin frequency tensors (MiniMind-style)."""
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    if rope_scaling is not None:
        orig_max = rope_scaling.get("original_max_position_embeddings", 2048)
        factor = rope_scaling.get("factor", 16)
        beta_fast = rope_scaling.get("beta_fast", 32.0)
        beta_slow = rope_scaling.get("beta_slow", 1.0)
        attn_factor = rope_scaling.get("attention_factor", 1.0)
        if end / orig_max > 1.0:
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low = max(math.floor(inv_dim(beta_fast)), 0)
            high = min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 1), 0, 1)
            freqs = freqs * (1 - ramp + ramp / factor)
    else:
        attn_factor = 1.0
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
    q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))).to(q.dtype)
    k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))).to(k.dtype)
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, num_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return x[:, :, :, None, :].expand(bs, slen, num_kv_heads, n_rep, head_dim).reshape(bs, slen, num_kv_heads * n_rep, head_dim)


# =============================================================================
# Attention
# =============================================================================


class DeepSleepAttention(nn.Module):
    """Multi-head attention with Grouped Query Attention (GQA)."""

    def __init__(self, config: DeepSleepConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads or config.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = config.head_dim
        self.is_causal = True
        self.dropout = config.dropout

        self.q_proj = nn.Linear(config.d_model, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, config.d_model, bias=False)

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        bsz, seq_len, _ = x.shape
        xq = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim)
        xk = self.k_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        xv = self.v_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim)

        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        xq = xq.transpose(1, 2)
        xk = repeat_kv(xk, self.n_rep).transpose(1, 2)
        xv = repeat_kv(xv, self.n_rep).transpose(1, 2)

        if self.flash and seq_len > 1 and (attention_mask is None or torch.all(attention_mask == 1)):
            output = F.scaled_dot_product_attention(
                xq, xk, xv,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=self.is_causal,
            )
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if self.is_causal and past_key_value is None:
                scores[:, :, :, -seq_len:] += torch.full(
                    (seq_len, seq_len), float("-inf"), device=scores.device
                ).triu(1)
            if attention_mask is not None:
                scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
            output = F.softmax(scores.float(), dim=-1).type_as(xq) @ xv

        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        return output, past_kv


# =============================================================================
# Feed-Forward and MoE
# =============================================================================


class DeepSleepFeedForward(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, config: DeepSleepConfig, intermediate_size: Optional[int] = None):
        super().__init__()
        intermediate = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.d_model, intermediate, bias=False)
        self.up_proj = nn.Linear(config.d_model, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, config.d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DeepSleepMoE(nn.Module):
    """Mixture-of-Experts with optional shared experts.

    Routing: softmax → top-k → renormalize → dispatch (no token dropping).
    Auxiliary losses: load balance + z-loss.
    """

    def __init__(self, config: DeepSleepConfig):
        super().__init__()
        self.config = config
        self.num_routed_experts = config.num_routed_experts
        self.top_k = config.top_k

        self.gate = nn.Linear(config.d_model, self.num_routed_experts, bias=False)
        self.experts = nn.ModuleList([
            DeepSleepFeedForward(config, intermediate_size=config.moe_intermediate_size)
            for _ in range(self.num_routed_experts)
        ])

        if config.num_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                DeepSleepFeedForward(config, intermediate_size=config.moe_intermediate_size)
                for _ in range(config.num_shared_experts)
            ])
        else:
            self.shared_experts = None

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)

        # Router
        router_logits = self.gate(x_flat)
        routing_probs = F.softmax(router_logits.float(), dim=-1)
        top_k_probs, top_k_indices = torch.topk(routing_probs, self.top_k, dim=-1, sorted=False)
        top_k_weights = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-9)
        top_k_weights = top_k_weights.to(x.dtype)

        # Dispatch to experts
        y = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (top_k_indices == i)
            if mask.any():
                token_idx = mask.any(dim=-1).nonzero().flatten()
                weight = top_k_weights[mask].view(-1, 1)
                y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))
            elif self.training:
                y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())

        # Shared experts (always active)
        if self.shared_experts is not None:
            for se in self.shared_experts:
                y = y + se(x_flat)

        # Auxiliary loss
        if self.training and self.config.router_aux_loss_coef > 0:
            load = F.one_hot(top_k_indices.view(-1), self.num_routed_experts).float().mean(0)
            self.aux_loss = (load * routing_probs.mean(0)).sum() * self.num_routed_experts * self.config.router_aux_loss_coef
        else:
            self.aux_loss = x.new_zeros(1).squeeze()

        return y.view(batch_size, seq_len, hidden_dim)


# =============================================================================
# Transformer Block
# =============================================================================


class DeepSleepBlock(nn.Module):
    """Pre-norm transformer block with attention + FFN or MoE."""

    def __init__(self, layer_id: int, config: DeepSleepConfig):
        super().__init__()
        self.layer_id = layer_id
        self.self_attn = DeepSleepAttention(config)
        self.input_layernorm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.d_model, eps=config.rms_norm_eps)

        is_moe = config.use_moe and layer_id in config.moe_layers
        self.mlp = DeepSleepMoE(config) if is_moe else DeepSleepFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask,
        )
        hidden_states = residual + hidden_states
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value


# =============================================================================
# Model
# =============================================================================


class DeepSleepModel(nn.Module):
    """Core DeepSleep transformer model."""

    def __init__(self, config: DeepSleepConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.n_layers
        self.gradient_checkpointing = False

        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([
            DeepSleepBlock(l, config) for l in range(config.n_layers)
        ])
        self.norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)

        # Precompute RoPE frequencies
        rope_scaling = config.rope_scaling
        if config.inference_rope_scaling and rope_scaling is None:
            rope_scaling = {
                "beta_fast": 32, "beta_slow": 1, "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0, "type": "yarn",
            }
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.head_dim,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=rope_scaling,
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        batch_size, seq_length = input_ids.shape
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        hidden_states = self.dropout(self.embed_tokens(input_ids))

        # Recompute RoPE buffers if needed (e.g. meta-device init)
        if self.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = precompute_freqs_cis(
                dim=self.config.head_dim,
                end=self.config.max_position_embeddings,
                rope_base=self.config.rope_theta,
                rope_scaling=self.config.rope_scaling,
            )
            self.freqs_cos = freqs_cos.to(hidden_states.device)
            self.freqs_sin = freqs_sin.to(hidden_states.device)

        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_length],
            self.freqs_sin[start_pos:start_pos + seq_length],
        )

        presents = []
        for layer, past_kv in zip(self.layers, past_key_values):
            if self.gradient_checkpointing and self.training:
                from torch.utils.checkpoint import checkpoint as ckpt
                hidden_states, present = ckpt(
                    layer,
                    hidden_states,
                    position_embeddings,
                    None,       # past_key_value
                    False,      # use_cache
                    attention_mask,
                    use_reentrant=False,
                )
                present = None
            else:
                hidden_states, present = layer(
                    hidden_states, position_embeddings,
                    past_key_value=past_kv,
                    use_cache=use_cache,
                    attention_mask=attention_mask,
                )
            presents.append(present)

        hidden_states = self.norm(hidden_states)

        # Aggregate MoE auxiliary loss
        aux_loss = sum(
            (l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, DeepSleepMoE)),
            hidden_states.new_zeros(1).squeeze(),
        )

        return hidden_states, presents, aux_loss


# =============================================================================
# Causal LM
# =============================================================================


class DeepSleepForCausalLM(PreTrainedModel, GenerationMixin):
    """DeepSleep model with a causal language modeling head."""

    config_class = DeepSleepConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    accepts_loss_kwargs = False  # Let HF Trainer handle loss normalization by GA steps
    supports_gradient_checkpointing = True

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, DeepSleepModel):
            module.gradient_checkpointing = value

    def __init__(self, config: DeepSleepConfig = None):
        self.config = config or DeepSleepConfig()
        super().__init__(self.config)
        self.model = DeepSleepModel(self.config)
        self.lm_head = nn.Linear(self.config.d_model, self.config.vocab_size, bias=False)
        if self.config.tie_word_embeddings:
            self.model.embed_tokens.weight = self.lm_head.weight
        # Fix GenerationConfig: prevent MoE top_k from being interpreted as generation top_k
        self.generation_config.do_sample = True
        self.post_init()

    def forward(
        self, input_ids=None, attention_mask=None, past_key_values=None,
        use_cache=False, labels=None, **kwargs,
    ):
        hidden_states, past_key_values, aux_loss = self.model(
            input_ids, attention_mask, past_key_values, use_cache, **kwargs,
        )
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
        )

    @torch.inference_mode()
    def generate(
        self, inputs=None, attention_mask=None, max_new_tokens=4096,
        temperature=0.85, top_p=0.85, top_k=None, eos_token_id=2,
        streamer=None, use_cache=True, do_sample=True,
        repetition_penalty=1.0, **kwargs,
    ):
        input_ids = kwargs.pop("input_ids", inputs)
        if input_ids is None:
            raise ValueError("input_ids or inputs is required")
        past_key_values = kwargs.pop("past_key_values", None)
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        if streamer:
            streamer.put(input_ids.cpu())

        for _ in range(max_new_tokens):
            past_len = past_key_values[0][0].shape[1] if past_key_values else 0
            outputs = self.forward(
                input_ids[:, past_len:], attention_mask,
                past_key_values, use_cache=use_cache, **kwargs,
            )
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1
                )
            logits = outputs.logits[:, -1, :] / max(temperature, 1e-8)

            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]):
                    seen = torch.unique(input_ids[i])
                    score = logits[i, seen]
                    logits[i, seen] = torch.where(
                        score > 0, score / repetition_penalty, score * repetition_penalty
                    )
            if top_k is not None and top_k > 0:
                logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
                logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1) if do_sample else torch.argmax(logits, dim=-1, keepdim=True)
            if eos_token_id is not None:
                next_token = torch.where(
                    finished.unsqueeze(-1),
                    next_token.new_full((next_token.shape[0], 1), eos_token_id),
                    next_token,
                )
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values if use_cache else None
            if streamer:
                streamer.put(next_token.cpu())
            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                if finished.all():
                    break

        if streamer:
            streamer.end()
        return input_ids

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings


# =============================================================================
# Tokenizer Registration
# =============================================================================

def register_tokenizer():
    """Register DeepSleepTokenizer with HuggingFace Auto* classes."""
    try:
        from transformers import AutoTokenizer
        AutoTokenizer.register(
            "deepsleep",
            trusted=True,
            tokenizer_class="DeepSleepTokenizer",
        )
    except Exception:
        pass


register_tokenizer()
