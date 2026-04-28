"""Main DeepSleep model implementation.

Assembles the full DeepSleep-2B causal language model from its constituent
components: embedding, decoder layers (dense + MoE), and language modeling head.

The model is fully compatible with the HuggingFace transformers ecosystem,
supporting:
- ``AutoConfig.register("deepsleep", DeepSleepConfig)``
- ``AutoModelForCausalLM.register(DeepSleepConfig, DeepSleepForCausalLM)``
- Standard HF training and inference workflows
- Gradient checkpointing for memory efficiency
- KV-caching and beam search
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

from .attention import DeepSleepAttention
from .config import DeepSleepConfig
from .embedding import DeepSleepEmbedding
from .layers import DeepSleepMLP, DeepSleepRMSNorm
from .moe import DeepSleepMoE

logger = logging.getLogger(__name__)

# Type alias for optional KV-cache entries
CacheType = Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]


class DeepSleepDecoderLayer(nn.Module):
    """A single transformer decoder layer with pre-normalization.

    Each layer consists of:
    1. RMSNorm -> Self-Attention -> Residual connection
    2. RMSNorm -> FFN (Dense or MoE) -> Residual connection

    The choice between dense FFN and MoE is determined by the layer index
    and the ``layer_pattern`` configuration.

    Args:
        config: Model configuration.
        layer_idx: Index of this layer in the stack (0-based).
    """

    def __init__(self, config: DeepSleepConfig, layer_idx: int) -> None:
        super().__init__()
        self.hidden_size = config.d_model
        self.layer_idx = layer_idx

        # Self-attention sublayer
        self.input_layernorm = DeepSleepRMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.self_attn = DeepSleepAttention(config, layer_idx=layer_idx)

        # Post-attention sublayer: dense FFN or MoE
        self.post_attention_layernorm = DeepSleepRMSNorm(
            config.d_model, eps=config.rms_norm_eps
        )

        use_moe = self._should_use_moe(config, layer_idx)
        if use_moe:
            self.mlp = DeepSleepMoE(config)
            self.is_moe = True
        else:
            self.mlp = DeepSleepMLP(config)
            self.is_moe = False

    @staticmethod
    def _should_use_moe(config: DeepSleepConfig, layer_idx: int) -> bool:
        """Determine whether this layer should use MoE or dense FFN.

        Args:
            config: Model configuration with ``layer_pattern``.
            layer_idx: Zero-based layer index.

        Returns:
            True if this layer should use MoE, False for dense FFN.
        """
        if config.layer_pattern == "all_moe":
            return True
        if config.layer_pattern == "all_dense":
            return False
        # Alternating: even indices are dense, odd indices are MoE
        return layer_idx % 2 == 1

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, ...]], Optional[torch.Tensor]]:
        """Forward pass through a single decoder layer.

        Args:
            hidden_states: Input of shape ``(batch, seq_len, d_model)``.
            attention_mask: Optional attention mask.
            position_ids: Optional position indices of shape ``(batch, seq_len)``.
            past_key_value: Optional KV-cache tuple from previous forward pass.
            output_attentions: Whether to return attention weights.
            use_cache: Whether to return updated KV cache.

        Returns:
            Tuple of:
            - ``hidden_states``: Output tensor of shape ``(batch, seq_len, d_model)``.
            - ``present_kv``: Optional KV-cache tuple.
            - ``attn_weights``: Optional attention weights.
        """
        residual = hidden_states

        # Pre-norm + self-attention
        normed_hidden = self.input_layernorm(hidden_states)
        cache_layer_idx = kwargs.get("cache_layer_idx", None)
        attn_output, attn_weights, present_kv = self.self_attn(
            hidden_states=normed_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_layer_idx=cache_layer_idx,
        )
        hidden_states = residual + attn_output

        # Pre-norm + FFN/MoE
        residual = hidden_states
        normed_hidden = self.post_attention_layernorm(hidden_states)

        if self.is_moe:
            ffn_output, aux_losses = self.mlp(normed_hidden)
            # Store aux losses as buffer for collection by the parent model
            # Using a dict attached to the module for out-of-band communication
            if not hasattr(self, "_aux_losses") or self._aux_losses is None:
                self._aux_losses: Dict[str, torch.Tensor] = {}
            self._aux_losses.update(aux_losses)
        else:
            ffn_output = self.mlp(normed_hidden)

        hidden_states = residual + ffn_output

        outputs = (hidden_states,)

        if use_cache:
            outputs += (present_kv,)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class DeepSleepModel(PreTrainedModel):
    """Core DeepSleep transformer model (no language modeling head).

    Stacks embedding, N decoder layers, and a final RMSNorm.
    Supports gradient checkpointing for memory-efficient training.

    Args:
        config: DeepSleep model configuration.
    """

    config_class = DeepSleepConfig
    _no_split_modules = ["DeepSleepDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _keys_to_ignore_on_load_missing = ["rotary_emb.inv_freq"]

    def __init__(self, config: DeepSleepConfig) -> None:
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Token embedding (no positional embedding; RoPE is applied in attention)
        self.embed_tokens = DeepSleepEmbedding(
            config, padding_idx=self.padding_idx
        )

        # Decoder layers
        self.layers = nn.ModuleList(
            [DeepSleepDecoderLayer(config, layer_idx=i) for i in range(config.n_layers)]
        )

        # Final RMSNorm
        self.norm = DeepSleepRMSNorm(config.d_model, eps=config.rms_norm_eps)

        # Gradient checkpointing flag
        self.gradient_checkpointing = False

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights with normal distribution.

        Follows the standard initialization scheme for transformer models.
        Embedding weights are also initialized from a normal distribution
        (rather than the default uniform in nn.Embedding).

        Args:
            module: Module to initialize.
        """
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if self.padding_idx is not None and module.padding_idx >= 0:
                module.weight.data[self.padding_idx].zero_()
        elif isinstance(module, DeepSleepRMSNorm):
            module.weight.data.fill_(1.0)

    def get_input_embeddings(self) -> nn.Embedding:
        """Return the token embedding layer.

        Returns:
            The ``nn.Embedding`` used for input tokens.
        """
        return self.embed_tokens.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        """Set the token embedding layer.

        Args:
            value: New embedding layer to use.
        """
        self.embed_tokens.embed_tokens = value

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, ...]:
        """Forward pass through the DeepSleep model backbone.

        Args:
            input_ids: Token IDs of shape ``(batch, seq_len)``.
            attention_mask: Optional attention mask of shape ``(batch, seq_len)``.
            position_ids: Optional position indices of shape ``(batch, seq_len)``.
            past_key_values: Optional list of KV-cache tuples per layer.
            inputs_embeds: Optional embedded inputs (bypasses embed_tokens).
            use_cache: Whether to use and return KV cache.
            output_attentions: Whether to return attention weights from all layers.
            output_hidden_states: Whether to return hidden states from all layers.

        Returns:
            Tuple containing:
            - ``hidden_states``: Final hidden states ``(batch, seq_len, d_model)``.
            - ``present_key_values``: Updated KV caches (if use_cache).
            - ``all_hidden_states``: All layer hidden states (if output_hidden_states).
            - ``all_self_attns``: All layer attention weights (if output_attentions).
            - ``moe_aux_losses``: Dictionary of aggregated MoE auxiliary losses.
        """
        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else False
        )
        use_cache = use_cache if use_cache is not None else False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = inputs_embeds.shape[:2]

        # Default position_ids
        if position_ids is None:
            past_length = 0
            if past_key_values is not None:
                if hasattr(past_key_values, 'get_seq_length'):
                    past_length = past_key_values.get_seq_length()
                elif isinstance(past_key_values, (list, tuple)) and len(past_key_values) > 0:
                    past_length = past_key_values[0][0].shape[2]
            position_ids = torch.arange(
                past_length,
                past_length + seq_length,
                device=inputs_embeds.device,
                dtype=torch.long,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

        # Build causal attention mask
        if attention_mask is not None:
            # Expand 2D mask to 4D: (batch, 1, 1, seq_len)
            expanded_mask = self._prepare_attention_mask(
                attention_mask, inputs_embeds.shape, past_key_values
            )
        else:
            expanded_mask = None

        # Collect hidden states and attentions if requested
        all_hidden_states: List[torch.Tensor] = ()
        all_self_attns: List[torch.Tensor] = ()
        moe_aux_losses: Dict[str, torch.Tensor] = {}

        hidden_states = inputs_embeds

        # Decoder layers
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # Extract per-layer KV cache (handle both list/tuple and DynamicCache)
            if past_key_values is not None:
                if hasattr(past_key_values, 'update'):
                    # DynamicCache (transformers 5.x) - pass the cache object directly
                    past_key_value = past_key_values
                elif isinstance(past_key_values, (list, tuple)):
                    past_key_value = past_key_values[idx]
                else:
                    past_key_value = None
            else:
                past_key_value = None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    expanded_mask,
                    position_ids,
                    past_key_value,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=expanded_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_layer_idx=idx,
                )

            hidden_states = layer_outputs[0]

            # Collect MoE auxiliary losses from MoE layers
            if decoder_layer.is_moe and hasattr(decoder_layer, "_aux_losses"):
                layer_aux = getattr(decoder_layer, "_aux_losses")
                for loss_name, loss_val in layer_aux.items():
                    if loss_name in moe_aux_losses:
                        moe_aux_losses[loss_name] = moe_aux_losses[loss_name] + loss_val
                    else:
                        moe_aux_losses[loss_name] = loss_val.clone()

            # Update cache
            if use_cache:
                if past_key_values is not None and hasattr(past_key_values, 'update'):
                    # DynamicCache - already updated in-place by attention
                    next_cache = past_key_values
                else:
                    next_cache = layer_outputs[1]
            else:
                next_cache = None

            if output_attentions:
                all_self_attns += (layer_outputs[2 if use_cache else 1],)

        # Final normalization
        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # Build return tuple
        outputs = (hidden_states,)
        if use_cache:
            outputs += (next_cache,)  # type: ignore[operator]
        if output_hidden_states:
            outputs += (all_hidden_states,)
        if output_attentions:
            outputs += (all_self_attns,)
        outputs += (moe_aux_losses,)

        return outputs

    def _prepare_attention_mask(
        self,
        attention_mask: torch.Tensor,
        input_shape: Tuple[int, ...],
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> torch.Tensor:
        """Convert a 2D attention mask to a 4D causal + padding mask for SDPA.

        Combines a lower-triangular causal mask with the padding mask so that
        each position can only attend to non-padding tokens at or before its
        own position.

        Args:
            attention_mask: Boolean mask of shape ``(batch, seq_len)``.
            input_shape: Shape of the input tensor ``(batch, seq_len, ...)``.
            past_key_values: Optional past KV caches (affects mask length).

        Returns:
            Float mask of shape ``(batch, 1, seq_len, total_seq_len)`` with
            0 at allowed positions and -inf at masked positions.
        """
        batch_size, seq_length = input_shape[:2]

        # Handle KV-cache extension
        if past_key_values is not None:
            # Support both tuple-based and DynamicCache
            if hasattr(past_key_values, "get_seq_length"):
                past_length = past_key_values.get_seq_length()
            elif isinstance(past_key_values, (list, tuple)) and len(past_key_values) > 0:
                past_length = past_key_values[0][0].shape[2]
            else:
                past_length = 0
        else:
            past_length = 0

        total_length = past_length + seq_length

        # Build 4D causal mask: (1, 1, seq_len, total_length)
        causal_mask = torch.zeros(
            (seq_length, total_length), dtype=torch.float32, device=attention_mask.device
        )
        if past_length > 0:
            causal_mask[:, :past_length] = 0.0
        causal_mask[:, past_length:] = torch.triu(
            torch.full((seq_length, seq_length), float("-inf"), device=attention_mask.device),
            diagonal=1,
        )
        causal_mask = causal_mask[None, None, :, :]  # (1, 1, seq_len, total_length)

        # Expand padding mask: (batch, 1, 1, total_length)
        # Pad attention_mask with 1s for past tokens if it's shorter than total_length
        mask_len = attention_mask.shape[-1]
        if mask_len < total_length:
            pad_len = total_length - mask_len
            ones = torch.ones(
                (batch_size, pad_len), dtype=attention_mask.dtype, device=attention_mask.device
            )
            attention_mask = torch.cat([ones, attention_mask], dim=1)
        elif mask_len > total_length:
            attention_mask = attention_mask[:, :total_length]
        padding_mask = attention_mask[:, None, None, :].to(dtype=torch.float32)
        padding_mask = (1.0 - padding_mask) * torch.finfo(torch.float32).min

        # Combine: causal + padding
        combined_mask = causal_mask + padding_mask

        return combined_mask


class DeepSleepForCausalLM(PreTrainedModel, GenerationMixin):
    """DeepSleep model with a causal language modeling head.

    Wraps the core transformer model and adds a linear projection head
    (``lm_head``) that maps hidden states to vocabulary logits.  When
    ``tie_word_embeddings`` is True, the lm_head weights are shared with
    the input embedding layer.

    The forward method computes the cross-entropy loss against ``labels``
    when provided, and combines MoE auxiliary losses with the LM loss.

    Args:
        config: DeepSleep model configuration.
    """

    config_class = DeepSleepConfig
    _no_split_modules = ["DeepSleepDecoderLayer"]
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: DeepSleepConfig) -> None:
        super().__init__(config)
        self.model = DeepSleepModel(config)

        # Language modeling head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights and handle weight tying
        self.apply(self._init_weights)
        if config.tie_word_embeddings:
            self.tie_weights()

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize the lm_head weights.

        The base model handles its own initialization; we only need to
        initialize the lm_head here (if not tied).

        Args:
            module: Module to initialize.
        """
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()

    def tie_weights(self) -> None:
        """Tie lm_head weights with input embedding weights."""
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.embed_tokens.weight

    def get_input_embeddings(self) -> nn.Embedding:
        """Return the input token embedding layer."""
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        """Set the input token embedding layer."""
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Linear:
        """Return the output projection layer (lm_head)."""
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        """Set the output projection layer."""
        self.lm_head = new_embeddings

    def set_decoder(self, decoder: PreTrainedModel) -> None:
        """Set the decoder model."""
        self.model = decoder

    def get_decoder(self) -> PreTrainedModel:
        """Return the decoder model."""
        return self.model

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        """Forward pass for causal language modeling.

        Computes logits and optionally the cross-entropy loss.  MoE auxiliary
        losses are added to the LM loss when labels are provided.

        Args:
            input_ids: Token IDs of shape ``(batch, seq_len)``.
            attention_mask: Optional attention mask.
            position_ids: Optional position indices.
            past_key_values: Optional KV-cache from previous forward passes.
            inputs_embeds: Optional embedded inputs (bypasses embed_tokens).
            labels: Optional target token IDs for loss computation.
                Shape ``(batch, seq_len)``.  Tokens with label == -100 are ignored.
            use_cache: Whether to use and return KV cache.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return hidden states.

        Returns:
            ``CausalLMOutputWithPast`` containing loss, logits,
            past_key_values, hidden_states, and attentions.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift logits and labels for causal LM: predict next token
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
            )

            # Add MoE auxiliary losses
            moe_aux_losses = outputs[-1]  # Last element is the aux losses dict
            if isinstance(moe_aux_losses, dict):
                for aux_loss_name, aux_loss_val in moe_aux_losses.items():
                    if isinstance(aux_loss_val, torch.Tensor) and aux_loss_val.requires_grad:
                        loss = loss + aux_loss_val

        # Extract optional outputs from the variable-length outputs tuple.
        # The model always returns (hidden_states, ...) with moe_aux_losses last.
        # Between them, the order depends on which optional outputs are enabled:
        #   [hidden_states, next_cache, all_hidden_states, all_self_attns, moe_aux_losses]
        #   but only the enabled ones are present.
        past_key_values_out = None
        all_hidden_states = None
        all_self_attns = None

        # Remove the always-present elements to inspect optional ones
        # outputs[-1] is always moe_aux_losses, outputs[0] is always hidden_states
        optional_outputs = outputs[1:-1]  # everything between hidden_states and aux_losses

        idx = 0
        if use_cache and idx < len(optional_outputs):
            past_key_values_out = optional_outputs[idx]
            idx += 1
        if output_hidden_states and idx < len(optional_outputs):
            all_hidden_states = optional_outputs[idx]
            idx += 1
        if output_attentions and idx < len(optional_outputs):
            all_self_attns = optional_outputs[idx]
            idx += 1

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values_out,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Prepare inputs for the next generation step."""
        past_length = 0
        if past_key_values is not None:
            # Handle both DynamicCache and tuple formats
            if hasattr(past_key_values, 'get_seq_length'):
                past_length = past_key_values.get_seq_length()
            elif isinstance(past_key_values, (list, tuple)) and len(past_key_values) > 0:
                past_length = past_key_values[0][0].shape[2]
            if past_length > 0:
                input_ids = input_ids[:, past_length:]

        position_ids = torch.arange(
            past_length,
            past_length + input_ids.shape[1],
            device=input_ids.device,
            dtype=torch.long,
        )
        position_ids = position_ids.unsqueeze(0)

        model_inputs = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }

        if inputs_embeds is not None:
            model_inputs["inputs_embeds"] = inputs_embeds

        return model_inputs

    def _reorder_cache(
        self,
        past_key_values: List[Tuple[torch.Tensor, torch.Tensor]],
        beam_idx: torch.Tensor,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Reorder KV-cache entries for beam search.

        When beam search prunes beams, the cache must be reordered to
        match the surviving beam indices.

        Args:
            past_key_values: List of (key, value) tuples per layer.
            beam_idx: Tensor of beam indices to keep, shape ``(batch_size * num_beams,)``.

        Returns:
            Reordered list of (key, value) tuples.
        """
        reordered_cache: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for layer_past in past_key_values:
            # layer_past is a tuple of (key, value), each of shape
            # (batch_size * num_beams, num_heads, seq_len, head_dim)
            reordered_key = layer_past[0].index_select(0, beam_idx)
            reordered_value = layer_past[1].index_select(0, beam_idx)
            reordered_cache.append((reordered_key, reordered_value))
        return reordered_cache


# Register with HuggingFace Auto classes
AutoConfig.register("deepsleep", DeepSleepConfig)
AutoModelForCausalLM.register(DeepSleepConfig, DeepSleepForCausalLM)
