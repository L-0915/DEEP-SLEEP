"""Configuration for the DeepSleep language model.

DeepSleepConfig defines all hyperparameters for the DeepSleep-2B model,
a Qwen2.5-MoE inspired architecture for the medical sleep health domain.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

from transformers import PretrainedConfig


class DeepSleepConfig(PretrainedConfig):
    """Configuration class for DeepSleep model.

    This config inherits from ``PretrainedConfig`` and is fully compatible
    with the HuggingFace transformers ecosystem.  All hyperparameters follow
    the design specification in ``docs/superpowers/specs/2026-04-07-deepsleep-design.md``.

    Attributes:
        model_type: Model identifier registered with HuggingFace AutoConfig.
        d_model: Hidden dimension size.
        n_layers: Number of transformer decoder layers.
        n_heads: Number of query attention heads.
        n_kv_heads: Number of key/value attention heads (GQA).
        vocab_size: Vocabulary size of the tokenizer.
        max_position_embeddings: Maximum sequence length supported.
        hidden_act: Activation function (only ``"silu"`` is supported).
        rms_norm_eps: Epsilon for RMS normalization.
        tie_word_embeddings: Whether input/output embeddings are shared.
        num_experts: Total number of routed experts.
        num_routed_experts: Number of routed (gated) experts.
        num_shared_experts: Number of always-active shared experts.
        top_k: Number of experts selected per token by the router.
        aux_loss_coeff: Coefficient for the MoE load-balancing auxiliary loss.
        z_loss_coeff: Coefficient for the MoE router z-loss.
        router_jitter_noise: Gaussian noise added to router logits during training.
        expert_capacity_factor: Multiplier for per-expert token capacity.
        router_dtype: Data type for router computations.
        use_flash_attention: Whether to use Flash Attention 2.
        rope_theta: Base frequency for Rotary Position Embeddings.
        rope_scaling: Optional RoPE scaling configuration for extended context.
        initializer_range: Standard deviation for weight initialization.
    """

    model_type: str = "deepsleep"

    def __init__(
        self,
        d_model: int = 2048,
        n_layers: int = 24,
        n_heads: int = 16,
        n_kv_heads: int = 4,
        vocab_size: int = 64000,
        max_position_embeddings: int = 8192,
        hidden_act: str = "silu",
        rms_norm_eps: float = 1e-6,
        tie_word_embeddings: bool = True,
        # MoE configuration
        num_experts: int = 6,
        num_routed_experts: int = 5,
        num_shared_experts: int = 1,
        top_k: int = 2,
        aux_loss_coeff: float = 0.1,
        z_loss_coeff: float = 0.01,
        router_jitter_noise: float = 0.1,
        expert_capacity_factor: float = 1.25,
        router_dtype: str = "float32",
        # Attention configuration
        use_flash_attention: bool = True,
        rope_theta: float = 10000.0,
        rope_scaling: Optional[Dict[str, Any]] = None,
        # Initialization
        initializer_range: float = 0.02,
        # Dense FFN intermediate size (defaults to ~2.75x d_model)
        intermediate_size: Optional[int] = None,
        # MoE FFN intermediate size (defaults to ~0.688x d_model per expert)
        moe_intermediate_size: Optional[int] = None,
        # Layer alternation pattern: 'alternating' means even=dense, odd=MoE
        layer_pattern: str = "alternating",
        pad_token_id: Optional[int] = None,
        bos_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        # Core architecture
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.tie_word_embeddings = tie_word_embeddings
        self.initializer_range = initializer_range

        # Derived dimensions
        self.head_dim = d_model // n_heads
        self.num_key_value_groups = n_heads // n_kv_heads

        # Dense FFN intermediate size (SwiGLU): 2/3 * 4 * d_model rounded to multiple of 256
        if intermediate_size is None:
            self.intermediate_size = self._compute_intermediate_size(d_model)
        else:
            self.intermediate_size = intermediate_size

        # MoE FFN intermediate size per expert
        if moe_intermediate_size is None:
            # Experts use a smaller intermediate size to keep total MoE params reasonable
            # Design doc specifies SwiGLU (2048 -> 1408) per expert
            self.moe_intermediate_size = self._compute_moe_intermediate_size(d_model)
        else:
            self.moe_intermediate_size = moe_intermediate_size

        # MoE configuration
        self.num_experts = num_experts
        self.num_routed_experts = num_routed_experts
        self.num_shared_experts = num_shared_experts
        self.top_k = top_k
        self.aux_loss_coeff = aux_loss_coeff
        self.z_loss_coeff = z_loss_coeff
        self.router_jitter_noise = router_jitter_noise
        self.expert_capacity_factor = expert_capacity_factor
        self.router_dtype = router_dtype

        # Attention configuration
        self.use_flash_attention = use_flash_attention
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.layer_pattern = layer_pattern

        # Validate configuration
        self._validate_config()
        if self.rope_scaling is not None:
            self._rope_scaling_validation()

    @staticmethod
    def _compute_intermediate_size(d_model: int) -> int:
        """Compute the SwiGLU intermediate size for dense FFN layers.

        Follows the Qwen2 convention: intermediate_size = (2/3 * 4 * d_model)
        rounded to the nearest multiple of 256.
        """
        hidden_dim = int(2.0 / 3.0 * 4.0 * d_model)
        # Round up to nearest multiple of 256
        return ((hidden_dim + 255) // 256) * 256

    @staticmethod
    def _compute_moe_intermediate_size(d_model: int) -> int:
        """Compute the SwiGLU intermediate size for MoE expert FFN layers.

        Per the design doc, experts use a smaller intermediate dimension.
        With d_model=2048, this yields 1408.
        """
        hidden_dim = int(2.0 / 3.0 * 4.0 * d_model) // 4
        # Round up to nearest multiple of 64 for efficiency
        return ((hidden_dim + 63) // 64) * 64

    @property
    def num_hidden_layers(self) -> int:
        return self.n_layers

    def _validate_config(self) -> None:
        """Validate that configuration values are internally consistent."""
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by "
                f"n_kv_heads ({self.n_kv_heads})."
            )
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by "
                f"n_heads ({self.n_heads})."
            )
        if self.num_routed_experts + self.num_shared_experts != self.num_experts:
            raise ValueError(
                f"num_routed_experts ({self.num_routed_experts}) + "
                f"num_shared_experts ({self.num_shared_experts}) must equal "
                f"num_experts ({self.num_experts})."
            )
        if self.top_k > self.num_routed_experts:
            raise ValueError(
                f"top_k ({self.top_k}) cannot exceed "
                f"num_routed_experts ({self.num_routed_experts})."
            )
        if self.hidden_act != "silu":
            raise ValueError(
                f"hidden_act must be 'silu', got '{self.hidden_act}'."
            )
        if self.head_dim * self.n_heads != self.d_model:
            raise ValueError(
                f"head_dim ({self.head_dim}) * n_heads ({self.n_heads}) "
                f"!= d_model ({self.d_model})."
            )
        if self.layer_pattern not in ("alternating", "all_dense", "all_moe"):
            raise ValueError(
                f"layer_pattern must be one of 'alternating', 'all_dense', 'all_moe', "
                f"got '{self.layer_pattern}'."
            )

    def _rope_scaling_validation(self) -> None:
        """Validate the RoPE scaling configuration.

        Supports ``"linear"`` and ``"dynamic"`` (NTK-aware) scaling types.
        """
        if self.rope_scaling is None:
            return

        if not isinstance(self.rope_scaling, dict):
            raise TypeError(
                f"rope_scaling must be a dict, got {type(self.rope_scaling).__name__}."
            )

        scaling_type = self.rope_scaling.get("type")
        if scaling_type not in ("linear", "dynamic"):
            raise ValueError(
                f"rope_scaling type must be 'linear' or 'dynamic', got '{scaling_type}'."
            )

        if "factor" not in self.rope_scaling:
            raise ValueError("rope_scaling must include a 'factor' key.")

        factor = self.rope_scaling["factor"]
        if not isinstance(factor, (int, float)) or factor <= 1.0:
            raise ValueError(
                f"rope_scaling factor must be a float > 1.0, got {factor}."
            )
