"""FSDP (Fully Sharded Data Parallel) configuration for DeepSleep.

Provides a factory function that returns a dictionary of FSDP kwargs suitable
for passing to ``torch.distributed.FSDP``.  The configuration is tuned for the
DeepSleep MoE architecture -- each transformer decoder layer is treated as an
independent sharding unit while embeddings, norms, and the LM head remain on
the owner rank.

Typical usage::

    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from src.utils.fsdp_config import get_fsdp_config

    fsdp_kwargs = get_fsdp_config()
    model = FSDP(model, **fsdp_kwargs)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

import torch
import torch.nn as nn


def _get_transformer_layer_cls() -> Optional[Type[nn.Module]]:
    """Attempt to import the DeepSleep transformer block class.

    Returns the class if it can be imported, otherwise ``None``.  The class is
    resolved lazily so that this module can be imported before the full model
    package is available.
    """
    try:
        # The model package may not be importable during early setup, so we
        # handle ImportError gracefully.
        from src.model.modeling_deepsleep import DeepSleepDecoderLayer  # type: ignore[import-untyped]
        return DeepSleepDecoderLayer
    except (ImportError, AttributeError):
        return None


def _default_auto_wrap_policy(
    module: nn.Module,
    recurse: bool,
    nonwrapped_numel: int,
) -> bool:
    """Default auto-wrap policy that targets transformer decoder layers.

    Layers are wrapped if their parameter count exceeds a configurable
    threshold (default 1e8, i.e. 100M parameters).  This prevents tiny layers
    from becoming their own FSDP unit which would add communication overhead.

    Args:
        module: The module being considered for wrapping.
        recurse: Whether to recurse into children (set by FSDP).
        nonwrapped_numel: Number of parameters not yet wrapped.

    Returns:
        ``True`` if the module should be wrapped as an FSDP unit.
    """
    # Only wrap leaf modules (those without children being wrapped).
    if len(list(module.children())) > 0:
        # Allow recursion into children but don't wrap the parent.
        return recurse

    # Threshold: wrap if the module has a substantial number of parameters.
    _MIN_PARAMS = 100_000_000  # 100M -- roughly one transformer layer
    numel = sum(p.numel() for p in module.parameters())
    return numel >= _MIN_PARAMS


def get_fsdp_config(
    sharding_strategy: str = "full_shard",
    mixed_precision_dtype: str = "bf16",
    offload_params: bool = False,
    backward_prefetch: str = "backward_pre",
    cpu_offload: bool = False,
) -> Dict[str, Any]:
    """Build and return an FSDP configuration dictionary.

    Args:
        sharding_strategy: One of ``"full_shard"``, ``"shard_grad_op"``, or
            ``"no_shard"``.  Controls the granularity of parameter sharding.
        mixed_precision_dtype: Mixed-precision compute dtype.  Supported
            values are ``"bf16"``, ``"fp16"``, and ``"fp32"``.
        offload_params: If ``True``, offload sharded parameters to CPU.
        backward_prefetch: Backward prefetch policy.  Supported values are
            ``"backward_pre"``, ``"backward_post"``, and ``"none"``.
        cpu_offload: Deprecated -- use *offload_params* instead.  Kept for
            backward compatibility.

    Returns:
        Dictionary suitable for ``**``-unpacking into
        ``torch.distributed.fsdp.FullyShardedDataParallel``.
    """
    from torch.distributed.fsdp import (  # type: ignore[import-untyped]
        BackwardPrefetch,
        CPUOffload,
        MixedPrecision,
        ShardingStrategy,
    )
    from torch.distributed.fsdp.wrap import (  # type: ignore[import-untyped]
        size_based_auto_wrap_policy,
        transformer_auto_wrap_policy,
    )

    # -- Sharding strategy --
    _strategy_map = {
        "full_shard": ShardingStrategy.FULL_SHARD,
        "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
        "no_shard": ShardingStrategy.NO_SHARD,
    }
    strategy = _strategy_map.get(sharding_strategy, ShardingStrategy.FULL_SHARD)

    # -- Mixed precision --
    _dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    compute_dtype = _dtype_map.get(mixed_precision_dtype, torch.bfloat16)
    mixed_precision = MixedPrecision(
        param_dtype=compute_dtype,
        reduce_dtype=compute_dtype,
        buffer_dtype=compute_dtype,
    )

    # -- CPU offload --
    cpu_offload_cfg = CPUOffload(offload_params=offload_params)

    # -- Backward prefetch --
    _prefetch_map = {
        "backward_pre": BackwardPrefetch.BACKWARD_PRE,
        "backward_post": BackwardPrefetch.BACKWARD_POST,
        "none": BackwardPrefetch.BACKWARD_POST,
    }
    prefetch = _prefetch_map.get(backward_prefetch, BackwardPrefetch.BACKWARD_PRE)

    # -- Auto wrap policy --
    transformer_layer_cls = _get_transformer_layer_cls()
    if transformer_layer_cls is not None:
        auto_wrap_policy = transformer_auto_wrap_policy(
            transformer_layer_cls,
        )
    else:
        # Fall back to a size-based policy when the model class is not yet
        # available (e.g. during unit tests or standalone usage).
        auto_wrap_policy = _default_auto_wrap_policy

    return {
        "sharding_strategy": strategy,
        "mixed_precision": mixed_precision,
        "cpu_offload": cpu_offload_cfg,
        "backward_prefetch": prefetch,
        "auto_wrap_policy": auto_wrap_policy,
        "use_orig_params": True,
    }
