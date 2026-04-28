"""DeepSleep-2B model package.

Provides the complete model architecture for the DeepSleep medical sleep health
domain language model, a Qwen2.5-MoE inspired architecture.

Public API:
    - ``DeepSleepConfig``: Model configuration (registered as "deepsleep" with AutoConfig).
    - ``DeepSleepModel``: Core transformer backbone (no LM head).
    - ``DeepSleepForCausalLM``: Full causal LM model (registered with AutoModelForCausalLM).

Submodules:
    - ``config``: DeepSleepConfig with all hyperparameters.
    - ``layers``: RMSNorm, RotaryEmbedding, SwiGLU MLP.
    - ``attention``: Grouped Query Attention with Flash Attention support.
    - ``moe``: Mixture-of-Experts layer with auxiliary losses.
    - ``embedding``: Token embedding layer.
    - ``modeling_deepsleep``: Full model assembly and HF integration.
"""

from .config import DeepSleepConfig
from .modeling_deepsleep import (
    DeepSleepDecoderLayer,
    DeepSleepForCausalLM,
    DeepSleepModel,
)

# Register the tokenizer with HuggingFace auto-classes.
# This import has a side effect (auto-registration) and is intentionally
# placed here so that ``import src.model`` or ``from src.model import ...``
# is sufficient to make ``AutoTokenizer.from_pretrained`` resolve
# "DeepSleepTokenizer".
try:
    from .tokenization_deepsleep import DeepSleepTokenizer as _DeepSleepTokenizerFast  # noqa: F401
except Exception:
    pass

__all__ = [
    "DeepSleepConfig",
    "DeepSleepModel",
    "DeepSleepForCausalLM",
    "DeepSleepDecoderLayer",
]
