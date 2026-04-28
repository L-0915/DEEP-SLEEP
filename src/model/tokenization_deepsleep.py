"""HuggingFace tokenizer integration for the DeepSleep model.

Registers ``DeepSleepTokenizer`` with HuggingFace auto-classes so that
``AutoTokenizer.from_pretrained("path/to/model")`` resolves correctly.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from transformers import PreTrainedTokenizerFast

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Special tokens
# ---------------------------------------------------------------------------

VOCAB_FILES_NAMES: Dict[str, str] = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
    "tokenizer_file": "tokenizer.json",
}

# PreTrainedTokenizerFast expects these as class-level attributes.
PRETRAINED_VOCAB_FILES_MAP: Dict[str, Dict[str, str]] = {
    "vocab_file": {
        "deepsleep/deepsleep-2b": "https://huggingface.co/deepsleep/deepsleep-2b/resolve/main/vocab.json",
    },
    "merges_file": {
        "deepsleep/deepsleep-2b": "https://huggingface.co/deepsleep/deepsleep-2b/resolve/main/merges.txt",
    },
    "tokenizer_file": {
        "deepsleep/deepsleep-2b": "https://huggingface.co/deepsleep/deepsleep-2b/resolve/main/tokenizer.json",
    },
}

# ChatML chat template for multi-turn conversations.
CHAT_TEMPLATE: str = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|im_start|>assistant\\n' }}"
    "{% endif %}"
)


class DeepSleepTokenizer(PreTrainedTokenizerFast):
    """HuggingFace-registered fast tokenizer for the DeepSleep model.

    This class integrates with the HuggingFace auto-classes so that
    ``AutoTokenizer.from_pretrained()`` resolves to this tokenizer when
    the model config specifies ``tokenizer_class": "DeepSleepTokenizer"``.

    The tokenizer uses ChatML-style special tokens (``<|im_start|>`` /
    ``<|im_end|>``) for multi-turn conversation formatting.

    Example::

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("path/to/deepsleep/model")

        messages = [
            {"role": "system", "content": "You are a sleep health advisor."},
            {"role": "user", "content": "How can I improve my sleep quality?"},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        # <|im_start|>system\\nYou are a sleep health advisor.<|im_end|>\\n
        # <|im_start|>user\\nHow can I improve my sleep quality?<|im_end|>\\n
        # <|im_start|>assistant\\n
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    model_input_names = ["input_ids", "attention_mask"]
    # This is the key that HuggingFace uses to look up the tokenizer class
    # in the tokenizer_config.json.
    slow_tokenizer_class = None

    def __init__(
        self,
        vocab_file: Optional[str] = None,
        merges_file: Optional[str] = None,
        tokenizer_file: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        # Apply ChatML chat template by default
        kwargs.setdefault("chat_template", CHAT_TEMPLATE)

        super().__init__(
            vocab_file=vocab_file,
            merges_file=merges_file,
            tokenizer_file=tokenizer_file,
            **kwargs,
        )

        # Ensure special tokens are set
        self._ensure_special_tokens()

    def _ensure_special_tokens(self) -> None:
        """Set special tokens if they are not already configured.

        This is called after init to make sure the ChatML tokens are
        properly registered even when loading from a tokenizer.json that
        does not explicitly list them.
        """
        special_tokens = {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": "<pad>",
            "unk_token": "<unk>",
        }
        additional = ["<|im_start|>", "<|im_end|>", ""]

        for attr, token in special_tokens.items():
            current = getattr(self, attr, None)
            if current is None:
                setattr(self, attr, token)
                setattr(self, f"{attr}_id", self.convert_tokens_to_ids(token))

        # Add additional special tokens if not present
        existing = set(self.all_special_tokens)
        new_tokens = [t for t in additional if t not in existing]
        if new_tokens:
            self.add_special_tokens({"additional_special_tokens": new_tokens})

    def build_inputs_with_special_tokens(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
    ) -> List[int]:
        """Build model inputs by prepending BOS and appending EOS.

        This is used by the model to know where sequences begin and end.
        """
        bos_id = self.bos_token_id if self.bos_token_id is not None else -1
        eos_id = self.eos_token_id if self.eos_token_id is not None else -1

        output = list(token_ids_0)
        if bos_id >= 0:
            output = [bos_id] + output
        if eos_id >= 0:
            output = output + [eos_id]

        if token_ids_1 is not None:
            output = output + [eos_id] if eos_id >= 0 else output
            output = output + list(token_ids_1)
            if eos_id >= 0:
                output = output + [eos_id]

        return output

    def get_chat_template(self, *args: Any, **kwargs: Any) -> str:  # type: ignore[override]
        """Return the ChatML chat template string."""
        return CHAT_TEMPLATE


# ---------------------------------------------------------------------------
# Auto-class registration
# ---------------------------------------------------------------------------

def register_deepsleep_tokenizer() -> None:
    """Register DeepSleepTokenizer with HuggingFace AutoTokenizer.

    This allows ``AutoTokenizer.from_pretrained()`` to resolve
    ``"DeepSleepTokenizer"`` from tokenizer_config.json.
    """
    try:
        from transformers import AutoTokenizer

        AutoTokenizer.register(
            DeepSleepTokenizer,
            exist_ok=True,
        )
        logger.info("DeepSleepTokenizer registered with HuggingFace AutoTokenizer")
    except Exception as exc:
        logger.warning("Failed to register DeepSleepTokenizer: %s", exc)


# Auto-register on module import
register_deepsleep_tokenizer()
