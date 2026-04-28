"""DeepSleep tokenizer package.

Provides ``DeepSleepTokenizer``, a HuggingFace-compatible fast tokenizer
with built-in ChatML chat template for multi-turn conversations.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Special tokens used by the DeepSleep model.
SPECIAL_TOKENS: List[str] = [
    "<|im_start|>",
    "<|im_end|>",
    "",
    "<pad>",
    "<unk>",
    "<s>",
    "</s>",
]

SPECIAL_TOKENS_MAP: Dict[str, str] = {
    "bos_token": "<s>",
    "eos_token": "</s>",
    "pad_token": "<pad>",
    "unk_token": "<unk>",
    "additional_special_tokens": ["<|im_start|>", "<|im_end|>", ""],
}


class DeepSleepTokenizer:
    """HuggingFace-compatible tokenizer for the DeepSleep model.

    Wraps the trained BPE tokenizer with ChatML template support for
    multi-turn conversations using ``<|im_start|>`` / ``<|im_end|>`` tokens.

    This class is a thin wrapper that delegates to ``PreTrainedTokenizerFast``
    when available.  For production use, prefer ``AutoTokenizer`` with the
    model path.

    Usage::

        tokenizer = DeepSleepTokenizer.from_pretrained("src/model/")
        prompt = tokenizer.apply_chat_template([
            {"role": "system", "content": "You are a sleep health advisor."},
            {"role": "user", "content": "How can I improve my sleep?"},
        ])
    """

    def __init__(
        self,
        tokenizer_path: Optional[str] = None,
        vocab_file: Optional[str] = None,
        merges_file: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self._backend: Optional[Any] = None

        # Resolve path
        if tokenizer_path is not None:
            path = Path(tokenizer_path)
            if path.is_dir():
                vocab_file = str(path / "vocab.json")
                merges_file = str(path / "merges.txt")

        if vocab_file and merges_file:
            self._load_from_legacy(vocab_file, merges_file, **kwargs)
        elif tokenizer_path is not None:
            self._load_from_json(tokenizer_path, **kwargs)
        else:
            logger.warning(
                "No tokenizer files provided.  DeepSleepTokenizer is not functional."
            )

    def _load_from_json(self, path: str, **kwargs: Any) -> None:
        """Load from a tokenizer.json file (fast tokenizer format)."""
        try:
            from transformers import PreTrainedTokenizerFast

            self._backend = PreTrainedTokenizerFast(
                tokenizer_file=path,
                **kwargs,
            )
            self._configure_backend()
        except ImportError:
            logger.error(
                "transformers is required for tokenizer loading.  "
                "Install it with: pip install transformers"
            )

    def _load_from_legacy(
        self,
        vocab_file: str,
        merges_file: str,
        **kwargs: Any,
    ) -> None:
        """Load from legacy vocab.json + merges.txt files."""
        try:
            from tokenizers import Tokenizer
            from transformers import PreTrainedTokenizerFast

            tok = Tokenizer.from_file(vocab_file.replace("vocab.json", "tokenizer.json"))
            self._backend = PreTrainedTokenizerFast(
                tokenizer_object=tok,
                **kwargs,
            )
            self._configure_backend()
        except Exception as exc:
            logger.warning("Failed to load tokenizer from legacy format: %s", exc)

    def _configure_backend(self) -> None:
        """Set up special tokens and chat template on the backend tokenizer."""
        if self._backend is None:
            return

        # Set special tokens
        if hasattr(self._backend, "bos_token") and self._backend.bos_token is None:
            self._backend.bos_token = "<s>"
            self._backend.bos_token_id = self._backend.convert_tokens_to_ids("<s>")

        if hasattr(self._backend, "eos_token") and self._backend.eos_token is None:
            self._backend.eos_token = "</s>"
            self._backend.eos_token_id = self._backend.convert_tokens_to_ids("</s>")

        if hasattr(self._backend, "pad_token") and self._backend.pad_token is None:
            self._backend.pad_token = "<pad>"
            self._backend.pad_token_id = self._backend.convert_tokens_to_ids("<pad>")

        if hasattr(self._backend, "unk_token") and self._backend.unk_token is None:
            self._backend.unk_token = "<unk>"
            self._backend.unk_token_id = self._backend.convert_tokens_to_ids("<unk>")

        # Set additional special tokens
        if hasattr(self._backend, "add_special_tokens"):
            existing = set(self._backend.all_special_tokens)
            new_tokens = [t for t in SPECIAL_TOKENS if t not in existing]
            if new_tokens:
                self._backend.add_special_tokens(
                    {"additional_special_tokens": new_tokens}
                )

        # Set chat template
        chat_template = (
            "{% for message in messages %}"
            "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|im_start|>assistant\\n' }}"
            "{% endif %}"
        )
        if hasattr(self._backend, "chat_template"):
            self._backend.chat_template = chat_template

    # ------------------------------------------------------------------
    # Public API - delegate to backend
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(cls, path: str, **kwargs: Any) -> "DeepSleepTokenizer":
        """Load a tokenizer from a directory path.

        Args:
            path: Directory containing tokenizer files.
            **kwargs: Additional arguments passed to the backend.

        Returns:
            A configured DeepSleepTokenizer instance.
        """
        instance = cls.__new__(cls)
        instance._backend = None

        # Try fast tokenizer
        try:
            from transformers import PreTrainedTokenizerFast

            p = Path(path)
            tok_json = p / "tokenizer.json"
            if tok_json.exists():
                instance._backend = PreTrainedTokenizerFast.from_pretrained(
                    path, **kwargs
                )
            else:
                instance._backend = PreTrainedTokenizerFast.from_pretrained(
                    path, **kwargs
                )
            instance._configure_backend()
            logger.info("Loaded tokenizer from %s", path)
            return instance
        except Exception as exc:
            logger.warning("from_pretrained failed: %s, falling back to manual load", exc)

        # Fallback to manual loading
        instance.__init__(tokenizer_path=path, **kwargs)
        return instance

    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
        tokenize: bool = False,
        **kwargs: Any,
    ) -> Union[str, Any]:
        """Format a list of messages into a ChatML prompt.

        Args:
            messages: List of dicts with ``role`` and ``content`` keys.
            add_generation_prompt: Whether to append the assistant prompt.
            tokenize: Whether to tokenize the output (returns tensor ids).
            **kwargs: Additional arguments for tokenization.

        Returns:
            Formatted prompt string, or token ids if *tokenize* is True.
        """
        if self._backend is not None and hasattr(self._backend, "apply_chat_template"):
            return self._backend.apply_chat_template(
                messages,
                add_generation_prompt=add_generation_prompt,
                tokenize=tokenize,
                **kwargs,
            )

        # Manual template application
        parts: List[str] = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")

        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")

        text = "".join(parts)

        if tokenize and self._backend is not None:
            return self._backend.encode(text, **kwargs)

        return text

    def encode(self, text: str, **kwargs: Any) -> List[int]:
        """Encode text to token ids."""
        if self._backend is None:
            raise RuntimeError("Tokenizer backend not loaded")
        return self._backend.encode(text, **kwargs)

    def decode(self, token_ids: List[int], **kwargs: Any) -> str:
        """Decode token ids to text."""
        if self._backend is None:
            raise RuntimeError("Tokenizer backend not loaded")
        return self._backend.decode(token_ids, **kwargs)

    def __call__(self, text: Any, **kwargs: Any) -> Any:
        """Forward calls to the backend tokenizer."""
        if self._backend is None:
            raise RuntimeError("Tokenizer backend not loaded")
        return self._backend(text, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the backend."""
        backend = object.__getattribute__(self, "_backend")
        if backend is not None and hasattr(backend, name):
            return getattr(backend, name)
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    @property
    def pad_token_id(self) -> Optional[int]:
        """Return the pad token id."""
        if self._backend is not None:
            return self._backend.pad_token_id
        return None

    @property
    def eos_token_id(self) -> Optional[int]:
        """Return the eos token id."""
        if self._backend is not None:
            return self._backend.eos_token_id
        return None

    @property
    def vocab_size(self) -> int:
        """Return the vocabulary size."""
        if self._backend is not None:
            return self._backend.vocab_size
        return 0
