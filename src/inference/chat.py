"""Interactive chat interface for the DeepSleep sleep-medicine LLM.

Provides a conversational interface with multi-turn history, streaming
generation, quantization support, and a sleep-medicine-focused system prompt.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Generator, Optional

import torch

logger = logging.getLogger(__name__)

_DEFAULT_SYSTEM_PROMPT = (
    "你是一位专业的睡眠健康顾问，拥有丰富的睡眠医学知识。"
    "请基于循证医学为用户提供准确、有用的睡眠健康建议。"
    "注意：你的建议仅供参考，不能替代专业医生的诊断和治疗。"
)


class DeepSleepChat:
    """Chat interface for the DeepSleep model.

    Loads a HuggingFace model with optional quantization, maintains
    conversation history, and supports both batch and streaming generation.

    Args:
        model_path: Path to the HuggingFace model directory.
        tokenizer_path: Path to the tokenizer directory.
        device: Device string (``"auto"``, ``"cuda"``, ``"cpu"``).
        max_length: Maximum total sequence length (prompt + generation).
        temperature: Sampling temperature (0.0 for greedy, > 0 for stochastic).
        top_p: Nucleus sampling probability threshold.
    """

    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        device: str = "auto",
        max_length: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> None:
        self._model_path = model_path
        self._tokenizer_path = tokenizer_path
        self._device = device
        self._max_length = max_length
        self._temperature = temperature
        self._top_p = top_p

        self._model: Optional[Any] = None
        self._tokenizer: Optional[Any] = None
        self._history: list[dict[str, str]] = []

        self._load_model()

    # -- model loading -----------------------------------------------------

    def _load_model(self) -> None:
        """Load model and tokenizer from disk."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading tokenizer from %s ...", self._tokenizer_path)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._tokenizer_path, trust_remote_code=True
        )
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        logger.info("Loading model from %s ...", self._model_path)
        device_map = "auto" if self._device == "auto" else self._device

        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_path,
            torch_dtype=torch.float16,
            device_map=device_map,
            trust_remote_code=True,
        )
        self._model.eval()
        logger.info("Model loaded successfully on %s", device_map)

    # -- public methods ----------------------------------------------------

    def generate(
        self,
        user_input: str,
        system_prompt: Optional[str] = None,
        history: Optional[list[dict[str, str]]] = None,
    ) -> str:
        """Generate a response for the given user input.

        Args:
            user_input: The user's message.
            system_prompt: Optional system prompt override.
            history: Optional pre-existing conversation history.  Each dict
                has ``role`` (``"user"`` or ``"assistant"``) and ``content``.
                When *None*, the internal history buffer is used.

        Returns:
            The model's response string.
        """
        effective_history = history if history is not None else self._history
        sys_prompt = system_prompt or _DEFAULT_SYSTEM_PROMPT

        # Build the conversation as a single prompt string
        prompt = self._build_prompt(sys_prompt, effective_history, user_input)

        device = next(self._model.parameters()).device
        inputs = self._tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=self._max_length
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Determine generation length budget
        input_len = inputs["input_ids"].shape[1]
        max_new_tokens = max(1, self._max_length - input_len)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=self._temperature if self._temperature > 0 else None,
                top_p=self._top_p if self._temperature > 0 else 1.0,
                do_sample=self._temperature > 0,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )

        response = self._tokenizer.decode(
            outputs[0][input_len:], skip_special_tokens=True
        ).strip()

        # Update internal history
        if history is None:
            self._history.append({"role": "user", "content": user_input})
            self._history.append({"role": "assistant", "content": response})

        return response

    def generate_stream(
        self,
        user_input: str,
        system_prompt: Optional[str] = None,
        history: Optional[list[dict[str, str]]] = None,
    ) -> Generator[str, None, None]:
        """Generate a response with streaming (token-by-token yield).

        Args:
            user_input: The user's message.
            system_prompt: Optional system prompt override.
            history: Optional conversation history.

        Yields:
            Incremental text chunks as they are generated.
        """
        from transformers import TextIteratorStreamer
        from threading import Thread

        effective_history = history if history is not None else self._history
        sys_prompt = system_prompt or _DEFAULT_SYSTEM_PROMPT
        prompt = self._build_prompt(sys_prompt, effective_history, user_input)

        device = next(self._model.parameters()).device
        inputs = self._tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=self._max_length
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        input_len = inputs["input_ids"].shape[1]
        max_new_tokens = max(1, self._max_length - input_len)

        streamer = TextIteratorStreamer(
            self._tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        generation_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "temperature": self._temperature if self._temperature > 0 else None,
            "top_p": self._top_p if self._temperature > 0 else 1.0,
            "do_sample": self._temperature > 0,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
            "streamer": streamer,
        }

        thread = Thread(target=self._model.generate, kwargs=generation_kwargs)
        thread.start()

        full_response = []
        for chunk in streamer:
            full_response.append(chunk)
            yield chunk

        thread.join()

        response_text = "".join(full_response).strip()

        if history is None:
            self._history.append({"role": "user", "content": user_input})
            self._history.append({"role": "assistant", "content": response_text})

    def chat_loop(self) -> None:
        """Start an interactive terminal chat loop.

        Supports commands:
            /quit   - Exit the chat.
            /clear  - Clear conversation history.
            /help   - Show available commands.
        """
        print("\n" + "=" * 60)
        print("  DeepSleep - Sleep Health AI Assistant")
        print("  Type /help for available commands, /quit to exit.")
        print("=" * 60 + "\n")

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            if user_input == "/quit":
                print("Goodbye!")
                break

            if user_input == "/clear":
                self._history.clear()
                print("[History cleared]\n")
                continue

            if user_input == "/help":
                print("Commands:")
                print("  /quit  - Exit the chat")
                print("  /clear - Clear conversation history")
                print("  /help  - Show this help message\n")
                continue

            print("DeepSleep: ", end="", flush=True)

            response_chunks: list[str] = []
            for chunk in self.generate_stream(user_input):
                print(chunk, end="", flush=True)
                response_chunks.append(chunk)

            print("\n")

    def clear_history(self) -> None:
        """Clear the conversation history."""
        self._history.clear()

    @property
    def history(self) -> list[dict[str, str]]:
        """Return a shallow copy of the current conversation history."""
        return list(self._history)

    # -- private helpers --------------------------------------------------

    def _build_prompt(
        self,
        system_prompt: str,
        history: list[dict[str, str]],
        user_input: str,
    ) -> str:
        """Assemble the full prompt from system prompt, history, and user message.

        Uses the ChatML template with ``<|im_start|>`` / ``<|im_end|>`` tokens.
        """
        parts: list[str] = []
        parts.append(f"<|im_start|>system\n{system_prompt}<|im_end|>")

        for turn in history:
            role = turn["role"]
            content = turn["content"]
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")

        parts.append(f"<|im_start|>user\n{user_input}<|im_end|>")
        parts.append("<|im_start|>assistant\n")

        return "\n".join(parts)
