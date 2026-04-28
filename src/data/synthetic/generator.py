"""Synthetic data generator for sleep medicine conversations.

Uses an external LLM (e.g., Claude, GPT-4) to generate doctor-patient
dialogues, Q&A pairs, and report-style text from seed prompts.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "claude-3-5-sonnet-20241022"

_CONVERSATION_SYSTEM_PROMPT = """\
You are an expert sleep medicine doctor.  Given a patient's query, generate
a realistic, helpful doctor-patient conversation in ChatML format.

Rules:
- The conversation should be medically accurate and follow current AASM/ICSD-3 guidelines.
- Always include an appropriate medical disclaimer.
- The response should be in the same language as the patient's query.
- Be empathetic, professional, and educational.
- If the query is dangerous (e.g., asking for specific drug dosages), have the
  doctor politely decline and redirect to proper medical care.
- Format the output as a JSON array of messages with "role" and "content" keys.
  Roles should be "user" (patient) and "assistant" (doctor).
"""

_QA_SYSTEM_PROMPT = """\
You are a sleep medicine expert.  Given a question about sleep health, generate
a high-quality question-answer pair.

Rules:
- The answer should be medically accurate and based on current evidence.
- Include appropriate medical disclaimers.
- Match the language of the question.
- Format the output as a JSON object with "question" and "answer" keys.
"""

_MAX_RETRIES = 3
_RETRY_DELAY_BASE = 2.0
_RATE_LIMIT_DELAY = 1.0  # seconds between API calls


class SyntheticDataGenerator:
    """Generate synthetic sleep medicine training data using an external LLM.

    Produces doctor-patient conversations and Q&A pairs from seed prompts,
    with retry logic and rate-limit handling.

    Args:
        api_key: API key for the external LLM service.  Reads the
            ``ANTHROPIC_API_KEY`` environment variable when *None*.
        model: Model identifier for the external LLM.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = _DEFAULT_MODEL,
    ) -> None:
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise ValueError(
                "An API key is required.  Pass it explicitly or set "
                "the ANTHROPIC_API_KEY environment variable."
            )

        self._api_key = resolved_key
        self._model = model
        self._client: Optional[Any] = None
        self._last_call_time: float = 0.0

    # -- lazy client -------------------------------------------------------

    def _get_client(self) -> Any:
        """Lazily import and return the Anthropic client."""
        if self._client is not None:
            return self._client

        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise ImportError(
                "The 'anthropic' package is required.  Install it with: "
                "pip install anthropic"
            ) from exc

        self._client = Anthropic(api_key=self._api_key)
        return self._client

    # -- public methods ----------------------------------------------------

    def generate_conversation(
        self,
        prompt: dict[str, str],
        system_prompt: Optional[str] = None,
    ) -> dict[str, Any]:
        """Generate a doctor-patient conversation from a seed prompt.

        Args:
            prompt: A seed prompt dict with at least ``"prompt"`` and
                ``"language"`` keys.
            system_prompt: Optional system prompt override.

        Returns:
            A dict with ``messages`` (list of ``{"role", "content"}`` dicts),
            ``metadata`` (the original prompt metadata), and ``type``.
        """
        sys = system_prompt or _CONVERSATION_SYSTEM_PROMPT
        user_content = prompt["prompt"]

        raw = self._call_api(sys, user_content)
        messages = self._parse_conversation(raw)

        return {
            "messages": messages,
            "metadata": {
                "category": prompt.get("category", "unknown"),
                "difficulty": prompt.get("difficulty", "unknown"),
                "language": prompt.get("language", "unknown"),
                "source_seed": prompt["prompt"],
                "generator_model": self._model,
            },
            "type": "conversation",
        }

    def generate_qa_pair(
        self,
        prompt: dict[str, str],
    ) -> dict[str, Any]:
        """Generate a Q&A pair from a seed prompt.

        Args:
            prompt: A seed prompt dict with at least ``"prompt"`` and
                ``"language"`` keys.

        Returns:
            A dict with ``question``, ``answer``, and ``metadata``.
        """
        raw = self._call_api(_QA_SYSTEM_PROMPT, prompt["prompt"])
        qa = self._parse_qa(raw)

        return {
            "question": qa["question"],
            "answer": qa["answer"],
            "metadata": {
                "category": prompt.get("category", "unknown"),
                "difficulty": prompt.get("difficulty", "unknown"),
                "language": prompt.get("language", "unknown"),
                "source_seed": prompt["prompt"],
                "generator_model": self._model,
            },
            "type": "qa",
        }

    def generate_batch(
        self,
        prompts: list[dict[str, str]],
        output_path: str,
        max_concurrent: int = 5,
    ) -> list[dict[str, Any]]:
        """Generate data for a batch of seed prompts.

        Processes prompts sequentially with rate limiting, retries failed
        generations, and writes results incrementally to a JSONL file.

        Args:
            prompts: List of seed prompt dicts.
            output_path: Path to save the output JSONL file.
            max_concurrent: Maximum concurrent API calls (currently sequential
                but the parameter is reserved for future async support).

        Returns:
            List of generated data dicts.
        """
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        results: list[dict[str, Any]] = []
        failed = 0

        for idx, prompt in enumerate(prompts):
            logger.info(
                "Generating %d/%d (category=%s, lang=%s) ...",
                idx + 1,
                len(prompts),
                prompt.get("category", "?"),
                prompt.get("language", "?"),
            )

            generated = self._generate_with_retry(prompt)

            if generated is not None:
                results.append(generated)
                self._append_jsonl(out, generated)
            else:
                failed += 1
                logger.error(
                    "Failed to generate for prompt %d after retries", idx
                )

            if (idx + 1) % 10 == 0:
                logger.info(
                    "Progress: %d/%d generated, %d failed",
                    idx + 1,
                    len(prompts),
                    failed,
                )

        logger.info(
            "Batch complete: %d/%d generated, %d failed, saved to %s",
            len(results),
            len(prompts),
            failed,
            output_path,
        )
        return results

    # -- private helpers --------------------------------------------------

    def _call_api(self, system_prompt: str, user_content: str) -> str:
        """Send an API request with rate limiting."""
        self._rate_limit()

        client = self._get_client()
        message = client.messages.create(
            model=self._model,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_content}],
        )

        # Extract text content
        blocks = message.content
        text_parts: list[str] = []
        for block in blocks:
            if block.type == "text":
                text_parts.append(block.text)

        return "\n".join(text_parts)

    def _rate_limit(self) -> None:
        """Enforce a minimum interval between API calls."""
        now = time.monotonic()
        elapsed = now - self._last_call_time
        if elapsed < _RATE_LIMIT_DELAY:
            time.sleep(_RATE_LIMIT_DELAY - elapsed)
        self._last_call_time = time.monotonic()

    def _generate_with_retry(
        self,
        prompt: dict[str, str],
    ) -> Optional[dict[str, Any]]:
        """Generate data for a single prompt with retry logic."""
        prompt_type = prompt.get("category", "consultation")

        for attempt in range(_MAX_RETRIES):
            try:
                if prompt_type == "qa":
                    return self.generate_qa_pair(prompt)
                else:
                    return self.generate_conversation(prompt)
            except Exception as exc:
                delay = _RETRY_DELAY_BASE ** (attempt + 1)
                error_str = str(exc).lower()

                if "rate" in error_str or "429" in error_str:
                    delay = max(delay, 10.0)
                    logger.warning(
                        "Rate limited (attempt %d/%d), waiting %.1fs: %s",
                        attempt + 1,
                        _MAX_RETRIES,
                        delay,
                        exc,
                    )
                else:
                    logger.warning(
                        "Generation failed (attempt %d/%d): %s",
                        attempt + 1,
                        _MAX_RETRIES,
                        exc,
                    )

                if attempt < _MAX_RETRIES - 1:
                    time.sleep(delay)
                else:
                    logger.error(
                        "All retries exhausted for prompt: %s",
                        prompt["prompt"][:80],
                    )
                    return None

        return None

    @staticmethod
    def _parse_conversation(raw: str) -> list[dict[str, str]]:
        """Parse the API response into a list of message dicts."""
        # Try to extract JSON from the response
        json_str = raw.strip()

        # Remove markdown code fences if present
        if json_str.startswith("```"):
            lines = json_str.split("\n")
            # Remove first and last lines (code fence)
            lines = [l for l in lines if not l.strip().startswith("```")]
            json_str = "\n".join(lines)

        try:
            data = json.loads(json_str)
            if isinstance(data, list):
                messages = data
            elif isinstance(data, dict) and "messages" in data:
                messages = data["messages"]
            else:
                # Treat as a single assistant response
                messages = [{"role": "assistant", "content": raw}]
        except json.JSONDecodeError:
            logger.warning("Could not parse conversation as JSON, wrapping as assistant message")
            messages = [{"role": "assistant", "content": raw}]

        # Validate message format
        validated: list[dict[str, str]] = []
        for msg in messages:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                validated.append({
                    "role": str(msg["role"]),
                    "content": str(msg["content"]),
                })

        return validated

    @staticmethod
    def _parse_qa(raw: str) -> dict[str, str]:
        """Parse the API response into a Q&A dict."""
        json_str = raw.strip()

        if json_str.startswith("```"):
            lines = json_str.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            json_str = "\n".join(lines)

        try:
            data = json.loads(json_str)
            return {
                "question": str(data.get("question", "")),
                "answer": str(data.get("answer", "")),
            }
        except json.JSONDecodeError:
            logger.warning("Could not parse Q&A as JSON, returning raw text")
            return {"question": "", "answer": raw}

    @staticmethod
    def _append_jsonl(path: Path, data: dict[str, Any]) -> None:
        """Append a record to a JSONL file."""
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(data, ensure_ascii=False) + "\n")
