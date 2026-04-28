"""Data augmentation for synthetic sleep medicine training data.

Provides transformation strategies: paraphrasing, follow-up addition,
difficulty adjustment, and language translation.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

_PARAPHRASE_PROMPT = """\
Rephrase the following sleep medicine conversation while preserving all
medical accuracy, disclaimers, and key information.  Maintain the same
language and tone.  Output the result as a JSON array of messages with
"role" and "content" keys.

Original conversation:
{conversation}
"""

_ADD_FOLLOWUP_PROMPT = """\
Given the following sleep medicine conversation, add one natural follow-up
question from the patient and the doctor's response.  The follow-up should
be medically relevant and build on the previous discussion.  Maintain the
same language.  Output the COMPLETE updated conversation (original + follow-up)
as a JSON array of messages.

Original conversation:
{conversation}
"""

_DIFFICULTY_PROMPT = """\
Adjust the difficulty level of the following sleep medicine conversation to:
{difficulty}

Difficulty levels:
- simple: Use plain language, avoid jargon, keep explanations brief.
- medium: Include some medical terminology with explanations.
- complex: Use clinical terminology, include detailed pathophysiology,
  reference specific guidelines (AASM, ICSD-3), discuss differential
  diagnoses and evidence levels.

Maintain the same language and all medical accuracy.  Output the updated
conversation as a JSON array of messages.

Original conversation:
{conversation}
"""

_TRANSLATE_PROMPT = """\
Translate the following sleep medicine conversation to {target_language}.
Preserve all medical accuracy, drug names, and clinical terminology.
Convert any culture-specific health system references appropriately.
Maintain appropriate medical disclaimers in the target language.

Output the translated conversation as a JSON array of messages.

Original:
{conversation}
"""


class DataAugmenter:
    """Augment synthetic sleep medicine training data.

    Uses an external LLM to transform conversations while preserving
    medical accuracy and safety.

    Args:
        api_key: API key for the external LLM.  Reads ``ANTHROPIC_API_KEY``
            when *None*.
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client: Optional[Any] = None

    # -- lazy client -------------------------------------------------------

    def _get_client(self) -> Any:
        """Lazily import and return the Anthropic client."""
        if self._client is not None:
            return self._client

        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise ImportError(
                "The 'anthropic' package is required for data augmentation.  "
                "Install it with: pip install anthropic"
            ) from exc

        self._client = Anthropic(api_key=self._api_key)
        return self._client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def paraphrase(self, conversation: dict[str, Any]) -> dict[str, Any]:
        """Rephrase a conversation while preserving medical accuracy.

        Args:
            conversation: A conversation dict with a ``messages`` list.

        Returns:
            A new conversation dict with paraphrased messages.
        """
        text = self._messages_to_text(conversation.get("messages", []))
        prompt = _PARAPHRASE_PROMPT.format(conversation=text)

        raw = self._call_api(prompt)
        new_messages = self._parse_messages(raw)

        return self._build_augmented(conversation, new_messages, "paraphrase")

    def add_followup(self, conversation: dict[str, Any]) -> dict[str, Any]:
        """Add a follow-up question and response to a conversation.

        Args:
            conversation: A conversation dict with a ``messages`` list.

        Returns:
            A new conversation dict with an additional turn.
        """
        text = self._messages_to_text(conversation.get("messages", []))
        prompt = _ADD_FOLLOWUP_PROMPT.format(conversation=text)

        raw = self._call_api(prompt)
        new_messages = self._parse_messages(raw)

        return self._build_augmented(conversation, new_messages, "followup")

    def change_difficulty(
        self,
        conversation: dict[str, Any],
        target_difficulty: str,
    ) -> dict[str, Any]:
        """Adjust the complexity level of a conversation.

        Args:
            conversation: A conversation dict.
            target_difficulty: One of ``"simple"``, ``"medium"``, ``"complex"``.

        Returns:
            A new conversation dict at the target difficulty.

        Raises:
            ValueError: If *target_difficulty* is not a valid level.
        """
        valid_levels = ("simple", "medium", "complex")
        if target_difficulty not in valid_levels:
            raise ValueError(
                f"Invalid difficulty '{target_difficulty}'. "
                f"Choose from: {', '.join(valid_levels)}"
            )

        text = self._messages_to_text(conversation.get("messages", []))
        prompt = _DIFFICULTY_PROMPT.format(
            difficulty=target_difficulty,
            conversation=text,
        )

        raw = self._call_api(prompt)
        new_messages = self._parse_messages(raw)

        augmented = self._build_augmented(conversation, new_messages, "difficulty_change")
        # Update difficulty in metadata
        augmented["metadata"]["difficulty"] = target_difficulty
        return augmented

    def translate(
        self,
        conversation: dict[str, Any],
        target_language: str,
    ) -> dict[str, Any]:
        """Translate a conversation between Chinese and English.

        Args:
            conversation: A conversation dict.
            target_language: Target language (``"zh"`` or ``"en"``).

        Returns:
            A new conversation dict in the target language.

        Raises:
            ValueError: If *target_language* is not ``"zh"`` or ``"en"``.
        """
        if target_language not in ("zh", "en"):
            raise ValueError(
                f"Unsupported target language '{target_language}'. "
                "Choose from: zh, en"
            )

        lang_map = {"zh": "Chinese", "en": "English"}
        text = self._messages_to_text(conversation.get("messages", []))
        prompt = _TRANSLATE_PROMPT.format(
            target_language=lang_map[target_language],
            conversation=text,
        )

        raw = self._call_api(prompt)
        new_messages = self._parse_messages(raw)

        augmented = self._build_augmented(conversation, new_messages, "translation")
        augmented["metadata"]["language"] = target_language
        return augmented

    def augment_batch(
        self,
        conversations: list[dict[str, Any]],
        augmentations_per_sample: int = 2,
    ) -> list[dict[str, Any]]:
        """Apply random augmentations to a batch of conversations.

        For each input conversation, applies *augmentations_per_sample*
        random augmentation strategies and returns all augmented items.

        Args:
            conversations: List of conversation dicts.
            augmentations_per_sample: Number of augmented variants to produce
                per input conversation.

        Returns:
            List of augmented conversation dicts.
        """
        import random

        rng = random.Random(42)
        strategies = ["paraphrase", "followup", "difficulty_change", "translate"]
        all_augmented: list[dict[str, Any]] = []

        for idx, conv in enumerate(conversations):
            logger.info(
                "Augmenting conversation %d/%d (%d variants) ...",
                idx + 1,
                len(conversations),
                augmentations_per_sample,
            )

            for _ in range(augmentations_per_sample):
                strategy = rng.choice(strategies)

                try:
                    if strategy == "paraphrase":
                        augmented = self.paraphrase(conv)
                    elif strategy == "followup":
                        augmented = self.add_followup(conv)
                    elif strategy == "difficulty_change":
                        current_diff = conv.get("metadata", {}).get(
                            "difficulty", "medium"
                        )
                        levels = ["simple", "medium", "complex"]
                        target = rng.choice([l for l in levels if l != current_diff])
                        augmented = self.change_difficulty(conv, target)
                    else:
                        current_lang = conv.get("metadata", {}).get("language", "zh")
                        target_lang = "en" if current_lang == "zh" else "zh"
                        augmented = self.translate(conv, target_lang)

                    all_augmented.append(augmented)

                except Exception as exc:
                    logger.error(
                        "Augmentation failed for conversation %d (%s): %s",
                        idx,
                        strategy,
                        exc,
                    )

        logger.info(
            "Batch augmentation complete: %d augmented from %d originals",
            len(all_augmented),
            len(conversations),
        )
        return all_augmented

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _call_api(self, user_content: str) -> str:
        """Send an API request and return the text response."""
        client = self._get_client()
        message = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=4096,
            messages=[{"role": "user", "content": user_content}],
        )

        blocks = message.content
        text_parts: list[str] = []
        for block in blocks:
            if block.type == "text":
                text_parts.append(block.text)

        return "\n".join(text_parts)

    @staticmethod
    def _messages_to_text(messages: list[dict[str, str]]) -> str:
        """Convert a messages list to a readable text format."""
        lines: list[str] = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            lines.append(f"[{role}]: {content}")
        return "\n\n".join(lines)

    @staticmethod
    def _parse_messages(raw: str) -> list[dict[str, str]]:
        """Parse JSON messages from the API response."""
        import json

        json_str = raw.strip()
        if json_str.startswith("```"):
            lines = json_str.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            json_str = "\n".join(lines)

        try:
            data = json.loads(json_str)
            if isinstance(data, list):
                msgs = data
            elif isinstance(data, dict) and "messages" in data:
                msgs = data["messages"]
            else:
                return [{"role": "assistant", "content": raw}]

            return [
                {"role": str(m["role"]), "content": str(m["content"])}
                for m in msgs
                if isinstance(m, dict) and "role" in m and "content" in m
            ]
        except json.JSONDecodeError:
            return [{"role": "assistant", "content": raw}]

    @staticmethod
    def _build_augmented(
        original: dict[str, Any],
        new_messages: list[dict[str, str]],
        augmentation_type: str,
    ) -> dict[str, Any]:
        """Build an augmented conversation dict from the original metadata."""
        metadata = dict(original.get("metadata", {}))
        metadata["augmentation_type"] = augmentation_type
        metadata["augmented_from"] = metadata.get("source_seed", "")

        return {
            "messages": new_messages,
            "metadata": metadata,
            "type": "conversation",
        }
