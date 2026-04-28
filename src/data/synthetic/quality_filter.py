"""Quality filtering for synthetic sleep medicine training data.

Provides multiple filter strategies:
- LLM-based self-evaluation (accuracy, relevance, safety).
- Length-based filtering.
- Embedding-similarity deduplication.
- Medical safety validation.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

_SELF_EVAL_SYSTEM_PROMPT = """\
You are a medical data quality evaluator specializing in sleep medicine.
Given a synthetic conversation or Q&A pair, rate it on three dimensions:
1. **Accuracy** (1-5): Is the medical information factually correct?
2. **Relevance** (1-5): Does the response directly address the query?
3. **Safety** (1-5): Does it include appropriate disclaimers and avoid
   dangerous advice?

Respond with ONLY a JSON object:
{"accuracy": <int>, "relevance": <int>, "safety": <int>, "overall": <float>}
"""

_MEDICAL_SAFETY_PHRASES: list[str] = [
    # English
    "take X mg of",
    "prescribe yourself",
    "self-diagnose",
    "stop your medication immediately",
    "ignore your doctor",
    "you definitely have",
    "this will cure",
    "dangerous to combine",
    "overdose",
    # Chinese
    "服用X毫克",
    "自己诊断",
    "自行停药",
    "确诊为",
    "可以治愈",
    "无视医生的建议",
    "过量服用",
    "自行处方",
]


class SyntheticQualityFilter:
    """Filter and validate synthetic sleep medicine training data.

    Args:
        api_key: API key for the self-evaluation LLM.  Reads
            ``ANTHROPIC_API_KEY`` when *None*.
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
                "The 'anthropic' package is required for self-evaluation.  "
                "Install it with: pip install anthropic"
            ) from exc

        self._client = Anthropic(api_key=self._api_key)
        return self._client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def filter_by_self_eval(
        self,
        conversations: list[dict[str, Any]],
        min_score: float = 3.5,
    ) -> list[dict[str, Any]]:
        """Filter conversations using LLM-based self-evaluation.

        Each conversation is sent to the evaluator LLM which scores it on
        accuracy, relevance, and safety.  Only conversations with an overall
        score >= *min_score* are retained.

        Args:
            conversations: List of conversation dicts.
            min_score: Minimum overall score to keep a conversation.

        Returns:
            Filtered list of conversations that passed quality threshold.
        """
        if self._api_key is None:
            logger.warning(
                "No API key provided; self-evaluation filter returns all items."
            )
            return list(conversations)

        passed: list[dict[str, Any]] = []
        rejected = 0

        for idx, conv in enumerate(conversations):
            logger.info(
                "Self-evaluating conversation %d/%d ...", idx + 1, len(conversations)
            )
            try:
                score = self._evaluate_single(conv)
                conv["_quality_score"] = score

                if score >= min_score:
                    passed.append(conv)
                else:
                    rejected += 1
                    logger.info(
                        "Rejected conversation %d: score=%.1f < %.1f",
                        idx, score, min_score,
                    )
            except Exception as exc:
                logger.error("Self-evaluation failed for conversation %d: %s", idx, exc)
                # Keep on evaluation failure (fail-open)
                passed.append(conv)

        logger.info(
            "Self-evaluation complete: %d/%d passed, %d rejected",
            len(passed),
            len(conversations),
            rejected,
        )
        return passed

    def filter_by_length(
        self,
        conversations: list[dict[str, Any]],
        min_turns: int = 2,
        min_response_length: int = 50,
    ) -> list[dict[str, Any]]:
        """Filter conversations by length criteria.

        For conversation-type data, requires at least *min_turns* exchanges.
        For Q&A data, requires the answer to be at least *min_response_length*
        characters.

        Args:
            conversations: List of conversation/QA dicts.
            min_turns: Minimum number of message turns for conversations.
            min_response_length: Minimum response text length in characters.

        Returns:
            Filtered list meeting the length criteria.
        """
        passed: list[dict[str, Any]] = []

        for conv in conversations:
            conv_type = conv.get("type", "conversation")

            if conv_type == "qa":
                answer = conv.get("answer", "")
                if len(answer) >= min_response_length:
                    passed.append(conv)
                else:
                    logger.debug(
                        "Rejected QA: answer length %d < %d",
                        len(answer),
                        min_response_length,
                    )
            else:
                messages = conv.get("messages", [])
                if len(messages) >= min_turns:
                    # Also check response lengths
                    response_lengths = [
                        len(m.get("content", ""))
                        for m in messages
                        if m.get("role") == "assistant"
                    ]
                    if all(l >= min_response_length for l in response_lengths):
                        passed.append(conv)
                    else:
                        logger.debug("Rejected conversation: some responses too short")
                else:
                    logger.debug(
                        "Rejected conversation: %d turns < %d",
                        len(messages),
                        min_turns,
                    )

        logger.info(
            "Length filter: %d/%d passed", len(passed), len(conversations)
        )
        return passed

    def filter_duplicates(
        self,
        conversations: list[dict[str, Any]],
        threshold: float = 0.85,
    ) -> list[dict[str, Any]]:
        """Remove near-duplicate conversations using embedding similarity.

        Computes sentence embeddings for each conversation's text content and
        removes items whose cosine similarity exceeds *threshold* with an
        already-seen item.

        Falls back to simple character-level Jaccard similarity if no embedding
        model is available.

        Args:
            conversations: List of conversation/QA dicts.
            threshold: Similarity threshold above which items are considered
                duplicates.

        Returns:
            Deduplicated list of conversations.
        """
        texts = [self._extract_text(c) for c in conversations]

        try:
            return self._dedup_with_embeddings(conversations, texts, threshold)
        except (ImportError, Exception) as exc:
            logger.info(
                "Embedding-based dedup not available (%s), using Jaccard fallback.",
                exc,
            )
            return self._dedup_with_jaccard(conversations, texts, threshold)

    def validate_medical_safety(
        self,
        conversations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Check for dangerous medical advice in the conversations.

        Flags conversations that contain potentially harmful phrases such as
        specific dosage recommendations, self-diagnosis encouragement, or
        instructions to ignore professional medical advice.

        Args:
            conversations: List of conversation/QA dicts.

        Returns:
            List of conversations that passed safety validation.
        """
        safe: list[dict[str, Any]] = []
        flagged = 0

        for conv in conversations:
            text = self._extract_text(conv).lower()

            if any(phrase.lower() in text for phrase in _MEDICAL_SAFETY_PHRASES):
                flagged += 1
                logger.warning(
                    "Flagged conversation for unsafe medical content: %s...",
                    text[:100],
                )
                continue

            safe.append(conv)

        logger.info(
            "Medical safety validation: %d/%d passed, %d flagged",
            len(safe),
            len(conversations),
            flagged,
        )
        return safe

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _evaluate_single(self, conv: dict[str, Any]) -> float:
        """Evaluate a single conversation and return its overall score."""
        text = self._extract_text(conv)

        client = self._get_client()
        message = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=256,
            system=_SELF_EVAL_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"Evaluate:\n\n{text[:3000]}"}],
        )

        raw = message.content[0].text if message.content else ""

        try:
            # Strip code fences
            json_str = raw.strip()
            if json_str.startswith("```"):
                lines = json_str.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                json_str = "\n".join(lines)

            data = json.loads(json_str)
            return float(data.get("overall", 0.0))
        except (json.JSONDecodeError, TypeError, ValueError):
            logger.warning("Could not parse evaluation score from: %s", raw[:100])
            return 0.0

    @staticmethod
    def _extract_text(conv: dict[str, Any]) -> str:
        """Extract all text content from a conversation or QA dict."""
        if conv.get("type") == "qa":
            parts = [conv.get("question", ""), conv.get("answer", "")]
        else:
            messages = conv.get("messages", [])
            parts = [m.get("content", "") for m in messages]

        return "\n".join(parts)

    @staticmethod
    def _dedup_with_embeddings(
        conversations: list[dict[str, Any]],
        texts: list[str],
        threshold: float,
    ) -> list[dict[str, Any]]:
        """Deduplicate using sentence-transformer embeddings."""
        from sentence_transformers import SentenceTransformer
        import numpy as np

        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

        # Compute cosine similarity matrix
        sim_matrix = np.dot(embeddings, embeddings.T)

        keep_indices: list[int] = []
        n = len(conversations)

        for i in range(n):
            is_dup = False
            for j in keep_indices:
                if sim_matrix[i][j] > threshold:
                    is_dup = True
                    break
            if not is_dup:
                keep_indices.append(i)

        logger.info(
            "Embedding dedup: %d/%d kept (%d duplicates removed)",
            len(keep_indices),
            n,
            n - len(keep_indices),
        )
        return [conversations[i] for i in keep_indices]

    @staticmethod
    def _dedup_with_jaccard(
        conversations: list[dict[str, Any]],
        texts: list[str],
        threshold: float,
    ) -> list[dict[str, Any]]:
        """Deduplicate using character-level Jaccard similarity (fallback)."""

        def _shingles(text: str, k: int = 4) -> set[str]:
            return {text[i:i + k] for i in range(len(text) - k + 1)}

        def _jaccard(a: set[str], b: set[str]) -> float:
            if not a or not b:
                return 0.0
            intersection = a & b
            union = a | b
            return len(intersection) / len(union)

        shingle_sets = [_shingles(t) for t in texts]
        keep_indices: list[int] = []

        for i, s_i in enumerate(shingle_sets):
            is_dup = False
            for j in keep_indices:
                if _jaccard(s_i, shingle_sets[j]) > threshold:
                    is_dup = True
                    break
            if not is_dup:
                keep_indices.append(i)

        logger.info(
            "Jaccard dedup: %d/%d kept (%d duplicates removed)",
            len(keep_indices),
            len(conversations),
            len(conversations) - len(keep_indices),
        )
        return [conversations[i] for i in keep_indices]
