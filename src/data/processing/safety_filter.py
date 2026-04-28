"""
Content safety filtering for the DeepSleep LLM data processing pipeline.

Detects and handles personally identifiable information (PII), harmful content,
non-text noise, and very short documents.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# PII detection patterns
# Phone numbers: Chinese mobile (11 digits starting with 1), landlines, international
_PHONE_PATTERNS = [
    re.compile(r"1[3-9]\d{9}"),                       # Chinese mobile
    re.compile(r"\d{3,4}[-\s]?\d{7,8}"),              # Chinese landline
    re.compile(r"\+?1?\s*\(?\d{3}\)?[-\s.]?\d{3}[-\s.]?\d{4}"),  # US/International
    re.compile(r"\(\d{2,4}\)\s*\d{5,8}"),             # Parenthesized area codes
]

# ID numbers: Chinese national ID (18 digits with optional X), passport
_ID_PATTERNS = [
    re.compile(r"\b[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]\b"),
    # Chinese national ID: 6 digit area + YYYYMMDD + 3 digit sequence + check digit
]

# Email addresses
_EMAIL_PATTERN = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
)

# IP addresses
_IP_PATTERN = re.compile(
    r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
)

# Harmful content keywords (basic blocklist approach)
_HARMFUL_KEYWORDS: list[str] = [
    # Violence-related (non-medical context)
    "恐怖袭击", "制造炸弹", "炸弹制作", "自制武器",
    "杀人方法", "自杀方法", "自残方法",
    # Illegal activities
    "洗钱方法", "毒品制作", "毒品走私", "伪造货币",
    # Explicit content
    "child pornography", "child exploitation",
    "儿童色情", "恋童",
]

# Non-text noise patterns
_NOISE_PATTERNS = [
    re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"),          # Control characters
    re.compile(r"&#\d+;"),                                       # HTML entities (raw)
    re.compile(r"\\u[0-9a-fA-F]{4}"),                            # Raw unicode escapes
    re.compile(r"(.)\1{20,}"),                                   # Repeated characters (20+)
]

# Minimum document length after cleaning
_MIN_DOCUMENT_LENGTH = 100


class SafetyFilter:
    """Content safety filter for training data.

    Provides filtering and masking of sensitive content in training data,
    including PII detection, harmful content blocklisting, noise removal,
    and minimum length requirements.

    Attributes:
        pii_action: How to handle PII. "mask" replaces PII with [REDACTED],
            "reject" removes documents containing PII.
        harmful_action: How to handle harmful content. "reject" removes documents,
            "mask" replaces harmful passages.
        min_length: Minimum character length for documents after cleaning.
    """

    def __init__(
        self,
        pii_action: str = "mask",
        harmful_action: str = "reject",
        min_length: int = _MIN_DOCUMENT_LENGTH,
    ) -> None:
        if pii_action not in ("mask", "reject"):
            raise ValueError(f"pii_action must be 'mask' or 'reject', got '{pii_action}'")
        if harmful_action not in ("mask", "reject"):
            raise ValueError(f"harmful_action must be 'mask' or 'reject', got '{harmful_action}'")

        self.pii_action = pii_action
        self.harmful_action = harmful_action
        self.min_length = min_length

    def mask_pii(self, text: str) -> str:
        """Replace personally identifiable information with [REDACTED] placeholders.

        Detects and masks phone numbers, national ID numbers, email addresses,
        and IP addresses in the input text.

        Args:
            text: Input text potentially containing PII.

        Returns:
            Text with PII replaced by [REDACTED_PHONE], [REDACTED_ID],
            [REDACTED_EMAIL], or [REDACTED_IP] placeholders.
        """
        masked = text

        # Mask phone numbers
        for pattern in _PHONE_PATTERNS:
            masked = pattern.sub("[REDACTED_PHONE]", masked)

        # Mask national ID numbers
        for pattern in _ID_PATTERNS:
            masked = pattern.sub("[REDACTED_ID]", masked)

        # Mask email addresses
        masked = _EMAIL_PATTERN.sub("[REDACTED_EMAIL]", masked)

        # Mask IP addresses
        masked = _IP_PATTERN.sub("[REDACTED_IP]", masked)

        return masked

    def _contains_pii(self, text: str) -> bool:
        """Check whether text contains any PII patterns.

        Args:
            text: Input text to check.

        Returns:
            True if any PII pattern is detected, False otherwise.
        """
        for pattern in _PHONE_PATTERNS:
            if pattern.search(text):
                return True

        for pattern in _ID_PATTERNS:
            if pattern.search(text):
                return True

        if _EMAIL_PATTERN.search(text):
            return True

        if _IP_PATTERN.search(text):
            return True

        return False

    def _contains_harmful_content(self, text: str) -> bool:
        """Check whether text contains harmful keywords.

        Args:
            text: Input text to check.

        Returns:
            True if any harmful keyword is found, False otherwise.
        """
        text_lower = text.lower()
        for keyword in _HARMFUL_KEYWORDS:
            if keyword.lower() in text_lower:
                return True
        return False

    def _has_excessive_noise(self, text: str) -> bool:
        """Check whether text contains excessive noise patterns.

        Args:
            text: Input text to check.

        Returns:
            True if noise patterns exceed acceptable thresholds.
        """
        # Check for control characters
        control_matches = _NOISE_PATTERNS[0].findall(text)
        if len(control_matches) > 10:
            return True

        # Check for repeated characters
        repeat_matches = _NOISE_PATTERNS[3].findall(text)
        if len(repeat_matches) > 5:
            return True

        # Check for excessive HTML entity sequences
        html_entities = _NOISE_PATTERNS[1].findall(text)
        if len(html_entities) > 20:
            return True

        return False

    def _clean_noise(self, text: str) -> str:
        """Remove common noise patterns from text.

        Args:
            text: Input text.

        Returns:
            Cleaned text with noise patterns removed.
        """
        cleaned = text
        # Remove control characters
        cleaned = _NOISE_PATTERNS[0].sub(" ", cleaned)
        # Collapse repeated characters (max 3 repetitions)
        cleaned = re.sub(r"(.)\1{4,}", r"\1\1\1\1", cleaned)
        return cleaned

    def _check_document_length(self, text: str) -> bool:
        """Check whether text meets minimum length requirement.

        Args:
            text: Input text.

        Returns:
            True if text length meets minimum requirement.
        """
        return len(text.strip()) >= self.min_length

    def filter_harmful_content(
        self,
        documents: list[dict[str, Any]],
        text_field: str = "text",
    ) -> list[dict[str, Any]]:
        """Filter documents for content safety.

        Applies a pipeline of safety checks:
        1. PII detection (mask or reject based on configuration)
        2. Harmful content detection (mask or reject)
        3. Noise removal
        4. Minimum length check

        Args:
            documents: List of document dictionaries.
            text_field: Field name containing document text.

        Returns:
            List of documents that pass all safety checks.
            When pii_action is "mask", PII is replaced in-place in the text.
        """
        filtered: list[dict[str, Any]] = []
        total = len(documents)
        stats = {
            "pii_detected": 0,
            "harmful_detected": 0,
            "noise_detected": 0,
            "too_short": 0,
        }

        logger.info("Filtering %d documents for content safety", total)

        for idx, doc in enumerate(documents):
            text = str(doc.get(text_field, ""))

            if not text.strip():
                continue

            # Check for harmful content first (reject or mask)
            if self._contains_harmful_content(text):
                stats["harmful_detected"] += 1
                if self.harmful_action == "reject":
                    logger.debug("Document %d rejected: harmful content", idx)
                    continue
                # For mask action, we still reject documents with harmful content
                # as partial masking may leave dangerous content
                continue

            # Check for PII
            has_pii = self._contains_pii(text)
            if has_pii:
                stats["pii_detected"] += 1
                if self.pii_action == "reject":
                    logger.debug("Document %d rejected: contains PII", idx)
                    continue
                # Mask PII in-place
                text = self.mask_pii(text)

            # Check for noise
            if self._has_excessive_noise(text):
                stats["noise_detected"] += 1
                text = self._clean_noise(text)

            # Check minimum length
            if not self._check_document_length(text):
                stats["too_short"] += 1
                logger.debug(
                    "Document %d rejected: too short (%d chars)",
                    idx,
                    len(text.strip()),
                )
                continue

            # Create a new dict with cleaned text (immutable pattern)
            filtered_doc = {**doc, text_field: text}
            filtered.append(filtered_doc)

            if (idx + 1) % 10000 == 0:
                logger.info("Processed %d/%d documents", idx + 1, total)

        logger.info(
            "Safety filtering complete: %d/%d documents kept. "
            "Rejected: %d PII, %d harmful, %d noise, %d too short",
            len(filtered),
            total,
            stats["pii_detected"],
            stats["harmful_detected"],
            stats["noise_detected"],
            stats["too_short"],
        )

        return filtered

    def get_safety_report(
        self,
        documents: list[dict[str, Any]],
        text_field: str = "text",
    ) -> dict[str, Any]:
        """Generate a safety analysis report without modifying documents.

        Args:
            documents: List of document dictionaries to analyze.
            text_field: Field name containing document text.

        Returns:
            Dictionary containing safety statistics:
                - total_documents: Number of documents analyzed.
                - pii_documents: Number containing PII.
                - harmful_documents: Number with harmful content.
                - noisy_documents: Number with excessive noise.
                - short_documents: Number below minimum length.
                - pii_examples: Sample of detected PII patterns.
                - safe_documents: Number passing all checks.
        """
        total = len(documents)
        pii_count = 0
        harmful_count = 0
        noisy_count = 0
        short_count = 0
        pii_examples: list[str] = []

        for doc in documents:
            text = str(doc.get(text_field, ""))
            if not text.strip():
                short_count += 1
                continue

            if self._contains_harmful_content(text):
                harmful_count += 1

            if self._contains_pii(text):
                pii_count += 1
                # Collect PII examples (masked)
                masked = self.mask_pii(text)
                if len(pii_examples) < 10:
                    pii_examples.append(masked[:200])

            if self._has_excessive_noise(text):
                noisy_count += 1

            if not self._check_document_length(text):
                short_count += 1

        report = {
            "total_documents": total,
            "pii_documents": pii_count,
            "harmful_documents": harmful_count,
            "noisy_documents": noisy_count,
            "short_documents": short_count,
            "safe_documents": total - max(pii_count, 1) - harmful_count - short_count,
            "pii_examples": pii_examples,
            "pii_percentage": round(pii_count / total * 100, 2) if total > 0 else 0.0,
            "harmful_percentage": round(harmful_count / total * 100, 2) if total > 0 else 0.0,
        }

        logger.info("Safety report: %s", report)
        return report


if __name__ == "__main__":
    import argparse
    import json as _json

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Content safety filtering")
    parser.add_argument("--mode", choices=["filter", "report"], default="filter")
    parser.add_argument("--input", required=True, help="Input JSONL file path")
    parser.add_argument("--output", required=True, help="Output JSONL file path")
    parser.add_argument("--text-field", default="text", help="Field name for document text")
    parser.add_argument("--pii-action", choices=["mask", "reject"], default="mask")
    parser.add_argument("--harmful-action", choices=["mask", "reject"], default="reject")
    parser.add_argument("--min-length", type=int, default=100, help="Minimum document length")
    args = parser.parse_args()

    safety_filter = SafetyFilter(
        pii_action=args.pii_action,
        harmful_action=args.harmful_action,
        min_length=args.min_length,
    )

    documents = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    documents.append(_json.loads(line))
                except _json.JSONDecodeError:
                    logger.warning("Skipping malformed JSON line")

    logger.info("Loaded %d documents", len(documents))

    if args.mode == "filter":
        filtered = safety_filter.filter_harmful_content(
            documents, text_field=args.text_field
        )

        with open(args.output, "w", encoding="utf-8") as f:
            for doc in filtered:
                f.write(_json.dumps(doc, ensure_ascii=False) + "\n")

        logger.info("Wrote %d safe documents to %s", len(filtered), args.output)

    elif args.mode == "report":
        report = safety_filter.get_safety_report(documents, text_field=args.text_field)

        with open(args.output, "w", encoding="utf-8") as f:
            _json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info("Safety report written to %s", args.output)
        print("\n=== Safety Report ===")
        for key, value in report.items():
            if key != "pii_examples":
                print(f"  {key}: {value}")

    logger.info("Done")
