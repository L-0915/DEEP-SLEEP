"""
Language detection module for the DeepSleep LLM data processing pipeline.

Provides language identification using fasttext with support for Chinese
and English language filtering, including handling of variant codes.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Map fasttext language codes to standardized codes.
# fasttext may return codes like "cmn", "zho", "eng" etc.
_FASTTEXT_CODE_MAP: dict[str, str] = {
    "cmn": "zh",
    "zho": "zh",
    "lzh": "zh",
    "yue": "zh",
    "wuu": "zh",
    "hak": "zh",
    "nan": "zh",
    "eng": "en",
    "en": "en",
}

# Default set of allowed languages
_DEFAULT_ALLOWED: list[str] = ["zh", "en"]


class LanguageDetector:
    """Language identification using fasttext.

    Wraps the fasttext language identification model to classify text
    by language and filter documents by allowed language set. Handles
    variant language codes (e.g., "cmn"/"zho" -> "zh").

    Attributes:
        model_path: Path to the fasttext language identification model.
        allowed_languages: List of standardized language codes to allow.
    """

    def __init__(
        self,
        model_path: str = "lid.176.bin",
        allowed_languages: list[str] | None = None,
    ) -> None:
        self.model_path = model_path
        self.allowed_languages = allowed_languages if allowed_languages is not None else _DEFAULT_ALLOWED
        self._model = None

    def _load_model(self) -> Any:
        """Lazily load the fasttext model.

        Returns:
            The loaded fasttext model instance.
        """
        if self._model is not None:
            return self._model

        try:
            import fasttext
        except ImportError:
            logger.error(
                "fasttext is not installed. Install it with: pip install fasttext"
            )
            raise

        try:
            self._model = fasttext.load_model(self.model_path)
            logger.info("Loaded fasttext language model from %s", self.model_path)
        except Exception as e:
            logger.error("Failed to load fasttext model from %s: %s", self.model_path, e)
            raise

        return self._model

    @staticmethod
    def _normalize_lang_code(raw_code: str) -> str:
        """Normalize a fasttext language code to a standardized form.

        FastText may return codes with "__label__" prefix. This function
        strips the prefix and maps variant codes to canonical forms.

        Args:
            raw_code: Raw language code from fasttext prediction.

        Returns:
            Normalized language code (e.g., "zh", "en").
        """
        code = raw_code.replace("__label__", "").strip().split("_")[0].split("-")[0]
        return _FASTTEXT_CODE_MAP.get(code, code)

    @staticmethod
    def _preprocess_text(text: str) -> str:
        """Preprocess text for language detection.

        Replaces newlines with spaces and limits text length to avoid
        performance issues with very long documents.

        Args:
            text: Raw input text.

        Returns:
            Preprocessed text suitable for fasttext.
        """
        text = text.replace("\n", " ").replace("\r", " ")
        # Limit to 10,000 characters for performance
        if len(text) > 10000:
            text = text[:10000]
        return text.strip()

    def detect(self, text: str) -> tuple[str, float]:
        """Detect the language of a text string.

        Args:
            text: Input text to classify.

        Returns:
            A tuple of (language_code, confidence) where language_code
            is a standardized code (e.g., "zh", "en") and confidence
            is a float between 0 and 1.
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for language detection")
            return ("unknown", 0.0)

        model = self._load_model()
        processed = self._preprocess_text(text)

        try:
            predictions = model.predict(processed)
        except Exception as e:
            logger.error("Language detection failed: %s", e)
            return ("unknown", 0.0)

        if not predictions or len(predictions[0]) == 0:
            return ("unknown", 0.0)

        raw_code = predictions[0][0]
        confidence = float(predictions[1][0])
        normalized_code = self._normalize_lang_code(raw_code)

        return (normalized_code, confidence)

    def detect_batch(self, texts: list[str]) -> list[tuple[str, float]]:
        """Detect languages for a batch of texts.

        Args:
            texts: List of text strings to classify.

        Returns:
            List of (language_code, confidence) tuples, one per input text.
        """
        results: list[tuple[str, float]] = []

        for text in texts:
            try:
                lang, conf = self.detect(text)
                results.append((lang, conf))
            except Exception as e:
                logger.warning("Failed to detect language for text: %s", e)
                results.append(("unknown", 0.0))

        return results

    def filter_by_language(
        self,
        documents: list[dict[str, Any]],
        text_field: str = "text",
        min_confidence: float = 0.8,
        allowed: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Filter documents by language, keeping only those in allowed languages.

        Args:
            documents: List of document dictionaries.
            text_field: Field name containing document text.
            min_confidence: Minimum confidence threshold for language detection.
                Documents below this threshold are kept only if they pass
                a character-based heuristic check.
            allowed: List of allowed language codes. Defaults to the instance's
                allowed_languages.

        Returns:
            List of documents written in allowed languages.
        """
        target_languages = allowed if allowed is not None else self.allowed_languages
        filtered: list[dict[str, Any]] = []
        total = len(documents)

        logger.info(
            "Filtering %d documents for languages: %s (min_confidence=%.2f)",
            total,
            target_languages,
            min_confidence,
        )

        for idx, doc in enumerate(documents):
            text = str(doc.get(text_field, ""))

            if not text.strip():
                logger.warning("Skipping empty document at index %d", idx)
                continue

            lang, confidence = self.detect(text)

            # If confidence is low, use character-based heuristic as fallback
            if confidence < min_confidence:
                lang = self._heuristic_language(text)
                logger.debug(
                    "Low confidence (%.2f) for doc %d, heuristic detected: %s",
                    confidence,
                    idx,
                    lang,
                )

            if lang in target_languages:
                filtered.append(doc)

            if (idx + 1) % 10000 == 0:
                logger.info("Processed %d/%d documents", idx + 1, total)

        removed = total - len(filtered)
        logger.info(
            "Language filtering complete: %d/%d documents removed (%.1f%%)",
            removed,
            total,
            (removed / total * 100) if total > 0 else 0,
        )

        return filtered

    @staticmethod
    def _heuristic_language(text: str) -> str:
        """Fallback language detection using character analysis.

        Counts Chinese characters vs Latin characters to make a simple
        heuristic classification when the model confidence is low.

        Args:
            text: Input text.

        Returns:
            "zh" if Chinese characters dominate, "en" otherwise.
        """
        chinese_chars = 0
        latin_chars = 0

        for char in text:
            code = ord(char)
            if 0x4E00 <= code <= 0x9FFF:
                chinese_chars += 1
            elif (0x0041 <= code <= 0x007A) or (0x00C0 <= code <= 0x024F):
                latin_chars += 1

        return "zh" if chinese_chars > latin_chars else "en"

    def get_language_distribution(
        self,
        documents: list[dict[str, Any]],
        text_field: str = "text",
    ) -> dict[str, int]:
        """Compute the language distribution of a document corpus.

        Args:
            documents: List of document dictionaries.
            text_field: Field name containing document text.

        Returns:
            Dictionary mapping language codes to document counts.
        """
        distribution: dict[str, int] = {}

        for doc in documents:
            text = str(doc.get(text_field, ""))
            if not text.strip():
                continue

            lang, _ = self.detect(text)
            distribution[lang] = distribution.get(lang, 0) + 1

        logger.info("Language distribution: %s", distribution)
        return distribution


if __name__ == "__main__":
    import argparse
    import json as _json

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Language detection and filtering")
    parser.add_argument("--input", required=True, help="Input JSONL file path")
    parser.add_argument("--output", required=True, help="Output JSONL file path")
    parser.add_argument("--model-path", default="lid.176.bin", help="Path to fasttext model")
    parser.add_argument("--text-field", default="text", help="Field name for document text")
    parser.add_argument("--min-confidence", type=float, default=0.8, help="Minimum confidence threshold")
    parser.add_argument("--allowed", nargs="+", default=["zh", "en"], help="Allowed language codes")
    args = parser.parse_args()

    detector = LanguageDetector(
        model_path=args.model_path,
        allowed_languages=args.allowed,
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
    distribution = detector.get_language_distribution(documents, text_field=args.text_field)
    logger.info("Pre-filter distribution: %s", distribution)

    filtered = detector.filter_by_language(
        documents,
        text_field=args.text_field,
        min_confidence=args.min_confidence,
        allowed=args.allowed,
    )

    logger.info("Writing %d filtered documents to %s", len(filtered), args.output)
    with open(args.output, "w", encoding="utf-8") as f:
        for doc in filtered:
            f.write(_json.dumps(doc, ensure_ascii=False) + "\n")

    logger.info("Done")
