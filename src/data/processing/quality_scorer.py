"""
Quality scoring module for the DeepSleep LLM data processing pipeline.

Provides two quality scoring approaches:
- PerplexityScorer: KenLM-based perplexity estimation for natural language quality.
- DocumentQualityScorer: Heuristic-based quality metrics for structural analysis.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import Any

logger = logging.getLogger(__name__)


class PerplexityScorer:
    """Perplexity-based quality scoring using KenLM language models.

    Uses a pre-trained KenLM language model to estimate the perplexity of
    text. Lower perplexity generally indicates more natural, well-formed text.

    Attributes:
        model_path: Path to the KenLM binary language model file.
        threshold: Maximum perplexity threshold. Documents exceeding this
            value are considered low quality.
    """

    def __init__(self, model_path: str, threshold: float = 1000.0) -> None:
        self.model_path = model_path
        self.threshold = threshold
        self._model = None

    def _load_model(self) -> Any:
        """Lazily load the KenLM language model.

        Returns:
            The loaded KenLM model instance.
        """
        if self._model is not None:
            return self._model

        try:
            import kenlm
        except ImportError:
            logger.error(
                "kenlm is not installed. Install it with: pip install https://github.com/kpu/kenlm/archive/master.zip"
            )
            raise

        try:
            self._model = kenlm.Model(self.model_path)
            logger.info("Loaded KenLM model from %s", self.model_path)
        except Exception as e:
            logger.error("Failed to load KenLM model from %s: %s", self.model_path, e)
            raise

        return self._model

    def score(self, text: str) -> float:
        """Compute the perplexity of a text.

        Args:
            text: Input text to score.

        Returns:
            Perplexity score as a float. Returns float('inf') if the text
            is empty or the model fails to score it.
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for perplexity scoring")
            return float("inf")

        model = self._load_model()

        try:
            # KenLM's score returns log10 probability; convert to perplexity
            log10_prob = model.score(text)
            # Perplexity = 10^(-log10_prob / num_chars) approximately
            # KenLM score already gives us what we need
            perplexity = 10.0 ** (-log10_prob)
            return perplexity
        except Exception as e:
            logger.error("Failed to compute perplexity: %s", e)
            return float("inf")

    def score_documents(
        self,
        documents: list[dict[str, Any]],
        text_field: str = "text",
    ) -> list[tuple[dict[str, Any], float]]:
        """Score a list of documents by perplexity.

        Args:
            documents: List of document dictionaries.
            text_field: Field name containing document text.

        Returns:
            List of (document, perplexity_score) tuples.
        """
        scored: list[tuple[dict[str, Any], float]] = []
        total = len(documents)

        logger.info("Scoring %d documents by perplexity", total)

        for idx, doc in enumerate(documents):
            text = str(doc.get(text_field, ""))
            perplexity = self.score(text)
            scored.append((doc, perplexity))

            if (idx + 1) % 10000 == 0:
                logger.info("Scored %d/%d documents", idx + 1, total)

        return scored

    def filter_by_perplexity(
        self,
        documents: list[dict[str, Any]],
        text_field: str = "text",
        max_perplexity: float | None = None,
    ) -> list[dict[str, Any]]:
        """Filter documents by maximum perplexity threshold.

        Documents with perplexity above the threshold are removed as
        they likely contain garbled, machine-generated, or nonsensical text.

        Args:
            documents: List of document dictionaries.
            text_field: Field name containing document text.
            max_perplexity: Maximum allowed perplexity. Defaults to the
                instance threshold.

        Returns:
            List of documents below the perplexity threshold.
        """
        threshold = max_perplexity if max_perplexity is not None else self.threshold
        filtered: list[dict[str, Any]] = []
        total = len(documents)

        logger.info("Filtering %d documents by perplexity (max=%.1f)", total, threshold)

        for idx, doc in enumerate(documents):
            text = str(doc.get(text_field, ""))

            if not text.strip():
                continue

            perplexity = self.score(text)

            if perplexity <= threshold:
                filtered.append(doc)
            else:
                logger.debug(
                    "Document %d rejected: perplexity %.1f > threshold %.1f",
                    idx,
                    perplexity,
                    threshold,
                )

            if (idx + 1) % 10000 == 0:
                logger.info("Processed %d/%d documents", idx + 1, total)

        removed = total - len(filtered)
        logger.info(
            "Perplexity filtering complete: %d/%d documents removed (%.1f%%)",
            removed,
            total,
            (removed / total * 100) if total > 0 else 0,
        )

        return filtered


class DocumentQualityScorer:
    """Heuristic-based document quality scoring.

    Computes multiple structural and statistical quality metrics for
    documents. Combines them into a single quality score and provides
    filtering based on configurable thresholds.

    Attributes:
        min_sentence_length: Reject documents with average sentence length below this.
        max_sentence_length: Reject documents with average sentence length above this.
        max_special_char_ratio: Reject documents with special character ratio above this.
        max_repetition_ratio: Reject documents with n-gram repetition ratio above this.
    """

    def __init__(
        self,
        min_sentence_length: int = 5,
        max_sentence_length: int = 200,
        max_special_char_ratio: float = 0.3,
        max_repetition_ratio: float = 0.3,
    ) -> None:
        self.min_sentence_length = min_sentence_length
        self.max_sentence_length = max_sentence_length
        self.max_special_char_ratio = max_special_char_ratio
        self.max_repetition_ratio = max_repetition_ratio

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences using common delimiters.

        Handles Chinese and English sentence-ending punctuation.

        Args:
            text: Input text.

        Returns:
            List of sentence strings.
        """
        # Split on Chinese and English sentence-ending punctuation
        pattern = r"[。！？.!?\n]+"
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]

    @staticmethod
    def _split_paragraphs(text: str) -> list[str]:
        """Split text into paragraphs.

        Args:
            text: Input text.

        Returns:
            List of non-empty paragraph strings.
        """
        paragraphs = re.split(r"\n\s*\n|\n", text)
        return [p.strip() for p in paragraphs if p.strip()]

    @staticmethod
    def _compute_avg_word_length(text: str) -> float:
        """Compute average word/token length.

        For Chinese text, treats each character as a word.
        For mixed text, segments by whitespace and averages.

        Args:
            text: Input text.

        Returns:
            Average word length as a float.
        """
        # Count Chinese characters
        chinese_chars = [c for c in text if 0x4E00 <= ord(c) <= 0x9FFF]
        # Count non-Chinese words (whitespace separated)
        non_chinese = text
        for char in chinese_chars:
            non_chinese = non_chinese.replace(char, " ", 1)
        words = [w for w in non_chinese.split() if w.strip()]

        total_items = len(chinese_chars) + len(words)
        if total_items == 0:
            return 0.0

        total_length = len(chinese_chars) + sum(len(w) for w in words)
        return total_length / total_items

    @staticmethod
    def _compute_special_char_ratio(text: str) -> float:
        """Compute the ratio of special characters in text.

        Special characters include punctuation, symbols, and control characters.

        Args:
            text: Input text.

        Returns:
            Ratio of special characters to total characters (0.0 to 1.0).
        """
        if not text:
            return 0.0

        # Normal characters: letters, digits, Chinese characters, common punctuation
        normal_pattern = re.compile(r"[\w\u4e00-\u9fff\u3000-\u303f\uff00-\uffef\s]")
        normal_count = len(normal_pattern.findall(text))
        special_count = len(text) - normal_count

        return special_count / len(text) if len(text) > 0 else 0.0

    @staticmethod
    def _compute_repetition_ratio(text: str, n_gram_size: int = 5) -> float:
        """Compute the ratio of repeated n-grams.

        Counts how many n-grams appear more than once relative to total n-grams.

        Args:
            text: Input text.
            n_gram_size: Size of n-grams for repetition detection.

        Returns:
            Ratio of repeated n-grams (0.0 to 1.0).
        """
        cleaned = re.sub(r"\s+", "", text.lower())
        if len(cleaned) < n_gram_size:
            return 0.0

        ngrams: list[str] = []
        for i in range(len(cleaned) - n_gram_size + 1):
            ngrams.append(cleaned[i : i + n_gram_size])

        if not ngrams:
            return 0.0

        counts = Counter(ngrams)
        repeated = sum(1 for c in counts.values() if c > 1)
        return repeated / len(counts)

    def score(self, text: str) -> dict[str, Any]:
        """Compute multiple quality metrics for a document.

        Args:
            text: Input document text.

        Returns:
            Dictionary containing quality metrics:
                - avg_sentence_length: Average characters per sentence.
                - avg_word_length: Average characters per word/token.
                - special_char_ratio: Ratio of special characters.
                - repetition_ratio: Ratio of repeated 5-grams.
                - line_count: Number of non-empty lines.
                - paragraph_count: Number of non-empty paragraphs.
                - sentence_count: Number of sentences detected.
                - quality_score: Aggregate score from 0 to 1.
                - flags: List of quality issue flags.
        """
        sentences = self._split_sentences(text)
        paragraphs = self._split_paragraphs(text)
        lines = [l.strip() for l in text.split("\n") if l.strip()]

        avg_sentence_len = (
            sum(len(s) for s in sentences) / len(sentences) if sentences else 0.0
        )
        avg_word_len = self._compute_avg_word_length(text)
        special_char_ratio = self._compute_special_char_ratio(text)
        repetition_ratio = self._compute_repetition_ratio(text)

        # Compute quality flags
        flags: list[str] = []
        if avg_sentence_len < self.min_sentence_length:
            flags.append("short_sentences")
        if avg_sentence_len > self.max_sentence_length:
            flags.append("long_sentences")
        if special_char_ratio > self.max_special_char_ratio:
            flags.append("high_special_chars")
        if repetition_ratio > self.max_repetition_ratio:
            flags.append("high_repetition")

        # Aggregate quality score (penalize each flag)
        quality_score = max(0.0, 1.0 - len(flags) * 0.25)

        return {
            "avg_sentence_length": round(avg_sentence_len, 2),
            "avg_word_length": round(avg_word_len, 2),
            "special_char_ratio": round(special_char_ratio, 4),
            "repetition_ratio": round(repetition_ratio, 4),
            "line_count": len(lines),
            "paragraph_count": len(paragraphs),
            "sentence_count": len(sentences),
            "quality_score": round(quality_score, 4),
            "flags": flags,
        }

    def filter_by_quality(
        self,
        documents: list[dict[str, Any]],
        text_field: str = "text",
        min_score: float = 0.5,
    ) -> list[dict[str, Any]]:
        """Filter documents by minimum quality score.

        Documents are scored on structural metrics and those below the
        minimum score threshold are removed.

        Args:
            documents: List of document dictionaries.
            text_field: Field name containing document text.
            min_score: Minimum aggregate quality score (0 to 1).

        Returns:
            List of documents meeting the quality threshold.
        """
        filtered: list[dict[str, Any]] = []
        total = len(documents)

        logger.info("Filtering %d documents by quality (min_score=%.2f)", total, min_score)

        for idx, doc in enumerate(documents):
            text = str(doc.get(text_field, ""))

            if not text.strip():
                continue

            metrics = self.score(text)
            quality = metrics["quality_score"]

            if quality >= min_score:
                filtered.append(doc)
            else:
                logger.debug(
                    "Document %d rejected: quality_score=%.2f, flags=%s",
                    idx,
                    quality,
                    metrics["flags"],
                )

            if (idx + 1) % 10000 == 0:
                logger.info("Processed %d/%d documents", idx + 1, total)

        removed = total - len(filtered)
        logger.info(
            "Quality filtering complete: %d/%d documents removed (%.1f%%)",
            removed,
            total,
            (removed / total * 100) if total > 0 else 0,
        )

        return filtered

    def get_quality_report(
        self,
        documents: list[dict[str, Any]],
        text_field: str = "text",
    ) -> dict[str, Any]:
        """Generate a summary quality report for a document corpus.

        Args:
            documents: List of document dictionaries.
            text_field: Field name containing document text.

        Returns:
            Dictionary containing aggregate statistics:
                - total_documents: Number of documents analyzed.
                - avg_sentence_length: Mean across all documents.
                - avg_word_length: Mean across all documents.
                - avg_special_char_ratio: Mean across all documents.
                - avg_repetition_ratio: Mean across all documents.
                - avg_quality_score: Mean quality score.
                - flag_distribution: Count of documents with each flag.
                - score_distribution: Count of documents in score buckets.
        """
        total = len(documents)
        if total == 0:
            return {"total_documents": 0}

        sentence_lengths: list[float] = []
        word_lengths: list[float] = []
        special_ratios: list[float] = []
        repetition_ratios: list[float] = []
        quality_scores: list[float] = []
        flag_counts: Counter = Counter()
        score_buckets: Counter = Counter()

        for idx, doc in enumerate(documents):
            text = str(doc.get(text_field, ""))
            if not text.strip():
                continue

            metrics = self.score(text)

            sentence_lengths.append(metrics["avg_sentence_length"])
            word_lengths.append(metrics["avg_word_length"])
            special_ratios.append(metrics["special_char_ratio"])
            repetition_ratios.append(metrics["repetition_ratio"])
            quality_scores.append(metrics["quality_score"])

            for flag in metrics["flags"]:
                flag_counts[flag] += 1

            bucket = "0.0-0.25"
            qs = metrics["quality_score"]
            if qs >= 0.75:
                bucket = "0.75-1.0"
            elif qs >= 0.5:
                bucket = "0.5-0.75"
            elif qs >= 0.25:
                bucket = "0.25-0.5"
            score_buckets[bucket] += 1

            if (idx + 1) % 10000 == 0:
                logger.info("Analyzed %d/%d documents for quality report", idx + 1, total)

        def _safe_mean(values: list[float]) -> float:
            return sum(values) / len(values) if values else 0.0

        report = {
            "total_documents": total,
            "documents_analyzed": len(quality_scores),
            "avg_sentence_length": round(_safe_mean(sentence_lengths), 2),
            "avg_word_length": round(_safe_mean(word_lengths), 2),
            "avg_special_char_ratio": round(_safe_mean(special_ratios), 4),
            "avg_repetition_ratio": round(_safe_mean(repetition_ratios), 4),
            "avg_quality_score": round(_safe_mean(quality_scores), 4),
            "flag_distribution": dict(flag_counts),
            "score_distribution": dict(score_buckets),
        }

        logger.info("Quality report generated: avg_score=%.2f", report["avg_quality_score"])
        return report


if __name__ == "__main__":
    import argparse
    import json as _json

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Document quality scoring and filtering")
    parser.add_argument("--mode", choices=["score", "filter", "report"], default="report", help="Operation mode")
    parser.add_argument("--input", required=True, help="Input JSONL file path")
    parser.add_argument("--output", required=True, help="Output JSONL file path")
    parser.add_argument("--text-field", default="text", help="Field name for document text")
    parser.add_argument("--min-score", type=float, default=0.5, help="Minimum quality score for filtering")
    parser.add_argument("--kenlm-model", default=None, help="Path to KenLM model for perplexity scoring")
    parser.add_argument("--max-perplexity", type=float, default=1000.0, help="Max perplexity threshold")
    args = parser.parse_args()

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

    scorer = DocumentQualityScorer()

    if args.mode == "report":
        report = scorer.get_quality_report(documents, text_field=args.text_field)
        with open(args.output, "w", encoding="utf-8") as f:
            _json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info("Quality report written to %s", args.output)

        print("\n=== Quality Report ===")
        for key, value in report.items():
            print(f"  {key}: {value}")

    elif args.mode == "filter":
        filtered = scorer.filter_by_quality(
            documents,
            text_field=args.text_field,
            min_score=args.min_score,
        )

        if args.kenlm_model:
            pp_scorer = PerplexityScorer(
                model_path=args.kenlm_model,
                threshold=args.max_perplexity,
            )
            filtered = pp_scorer.filter_by_perplexity(
                filtered,
                text_field=args.text_field,
                max_perplexity=args.max_perplexity,
            )

        logger.info("Writing %d filtered documents to %s", len(filtered), args.output)
        with open(args.output, "w", encoding="utf-8") as f:
            for doc in filtered:
                f.write(_json.dumps(doc, ensure_ascii=False) + "\n")

    elif args.mode == "score":
        results = []
        for idx, doc in enumerate(documents):
            text = str(doc.get(args.text_field, ""))
            metrics = scorer.score(text)
            result = {**doc, "quality_metrics": metrics}
            results.append(result)

        logger.info("Writing %d scored documents to %s", len(results), args.output)
        with open(args.output, "w", encoding="utf-8") as f:
            for doc in results:
                f.write(_json.dumps(doc, ensure_ascii=False) + "\n")

    logger.info("Done")
