"""
Data mixing module for the DeepSleep LLM data processing pipeline.

Handles proportional sampling from multiple data sources to create a
balanced training corpus, with support for distribution analysis and
resampling to target proportions.
"""

from __future__ import annotations

import logging
import random
from typing import Any

logger = logging.getLogger(__name__)


class DataMixer:
    """Data mixing and resampling for multi-source training corpora.

    Samples from multiple data categories according to configured mixing
    proportions, producing a shuffled training corpus. Supports both
    count-based and token-based mixing targets.

    Attributes:
        mixing_config: Dictionary mapping source names to their target
            proportions (must sum to 1.0).
        seed: Random seed for reproducibility.
    """

    def __init__(self, mixing_config: dict[str, float], seed: int = 42) -> None:
        self._validate_config(mixing_config)
        self.mixing_config = mixing_config
        self.seed = seed

    @staticmethod
    def _validate_config(mixing_config: dict[str, float]) -> None:
        """Validate the mixing configuration.

        Args:
            mixing_config: Dictionary mapping source names to proportions.

        Raises:
            ValueError: If proportions are invalid (negative, empty, or
                do not sum to approximately 1.0).
        """
        if not mixing_config:
            raise ValueError("Mixing config cannot be empty")

        for source, proportion in mixing_config.items():
            if proportion < 0:
                raise ValueError(f"Proportion for '{source}' cannot be negative: {proportion}")

        total = sum(mixing_config.values())
        if total == 0:
            raise ValueError("Sum of proportions cannot be zero")

        # Warn if proportions don't sum to 1.0 but allow it (we will normalize)
        if abs(total - 1.0) > 0.01:
            logger.warning(
                "Mixing proportions sum to %.4f (expected 1.0). Will normalize.",
                total,
            )

    def _normalize_config(self) -> dict[str, float]:
        """Normalize mixing proportions so they sum to 1.0.

        Returns:
            Normalized copy of the mixing configuration.
        """
        total = sum(self.mixing_config.values())
        if total == 0:
            return {}
        return {source: prop / total for source, prop in self.mixing_config.items()}

    def mix(
        self,
        corpus_dict: dict[str, list[dict[str, Any]]],
        total_tokens: int | None = None,
        total_documents: int | None = None,
    ) -> list[dict[str, Any]]:
        """Mix data from multiple sources according to configured proportions.

        Samples documents from each source category to match the target
        distribution. If total_tokens is provided, estimates document count
        using average character counts (rough approximation: 1 token ~ 2 chars
        for Chinese, ~ 4 chars for English).

        Args:
            corpus_dict: Dictionary mapping source names to lists of documents.
                Each source name should correspond to a key in mixing_config.
            total_tokens: Optional total target token count. If provided,
                sampling is adjusted to approximately meet this target.
            total_documents: Optional total target document count. Used when
                total_tokens is not provided.

        Returns:
            Shuffled list of sampled documents from all sources, each annotated
            with a "source" field.
        """
        normalized = self._normalize_config()
        rng = random.Random(self.seed)

        # Determine total sample size
        if total_tokens is not None:
            total_chars = total_tokens * 2  # Rough estimate
            avg_doc_chars = self._estimate_avg_doc_length(corpus_dict)
            if avg_doc_chars > 0:
                target_count = int(total_chars / avg_doc_chars)
            else:
                target_count = total_documents if total_documents is not None else 100000
        elif total_documents is not None:
            target_count = total_documents
        else:
            # Default: sample all available, proportional to config
            target_count = sum(len(docs) for docs in corpus_dict.values())

        logger.info(
            "Mixing data from %d sources, target count: %d documents",
            len(corpus_dict),
            target_count,
        )

        mixed: list[dict[str, Any]] = []
        sampling_stats: dict[str, dict[str, int]] = {}

        for source, proportion in normalized.items():
            docs = corpus_dict.get(source, [])

            if not docs:
                logger.warning("No documents found for source '%s', skipping", source)
                continue

            # Calculate how many documents to sample from this source
            target_for_source = int(target_count * proportion)

            if target_for_source >= len(docs):
                # Use all available documents
                sampled = list(docs)
                logger.info(
                    "Source '%s': using all %d documents (requested %d)",
                    source,
                    len(docs),
                    target_for_source,
                )
            else:
                # Random sample without replacement
                sampled = rng.sample(docs, target_for_source)
                logger.info(
                    "Source '%s': sampled %d/%d documents (%.1f%%)",
                    source,
                    target_for_source,
                    len(docs),
                    (target_for_source / len(docs) * 100) if docs else 0,
                )

            # Annotate each document with its source (create new dicts)
            for doc in sampled:
                annotated = {**doc, "source": source}
                mixed.append(annotated)

            sampling_stats[source] = {
                "available": len(docs),
                "sampled": len(sampled),
                "proportion": proportion,
            }

        # Shuffle the mixed dataset
        rng.shuffle(mixed)

        logger.info(
            "Mixing complete: %d total documents from %d sources",
            len(mixed),
            len(sampling_stats),
        )
        for source, stats in sampling_stats.items():
            logger.info(
                "  %s: %d docs (%.1f%% of mix)",
                source,
                stats["sampled"],
                (stats["sampled"] / len(mixed) * 100) if mixed else 0,
            )

        return mixed

    @staticmethod
    def _estimate_avg_doc_length(corpus_dict: dict[str, list[dict[str, Any]]]) -> float:
        """Estimate the average document length across all sources.

        Args:
            corpus_dict: Dictionary mapping source names to document lists.

        Returns:
            Average character count per document across all sources.
        """
        total_chars = 0
        total_docs = 0

        for source, docs in corpus_dict.items():
            for doc in docs:
                text = str(doc.get("text", ""))
                total_chars += len(text)
                total_docs += 1

        return total_chars / total_docs if total_docs > 0 else 0.0

    def compute_mixing_stats(
        self,
        mixed_data: list[dict[str, Any]],
        source_field: str = "source",
    ) -> dict[str, Any]:
        """Compute the actual distribution of a mixed dataset.

        Args:
            mixed_data: List of documents from a mixed dataset.
            source_field: Field name containing the source identifier.

        Returns:
            Dictionary containing:
                - total_documents: Total number of documents.
                - distribution: Dictionary mapping source names to counts.
                - proportions: Dictionary mapping source names to actual proportions.
                - target_proportions: The configured target proportions.
                - divergence: Dictionary mapping source names to the difference
                    between actual and target proportions.
        """
        total = len(mixed_data)
        if total == 0:
            return {
                "total_documents": 0,
                "distribution": {},
                "proportions": {},
                "target_proportions": self.mixing_config,
                "divergence": {},
            }

        source_counts: dict[str, int] = {}
        for doc in mixed_data:
            source = str(doc.get(source_field, "unknown"))
            source_counts[source] = source_counts.get(source, 0) + 1

        proportions = {
            source: count / total for source, count in source_counts.items()
        }

        normalized = self._normalize_config()
        divergence: dict[str, float] = {}
        for source in set(list(proportions.keys()) + list(normalized.keys())):
            actual = proportions.get(source, 0.0)
            target = normalized.get(source, 0.0)
            divergence[source] = round(actual - target, 4)

        stats = {
            "total_documents": total,
            "distribution": source_counts,
            "proportions": {k: round(v, 4) for k, v in proportions.items()},
            "target_proportions": {k: round(v, 4) for k, v in normalized.items()},
            "divergence": divergence,
        }

        logger.info("Mixing statistics: %s", stats)
        return stats

    def resample(
        self,
        mixed_data: list[dict[str, Any]],
        source_field: str = "source",
        target_distribution: dict[str, float] | None = None,
    ) -> list[dict[str, Any]]:
        """Rebalance a mixed dataset to match a target distribution.

        Takes an existing mixed dataset and resamples it to match the
        specified target proportions. Sources with excess documents are
        downsampled; sources with too few documents use all available
        data.

        Args:
            mixed_data: List of documents from a mixed dataset.
            source_field: Field name containing the source identifier.
            target_distribution: Target proportions to rebalance towards.
                If None, uses the instance's mixing_config.

        Returns:
            Resampled and shuffled list of documents.
        """
        target = target_distribution if target_distribution is not None else self.mixing_config
        rng = random.Random(self.seed)

        # Validate and normalize target distribution
        if not target:
            raise ValueError("Target distribution cannot be empty")

        total_proportion = sum(target.values())
        normalized_target = {
            source: prop / total_proportion for source, prop in target.items()
        }

        # Separate documents by source
        by_source: dict[str, list[dict[str, Any]]] = {}
        for doc in mixed_data:
            source = str(doc.get(source_field, "unknown"))
            by_source.setdefault(source, []).append(doc)

        # Determine total count based on the smallest adequately-sourced category
        total_docs = len(mixed_data)
        adjusted_counts: dict[str, int] = {}

        for source, proportion in normalized_target.items():
            available = len(by_source.get(source, []))
            needed = int(total_docs * proportion)

            if needed <= available:
                adjusted_counts[source] = needed
            else:
                # Use all available, warn about shortage
                adjusted_counts[source] = available
                if available > 0:
                    logger.warning(
                        "Source '%s': need %d docs but only %d available (%.1f%% shortfall)",
                        source,
                        needed,
                        available,
                        ((needed - available) / needed * 100) if needed > 0 else 0,
                    )
                else:
                    logger.warning("Source '%s': no documents available", source)

        # Resample from each source
        resampled: list[dict[str, Any]] = []
        for source, count in adjusted_counts.items():
            docs = by_source.get(source, [])
            if count >= len(docs):
                resampled.extend(docs)
            else:
                sampled = rng.sample(docs, count)
                resampled.extend(sampled)

        # Shuffle
        rng.shuffle(resampled)

        logger.info(
            "Resampling complete: %d documents from %d sources",
            len(resampled),
            len(adjusted_counts),
        )

        return resampled

    def compute_source_stats(
        self,
        corpus_dict: dict[str, list[dict[str, Any]]],
        text_field: str = "text",
    ) -> dict[str, dict[str, Any]]:
        """Compute statistics for each source corpus.

        Args:
            corpus_dict: Dictionary mapping source names to document lists.
            text_field: Field name containing document text.

        Returns:
            Dictionary mapping source names to statistics:
                - document_count: Number of documents.
                - total_chars: Total characters across all documents.
                - avg_chars: Average characters per document.
                - estimated_tokens: Rough token count estimate.
        """
        stats: dict[str, dict[str, Any]] = {}

        for source, docs in corpus_dict.items():
            total_chars = sum(len(str(doc.get(text_field, ""))) for doc in docs)
            doc_count = len(docs)

            stats[source] = {
                "document_count": doc_count,
                "total_chars": total_chars,
                "avg_chars": round(total_chars / doc_count, 1) if doc_count > 0 else 0,
                "estimated_tokens": total_chars // 2,  # Rough estimate
            }

            logger.info(
                "Source '%s': %d docs, %d chars, ~%d tokens",
                source,
                doc_count,
                total_chars,
                stats[source]["estimated_tokens"],
            )

        return stats


if __name__ == "__main__":
    import argparse
    import json as _json

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Data mixing and resampling")
    parser.add_argument("--mode", choices=["mix", "stats", "resample"], default="mix")
    parser.add_argument("--input", required=True, help="Input JSONL file path (or directory for mix mode)")
    parser.add_argument("--output", required=True, help="Output JSONL file path")
    parser.add_argument("--source-field", default="source", help="Field name for source identifier")
    parser.add_argument("--config", default=None, help="JSON file with mixing proportions")
    parser.add_argument("--total-documents", type=int, default=None, help="Target total document count")
    parser.add_argument("--total-tokens", type=int, default=None, help="Target total token count")
    args = parser.parse_args()

    # Default mixing configuration for DeepSleep
    default_config = {
        "sleep_domain": 0.4,
        "general_medical": 0.15,
        "zh_general": 0.25,
        "en_general": 0.15,
        "code": 0.05,
    }

    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            config = _json.load(f)
    else:
        config = default_config

    mixer = DataMixer(mixing_config=config)

    if args.mode == "mix":
        # In mix mode, input should be a directory with subdirectories
        # or a JSONL with source annotations
        input_path = Path(args.input)

        if input_path.is_dir():
            corpus_dict: dict[str, list[dict[str, Any]]] = {}
            for source_file in input_path.glob("*.jsonl"):
                source_name = source_file.stem
                docs = []
                with open(source_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                docs.append(_json.loads(line))
                            except _json.JSONDecodeError:
                                logger.warning("Skipping malformed line in %s", source_file)
                corpus_dict[source_name] = docs
                logger.info("Loaded %d documents from %s", len(docs), source_name)

            mixed = mixer.mix(
                corpus_dict,
                total_tokens=args.total_tokens,
                total_documents=args.total_documents,
            )
        else:
            # Single file with source annotations
            documents = []
            with open(args.input, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            documents.append(_json.loads(line))
                        except _json.JSONDecodeError:
                            logger.warning("Skipping malformed JSON line")

            corpus_dict = {}
            for doc in documents:
                source = str(doc.get(args.source_field, "unknown"))
                corpus_dict.setdefault(source, []).append(doc)

            mixed = mixer.mix(
                corpus_dict,
                total_tokens=args.total_tokens,
                total_documents=args.total_documents,
            )

        with open(args.output, "w", encoding="utf-8") as f:
            for doc in mixed:
                f.write(_json.dumps(doc, ensure_ascii=False) + "\n")

        stats = mixer.compute_mixing_stats(mixed, source_field=args.source_field)
        logger.info("Mixing stats: %s", stats)

    elif args.mode == "stats":
        documents = []
        with open(args.input, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        documents.append(_json.loads(line))
                    except _json.JSONDecodeError:
                        logger.warning("Skipping malformed JSON line")

        stats = mixer.compute_mixing_stats(documents, source_field=args.source_field)
        with open(args.output, "w", encoding="utf-8") as f:
            _json.dump(stats, f, indent=2, ensure_ascii=False)

        print("\n=== Mixing Statistics ===")
        for key, value in stats.items():
            print(f"  {key}: {value}")

    elif args.mode == "resample":
        documents = []
        with open(args.input, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        documents.append(_json.loads(line))
                    except _json.JSONDecodeError:
                        logger.warning("Skipping malformed JSON line")

        resampled = mixer.resample(documents, source_field=args.source_field)

        with open(args.output, "w", encoding="utf-8") as f:
            for doc in resampled:
                f.write(_json.dumps(doc, ensure_ascii=False) + "\n")

        stats = mixer.compute_mixing_stats(resampled, source_field=args.source_field)
        logger.info("Resampled stats: %s", stats)

    logger.info("Done")
