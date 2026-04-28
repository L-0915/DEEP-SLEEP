"""
Deduplication module for the DeepSleep LLM data processing pipeline.

Provides three deduplication strategies:
- MinHashDeduplicator: Approximate near-duplicate detection using MinHash LSH
- ExactDeduplicator: Exact duplicate detection via n-gram hashing
- URLDeduplicator: URL-based deduplication
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

from datasketch import MinHash, MinHashLSH

logger = logging.getLogger(__name__)


class MinHashDeduplicator:
    """Near-duplicate document detection using MinHash LSH.

    Uses locality-sensitive hashing to efficiently find documents that are
    approximately similar without pairwise comparisons. Suitable for large
    corpora where exact deduplication is too expensive.

    Attributes:
        n_grams: Number of characters per n-gram for MinHash computation.
        num_perm: Number of random permutations for MinHash (higher = more accurate).
        threshold: Jaccard similarity threshold for considering documents duplicates.
    """

    def __init__(self, n_grams: int = 5, num_perm: int = 128, threshold: float = 0.7) -> None:
        self.n_grams = n_grams
        self.num_perm = num_perm
        self.threshold = threshold
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.minhashes: dict[str, MinHash] = {}

    def compute_minhash(self, text: str) -> MinHash:
        """Compute MinHash signature for a document.

        Args:
            text: The document text to compute MinHash for.

        Returns:
            A MinHash object representing the document's fingerprint.
        """
        minhash = MinHash(num_perm=self.num_perm)
        for gram in self._extract_ngrams(text):
            minhash.update(gram.encode("utf-8"))
        return minhash

    def _extract_ngrams(self, text: str) -> list[str]:
        """Extract character-level n-grams from text.

        Args:
            text: Input text.

        Returns:
            List of n-gram strings.
        """
        cleaned = self._normalize_for_hashing(text)
        if len(cleaned) < self.n_grams:
            return [cleaned]
        return [cleaned[i : i + self.n_grams] for i in range(len(cleaned) - self.n_grams + 1)]

    @staticmethod
    def _normalize_for_hashing(text: str) -> str:
        """Normalize text for hashing by removing whitespace and lowercasing."""
        return "".join(text.lower().split())

    def add_document(self, doc_id: str, text: str) -> None:
        """Add a document to the LSH index.

        Args:
            doc_id: Unique identifier for the document.
            text: Document text to index.
        """
        minhash = self.compute_minhash(text)
        self.minhashes[doc_id] = minhash
        self.lsh.insert(doc_id, minhash)
        logger.debug("Added document %s to MinHash LSH index", doc_id)

    def query_duplicates(self, doc_id: str, text: str) -> set[str]:
        """Find near-duplicate documents for a given document.

        Args:
            doc_id: Unique identifier for the query document.
            text: Document text to query against the index.

        Returns:
            Set of doc IDs that are near-duplicates of the query document.
        """
        minhash = self.compute_minhash(text)
        result = set(self.lsh.query(minhash))
        result.discard(doc_id)
        return result

    def deduplicate_corpus(
        self, documents: list[dict[str, Any]], text_field: str = "text"
    ) -> list[dict[str, Any]]:
        """Deduplicate an entire corpus using MinHash LSH.

        Processes documents sequentially. The first occurrence of a near-duplicate
        cluster is kept; subsequent duplicates are removed.

        Args:
            documents: List of document dictionaries. Each must contain a unique "id"
                field and the specified text_field.
            text_field: Field name containing document text.

        Returns:
            List of unique documents with duplicates removed.
        """
        unique_docs: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        total = len(documents)

        logger.info("Starting MinHash deduplication on %d documents", total)

        for idx, doc in enumerate(documents):
            doc_id = str(doc.get("id", idx))

            if doc_id in seen_ids:
                continue

            text = str(doc.get(text_field, ""))

            if not text.strip():
                logger.warning("Skipping empty document %s", doc_id)
                continue

            minhash = self.compute_minhash(text)
            duplicates = set(self.lsh.query(minhash))
            duplicates.discard(doc_id)

            if duplicates:
                logger.debug(
                    "Document %s has %d near-duplicate(s): %s",
                    doc_id,
                    len(duplicates),
                    duplicates,
                )
                continue

            self.minhashes[doc_id] = minhash
            self.lsh.insert(doc_id, minhash)
            unique_docs.append(doc)
            seen_ids.add(doc_id)

            if (idx + 1) % 10000 == 0:
                logger.info(
                    "Processed %d/%d documents, %d unique so far",
                    idx + 1,
                    total,
                    len(unique_docs),
                )

        removed = total - len(unique_docs)
        logger.info(
            "Deduplication complete: %d/%d documents removed (%.1f%%)",
            removed,
            total,
            (removed / total * 100) if total > 0 else 0,
        )

        return unique_docs

    def save_index(self, path: str | Path) -> None:
        """Save the LSH index and MinHash signatures to disk.

        Args:
            path: Directory path to save the index files.
        """
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        lsh_path = save_dir / "minhash_lsh.pkl"
        minhashes_path = save_dir / "minhashes.pkl"
        config_path = save_dir / "dedup_config.json"

        with open(lsh_path, "wb") as f:
            pickle.dump(self.lsh, f)
        with open(minhashes_path, "wb") as f:
            pickle.dump(self.minhashes, f)

        config = {
            "n_grams": self.n_grams,
            "num_perm": self.num_perm,
            "threshold": self.threshold,
            "document_count": len(self.minhashes),
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        logger.info("Saved MinHash index to %s (%d documents)", save_dir, len(self.minhashes))

    def load_index(self, path: str | Path) -> None:
        """Load a previously saved LSH index from disk.

        Args:
            path: Directory path containing the saved index files.
        """
        load_dir = Path(path)

        lsh_path = load_dir / "minhash_lsh.pkl"
        minhashes_path = load_dir / "minhashes.pkl"
        config_path = load_dir / "dedup_config.json"

        if not lsh_path.exists():
            raise FileNotFoundError(f"LSH index not found at {lsh_path}")

        with open(lsh_path, "rb") as f:
            self.lsh = pickle.load(f)
        with open(minhashes_path, "rb") as f:
            self.minhashes = pickle.load(f)

        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            logger.info(
                "Loaded MinHash index config: n_grams=%d, num_perm=%d, threshold=%.2f",
                config.get("n_grams", self.n_grams),
                config.get("num_perm", self.num_perm),
                config.get("threshold", self.threshold),
            )

        logger.info("Loaded MinHash index from %s (%d documents)", load_dir, len(self.minhashes))


class ExactDeduplicator:
    """Exact duplicate detection using n-gram hash sets.

    Compares documents by computing hash sets of their character-level n-grams.
    Two documents are considered exact duplicates if their n-gram hash sets
    are identical.

    Attributes:
        n_gram_size: Size of character n-grams used for hashing.
    """

    def __init__(self, n_gram_size: int = 5) -> None:
        self.n_gram_size = n_gram_size
        self.seen_hashes: set[int] = set()

    def compute_ngram_hash(self, text: str) -> set[int]:
        """Compute hash set of n-grams for a document.

        Args:
            text: Input text to hash.

        Returns:
            Set of integer hashes, one per unique n-gram.
        """
        cleaned = "".join(text.lower().split())
        if len(cleaned) < self.n_gram_size:
            return {hash(cleaned)}

        hashes: set[int] = set()
        for i in range(len(cleaned) - self.n_gram_size + 1):
            ngram = cleaned[i : i + self.n_gram_size]
            hashes.add(hash(ngram))
        return hashes

    def _hashset_to_hash(self, ngram_hashes: set[int]) -> int:
        """Convert a set of n-gram hashes to a single frozen hash for comparison."""
        return hash(frozenset(ngram_hashes))

    def deduplicate_paragraphs(self, texts: list[str]) -> list[str]:
        """Remove exact duplicate paragraphs from a list of texts.

        Args:
            texts: List of paragraph strings.

        Returns:
            List of unique paragraph strings, preserving first occurrence order.
        """
        unique_texts: list[str] = []
        seen: set[int] = set()

        for text in texts:
            if not text.strip():
                continue

            ngram_hashes = self.compute_ngram_hash(text)
            fingerprint = self._hashset_to_hash(ngram_hashes)

            if fingerprint in seen:
                continue

            seen.add(fingerprint)
            unique_texts.append(text)

        removed = len(texts) - len(unique_texts)
        if removed > 0:
            logger.info("Removed %d duplicate paragraphs (%d kept)", removed, len(unique_texts))

        return unique_texts

    def deduplicate_documents(
        self, documents: list[dict[str, Any]], text_field: str = "text"
    ) -> list[dict[str, Any]]:
        """Remove exact duplicate documents from a list.

        Args:
            documents: List of document dictionaries.
            text_field: Field name containing document text.

        Returns:
            List of unique documents, preserving first occurrence order.
        """
        unique_docs: list[dict[str, Any]] = []
        seen: set[int] = set()
        total = len(documents)

        logger.info("Starting exact deduplication on %d documents", total)

        for idx, doc in enumerate(documents):
            text = str(doc.get(text_field, ""))

            if not text.strip():
                logger.warning("Skipping empty document at index %d", idx)
                continue

            ngram_hashes = self.compute_ngram_hash(text)
            fingerprint = self._hashset_to_hash(ngram_hashes)

            if fingerprint in seen:
                logger.debug("Duplicate document found at index %d", idx)
                continue

            seen.add(fingerprint)
            unique_docs.append(doc)

        removed = total - len(unique_docs)
        logger.info(
            "Exact deduplication complete: %d/%d documents removed (%.1f%%)",
            removed,
            total,
            (removed / total * 100) if total > 0 else 0,
        )

        return unique_docs


class URLDeduplicator:
    """URL-based deduplication for web-scraped documents.

    Removes documents that share the same normalized URL. Handles common
    URL variations such as trailing slashes, query parameter ordering,
    and HTTP vs HTTPS schemes.
    """

    def __init__(self) -> None:
        self.seen_urls: set[str] = set()

    @staticmethod
    def _normalize_url(url: str) -> str:
        """Normalize a URL for deduplication comparison.

        Handles scheme normalization, trailing slashes, and removes
        common tracking query parameters.

        Args:
            url: Raw URL string.

        Returns:
            Normalized URL string.
        """
        normalized = url.strip().lower()
        normalized = normalized.replace("http://", "https://")
        normalized = normalized.rstrip("/")

        tracking_params = [
            "utm_source",
            "utm_medium",
            "utm_campaign",
            "utm_content",
            "utm_term",
            "fbclid",
            "gclid",
            "ref",
            "source",
        ]

        if "?" in normalized:
            base, query = normalized.split("?", 1)
            params = [p for p in query.split("&") if p.split("=")[0] not in tracking_params]
            if params:
                params.sort()
                normalized = f"{base}?{'&'.join(params)}"
            else:
                normalized = base

        return normalized

    def deduplicate_by_url(
        self, documents: list[dict[str, Any]], url_field: str = "url"
    ) -> list[dict[str, Any]]:
        """Remove documents with duplicate URLs.

        Args:
            documents: List of document dictionaries, each containing a URL field.
            url_field: Field name containing the document URL.

        Returns:
            List of documents with unique URLs, preserving first occurrence order.
        """
        unique_docs: list[dict[str, Any]] = []
        seen: set[str] = set()
        total = len(documents)

        logger.info("Starting URL deduplication on %d documents", total)

        for idx, doc in enumerate(documents):
            url = doc.get(url_field, "")

            if not url or not isinstance(url, str):
                logger.warning("Missing or invalid URL at index %d, skipping", idx)
                unique_docs.append(doc)
                continue

            normalized = self._normalize_url(url)

            if normalized in seen:
                logger.debug("Duplicate URL found: %s", url)
                continue

            seen.add(normalized)
            unique_docs.append(doc)

        removed = total - len(unique_docs)
        logger.info(
            "URL deduplication complete: %d/%d documents removed (%.1f%%)",
            removed,
            total,
            (removed / total * 100) if total > 0 else 0,
        )

        return unique_docs


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Document deduplication tools")
    parser.add_argument("--mode", choices=["minhash", "exact", "url"], default="exact", help="Deduplication mode")
    parser.add_argument("--input", required=True, help="Input JSONL file path")
    parser.add_argument("--output", required=True, help="Output JSONL file path")
    parser.add_argument("--text-field", default="text", help="Field name for document text")
    parser.add_argument("--url-field", default="url", help="Field name for document URL")
    parser.add_argument("--threshold", type=float, default=0.7, help="MinHash similarity threshold")
    parser.add_argument("--n-grams", type=int, default=5, help="N-gram size")
    args = parser.parse_args()

    import json as _json

    logger.info("Loading documents from %s", args.input)
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

    if args.mode == "minhash":
        deduplicator = MinHashDeduplicator(
            n_grams=args.n_grams,
            threshold=args.threshold,
        )
        result = deduplicator.deduplicate_corpus(documents, text_field=args.text_field)
    elif args.mode == "exact":
        deduplicator = ExactDeduplicator(n_gram_size=args.n_grams)
        result = deduplicator.deduplicate_documents(documents, text_field=args.text_field)
    else:
        deduplicator = URLDeduplicator()
        result = deduplicator.deduplicate_by_url(documents, url_field=args.url_field)

    logger.info("Writing %d deduplicated documents to %s", len(result), args.output)
    with open(args.output, "w", encoding="utf-8") as f:
        for doc in result:
            f.write(_json.dumps(doc, ensure_ascii=False) + "\n")

    logger.info("Done")
