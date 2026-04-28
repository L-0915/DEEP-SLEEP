"""
arXiv crawler for collecting sleep-related research papers.

Uses the arxiv Python package to search for papers in sleep signal processing,
circadian biology, NLP applications in sleep medicine, and related fields.
Downloads PDFs and extracts text content using PyMuPDF (fitz).

Deduplicates results by arXiv ID and supports resumable crawling.
"""

import json
import logging
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Search queries focused on sleep medicine and related fields
SLEEP_SEARCH_QUERIES = [
    "sleep",
    "circadian",
    "polysomnography",
    "sleep disorder",
    "EEG sleep",
    "sleep stage",
    "sleep apnea detection",
    "sleep quality assessment",
    "insomnia",
    "narcolepsy",
    "restless legs syndrome",
    "sleep architecture",
    "sleep spindle",
    "sleep scoring",
    "actigraphy",
    "sleep breathing disorder",
]

# Relevant arXiv categories for sleep research
RELEVANT_CATEGORIES = [
    "q-bio.NC",   # Neurons and Cognition
    "cs.CL",      # Computation and Language (NLP)
    "eess.SP",    # Signal Processing
    "cs.LG",      # Machine Learning
    "stat.ML",    # Machine Learning (Statistics)
    "q-bio.QM",   # Quantitative Methods
    "cs.AI",      # Artificial Intelligence
    "eess.IV",    # Image and Video Processing
]

# Rate limiting for arXiv API (be a good citizen)
ARXIV_REQUEST_INTERVAL = 3.0  # seconds between requests
MAX_RETRY_ATTEMPTS = 5
RETRY_BACKOFF_BASE = 2.0


@dataclass(frozen=True)
class ArxivPaper:
    """Structured representation of an arXiv paper record."""

    arxiv_id: str
    title: str
    abstract: str
    authors: list[str]
    published: str
    updated: str
    categories: list[str]
    primary_category: str
    doi: Optional[str]
    comments: Optional[str]
    journal_ref: Optional[str]
    pdf_url: str
    full_text: Optional[str] = None
    download_path: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ArxivCrawler:
    """Crawler for sleep-related research papers from arXiv.

    Searches arXiv using sleep-related queries and relevant categories,
    fetches paper metadata, downloads PDFs, and extracts text content.

    Deduplicates results by arXiv ID and supports resumable crawling
    by tracking processed IDs in a local state file.

    Attributes:
        output_dir: Directory for JSONL output and downloaded PDFs.
    """

    def __init__(self, output_dir: str = "data/raw/arxiv") -> None:
        self._output_dir = Path(output_dir)
        self._pdf_dir = self._output_dir / "pdfs"
        self._state_file = self._output_dir / "_arxiv_state.json"

        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._pdf_dir.mkdir(parents=True, exist_ok=True)

        # Track processed IDs for deduplication and resumability
        self._processed_ids: set[str] = set()
        self._load_state()

        self._last_request_time: float = 0.0

    def _load_state(self) -> None:
        """Load previously processed arXiv IDs from the state file."""
        if not self._state_file.exists():
            return

        try:
            with open(self._state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
            self._processed_ids = set(state.get("processed_ids", []))
            logger.info(
                "Resuming: %d arXiv papers already processed",
                len(self._processed_ids),
            )
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Failed to load state file, starting fresh: %s", exc)

    def _save_state(self) -> None:
        """Persist processed arXiv IDs to the state file."""
        state = {
            "processed_ids": sorted(self._processed_ids),
            "processed_count": len(self._processed_ids),
            "last_updated": datetime.now().isoformat(),
        }
        with open(self._state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def _rate_limit(self) -> None:
        """Enforce rate limiting between arXiv API requests."""
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < ARXIV_REQUEST_INTERVAL:
            time.sleep(ARXIV_REQUEST_INTERVAL - elapsed)
        self._last_request_time = time.monotonic()

    def _extract_arxiv_id(self, url_or_id: str) -> str:
        """Extract the canonical arXiv ID from a URL or ID string.

        Handles various arXiv URL formats and bare IDs (e.g., '2301.00001v1').

        Args:
            url_or_id: arXiv URL or ID string.

        Returns:
            Canonical arXiv ID without version suffix.
        """
        # Match patterns like 2301.00001 or 2301.00001v1
        match = re.search(r"(\d{4}\.\d{4,5})(?:v\d+)?", url_or_id)
        if match:
            return match.group(1)

        # Match older format: archive/YYMMNNN
        match = re.search(r"([a-z-]+/\d{7})", url_or_id)
        if match:
            return match.group(1)

        return url_or_id

    def search_papers(
        self,
        query: str,
        max_results: int = 5000,
        categories: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        """Search arXiv for papers matching the given query.

        Uses the arxiv Python package to search by keyword and optionally
        filter by arXiv categories. Results are deduplicated by arXiv ID.

        Args:
            query: Search query string.
            max_results: Maximum number of papers to retrieve.
            categories: Optional list of arXiv categories to filter by.

        Returns:
            List of paper metadata dictionaries.
        """
        import arxiv  # noqa: F401 (import here to avoid hard dep at module level)

        logger.info(
            "Searching arXiv for: '%s' (max_results=%d)", query, max_results
        )

        papers: list[dict[str, Any]] = []
        seen_ids: set[str] = set(self._processed_ids)

        try:
            # Build category filter if specified
            cat_query = ""
            if categories:
                cat_query = " AND (" + " OR ".join(
                    f"cat:{cat}" for cat in categories
                ) + ")"

            search_query = f"all:{query}{cat_query}"

            client = arxiv.Client()
            search = arxiv.Search(
                query=search_query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance,
            )

            for result in search.results():
                arxiv_id = self._extract_arxiv_id(result.entry_id)
                if arxiv_id in seen_ids:
                    continue

                # Extract authors
                authors = [str(a) for a in result.authors]

                # Extract categories
                cats = list(result.categories) if result.categories else []
                primary_cat = result.primary_category or (cats[0] if cats else "")

                # Find DOI
                doi = result.doi

                paper = ArxivPaper(
                    arxiv_id=arxiv_id,
                    title=result.title.strip(),
                    abstract=result.summary.strip(),
                    authors=authors,
                    published=(
                        result.published.isoformat()
                        if result.published
                        else ""
                    ),
                    updated=(
                        result.updated.isoformat()
                        if result.updated
                        else ""
                    ),
                    categories=cats,
                    primary_category=primary_cat,
                    doi=doi,
                    comments=getattr(result, "comments", ""),
                    journal_ref=getattr(result, "journal_ref", ""),
                    pdf_url=result.pdf_url or "",
                )

                papers.append(paper.to_dict())
                seen_ids.add(arxiv_id)

                if len(papers) % 100 == 0:
                    logger.info(
                        "Found %d papers so far for query '%s'",
                        len(papers),
                        query,
                    )

            logger.info(
                "Search complete for '%s': %d papers found", query, len(papers)
            )

        except Exception as exc:
            logger.error("arXiv search failed for '%s': %s", query, exc)

        return papers

    def download_papers(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Download PDFs and extract text for a list of arXiv papers.

        Downloads PDFs using the arxiv package, then extracts text content
        using PyMuPDF (fitz). Papers that have already been downloaded
        are skipped.

        Args:
            results: List of paper metadata dictionaries from search_papers.

        Returns:
            Updated list of paper dictionaries with full_text and download_path.
        """
        import arxiv  # noqa: F401

        enriched = []
        new_downloads = 0

        for paper in results:
            arxiv_id = paper["arxiv_id"]
            pdf_path = self._pdf_dir / f"{arxiv_id}.pdf"

            if arxiv_id in self._processed_ids:
                # Already processed; check if we have the extracted text
                text_path = self._pdf_dir / f"{arxiv_id}.txt"
                if text_path.exists():
                    with open(text_path, "r", encoding="utf-8") as f:
                        paper["full_text"] = f.read()
                    paper["download_path"] = str(text_path)
                    enriched.append(paper)
                continue

            # Download PDF
            if not pdf_path.exists():
                try:
                    paper_obj = next(
                        arxiv.Client().results(
                            arxiv.Search(id_list=[arxiv_id])
                        ),
                        None,
                    )
                    if paper_obj is None:
                        logger.warning("Paper %s not found on arXiv", arxiv_id)
                        continue

                    paper_obj.download_pdf(
                        dirpath=str(self._pdf_dir), filename=f"{arxiv_id}.pdf"
                    )
                    new_downloads += 1
                    logger.info("Downloaded PDF: %s", arxiv_id)

                except Exception as exc:
                    logger.error("Failed to download %s: %s", arxiv_id, exc)
                    continue

            # Extract text from PDF
            text_path = self._pdf_dir / f"{arxiv_id}.txt"
            if not text_path.exists():
                extracted_text = self._extract_text_from_pdf(pdf_path)
                if extracted_text:
                    with open(text_path, "w", encoding="utf-8") as f:
                        f.write(extracted_text)
                    paper["full_text"] = extracted_text
                    paper["download_path"] = str(text_path)
                else:
                    logger.warning("No text extracted from %s", arxiv_id)
                    paper["full_text"] = None
                    paper["download_path"] = str(pdf_path)
            else:
                with open(text_path, "r", encoding="utf-8") as f:
                    paper["full_text"] = f.read()
                paper["download_path"] = str(text_path)

            enriched.append(paper)
            self._processed_ids.add(arxiv_id)

            if new_downloads % 10 == 0 and new_downloads > 0:
                logger.info("Downloaded %d new PDFs", new_downloads)

        logger.info(
            "Download complete: %d papers enriched (%d new downloads)",
            len(enriched),
            new_downloads,
        )
        self._save_state()
        return enriched

    def _extract_text_from_pdf(self, pdf_path: Path) -> Optional[str]:
        """Extract text content from a PDF file using PyMuPDF.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Extracted text content, or None on failure.
        """
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(str(pdf_path))
            text_parts: list[str] = []

            for page_num, page in enumerate(doc):
                page_text = page.get_text("text")
                if page_text.strip():
                    text_parts.append(f"--- Page {page_num + 1} ---\n{page_text}")

            doc.close()

            if text_parts:
                return "\n\n".join(text_parts)
            return None

        except ImportError:
            logger.error(
                "PyMuPDF (fitz) not installed. "
                "Install with: pip install pymupdf"
            )
            return None
        except Exception as exc:
            logger.error("Failed to extract text from %s: %s", pdf_path, exc)
            return None

    def _save_results(
        self, results: list[dict[str, Any]], filename: str
    ) -> int:
        """Append paper results to a JSONL output file.

        Skips papers whose arXiv IDs have already been written to the file.

        Args:
            results: List of paper dictionaries to save.
            filename: Name of the JSONL output file.

        Returns:
            Number of new records written.
        """
        if not results:
            return 0

        output_path = self._output_dir / filename
        written = 0

        with open(output_path, "a", encoding="utf-8") as f:
            for paper in results:
                arxiv_id = paper.get("arxiv_id", "")
                if arxiv_id in self._processed_ids:
                    continue

                f.write(json.dumps(paper, ensure_ascii=False) + "\n")
                self._processed_ids.add(arxiv_id)
                written += 1

        logger.info(
            "Wrote %d new records to %s", written, output_path
        )
        return written

    def crawl_sleep_arxiv(self, max_papers: int = 10000) -> list[dict[str, Any]]:
        """Main entry point for crawling sleep-related papers from arXiv.

        Executes all configured sleep-related search queries across relevant
        arXiv categories, deduplicates results, downloads PDFs, and extracts
        text content. Results are saved incrementally as JSONL.

        Args:
            max_papers: Maximum total number of unique papers to collect.

        Returns:
            List of all collected paper dictionaries.
        """
        logger.info(
            "Starting arXiv sleep paper crawl (max_papers=%d)", max_papers
        )
        start_time = time.time()
        all_papers: list[dict[str, Any]] = []
        seen_ids: set[str] = set(self._processed_ids)

        for query_idx, query in enumerate(SLEEP_SEARCH_QUERIES):
            if len(seen_ids) >= max_papers:
                logger.info(
                    "Reached max_papers limit (%d), stopping", max_papers
                )
                break

            logger.info(
                "Processing query %d/%d: '%s'",
                query_idx + 1,
                len(SLEEP_SEARCH_QUERIES),
                query,
            )

            remaining = max_papers - len(seen_ids)
            papers = self.search_papers(
                query,
                max_results=min(5000, remaining),
                categories=RELEVANT_CATEGORIES,
            )

            if not papers:
                continue

            # Deduplicate against global set
            new_papers = [
                p for p in papers if p["arxiv_id"] not in seen_ids
            ]
            logger.info(
                "%d new papers from query '%s' (%d duplicates skipped)",
                len(new_papers),
                query,
                len(papers) - len(new_papers),
            )

            # Save metadata immediately
            metadata_file = f"arxiv_metadata_{query_idx}.jsonl"
            self._save_results(new_papers, metadata_file)
            all_papers.extend(new_papers)
            seen_ids.update(p["arxiv_id"] for p in new_papers)

        logger.info(
            "Metadata collection complete: %d unique papers. "
            "Starting PDF downloads...",
            len(all_papers),
        )

        # Download PDFs and extract text
        enriched_papers = self.download_papers(all_papers)

        # Save final enriched output
        self._save_results(enriched_papers, "arxiv_sleep_papers_all.jsonl")

        elapsed = time.time() - start_time
        papers_with_text = sum(
            1 for p in enriched_papers if p.get("full_text")
        )
        logger.info(
            "arXiv crawl complete. %d papers collected (%d with full text) "
            "in %.1f seconds",
            len(enriched_papers),
            papers_with_text,
            elapsed,
        )

        return enriched_papers
