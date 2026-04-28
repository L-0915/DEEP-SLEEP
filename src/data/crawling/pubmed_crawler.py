"""
PubMed/MEDLINE crawler for collecting sleep medicine domain research papers.

Uses Bio.Entrez from biopython to search PubMed with MeSH terms, fetch article
metadata, and retrieve full text from PMC when available. Implements NCBI-compliant
rate limiting (max 3 requests/second) and exponential backoff retry logic.

Output is saved as JSONL files with one article record per line.
"""

import json
import logging
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import requests
from Bio import Entrez

logger = logging.getLogger(__name__)

# NCBI requires max 3 requests per second
NCBI_RATE_LIMIT = 3
REQUEST_INTERVAL = 1.0 / NCBI_RATE_LIMIT
MAX_RETRY_ATTEMPTS = 5
RETRY_BACKOFF_BASE = 2.0
RETRY_BACKOFF_MAX = 60.0

# MeSH terms for sleep medicine domain
SLEEP_MESH_TERMS = [
    '"Sleep Disorders"[MeSH]',
    '"Insomnia"[MeSH]',
    '"Sleep Apnea, Obstructive"[MeSH]',
    '"Narcolepsy"[MeSH]',
    '"Restless Legs Syndrome"[MeSH]',
    '"Circadian Rhythm"[MeSH]',
    '"Polysomnography"[MeSH]',
    '"CPAP"[MeSH]',
    '"Melatonin"[MeSH]',
    '"Sleep Initiation and Maintenance Disorders"[MeSH]',
    '"Hypersomnia"[MeSH]',
    '"Parasomnias"[MeSH]',
    '"Shift Work Disorder"[MeSH]',
    '"REM Sleep Behavior Disorder"[MeSH]',
    '"Sleep Bruxism"[MeSH]',
]

# Additional keyword queries for broader coverage
SLEEP_KEYWORD_QUERIES = [
    "sleep quality",
    "sleep architecture",
    "sleep deprivation",
    "slow wave sleep",
    "rapid eye movement sleep",
    "sleep electroencephalography",
    "continuous positive airway pressure",
    "oral appliance therapy sleep apnea",
    "sleep-disordered breathing",
]


@dataclass(frozen=True)
class PubmedArticle:
    """Structured representation of a PubMed article record."""

    pmid: str
    pmc_id: Optional[str]
    title: str
    abstract: str
    authors: list[str]
    journal: str
    publication_year: Optional[int]
    keywords: list[str]
    mesh_terms: list[str]
    doi: Optional[str]
    article_type: str
    full_text: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RateLimiter:
    """Token bucket rate limiter for NCBI API compliance."""

    def __init__(self, max_requests_per_second: float = NCBI_RATE_LIMIT) -> None:
        self._min_interval = 1.0 / max_requests_per_second
        self._last_request_time: float = 0.0

    def wait(self) -> None:
        """Block until it is safe to make the next request."""
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < self._min_interval:
            sleep_time = self._min_interval - elapsed
            time.sleep(sleep_time)
        self._last_request_time = time.monotonic()


def _retry_with_backoff(func: Any) -> Any:
    """Decorator that retries an Entrez call with exponential backoff.

    Handles NCBI rate limit errors (HTTP 429) and transient network errors.
    """
    # This is returned as a closure rather than a true decorator to keep
    # the implementation straightforward without functools.wraps complexity.
    return func  # Placeholder; actual retry logic is in _entrez_request


class PubmedCrawler:
    """Crawler for PubMed/MEDLINE sleep medicine literature.

    Searches PubMed using sleep-related MeSH terms and keywords, fetches
    article metadata (title, abstract, keywords, MeSH terms), and attempts
    to retrieve full text from PMC for open-access articles.

    All output is saved as JSONL files in the specified output directory.

    Attributes:
        email: Email address required by NCBI for API access.
        api_key: Optional NCBI API key for higher rate limits.
        output_dir: Directory where JSONL output files are written.
    """

    def __init__(
        self,
        email: str,
        api_key: Optional[str] = None,
        output_dir: str = "data/raw/pubmed",
    ) -> None:
        self._email = email
        self._api_key = api_key
        self._output_dir = Path(output_dir)
        self._rate_limiter = RateLimiter(max_requests_per_second=NCBI_RATE_LIMIT)

        # Configure Entrez with credentials
        Entrez.email = email
        if api_key is not None:
            Entrez.api_key = api_key

        # Track processed PMIDs for resumability
        self._processed_pmids: set[str] = set()
        self._load_processed_ids()

        # Create output directory
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def _load_processed_ids(self) -> None:
        """Load previously processed PMIDs from the output directory.

        Enables resumable crawling by skipping already-processed articles.
        """
        for jsonl_file in self._output_dir.glob("*.jsonl"):
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        pmid = record.get("pmid")
                        if pmid is not None:
                            self._processed_pmids.add(str(pmid))
                    except json.JSONDecodeError:
                        logger.warning(
                            "Skipping malformed line in %s", jsonl_file
                        )
        if self._processed_pmids:
            logger.info(
                "Loaded %d previously processed PMIDs for resumption",
                len(self._processed_pmids),
            )

    def _entrez_request(self, func_name: str, **kwargs: Any) -> Any:
        """Execute an Entrez API request with rate limiting and retry logic.

        Args:
            func_name: Name of the Entrez function to call (e.g., 'esearch', 'efetch').
            **kwargs: Keyword arguments forwarded to the Entrez function.

        Returns:
            Entrez response handle.

        Raises:
            RuntimeError: If all retry attempts are exhausted.
        """
        func = getattr(Entrez, func_name)
        last_error: Optional[Exception] = None

        for attempt in range(MAX_RETRY_ATTEMPTS):
            self._rate_limiter.wait()
            try:
                handle = func(**kwargs)
                return handle
            except Exception as exc:
                last_error = exc
                error_str = str(exc).lower()
                # Retry on rate limit or transient errors
                if "429" in error_str or "rate" in error_str or "timeout" in error_str:
                    backoff = min(
                        RETRY_BACKOFF_BASE ** (attempt + 1),
                        RETRY_BACKOFF_MAX,
                    )
                    logger.warning(
                        "NCBI rate limit hit (attempt %d/%d), backing off %.1fs: %s",
                        attempt + 1,
                        MAX_RETRY_ATTEMPTS,
                        backoff,
                        exc,
                    )
                    time.sleep(backoff)
                else:
                    raise

        raise RuntimeError(
            f"Failed Entrez.{func_name} after {MAX_RETRY_ATTEMPTS} attempts"
        ) from last_error

    def search(self, query: str, max_results: int = 10000) -> list[str]:
        """Search PubMed with a given query string.

        Uses NCBI E-utilities REST API directly via requests to avoid XML
        parsing issues with Bio.Entrez.

        Args:
            query: PubMed search query (supports MeSH, title/abstract, etc.).
            max_results: Maximum number of PMIDs to return.

        Returns:
            List of PubMed IDs matching the query.
        """
        logger.info(
            "Searching PubMed for: %s (max_results=%d)", query, max_results
        )
        all_pmids: list[str] = []
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

        try:
            params = {
                "db": "pubmed",
                "term": query,
                "retmax": min(max_results, 10000),
                "retmode": "json",
                "sort": "pub_date",
                "email": self.email,
            }
            if self._api_key:
                params["api_key"] = self._api_key

            self._rate_limiter.wait()
            resp = requests.get(base_url, params=params, timeout=30)
            resp.raise_for_status()
            search_results = resp.json()

            count = int(search_results["esearchresult"]["count"])
            pmid_list = search_results["esearchresult"]["idlist"]
            all_pmids.extend(pmid_list)

            logger.info(
                "Found %d total results, retrieved %d PMIDs (batch 1)",
                count,
                len(pmid_list),
            )

            # Paginate through remaining results using WebEnv
            if count > len(pmid_list) and len(all_pmids) < max_results:
                web_env = search_results["esearchresult"]["webenv"]
                query_key = search_results["esearchresult"]["querykey"]
                batch_size = 10000

                while len(all_pmids) < max_results:
                    ret_start = len(all_pmids)
                    ret_count = min(batch_size, max_results - ret_start)

                    params_page = {
                        "db": "pubmed",
                        "term": query,
                        "retstart": ret_start,
                        "retmax": ret_count,
                        "retmode": "json",
                        "webenv": web_env,
                        "query_key": query_key,
                        "email": self.email,
                    }
                    if self._api_key:
                        params_page["api_key"] = self._api_key

                    self._rate_limiter.wait()
                    resp = requests.get(base_url, params=params_page, timeout=30)
                    resp.raise_for_status()
                    batch_results = resp.json()

                    batch_ids = batch_results["esearchresult"]["idlist"]
                    if not batch_ids:
                        break

                    all_pmids.extend(batch_ids)
                    logger.info(
                        "Retrieved batch: %d/%d PMIDs", len(all_pmids), max_results
                    )

        except Exception as exc:
            logger.error("Search failed for query '%s': %s", query, exc)

        return all_pmids[:max_results]

    def fetch_details(self, pmids: list[str]) -> list[dict[str, Any]]:
        """Fetch article metadata for a list of PMIDs.

        Retrieves title, abstract, authors, journal, keywords, MeSH terms,
        and other bibliographic information from PubMed.

        Args:
            pmids: List of PubMed IDs to fetch details for.

        Returns:
            List of dictionaries containing article metadata.
        """
        if not pmids:
            return []

        # Filter out already-processed PMIDs
        new_pmids = [p for p in pmids if p not in self._processed_pmids]
        if not new_pmids:
            logger.info("All %d PMIDs already processed, skipping", len(pmids))
            return []

        logger.info(
            "Fetching details for %d PMIDs (%d already processed)",
            len(new_pmids),
            len(pmids) - len(new_pmids),
        )

        articles: list[dict[str, Any]] = []
        batch_size = 200  # Entrez efetch limit per request

        for i in range(0, len(new_pmids), batch_size):
            batch = new_pmids[i : i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(new_pmids) + batch_size - 1) // batch_size

            logger.info(
                "Fetching batch %d/%d (%d PMIDs)", batch_num, total_batches, len(batch)
            )

            try:
                efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
                params = {
                    "db": "pubmed",
                    "id": ",".join(batch),
                    "rettype": "medline",
                    "retmode": "text",
                    "email": self.email,
                }
                if self._api_key:
                    params["api_key"] = self._api_key

                self._rate_limiter.wait()
                resp = requests.get(efetch_url, params=params, timeout=60)
                resp.raise_for_status()
                batch_articles = self._parse_pubmed_text(resp.text)

                articles.extend(batch_articles)
                logger.info(
                    "Parsed %d articles from batch %d", len(batch_articles), batch_num
                )

            except Exception as exc:
                logger.error(
                    "Failed to fetch details for batch %d: %s", batch_num, exc
                )

        return articles

    def _parse_pubmed_text(self, text: str) -> list[dict[str, Any]]:
        """Parse PubMed Medline text format into structured article records.

        Extracts title, abstract, authors, journal, publication year,
        keywords, MeSH terms, DOI, and article type.

        Args:
            text: Medline format text response from NCBI E-utilities.

        Returns:
            List of article dictionaries.
        """
        from Bio import Medline
        from io import StringIO

        try:
            records = list(Medline.parse(StringIO(text)))
            articles = []

            for record in records:
                if not record.get("PMID"):
                    continue

                # Extract authors
                authors_raw = record.get("AU", [])
                authors = list(authors_raw) if isinstance(authors_raw, list) else [authors_raw]

                # Extract publication year
                pub_year = None
                dp = record.get("DP", "")
                if dp:
                    year_str = dp.split()[0] if dp else ""
                    if year_str.isdigit():
                        pub_year = int(year_str)

                # Extract MeSH terms
                mesh_raw = record.get("MH", [])
                mesh_terms = list(mesh_raw) if isinstance(mesh_raw, list) else [mesh_raw]

                # Extract keywords
                kw_raw = record.get("OT", [])
                keywords = list(kw_raw) if isinstance(kw_raw, list) else [kw_raw]
                # Also include MH terms as keywords
                keywords.extend(mesh_raw if isinstance(mesh_raw, list) else [mesh_raw])

                # Extract PMC ID from PMCID field
                pmc_id = record.get("PMC")
                if pmc_id:
                    pmc_id = str(pmc_id).replace("PMC", "")

                article = {
                    "pmid": record.get("PMID", ""),
                    "pmc_id": pmc_id,
                    "title": record.get("TI", ""),
                    "abstract": record.get("AB", ""),
                    "authors": authors,
                    "journal": record.get("TA", ""),
                    "publication_year": pub_year,
                    "keywords": keywords,
                    "mesh_terms": mesh_terms,
                    "doi": record.get("LID", "").split(" ")[0] if record.get("LID") else None,
                    "article_type": record.get("PT", [""])[0] if record.get("PT") else "",
                }
                articles.append(article)

            return articles

        except Exception as exc:
            logger.error("Failed to parse Medline records: %s", exc)
            return []

    def fetch_fulltext(self, pmids: list[str]) -> list[dict[str, Any]]:
        """Fetch full text from PMC for articles with PMC IDs.

        Uses the Entrez efetch API to retrieve PMC open-access full text
        in XML format and extracts the body text.

        Args:
            pmids: List of PubMed IDs. Only those with PMC IDs will be fetched.

        Returns:
            List of article dictionaries with full_text field populated.
        """
        # First get details to identify which have PMC IDs
        details = self.fetch_details(pmids)
        pmc_articles = [
            a for a in details if a.get("pmc_id") and a["pmc_id"].startswith("PMC")
        ]

        if not pmc_articles:
            logger.info("No PMC IDs found among %d articles", len(details))
            return details

        logger.info(
            "Found %d articles with PMC IDs, fetching full text", len(pmc_articles)
        )

        pmc_ids = [a["pmc_id"] for a in pmc_articles]
        batch_size = 50

        for i in range(0, len(pmc_ids), batch_size):
            batch = pmc_ids[i : i + batch_size]
            batch_num = i // batch_size + 1

            logger.info(
                "Fetching PMC full text batch %d/%d", batch_num,
                (len(pmc_ids) + batch_size - 1) // batch_size,
            )

            try:
                handle = self._entrez_request(
                    "efetch",
                    db="pmc",
                    id=",".join(batch),
                    rettype="full",
                    retmode="xml",
                )
                fulltext_data = self._parse_pmc_xml(handle)
                handle.close()

                # Merge full text into article records
                for pmc_id, text in fulltext_data.items():
                    for article in pmc_articles:
                        if article.get("pmc_id") == pmc_id:
                            article["full_text"] = text
                            break

            except Exception as exc:
                logger.error("Failed to fetch PMC full text batch %d: %s", batch_num, exc)

        return details

    def _parse_pmc_xml(self, handle: Any) -> dict[str, str]:
        """Parse PMC full-text XML and extract body text.

        Navigates the PMC article XML structure to extract text from
        body sections, figure captions, and table content.

        Args:
            handle: Entrez efetch XML response handle for PMC.

        Returns:
            Dictionary mapping PMC ID to extracted full text string.
        """
        import xml.etree.ElementTree as ET

        result: dict[str, str] = {}
        try:
            tree = ET.parse(handle)
            root = tree.getroot()

            # PMC XML namespace handling
            ns = {
                "pmc": "http://www.ncbi.nlm.nih.gov/pmc/articles/",
                "mml": "http://www.w3.org/1998/Math/MathML",
                "xlink": "http://www.w3.org/1999/xlink",
            }

            for article in root.findall(".//article", ns) or root.findall(".//article"):
                pmc_id_el = article.find(".//article-id[@pub-id-type='pmc']")
                if pmc_id_el is None or not pmc_id_el.text:
                    continue
                pmc_id = f"PMC{pmc_id_el.text}"

                # Extract body text
                body_parts: list[str] = []
                body = article.find(".//body") or article.find(".//body", ns)
                if body is not None:
                    for p in body.iter():
                        if p.tag.endswith("}p") or p.tag == "p":
                            if p.text and p.text.strip():
                                body_parts.append(p.text.strip())

                # Extract abstract
                abstract_parts: list[str] = []
                abstract = article.find(".//abstract") or article.find(".//abstract", ns)
                if abstract is not None:
                    for p in abstract.iter():
                        if p.tag.endswith("}p") or p.tag == "p":
                            if p.text and p.text.strip():
                                abstract_parts.append(p.text.strip())

                # Extract figure captions
                fig_captions: list[str] = []
                for fig in article.iter():
                    if fig.tag.endswith("}fig") or fig.tag == "fig":
                        caption = fig.find(".//caption") or fig.find(".//caption", ns)
                        if caption is not None and caption.text:
                            fig_captions.append(caption.text.strip())

                # Extract table content
                table_parts: list[str] = []
                for table in article.iter():
                    if table.tag.endswith("}table") or table.tag == "table":
                        # Get table label and caption
                        label_el = table.find(".//label") or table.find(".//label", ns)
                        caption_el = (
                            table.find(".//caption") or table.find(".//caption", ns)
                        )
                        if caption_el is not None and caption_el.text:
                            table_text = caption_el.text.strip()
                            if label_el is not None and label_el.text:
                                table_text = f"{label_el.text.strip()}: {table_text}"
                            table_parts.append(table_text)

                full_text_sections: list[str] = []
                if abstract_parts:
                    full_text_sections.append("ABSTRACT\n" + "\n".join(abstract_parts))
                if body_parts:
                    full_text_sections.append("BODY\n" + "\n".join(body_parts))
                if fig_captions:
                    full_text_sections.append(
                        "FIGURES\n" + "\n".join(fig_captions)
                    )
                if table_parts:
                    full_text_sections.append(
                        "TABLES\n" + "\n".join(table_parts)
                    )

                if full_text_sections:
                    result[pmc_id] = "\n\n".join(full_text_sections)

        except ET.ParseError as exc:
            logger.error("Failed to parse PMC XML: %s", exc)

        return result

    def _save_articles(self, articles: list[dict[str, Any]], filename: str) -> int:
        """Append articles to a JSONL output file.

        Each article is written as a single JSON object on its own line.
        Already-processed PMIDs are skipped to support resumable crawling.

        Args:
            articles: List of article dictionaries to save.
            filename: Name of the JSONL output file.

        Returns:
            Number of new articles actually written to the file.
        """
        if not articles:
            return 0

        output_path = self._output_dir / filename
        written = 0

        with open(output_path, "a", encoding="utf-8") as f:
            for article in articles:
                pmid = article.get("pmid", "")
                if pmid in self._processed_pmids:
                    continue

                f.write(json.dumps(article, ensure_ascii=False) + "\n")
                self._processed_pmids.add(pmid)
                written += 1

        logger.info(
            "Wrote %d new articles to %s (total processed: %d)",
            written,
            output_path,
            len(self._processed_pmids),
        )
        return written

    def crawl_sleep_papers(self, max_papers: int = 50000) -> list[dict[str, Any]]:
        """Main entry point for crawling sleep medicine literature from PubMed.

        Searches PubMed using comprehensive sleep-related MeSH terms and
        keyword queries, fetches article metadata, and attempts to retrieve
        full text from PMC for open-access articles.

        Results are saved incrementally to JSONL files as they are retrieved,
        enabling resumption if the crawl is interrupted.

        Args:
            max_papers: Maximum total number of unique papers to collect.

        Returns:
            List of all collected article dictionaries.
        """
        logger.info(
            "Starting PubMed sleep literature crawl (max_papers=%d)", max_papers
        )
        start_time = time.time()
        all_articles: list[dict[str, Any]] = []
        seen_pmids: set[str] = set(self._processed_pmids)

        # Build combined query from MeSH terms
        mesh_query = " OR ".join(SLEEP_MESH_TERMS)
        all_queries = [mesh_query] + SLEEP_KEYWORD_QUERIES

        for query_idx, query in enumerate(all_queries):
            if len(seen_pmids) >= max_papers:
                logger.info(
                    "Reached max_papers limit (%d), stopping search", max_papers
                )
                break

            logger.info(
                "Processing query %d/%d: %s",
                query_idx + 1,
                len(all_queries),
                query[:80],
            )

            # Search for PMIDs
            remaining = max_papers - len(seen_pmids)
            pmids = self.search(query, max_results=remaining)

            # Deduplicate
            new_pmids = [p for p in pmids if p not in seen_pmids]
            if not new_pmids:
                logger.info("No new PMIDs from this query")
                continue

            logger.info(
                "Found %d new PMIDs (total unique: %d)",
                len(new_pmids),
                len(seen_pmids) + len(new_pmids),
            )

            # Fetch article details
            articles = self.fetch_details(new_pmids)

            if articles:
                # Save metadata
                filename = f"sleep_papers_metadata_{query_idx}.jsonl"
                self._save_articles(articles, filename)
                all_articles.extend(articles)
                seen_pmids.update(a.get("pmid", "") for a in articles)

        logger.info(
            "Metadata collection complete. Total articles: %d. Fetching PMC full text...",
            len(all_articles),
        )

        # Fetch full text from PMC for articles with PMC IDs
        pmids_with_pmc = [
            a["pmid"]
            for a in all_articles
            if a.get("pmc_id") and a["pmid"] not in self._processed_pmids
        ]

        if pmids_with_pmc:
            fulltext_articles = self.fetch_fulltext(pmids_with_pmc)
            # Save articles with full text
            if fulltext_articles:
                self._save_articles(fulltext_articles, "sleep_papers_fulltext.jsonl")

        elapsed = time.time() - start_time
        logger.info(
            "PubMed crawl complete. %d articles collected in %.1f seconds "
            "(%d with full text)",
            len(all_articles),
            elapsed,
            sum(1 for a in all_articles if a.get("full_text")),
        )

        # Save final merged output
        self._save_articles(all_articles, "sleep_papers_all.jsonl")

        return all_articles
