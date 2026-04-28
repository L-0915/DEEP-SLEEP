"""
PMC Open Access crawler for bulk downloading and parsing PMC articles.

Downloads open-access bulk packages from the PMC FTP server, extracts
archives, and parses PMC XML format to extract title, abstract, body text,
figure captions, and tables. Filters downloaded articles for sleep medicine
relevance using keyword matching.

Supports resumable downloads by tracking already-downloaded packages.
"""

import ftplib
import gzip
import json
import logging
import re
import shutil
import tarfile
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# PMC FTP configuration
PMC_FTP_HOST = "ftp.ncbi.nlm.nih.gov"
PMC_FTP_OA_DIR = "/pub/pmc/"
PMC_FTP_BULK_DIR = "/pub/pmc/open_access/"

# Sleep relevance keywords for filtering
SLEEP_KEYWORDS = [
    "sleep", "insomnia", "apnea", "narcolepsy", "circadian",
    "polysomnography", "cpap", "melatonin", "hypersomnia",
    "parasomnia", "restless legs", "rem sleep", "slow wave sleep",
    "sleep stage", "sleep disorder", "sleep quality", "sleep deprivation",
    "sleep architecture", "sleep hygiene", "sleep onset", "sleep maintenance",
    "obstructive sleep", "central sleep", "sleep breathing",
    "sleep fragmentation", "sleep efficiency", "sleep latency",
    "rapid eye movement", "non-rem", "dream", "somnolence",
    "nocturnal", "dyssomnia", "bruxism", "sleepwalking", "night terror",
    "shift work", "jet lag", "chronotherapy",
]

RELEVANCE_THRESHOLD = 3  # Minimum keyword matches to consider an article relevant


@dataclass(frozen=True)
class PMCArticle:
    """Structured representation of a PMC article."""

    pmc_id: str
    pmid: Optional[str]
    doi: Optional[str]
    title: str
    abstract: str
    body_text: str
    sections: list[dict[str, str]]
    figure_captions: list[str]
    table_captions: list[str]
    authors: list[str]
    journal: str
    publication_year: Optional[int]
    article_type: str
    keywords: list[str]
    relevance_score: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PMCCrawler:
    """Crawler for PMC Open Access bulk download packages.

    Downloads OA bulk packages from the PMC FTP server, extracts XML files,
    parses article content, and filters for sleep medicine relevance.

    Supports resumable downloads by tracking already-processed package names
    in a local state file.

    Attributes:
        output_dir: Directory for JSONL output files.
        download_dir: Directory for downloaded and extracted archives.
    """

    def __init__(self, output_dir: str = "data/raw/pmc") -> None:
        self._output_dir = Path(output_dir)
        self._download_dir = self._output_dir / "downloads"
        self._state_file = self._output_dir / "_download_state.json"

        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._download_dir.mkdir(parents=True, exist_ok=True)

        # Load download state for resumability
        self._downloaded_packages: set[str] = set()
        self._processed_articles: set[str] = set()
        self._load_state()

    def _load_state(self) -> None:
        """Load download state from the state file.

        Tracks which packages have been downloaded and which articles have
        been processed, enabling resumable crawling.
        """
        if not self._state_file.exists():
            return

        try:
            with open(self._state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
            self._downloaded_packages = set(state.get("downloaded_packages", []))
            self._processed_articles = set(state.get("processed_articles", []))
            logger.info(
                "Resuming: %d packages downloaded, %d articles processed",
                len(self._downloaded_packages),
                len(self._processed_articles),
            )
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Failed to load state file, starting fresh: %s", exc)

    def _save_state(self) -> None:
        """Persist download state to the state file."""
        state = {
            "downloaded_packages": sorted(self._downloaded_packages),
            "processed_articles": sorted(list(self._processed_articles)[:100000]),
            "downloaded_packages_count": len(self._downloaded_packages),
            "processed_articles_count": len(self._processed_articles),
        }
        with open(self._state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def list_available_packages(self) -> list[str]:
        """List available OA bulk download packages on the PMC FTP server.

        Connects to the PMC FTP server and enumerates the available
        open-access bulk download archives.

        Returns:
            List of package filenames available for download.
        """
        packages: list[str] = []

        try:
            ftp = ftplib.FTP(PMC_FTP_HOST)
            ftp.login()
            ftp.cwd(PMC_FTP_BULK_DIR)

            ftp.retrlines("NLST", lambda line: packages.append(line.strip()))
            ftp.quit()

            # Filter for archive files only
            packages = [
                p for p in packages
                if p.endswith((".tar.gz", ".tgz", ".tar")) and p
            ]

            logger.info("Found %d available OA bulk packages", len(packages))

        except ftplib.all_errors as exc:
            logger.error("FTP connection failed: %s", exc)

        return sorted(packages)

    def download_package(self, package_name: str) -> Optional[Path]:
        """Download and extract a PMC OA bulk package.

        Downloads the archive from the PMC FTP server, extracts it, and
        cleans up the archive file after extraction. Skips packages that
        have already been downloaded (resumable).

        Args:
            package_name: Name of the package file on the FTP server.

        Returns:
            Path to the extracted package directory, or None on failure.
        """
        if package_name in self._downloaded_packages:
            logger.info("Skipping already downloaded package: %s", package_name)
            extract_dir = self._download_dir / package_name.replace(".tar.gz", "")
            if extract_dir.exists():
                return extract_dir
            # Package was tracked but files are missing; redownload
            self._downloaded_packages.discard(package_name)

        extract_dir = self._download_dir / package_name.replace(".tar.gz", "").replace(
            ".tgz", ""
        )
        archive_path = self._download_dir / package_name

        try:
            logger.info("Downloading package: %s", package_name)

            ftp = ftplib.FTP(PMC_FTP_HOST)
            ftp.login()
            ftp.cwd(PMC_FTP_BULK_DIR)

            with open(archive_path, "wb") as f:
                ftp.retrbinary(f"RETR {package_name}", f.write)

            ftp.quit()
            logger.info("Downloaded %s (%.2f MB)", package_name, archive_path.stat().st_size / 1e6)

            # Extract archive
            if package_name.endswith(".tar.gz") or package_name.endswith(".tgz"):
                with tarfile.open(archive_path, "r:gz") as tar:
                    tar.extractall(extract_dir, filter="data")
            elif package_name.endswith(".tar"):
                with tarfile.open(archive_path, "r") as tar:
                    tar.extractall(extract_dir, filter="data")

            # Clean up archive to save disk space
            archive_path.unlink()
            logger.info("Extracted package to: %s", extract_dir)

            self._downloaded_packages.add(package_name)
            self._save_state()
            return extract_dir

        except (ftplib.all_errors, tarfile.TarError, OSError) as exc:
            logger.error("Failed to download/extract %s: %s", package_name, exc)
            if archive_path.exists():
                archive_path.unlink()
            return None

    def parse_pmc_article(self, xml_path: Path) -> Optional[PMCArticle]:
        """Parse a PMC XML article file into a structured record.

        Extracts title, abstract, body text (organized by sections),
        figure captions, table captions, authors, journal, and other
        metadata from the PMC XML format.

        Args:
            xml_path: Path to the PMC article XML file.

        Returns:
            PMCArticle dataclass instance, or None if parsing fails.
        """
        if not xml_path.exists():
            return None

        pmc_id = xml_path.stem  # Usually the PMC ID from the filename

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Handle XML namespaces
            ns_map: dict[str, str] = {}
            for event, elem in ET.iterparse(str(xml_path), events=["start-ns"]):
                if elem[0] and elem[0] not in ns_map:
                    ns_map[elem[0]] = elem[1]

            # Find article element (may be nested in different structures)
            article = root.find(".//article")
            if article is None:
                return None

            # Extract PMC ID
            pmc_id_el = article.find(".//article-id[@pub-id-type='pmc']")
            if pmc_id_el is not None and pmc_id_el.text:
                pmc_id = f"PMC{pmc_id_el.text}"

            # Extract PMID
            pmid = None
            pmid_el = article.find(".//article-id[@pub-id-type='pmid']")
            if pmid_el is not None and pmid_el.text:
                pmid = pmid_el.text

            # Extract DOI
            doi = None
            doi_el = article.find(".//article-id[@pub-id-type='doi']")
            if doi_el is not None and doi_el.text:
                doi = doi_el.text

            # Extract title
            title = ""
            title_el = article.find(".//article-title") or article.find(".//title-group/article-title")
            if title_el is not None:
                title = _get_element_text(title_el).strip()

            # Extract abstract
            abstract = ""
            abstract_el = article.find(".//abstract")
            if abstract_el is not None:
                abstract = _get_element_text(abstract_el).strip()

            # Extract body sections
            sections: list[dict[str, str]] = []
            body = article.find(".//body")
            if body is not None:
                for sec in body.iter():
                    tag = _strip_ns(sec.tag)
                    if tag == "sec":
                        sec_title_el = sec.find("title")
                        sec_title = (
                            _get_element_text(sec_title_el).strip()
                            if sec_title_el is not None
                            else ""
                        )
                        sec_text = _get_element_text(sec).strip()
                        if sec_text:
                            sections.append({"title": sec_title, "text": sec_text})

            # Extract body text (flat)
            body_text = "\n".join(s["text"] for s in sections if s["text"])
            if not body_text:
                body_el = article.find(".//body")
                if body_el is not None:
                    body_text = _get_element_text(body_el).strip()

            # Extract figure captions
            figure_captions: list[str] = []
            for fig in root.iter():
                tag = _strip_ns(fig.tag)
                if tag == "fig":
                    caption = fig.find("caption")
                    if caption is not None:
                        caption_text = _get_element_text(caption).strip()
                        if caption_text:
                            figure_captions.append(caption_text)

            # Extract table captions
            table_captions: list[str] = []
            for table in root.iter():
                tag = _strip_ns(table.tag)
                if tag == "table-wrap":
                    label_el = table.find("label")
                    caption_el = table.find("caption")
                    caption_text = ""
                    if caption_el is not None:
                        caption_text = _get_element_text(caption_el).strip()
                    if label_el is not None and label_el.text:
                        caption_text = f"{label_el.text.strip()}: {caption_text}"
                    if caption_text.strip():
                        table_captions.append(caption_text.strip())

            # Extract authors
            authors: list[str] = []
            for contrib in article.iter():
                tag = _strip_ns(contrib.tag)
                if tag == "contrib" and contrib.get("contrib-type") == "author":
                    surname = contrib.find(".//surname")
                    given_names = contrib.find(".//given-names")
                    if surname is not None:
                        name = surname.text or ""
                        if given_names is not None and given_names.text:
                            name = f"{given_names.text} {name}"
                        if name.strip():
                            authors.append(name.strip())

            # Extract journal
            journal = ""
            journal_el = article.find(".//journal-title")
            if journal_el is not None and journal_el.text:
                journal = journal_el.text.strip()

            # Extract publication year
            pub_year = None
            year_el = article.find(".//pub-date/year")
            if year_el is not None and year_el.text and year_el.text.isdigit():
                pub_year = int(year_el.text)

            # Extract article type
            article_type = article.get("article-type", "")

            # Extract keywords
            keywords: list[str] = []
            for kwp in article.iter():
                tag = _strip_ns(kwp.tag)
                if tag == "kwd-group":
                    for kwd in kwp.iter():
                        if _strip_ns(kwd.tag) == "kwd" and kwd.text:
                            keywords.append(kwd.text.strip())

            # Calculate sleep relevance score
            all_text = f"{title} {abstract} {body_text}".lower()
            relevance_score = sum(
                1 for kw in SLEEP_KEYWORDS if kw.lower() in all_text
            )

            return PMCArticle(
                pmc_id=pmc_id,
                pmid=pmid,
                doi=doi,
                title=title,
                abstract=abstract,
                body_text=body_text,
                sections=sections,
                figure_captions=figure_captions,
                table_captions=table_captions,
                authors=authors,
                journal=journal,
                publication_year=pub_year,
                article_type=article_type,
                keywords=keywords,
                relevance_score=relevance_score,
            )

        except ET.ParseError as exc:
            logger.error("XML parse error for %s: %s", xml_path, exc)
            return None
        except Exception as exc:
            logger.error("Error parsing PMC article %s: %s", xml_path, exc)
            return None

    def _process_package(
        self, package_dir: Path, output_file: Path
    ) -> list[dict[str, Any]]:
        """Process all XML articles in a downloaded package directory.

        Parses each XML file, filters for sleep relevance, and writes
        matching articles to the output JSONL file.

        Args:
            package_dir: Path to the extracted package directory.
            output_file: Path to the JSONL output file.

        Returns:
            List of relevant article dictionaries from this package.
        """
        articles: list[dict[str, Any]] = []
        xml_files = list(package_dir.rglob("*.xml")) + list(package_dir.rglob("*.nxml"))

        logger.info(
            "Processing package: %s (%d XML files)", package_dir.name, len(xml_files)
        )

        for xml_path in xml_files:
            # Skip already-processed articles
            article_key = xml_path.stem
            if article_key in self._processed_articles:
                continue

            try:
                article = self.parse_pmc_article(xml_path)
                if article is None:
                    continue

                # Filter by sleep relevance
                if article.relevance_score >= RELEVANCE_THRESHOLD:
                    article_dict = article.to_dict()
                    articles.append(article_dict)

                    # Write immediately for resumability
                    with open(output_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(article_dict, ensure_ascii=False) + "\n")

            except Exception as exc:
                logger.error("Error processing %s: %s", xml_path, exc)

            self._processed_articles.add(article_key)

        logger.info(
            "Found %d sleep-relevant articles from %s",
            len(articles),
            package_dir.name,
        )
        return articles

    def _get_disk_usage(self) -> float:
        """Calculate total disk usage of the download directory in GB."""
        total_size = 0
        for path in self._download_dir.rglob("*"):
            if path.is_file():
                total_size += path.stat().st_size
        return total_size / (1024 ** 3)

    def crawl_oa_sleep(self, max_gb: float = 50.0) -> list[dict[str, Any]]:
        """Main entry point for crawling PMC OA articles relevant to sleep.

        Downloads OA bulk packages from the PMC FTP server, extracts and
        parses XML articles, and filters for sleep medicine relevance.
        Respects the disk space limit and supports resumable downloads.

        Args:
            max_gb: Maximum disk space to use for downloads in gigabytes.

        Returns:
            List of all sleep-relevant article dictionaries collected.
        """
        logger.info(
            "Starting PMC OA sleep crawl (max_disk=%.1f GB)", max_gb
        )
        start_time = time.time()
        all_articles: list[dict[str, Any]] = []

        output_file = self._output_dir / "pmc_sleep_articles.jsonl"

        # List available packages
        packages = self.list_available_packages()
        if not packages:
            logger.error("No packages available on PMC FTP")
            return all_articles

        for package_name in packages:
            # Check disk usage limit
            current_gb = self._get_disk_usage()
            if current_gb >= max_gb:
                logger.info(
                    "Disk usage limit reached (%.1f / %.1f GB), stopping",
                    current_gb,
                    max_gb,
                )
                break

            if package_name in self._downloaded_packages:
                logger.info("Skipping already downloaded: %s", package_name)
                # Still process if not already done
                extract_dir = (
                    self._download_dir
                    / package_name.replace(".tar.gz", "").replace(".tgz", "")
                )
                if extract_dir.exists():
                    articles = self._process_package(extract_dir, output_file)
                    all_articles.extend(articles)
                continue

            # Download and extract package
            extract_dir = self.download_package(package_name)
            if extract_dir is None:
                continue

            # Process articles in the package
            articles = self._process_package(extract_dir, output_file)
            all_articles.extend(articles)

            # Save state periodically
            self._save_state()

        elapsed = time.time() - start_time
        logger.info(
            "PMC OA crawl complete. %d sleep-relevant articles collected in %.1f seconds",
            len(all_articles),
            elapsed,
        )
        return all_articles


def _get_element_text(element: ET.Element) -> str:
    """Recursively extract all text content from an XML element and its children.

    Handles nested elements by concatenating text content with appropriate
    whitespace separation.

    Args:
        element: XML element to extract text from.

    Returns:
        Concatenated text content of the element and its descendants.
    """
    text_parts: list[str] = []
    if element.text and element.text.strip():
        text_parts.append(element.text.strip())

    for child in element:
        child_text = _get_element_text(child)
        if child_text:
            text_parts.append(child_text)
        if child.tail and child.tail.strip():
            text_parts.append(child.tail.strip())

    return " ".join(text_parts)


def _strip_ns(tag: str) -> str:
    """Remove XML namespace prefix from a tag name.

    Args:
        tag: XML tag string potentially containing a namespace prefix.

    Returns:
        Tag name without namespace prefix.
    """
    if "}" in tag:
        return tag.split("}", 1)[1]
    if ":" in tag:
        return tag.split(":", 1)[1]
    return tag
