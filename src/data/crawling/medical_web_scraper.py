"""
Medical website scraper for sleep medicine domain content.

Crawls authoritative medical websites (sleepfoundation.org, sleepeducation.org,
ninds.nih.gov) for sleep medicine content. Uses trafilatura for robust main
content extraction and respects robots.txt with per-domain rate limiting.

Designed to collect patient-facing educational content and clinical guidelines
for training data enrichment.
"""

import json
import logging
import re
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests

logger = logging.getLogger(__name__)

# Target medical websites for sleep content
TARGET_SITES: dict[str, dict[str, Any]] = {
    "sleepfoundation": {
        "base_url": "https://www.sleepfoundation.org",
        "max_pages": 2000,
        "allowed_paths": ["/sleep/", "/insomnia/", "/sleep-apnea/", "/narcolepsy/"],
        "rate_limit": 2.0,  # seconds between requests
        "description": "National Sleep Foundation - patient education",
    },
    "sleepeducation": {
        "base_url": "https://www.sleepeducation.org",
        "max_pages": 1500,
        "allowed_paths": ["/essentials", "/disorders", "/treatment"],
        "rate_limit": 2.0,
        "description": "AASM Sleep Education - clinical education",
    },
    "ninds_sleep": {
        "base_url": "https://www.ninds.nih.gov",
        "max_pages": 500,
        "allowed_paths": ["/health-information/disorders/sleep"],
        "rate_limit": 3.0,
        "description": "NINDS Sleep Disorders - NIH research content",
    },
}

# Per-domain rate limit defaults
DEFAULT_RATE_LIMIT = 2.0
DEFAULT_TIMEOUT = 30
MAX_DEPTH = 4  # Max link depth from base URL
MAX_REDIRECTS = 5

# Request headers to mimic a browser
DEFAULT_HEADERS = {
    "User-Agent": (
        "DeepSleepLLM/1.0 (Educational sleep medicine research; "
        "+https://github.com/deepsleep)"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
}

MAX_RETRY_ATTEMPTS = 3
RETRY_BACKOFF_BASE = 2.0


@dataclass(frozen=True)
class ScrapedPage:
    """Structured representation of a scraped web page."""

    url: str
    domain: str
    title: str
    content: str
    author: Optional[str]
    publish_date: Optional[str]
    meta_description: Optional[str]
    crawl_date: str
    status_code: int
    content_type: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MedicalWebScraper:
    """Scraper for authoritative medical websites focused on sleep medicine.

    Crawls target sites with depth-limited BFS, respects robots.txt, applies
    per-domain rate limiting, and uses trafilatura for content extraction.

    All output is saved as JSONL files organized by source domain.

    Attributes:
        output_dir: Directory for JSONL output files.
    """

    def __init__(self, output_dir: str = "data/raw/websites") -> None:
        self._output_dir = Path(output_dir)
        self._state_file = self._output_dir / "_scraper_state.json"

        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Track processed URLs per domain for resumability
        self._processed_urls: dict[str, set[str]] = defaultdict(set)
        self._load_state()

        # Per-domain rate limiting
        self._last_request_time: dict[str, float] = defaultdict(float)

        # Per-domain robots.txt parser cache
        self._robots_parsers: dict[str, RobotFileParser] = {}

        # HTTP session with connection pooling
        self._session = requests.Session()
        self._session.headers.update(DEFAULT_HEADERS)
        self._session.max_redirects = MAX_REDIRECTS

    def _load_state(self) -> None:
        """Load previously processed URLs from the state file."""
        if not self._state_file.exists():
            return

        try:
            with open(self._state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
            for domain, urls in state.get("processed_urls", {}).items():
                self._processed_urls[domain] = set(urls)
            logger.info(
                "Resuming: %d URLs already processed across %d domains",
                sum(len(v) for v in self._processed_urls.values()),
                len(self._processed_urls),
            )
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Failed to load state file, starting fresh: %s", exc)

    def _save_state(self) -> None:
        """Persist processed URLs to state file."""
        state = {
            "processed_urls": {
                domain: sorted(urls)
                for domain, urls in self._processed_urls.items()
            },
            "total_urls": sum(len(v) for v in self._processed_urls.values()),
            "last_updated": datetime.now().isoformat(),
        }
        with open(self._state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def _check_robots_txt(self, url: str) -> bool:
        """Check if the URL is allowed by the domain's robots.txt.

        Args:
            url: Full URL to check.

        Returns:
            True if the URL is allowed to be crawled, False otherwise.
        """
        parsed = urlparse(url)
        domain = parsed.netloc

        if domain not in self._robots_parsers:
            rp = RobotFileParser()
            robots_url = f"{parsed.scheme}://{domain}/robots.txt"
            try:
                rp.set_url(robots_url)
                rp.read()
                self._robots_parsers[domain] = rp
            except Exception as exc:
                logger.debug("Could not fetch robots.txt for %s: %s", domain, exc)
                # If robots.txt is unreachable, allow crawling by default
                rp = RobotFileParser()
                self._robots_parsers[domain] = rp

        rp = self._robots_parsers[domain]
        try:
            return rp.can_fetch(DEFAULT_HEADERS["User-Agent"], url)
        except Exception:
            return True

    def _rate_limit_wait(self, domain: str, rate_limit: float = DEFAULT_RATE_LIMIT) -> None:
        """Wait to satisfy the per-domain rate limit.

        Args:
            domain: Domain name for per-domain rate limiting.
            rate_limit: Minimum seconds between requests to this domain.
        """
        now = time.monotonic()
        elapsed = now - self._last_request_time[domain]
        if elapsed < rate_limit:
            time.sleep(rate_limit - elapsed)
        self._last_request_time[domain] = time.monotonic()

    def _fetch_page(
        self, url: str, domain: str, rate_limit: float = DEFAULT_RATE_LIMIT
    ) -> Optional[requests.Response]:
        """Fetch a web page with rate limiting and retry logic.

        Args:
            url: URL to fetch.
            domain: Domain name for rate limiting.
            rate_limit: Minimum seconds between requests.

        Returns:
            HTTP response object, or None on failure.
        """
        # Check robots.txt
        if not self._check_robots_txt(url):
            logger.debug("Blocked by robots.txt: %s", url)
            return None

        for attempt in range(MAX_RETRY_ATTEMPTS):
            self._rate_limit_wait(domain, rate_limit)
            try:
                response = self._session.get(
                    url,
                    timeout=DEFAULT_TIMEOUT,
                    allow_redirects=True,
                )
                response.raise_for_status()
                return response

            except requests.exceptions.HTTPError as exc:
                if exc.response is not None and exc.response.status_code == 429:
                    backoff = RETRY_BACKOFF_BASE ** (attempt + 1)
                    logger.warning(
                        "Rate limited by %s (429), backing off %.1fs",
                        domain,
                        backoff,
                    )
                    time.sleep(backoff)
                elif exc.response is not None and exc.response.status_code in (403, 404):
                    logger.debug("HTTP %d for %s", exc.response.status_code, url)
                    return None
                else:
                    logger.warning(
                        "HTTP error for %s (attempt %d): %s",
                        url,
                        attempt + 1,
                        exc,
                    )

            except requests.exceptions.RequestException as exc:
                logger.warning(
                    "Request failed for %s (attempt %d): %s",
                    url,
                    attempt + 1,
                    exc,
                )

            if attempt < MAX_RETRY_ATTEMPTS - 1:
                time.sleep(RETRY_BACKOFF_BASE ** (attempt + 1))

        logger.error("Failed to fetch %s after %d attempts", url, MAX_RETRY_ATTEMPTS)
        return None

    def _extract_content(self, html: str, url: str) -> dict[str, Any]:
        """Extract main content from HTML using trafilatura.

        Falls back to a basic regex extraction if trafilatura fails.

        Args:
            html: Raw HTML content of the page.
            url: Source URL for content extraction metadata.

        Returns:
            Dictionary with 'title', 'content', 'author', 'publish_date',
            and 'meta_description' keys.
        """
        result: dict[str, Any] = {
            "title": "",
            "content": "",
            "author": None,
            "publish_date": None,
            "meta_description": None,
        }

        try:
            import trafilatura

            downloaded = trafilatura.fetch_url(url)
            if downloaded is None:
                # Use the provided HTML directly
                downloaded = html

            metadata = trafilatura.extract_metadata(downloaded)
            content = trafilatura.extract(downloaded)

            if content:
                result["content"] = content.strip()

            if metadata:
                result["title"] = metadata.title or ""
                result["author"] = metadata.author
                result["publish_date"] = metadata.date
                result["meta_description"] = metadata.description

        except ImportError:
            logger.warning(
                "trafilatura not installed, using fallback extraction. "
                "Install with: pip install trafilatura"
            )
            result.update(self._fallback_extraction(html))

        except Exception as exc:
            logger.warning("trafilatura extraction failed: %s", exc)
            result.update(self._fallback_extraction(html))

        return result

    def _fallback_extraction(self, html: str) -> dict[str, Any]:
        """Basic fallback content extraction when trafilatura is unavailable.

        Extracts title from <title> tag and strips HTML tags for content.

        Args:
            html: Raw HTML string.

        Returns:
            Dictionary with 'title' and 'content' keys.
        """
        title = ""
        title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.DOTALL | re.I)
        if title_match:
            title = re.sub(r"<[^>]+>", "", title_match.group(1)).strip()

        # Remove script and style tags
        cleaned = re.sub(
            r"<(script|style|nav|header|footer)[^>]*>.*?</\1>",
            "",
            html,
            flags=re.DOTALL | re.I,
        )
        # Remove HTML tags
        content = re.sub(r"<[^>]+>", " ", cleaned)
        # Collapse whitespace
        content = re.sub(r"\s+", " ", content).strip()

        return {"title": title, "content": content}

    def _extract_links(
        self, html: str, base_url: str, allowed_paths: list[str]
    ) -> list[str]:
        """Extract and filter links from an HTML page.

        Only returns links that match the allowed path prefixes and belong
        to the same domain as the base URL.

        Args:
            html: Raw HTML content.
            base_url: Base URL of the page for resolving relative links.
            allowed_paths: List of allowed URL path prefixes.

        Returns:
            List of absolute URLs that match the filtering criteria.
        """
        parsed_base = urlparse(base_url)
        links: list[str] = []
        seen: set[str] = set()

        # Find all href attributes
        href_pattern = re.compile(r'href=["\']([^"\']+)["\']', re.I)
        for match in href_pattern.finditer(html):
            href = match.group(1).strip()
            if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
                continue

            # Resolve relative URLs
            absolute_url = urljoin(base_url, href)
            parsed = urlparse(absolute_url)

            # Must be same domain
            if parsed.netloc != parsed_base.netloc:
                continue

            # Must be HTTP/HTTPS
            if parsed.scheme not in ("http", "https"):
                continue

            # Must match at least one allowed path (or have no specific filter)
            if allowed_paths:
                if not any(parsed.path.startswith(p) for p in allowed_paths):
                    # Also allow the root and base path
                    if parsed.path not in ("/", parsed_base.path):
                        continue

            # Deduplicate
            if absolute_url not in seen:
                seen.add(absolute_url)
                links.append(absolute_url)

        return links

    def scrape_site(
        self,
        site_key: str,
        max_pages: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Scrape all crawlable pages from a target medical website.

        Performs a breadth-first crawl from the site's base URL, respecting
        robots.txt, depth limits, and per-domain rate limits. Extracts main
        content using trafilatura.

        Args:
            site_key: Key into TARGET_SITES dict identifying the site.
            max_pages: Override max pages from TARGET_SITES config.

        Returns:
            List of scraped page dictionaries.
        """
        if site_key not in TARGET_SITES:
            logger.error("Unknown site key: %s", site_key)
            return []

        config = TARGET_SITES[site_key]
        base_url = config["base_url"]
        max_pages = max_pages or config["max_pages"]
        allowed_paths = config["allowed_paths"]
        rate_limit = config["rate_limit"]
        domain = urlparse(base_url).netloc

        logger.info(
            "Scraping %s: %s (max_pages=%d, rate_limit=%.1fs)",
            site_key,
            base_url,
            max_pages,
            rate_limit,
        )

        pages: list[dict[str, Any]] = []
        to_visit: list[tuple[str, int]] = [(base_url, 0)]  # (url, depth)
        visited: set[str] = set(self._processed_urls.get(domain, set()))
        output_file = self._output_dir / f"{site_key}.jsonl"

        while to_visit and len(pages) < max_pages:
            url, depth = to_visit.pop(0)

            if url in visited:
                continue
            visited.add(url)

            # Respect depth limit
            if depth > MAX_DEPTH:
                continue

            # Fetch page
            response = self._fetch_page(url, domain, rate_limit)
            if response is None:
                continue

            # Only process HTML content
            content_type = response.headers.get("Content-Type", "")
            if "text/html" not in content_type and "application/xhtml" not in content_type:
                continue

            html = response.text

            # Extract content
            content_data = self._extract_content(html, url)

            if not content_data["content"] or len(content_data["content"]) < 100:
                logger.debug("Skipping low-content page: %s", url)
                # Still extract links for crawling even if content is thin
                links = self._extract_links(html, base_url, allowed_paths)
                for link in links:
                    if link not in visited:
                        to_visit.append((link, depth + 1))
                continue

            page = ScrapedPage(
                url=url,
                domain=domain,
                title=content_data["title"],
                content=content_data["content"],
                author=content_data.get("author"),
                publish_date=content_data.get("publish_date"),
                meta_description=content_data.get("meta_description"),
                crawl_date=datetime.now().isoformat(),
                status_code=response.status_code,
                content_type=content_type,
            )

            page_dict = page.to_dict()
            pages.append(page_dict)

            # Save incrementally for resumability
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(page_dict, ensure_ascii=False) + "\n")

            if len(pages) % 50 == 0:
                logger.info(
                    "Scraped %d/%d pages from %s", len(pages), max_pages, site_key
                )

            # Extract links for further crawling
            links = self._extract_links(html, base_url, allowed_paths)
            for link in links:
                if link not in visited and len(pages) + len(to_visit) < max_pages:
                    to_visit.append((link, depth + 1))

        # Update state
        self._processed_urls[domain] = visited
        self._save_state()

        logger.info(
            "Finished scraping %s: %d pages collected", site_key, len(pages)
        )
        return pages

    def scrape_all_sites(self) -> list[dict[str, Any]]:
        """Main entry point for scraping all target medical websites.

        Iterates through all configured target sites, scrapes each one,
        and returns the combined results.

        Returns:
            List of all scraped page dictionaries from all sites.
        """
        logger.info("Starting medical website scraping for all target sites")
        start_time = time.time()
        all_pages: list[dict[str, Any]] = []

        for site_key in TARGET_SITES:
            try:
                pages = self.scrape_site(site_key)
                all_pages.extend(pages)
            except Exception as exc:
                logger.error("Failed to scrape %s: %s", site_key, exc)

        # Save combined output
        combined_file = self._output_dir / "all_sites_combined.jsonl"
        with open(combined_file, "w", encoding="utf-8") as f:
            for page in all_pages:
                f.write(json.dumps(page, ensure_ascii=False) + "\n")

        elapsed = time.time() - start_time
        domain_counts: dict[str, int] = defaultdict(int)
        for page in all_pages:
            domain_counts[page["domain"]] += 1

        logger.info(
            "All site scraping complete. %d pages in %.1f seconds. "
            "Per-domain: %s",
            len(all_pages),
            elapsed,
            dict(domain_counts),
        )

        return all_pages
