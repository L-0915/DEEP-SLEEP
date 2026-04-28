"""
General-purpose data crawler for pretraining.

Crawls diverse, high-quality content from multiple sources without domain
restrictions. Designed to build a broad pretraining corpus.

Sources:
1. Wikipedia (EN/ZH) - broad category crawl, not limited to sleep
2. arXiv - expanded queries across CS, math, physics, bio, etc.
3. PubMed/PMC Open Access - broader medical/scientific content
4. Project Gutenberg - public domain books
5. Stack Exchange Data Dump - Q&A across topics
"""

import json
import hashlib
import logging
import re
import time
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote, urljoin

import requests

logger = logging.getLogger(__name__)

OUTPUT_BASE = "data/raw/general"

# Wikipedia categories for broad coverage
WIKI_EN_CATEGORIES = [
    # Science
    "Computer science", "Mathematics", "Physics", "Chemistry", "Biology",
    "Medicine", "Neuroscience", "Psychology", "Genetics", "Immunology",
    "Pharmacology", "Epidemiology", "Anatomy", "Physiology", "Pathology",
    # Technology
    "Artificial intelligence", "Machine learning", "Software engineering",
    "Computer security", "Data science", "Natural language processing",
    "Computer vision", "Robotics", "Internet", "Web technology",
    # Humanities
    "Philosophy", "History", "Economics", "Sociology", "Anthropology",
    "Political science", "Law", "Education", "Linguistics",
    # Engineering
    "Electrical engineering", "Mechanical engineering", "Civil engineering",
    "Chemical engineering", "Aerospace engineering", "Biomedical engineering",
    # Arts
    "Music", "Literature", "Visual arts", "Architecture", "Film",
]

WIKI_ZH_CATEGORIES = [
    "计算机科学", "数学", "物理学", "化学", "生物学",
    "医学", "心理学", "人工智能", "机器学习", "经济学",
    "哲学", "历史", "文学", "教育学", "工程学",
    "电子工程", "机械工程", "生物医学工程", "法律", "政治学",
]

# arXiv categories for broad coverage
ARXIV_CATEGORIES = [
    "cs.AI", "cs.CL", "cs.CV", "cs.LG", "cs.NE", "cs.SE",
    "stat.ML", "stat.AP", "stat.CO",
    "q-bio.NC", "q-bio.QM", "q-bio.BM",
    "eess.SP", "eess.IV", "eess.AS",
    "math.ST", "math.PR", "math.OC",
    "physics.med-ph", "physics.bio-ph",
]

ARXIV_QUERIES = [
    "deep learning", "transformer", "neural network", "large language model",
    "reinforcement learning", "computer vision", "natural language processing",
    "graph neural network", "generative model", "attention mechanism",
    "medical imaging", "drug discovery", "protein structure", "genomics",
    "time series", "anomaly detection", "optimization", "causal inference",
    "Bayesian inference", "transfer learning", "self-supervised learning",
    "federated learning", "knowledge graph", "multimodal",
]

# PMC / PubMed broader queries
PUBMED_QUERIES = [
    "artificial intelligence medicine", "machine learning clinical",
    "deep learning diagnosis", "natural language processing medical",
    "computer vision radiology", "drug repurposing", "precision medicine",
    "electronic health records", "clinical decision support",
    "medical image analysis", "genomics analysis", "epidemiology",
    "public health", "mental health", "neuroscience", "immunology",
]

REQUEST_INTERVAL = 1.0


@dataclass(frozen=True)
class CrawledDocument:
    """Unified document format for all crawled data."""
    id: str
    source: str
    title: str
    url: str
    text: str
    lang: str
    crawl_date: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class WikipediaGeneralCrawler:
    """Crawl Wikipedia articles from broad categories."""

    def __init__(self, output_dir: str = f"{OUTPUT_BASE}/wikipedia") -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._processed: set[str] = set()
        self._state_file = self._output_dir / "_state.json"
        self._load_state()
        self._last_request: float = 0.0

    def _load_state(self) -> None:
        if self._state_file.exists():
            try:
                with open(self._state_file, "r", encoding="utf-8") as f:
                    state = json.load(f)
                self._processed = set(state.get("processed", []))
                logger.info("Wikipedia: resuming with %d articles processed", len(self._processed))
            except Exception:
                pass

    def _save_state(self) -> None:
        state = {
            "processed": sorted(self._processed),
            "count": len(self._processed),
            "last_updated": datetime.now().isoformat(),
        }
        with open(self._state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def _rate_limit(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_request
        if elapsed < REQUEST_INTERVAL:
            time.sleep(REQUEST_INTERVAL - elapsed)
        self._last_request = time.monotonic()

    def _fetch_article(self, title: str, lang: str) -> Optional[dict]:
        """Fetch a Wikipedia article via the MediaWiki API."""
        self._rate_limit()
        base = f"https://{lang}.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "titles": title,
            "prop": "extracts|info|categories",
            "explaintext": "1",
            "inprop": "url",
            "format": "json",
            "formatversion": "2",
        }
        try:
            resp = requests.get(base, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            page = data["query"]["pages"][0]
            if "missing" in page:
                return None
            return {
                "title": page.get("title", title),
                "url": page.get("fullurl", ""),
                "text": page.get("extract", ""),
            }
        except Exception as exc:
            logger.warning("Failed to fetch '%s' (%s): %s", title, lang, exc)
            return None

    def _get_category_members(self, category: str, lang: str, limit: int = 500) -> list[str]:
        """Get article titles from a Wikipedia category."""
        self._rate_limit()
        base = f"https://{lang}.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{category}",
            "cmlimit": min(limit, 500),
            "cmtype": "page",
            "format": "json",
        }
        titles = []
        try:
            resp = requests.get(base, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            for member in data.get("query", {}).get("categorymembers", []):
                titles.append(member["title"])
        except Exception as exc:
            logger.warning("Failed to get category '%s': %s", category, exc)
        return titles

    def crawl(self, max_articles: int = 5000) -> int:
        """Crawl Wikipedia articles from broad categories."""
        logger.info("Starting Wikipedia general crawl (target: %d articles)", max_articles)
        output_file = self._output_dir / "wikipedia_general.jsonl"
        total = 0

        for lang, categories in [("en", WIKI_EN_CATEGORIES), ("zh", WIKI_ZH_CATEGORIES)]:
            for cat in categories:
                if total >= max_articles:
                    break

                logger.info("  Category: %s (%s)", cat, lang)
                titles = self._get_category_members(cat, lang, limit=200)

                for title in titles:
                    if total >= max_articles:
                        break
                    key = f"{lang}:{title}"
                    if key in self._processed:
                        continue

                    article = self._fetch_article(title, lang)
                    if article is None or len(article["text"]) < 200:
                        continue

                    doc = CrawledDocument(
                        id=f"wiki_{lang}_{hashlib.md5(key.encode()).hexdigest()[:10]}",
                        source="wikipedia",
                        title=article["title"],
                        url=article["url"],
                        text=article["text"],
                        lang=lang,
                        crawl_date=datetime.now().isoformat(),
                    )

                    with open(output_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(doc.to_dict(), ensure_ascii=False) + "\n")

                    self._processed.add(key)
                    total += 1

                    if total % 100 == 0:
                        logger.info("  Progress: %d/%d articles", total, max_articles)
                        self._save_state()

            if total >= max_articles:
                break

        self._save_state()
        logger.info("Wikipedia crawl complete: %d articles", total)
        return total


class ArxivGeneralCrawler:
    """Crawl arXiv papers via the API with broader queries."""

    def __init__(self, output_dir: str = f"{OUTPUT_BASE}/arxiv") -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._processed: set[str] = set()
        self._state_file = self._output_dir / "_state.json"
        self._load_state()
        self._last_request: float = 0.0

    def _load_state(self) -> None:
        if self._state_file.exists():
            try:
                with open(self._state_file, "r", encoding="utf-8") as f:
                    state = json.load(f)
                self._processed = set(state.get("processed", []))
                logger.info("arXiv: resuming with %d papers processed", len(self._processed))
            except Exception:
                pass

    def _save_state(self) -> None:
        state = {
            "processed": sorted(self._processed),
            "count": len(self._processed),
            "last_updated": datetime.now().isoformat(),
        }
        with open(self._state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def _rate_limit(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_request
        if elapsed < 3.0:
            time.sleep(3.0 - elapsed)
        self._last_request = time.monotonic()

    def _search(self, query: str, max_results: int = 200) -> list[dict]:
        """Search arXiv via the Atom API (no extra deps needed)."""
        self._rate_limit()
        base = "http://export.arxiv.org/api/query"
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        try:
            resp = requests.get(base, params=params, timeout=60)
            resp.raise_for_status()
        except Exception as exc:
            logger.warning("arXiv search failed for '%s': %s", query, exc)
            return []

        ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
        root = ET.fromstring(resp.text)
        papers = []

        for entry in root.findall("atom:entry", ns):
            title_el = entry.find("atom:title", ns)
            summary_el = entry.find("atom:summary", ns)
            id_el = entry.find("atom:id", ns)
            published_el = entry.find("atom:published", ns)

            if title_el is None or summary_el is None:
                continue

            arxiv_id = id_el.text.split("/")[-1] if id_el is not None else ""
            if arxiv_id in self._processed:
                continue

            title = re.sub(r'\s+', ' ', title_el.text.strip())
            abstract = re.sub(r'\s+', ' ', summary_el.text.strip())
            published = published_el.text[:10] if published_el is not None else ""

            # Build full text from title + abstract (we don't download PDFs for speed)
            full_text = f"{title}\n\nAbstract\n{abstract}"

            if len(full_text) < 200:
                continue

            doc = CrawledDocument(
                id=f"arxiv_{arxiv_id.replace('/', '_')}",
                source="arxiv",
                title=title,
                url=f"https://arxiv.org/abs/{arxiv_id}",
                text=full_text,
                lang="en",
                crawl_date=datetime.now().isoformat(),
            )
            papers.append(doc.to_dict())
            self._processed.add(arxiv_id)

        return papers

    def crawl(self, max_papers: int = 5000) -> int:
        """Crawl arXiv papers from broad queries."""
        logger.info("Starting arXiv general crawl (target: %d papers)", max_papers)
        output_file = self._output_dir / "arxiv_general.jsonl"
        total = len(self._processed)

        for query in ARXIV_QUERIES:
            if total >= max_papers:
                break

            remaining = max_papers - total
            logger.info("  Query: '%s' (remaining: %d)", query, remaining)
            papers = self._search(query, max_results=min(200, remaining))

            with open(output_file, "a", encoding="utf-8") as f:
                for paper in papers:
                    f.write(json.dumps(paper, ensure_ascii=False) + "\n")

            total += len(papers)
            logger.info("  Found %d papers (total: %d)", len(papers), total)
            self._save_state()

        logger.info("arXiv crawl complete: %d papers", total)
        return total


class PubMedGeneralCrawler:
    """Crawl PubMed abstracts via E-utilities API."""

    def __init__(self, output_dir: str = f"{OUTPUT_BASE}/pubmed") -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._processed: set[str] = set()
        self._state_file = self._output_dir / "_state.json"
        self._load_state()
        self._last_request: float = 0.0

    def _load_state(self) -> None:
        if self._state_file.exists():
            try:
                with open(self._state_file, "r", encoding="utf-8") as f:
                    state = json.load(f)
                self._processed = set(state.get("processed", []))
                logger.info("PubMed: resuming with %d articles processed", len(self._processed))
            except Exception:
                pass

    def _save_state(self) -> None:
        state = {
            "processed": sorted(list(self._processed)[:50000]),
            "count": len(self._processed),
            "last_updated": datetime.now().isoformat(),
        }
        with open(self._state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def _rate_limit(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_request
        if elapsed < 0.4:  # PubMed allows up to 3/sec without API key
            time.sleep(0.4 - elapsed)
        self._last_request = time.monotonic()

    def _search_ids(self, query: str, max_results: int = 5000) -> list[str]:
        """Search PubMed for PMIDs matching a query."""
        self._rate_limit()
        base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "sort": "relevance",
        }
        try:
            resp = requests.get(base, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            return data.get("esearchresult", {}).get("idlist", [])
        except Exception as exc:
            logger.warning("PubMed search failed for '%s': %s", query, exc)
            return []

    def _fetch_abstracts(self, pmids: list[str]) -> list[dict]:
        """Fetch abstracts for a batch of PMIDs."""
        self._rate_limit()
        base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "rettype": "abstract",
            "retmode": "xml",
        }
        try:
            resp = requests.get(base, params=params, timeout=60)
            resp.raise_for_status()
        except Exception as exc:
            logger.warning("PubMed fetch failed: %s", exc)
            return []

        docs = []
        try:
            root = ET.fromstring(resp.text)
            for article in root.findall(".//PubmedArticle"):
                pmid_el = article.find(".//PMID")
                if pmid_el is None:
                    continue
                pmid = pmid_el.text

                if pmid in self._processed:
                    continue

                # Title
                title = ""
                title_el = article.find(".//ArticleTitle")
                if title_el is not None:
                    title = "".join(title_el.itertext()).strip()

                # Abstract
                abstract_parts = []
                for abs_el in article.findall(".//AbstractText"):
                    label = abs_el.get("Label", "")
                    text = "".join(abs_el.itertext()).strip()
                    if label:
                        abstract_parts.append(f"{label}: {text}")
                    else:
                        abstract_parts.append(text)
                abstract = "\n".join(abstract_parts)

                # Journal
                journal = ""
                journal_el = article.find(".//Journal/Title")
                if journal_el is not None:
                    journal = journal_el.text or ""

                # Year
                year = ""
                year_el = article.find(".//PubDate/Year")
                if year_el is not None:
                    year = year_el.text or ""

                full_text = f"{title}\n\n"
                if journal:
                    full_text += f"Journal: {journal}"
                    if year:
                        full_text += f" ({year})"
                    full_text += "\n\n"
                full_text += abstract

                if len(full_text) < 200:
                    continue

                doc = CrawledDocument(
                    id=f"pubmed_PMID{pmid}",
                    source="pubmed",
                    title=title,
                    url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    text=full_text,
                    lang="en",
                    crawl_date=datetime.now().isoformat(),
                )
                docs.append(doc.to_dict())
                self._processed.add(pmid)

        except ET.ParseError as exc:
            logger.warning("PubMed XML parse error: %s", exc)

        return docs

    def crawl(self, max_articles: int = 10000) -> int:
        """Crawl PubMed abstracts from broad medical queries."""
        logger.info("Starting PubMed general crawl (target: %d articles)", max_articles)
        output_file = self._output_dir / "pubmed_general.jsonl"
        total = len(self._processed)

        for query in PUBMED_QUERIES:
            if total >= max_articles:
                break

            remaining = max_articles - total
            logger.info("  Query: '%s' (remaining: %d)", query, remaining)
            pmids = self._search_ids(query, max_results=min(5000, remaining))
            new_pmids = [p for p in pmids if p not in self._processed]

            # Fetch in batches of 100
            batch_size = 100
            for i in range(0, len(new_pmids), batch_size):
                batch = new_pmids[i:i+batch_size]
                docs = self._fetch_abstracts(batch)

                with open(output_file, "a", encoding="utf-8") as f:
                    for doc in docs:
                        f.write(json.dumps(doc, ensure_ascii=False) + "\n")

                total += len(docs)
                if total % 500 == 0:
                    logger.info("  Progress: %d articles", total)
                    self._save_state()

            logger.info("  Got %d articles from '%s' (total: %d)", len(docs) if docs else 0, query, total)
            self._save_state()

        logger.info("PubMed crawl complete: %d articles", total)
        return total


class GutenbergCrawler:
    """Crawl public domain books from Project Gutenberg."""

    def __init__(self, output_dir: str = f"{OUTPUT_BASE}/gutenberg") -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._processed: set[str] = set()
        self._state_file = self._output_dir / "_state.json"
        self._load_state()
        self._last_request: float = 0.0

    def _load_state(self) -> None:
        if self._state_file.exists():
            try:
                with open(self._state_file, "r", encoding="utf-8") as f:
                    state = json.load(f)
                self._processed = set(state.get("processed", []))
                logger.info("Gutenberg: resuming with %d books processed", len(self._processed))
            except Exception:
                pass

    def _save_state(self) -> None:
        state = {
            "processed": sorted(self._processed),
            "count": len(self._processed),
            "last_updated": datetime.now().isoformat(),
        }
        with open(self._state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def _rate_limit(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_request
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)
        self._last_request = time.monotonic()

    def _fetch_book(self, book_id: str) -> Optional[str]:
        """Fetch plain text of a Gutenberg book."""
        self._rate_limit()
        url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code != 200:
                # Try alternate URL
                url2 = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
                resp = requests.get(url2, timeout=30)
                if resp.status_code != 200:
                    return None
            return resp.text
        except Exception:
            return None

    def _clean_gutenberg_text(self, text: str) -> str:
        """Remove Gutenberg headers/footers."""
        # Remove header
        start_markers = ["*** START OF", "***START OF"]
        for marker in start_markers:
            idx = text.find(marker)
            if idx >= 0:
                end = text.find("\n", idx)
                if end >= 0:
                    text = text[end+1:]
                break

        # Remove footer
        end_markers = ["*** END OF", "***END OF", "End of the Project Gutenberg"]
        for marker in end_markers:
            idx = text.find(marker)
            if idx >= 0:
                text = text[:idx]
                break

        return text.strip()

    def crawl(self, book_ids: list[int], max_books: int = 200) -> int:
        """Crawl specific books from Project Gutenberg."""
        logger.info("Starting Gutenberg crawl (target: %d books)", max_books)
        output_file = self._output_dir / "gutenberg_books.jsonl"
        total = 0

        # Popular classic books across genres
        default_ids = [
            11,  # Alice in Wonderland
            1342,  # Pride and Prejudice
            84,  # Frankenstein
            1080,  # Modest Proposal
            2701,  # Moby Dick
            98,  # Tale of Two Cities
            84,  # Frankenstein
            1661,  # Sherlock Holmes
            345,  # Dracula
            16,  # Peter Pan
            174,  # Dorian Gray
            2591,  # Grimms Fairy Tales
            1232,  # Prince by Machiavelli
            46,  # Christmas Carol
            74,  # Tom Sawyer
            76,  # Huck Finn
            1952,  # Charlotte's Web
            1260,  # Jane Eyre
            1400,  # Great Expectations
            5200,  # Metamorphosis
            16328,  # Beowulf
            1497,  # Republic (Plato)
            25344,  # Art of War (Sun Tzu)
            768,  # Wuthering Heights
            786,  # Blue Hotel
            4300,  # Ulysses
            105,  # Persuasion
            121,  # Northanger Abbey
            158,  # Sense and Sensibility
            2814,  # World Set Free (H.G. Wells)
            36,  # War of the Worlds
            8292,  # Varied Types (Chesterton)
            4671,  # Count of Monte Cristo (French)
            244,  # Five Weeks in a Balloon (Verne)
            164,  # Heart of Darkness (Polish-English)
            2554,  # Crime and Punishment
            2511,  # Emerson essays
            737,  # Time Machine
            1250,  # Siddhartha (German)
            8800,  # Edgar Allan Poe complete
            1064,  # Tarzan
            55,  # Three Men in a Boat
            203,  # Uncle Tom's Cabin
            815,  # Fenelon's Treatise
            3600,  # Othello
            1112,  # Little Women
            521,  # Robinson Crusoe
            6130,  # Siddhartha (English)
            14287,  # Plutarch's Lives
        ]

        ids_to_fetch = book_ids if book_ids else default_ids

        for book_id in ids_to_fetch:
            if total >= max_books:
                break
            bid = str(book_id)
            if bid in self._processed:
                continue

            logger.info("  Fetching book %s...", bid)
            raw = self._fetch_book(bid)
            if raw is None or len(raw) < 1000:
                continue

            text = self._clean_gutenberg_text(raw)
            if len(text) < 500:
                continue

            # Extract title from first line
            title = text.split("\n")[0].strip()[:200]

            doc = CrawledDocument(
                id=f"gutenberg_{bid}",
                source="gutenberg",
                title=title,
                url=f"https://www.gutenberg.org/ebooks/{bid}",
                text=text,
                lang="en",
                crawl_date=datetime.now().isoformat(),
            )

            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(doc.to_dict(), ensure_ascii=False) + "\n")

            self._processed.add(bid)
            total += 1

            if total % 10 == 0:
                logger.info("  Progress: %d books", total)
                self._save_state()

        self._save_state()
        logger.info("Gutenberg crawl complete: %d books", total)
        return total


def run_all_crawlers(
    wiki_max: int = 5000,
    arxiv_max: int = 5000,
    pubmed_max: int = 10000,
    gutenberg_max: int = 100,
) -> dict[str, int]:
    """Run all crawlers and return counts."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    results = {}

    logger.info("=" * 60)
    logger.info("Starting general-purpose data crawl")
    logger.info("=" * 60)

    # 1. Wikipedia
    logger.info("\n--- Wikipedia ---")
    wiki_crawler = WikipediaGeneralCrawler()
    results["wikipedia"] = wiki_crawler.crawl(max_articles=wiki_max)

    # 2. arXiv
    logger.info("\n--- arXiv ---")
    arxiv_crawler = ArxivGeneralCrawler()
    results["arxiv"] = arxiv_crawler.crawl(max_papers=arxiv_max)

    # 3. PubMed
    logger.info("\n--- PubMed ---")
    pubmed_crawler = PubMedGeneralCrawler()
    results["pubmed"] = pubmed_crawler.crawl(max_articles=pubmed_max)

    # 4. Gutenberg
    logger.info("\n--- Gutenberg ---")
    gut_crawler = GutenbergCrawler()
    results["gutenberg"] = gut_crawler.crawl(book_ids=[], max_books=gutenberg_max)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("CRAWL SUMMARY")
    logger.info("=" * 60)
    for source, count in results.items():
        logger.info("  %s: %d documents", source, count)
    logger.info("  TOTAL: %d documents", sum(results.values()))

    return results


if __name__ == "__main__":
    results = run_all_crawlers()
    print(f"\nResults: {results}")
