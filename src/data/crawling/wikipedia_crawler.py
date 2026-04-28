"""
Wikipedia medical article crawler for sleep medicine domain content.

Crawls sleep-related Wikipedia categories and articles from both English
and Chinese Wikipedia, extracting structured article content including
sections, links, and references. Uses the wikipedia-api package.

Provides high-quality, well-structured medical content for training data,
particularly useful for Chinese language sleep medicine terminology.
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

# Sleep-related Wikipedia categories for English Wikipedia
EN_SLEEP_CATEGORIES = [
    "Sleep disorders",
    "Sleep medicine",
    "Neurology",
    "Pulmonology",
    "Circadian rhythms",
    "Sleep physiology",
    "Chronobiology",
]

# Sleep-related Wikipedia categories for Chinese Wikipedia
ZH_SLEEP_CATEGORIES = [
    "睡眠障碍",
    "睡眠医学",
    "神经学",
    "呼吸系统疾病",
    "昼夜节律",
    "睡眠生理",
    "时间生物学",
]

# Individual article titles known to be sleep-related (fallback)
EN_SLEEP_ARTICLES = [
    "Sleep",
    "Insomnia",
    "Obstructive sleep apnea",
    "Narcolepsy",
    "Restless legs syndrome",
    "Circadian rhythm sleep disorder",
    "Rapid eye movement sleep",
    "Slow-wave sleep",
    "Sleep deprivation",
    "Polysomnography",
    "Continuous positive airway pressure",
    "Melatonin",
    "Hypersomnia",
    "Parasomnia",
    "Sleep apnea",
    "Shift work sleep disorder",
    "Delayed sleep phase disorder",
    "Advanced sleep phase disorder",
    "Non-rapid eye movement sleep",
    "Sleep spindle",
    "K-complex",
    "Sleep architecture",
    "Sleep hygiene",
    "Bruxism",
    "Sleepwalking",
    "Night terror",
    "REM sleep behavior disorder",
    "Idiopathic hypersomnia",
    "Sleep and health",
    "Jet lag",
    "Actigraphy",
    "Sleep study",
]

ZH_SLEEP_ARTICLES = [
    "睡眠",
    "失眠",
    "阻塞性睡眠呼吸暂停",
    "发作性睡病",
    "不宁腿综合征",
    "昼夜节律睡眠障碍",
    "快速眼动睡眠",
    "慢波睡眠",
    "睡眠剥夺",
    "多导睡眠图",
    "持续气道正压通气",
    "褪黑素",
    "嗜睡症",
    "异态睡眠",
    "睡眠呼吸暂停",
    "倒班工作睡眠障碍",
    "睡眠相位延迟障碍",
    "睡眠相位提前障碍",
    "非快速眼动睡眠",
    "睡眠纺锤波",
    "K复合波",
    "睡眠结构",
    "睡眠卫生",
    "磨牙症",
    "梦游",
    "夜惊",
    "REM睡眠行为障碍",
    "特发性嗜睡症",
    "睡眠与健康",
    "时差反应",
    "体动记录仪",
    "睡眠检查",
]

REQUEST_INTERVAL = 1.0  # Seconds between Wikipedia API requests
MAX_RETRY_ATTEMPTS = 3
RETRY_BACKOFF_BASE = 2.0


@dataclass(frozen=True)
class WikipediaArticle:
    """Structured representation of a Wikipedia article."""

    title: str
    language: str
    url: str
    summary: str
    sections: list[dict[str, str]]
    full_text: str
    links: list[str]
    categories: list[str]
    references: list[str]
    crawl_date: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class WikipediaCrawler:
    """Crawler for sleep-related Wikipedia articles.

    Crawls Wikipedia categories and individual articles related to sleep
    medicine from both English and Chinese Wikipedia. Extracts structured
    content including sections, links, references, and category membership.

    Attributes:
        output_dir: Directory for JSONL output files.
    """

    def __init__(self, output_dir: str = "data/raw/wikipedia") -> None:
        self._output_dir = Path(output_dir)
        self._state_file = self._output_dir / "_wikipedia_state.json"

        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Track processed articles for deduplication and resumability
        self._processed_titles: dict[str, set[str]] = {"en": set(), "zh": set()}
        self._load_state()

        self._last_request_time: float = 0.0

        # Lazy-loaded Wikipedia API instances
        self._wiki_en: Any = None
        self._wiki_zh: Any = None

    def _load_state(self) -> None:
        """Load previously processed article titles from state file."""
        if not self._state_file.exists():
            return

        try:
            with open(self._state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
            self._processed_titles["en"] = set(state.get("processed_en", []))
            self._processed_titles["zh"] = set(state.get("processed_zh", []))
            logger.info(
                "Resuming: %d EN and %d ZH articles already processed",
                len(self._processed_titles["en"]),
                len(self._processed_titles["zh"]),
            )
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Failed to load state file, starting fresh: %s", exc)

    def _save_state(self) -> None:
        """Persist processed article titles to state file."""
        state = {
            "processed_en": sorted(self._processed_titles["en"]),
            "processed_zh": sorted(self._processed_titles["zh"]),
            "en_count": len(self._processed_titles["en"]),
            "zh_count": len(self._processed_titles["zh"]),
            "last_updated": datetime.now().isoformat(),
        }
        with open(self._state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def _rate_limit(self) -> None:
        """Enforce rate limiting between Wikipedia API requests."""
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < REQUEST_INTERVAL:
            time.sleep(REQUEST_INTERVAL - elapsed)
        self._last_request_time = time.monotonic()

    def _get_wiki(self, language: str) -> Any:
        """Get or create a Wikipedia API instance for the given language.

        Args:
            language: Wikipedia language code ('en' or 'zh').

        Returns:
            wikipediaapi.Wikipedia instance.

        Raises:
            ImportError: If wikipedia-api is not installed.
        """
        try:
            import wikipediaapi
        except ImportError as exc:
            raise ImportError(
                "wikipedia-api is required. Install with: pip install wikipedia-api"
            ) from exc

        if language == "en":
            if self._wiki_en is None:
                self._wiki_en = wikipediaapi.Wikipedia(
                    user_agent="DeepSleepLLM/1.0 (sleep medicine research)",
                    language="en",
                )
            return self._wiki_en
        elif language == "zh":
            if self._wiki_zh is None:
                self._wiki_zh = wikipediaapi.Wikipedia(
                    user_agent="DeepSleepLLM/1.0 (sleep medicine research)",
                    language="zh",
                )
            return self._wiki_zh
        else:
            raise ValueError(f"Unsupported language: {language}")

    def get_sleep_categories(self) -> list[str]:
        """Return the configured list of sleep-related Wikipedia categories.

        Returns a flat list of all categories for both English and Chinese.

        Returns:
            List of category name strings.
        """
        return EN_SLEEP_CATEGORIES + ZH_SLEEP_CATEGORIES

    def get_article(self, title: str, language: str = "en") -> Optional[WikipediaArticle]:
        """Fetch a single Wikipedia article by title.

        Extracts the article summary, all sections with their text,
        links, categories, and references.

        Args:
            title: Exact title of the Wikipedia article.
            language: Wikipedia language code ('en' or 'zh').

        Returns:
            WikipediaArticle dataclass instance, or None if the article
            does not exist or cannot be fetched.
        """
        if title in self._processed_titles[language]:
            logger.debug("Skipping already processed: %s (%s)", title, language)
            return None

        self._rate_limit()

        try:
            wiki = self._get_wiki(language)
            page = wiki.page(title)

            if not page.exists():
                logger.debug("Article not found: %s (%s)", title, language)
                return None

            # Extract sections recursively
            sections = self._extract_sections(page.sections)

            # Extract links
            links = sorted(set(page.links.keys()))

            # Extract categories
            categories = sorted(set(page.categories.keys()))

            # Build full text from sections
            full_text = page.summary + "\n\n"
            for section in sections:
                if section["text"].strip():
                    full_text += f"\n{section['title']}\n{section['text']}\n"

            # Extract references from the full text
            references = self._extract_references(full_text)

            article = WikipediaArticle(
                title=title,
                language=language,
                url=page.fullurl,
                summary=page.summary,
                sections=sections,
                full_text=full_text,
                links=links,
                categories=categories,
                references=references,
                crawl_date=datetime.now().isoformat(),
            )

            self._processed_titles[language].add(title)
            return article

        except Exception as exc:
            logger.error(
                "Failed to fetch article '%s' (%s): %s", title, language, exc
            )
            return None

    def _extract_sections(
        self, section_list: Any, depth: int = 0
    ) -> list[dict[str, str]]:
        """Recursively extract sections from a Wikipedia page.

        Args:
            section_list: wikipediaapi.Section list or single Section.
            depth: Current recursion depth for indentation tracking.

        Returns:
            List of section dictionaries with 'title' and 'text' keys.
        """
        sections: list[dict[str, str]] = []

        if section_list is None:
            return sections

        # Handle both list and single section
        if not hasattr(section_list, "__iter__"):
            section_list = [section_list]

        for section in section_list:
            title_text = section.title.strip()
            section_text = section.text.strip() if section.text else ""

            if title_text or section_text:
                sections.append({
                    "title": title_text,
                    "text": section_text,
                })

            # Recursively extract subsections
            if section.sections:
                subsections = self._extract_sections(section.sections, depth + 1)
                sections.extend(subsections)

        return sections

    def _extract_references(self, text: str) -> list[str]:
        """Extract reference-like patterns from Wikipedia article text.

        Looks for common citation patterns in Wikipedia markup.

        Args:
            text: Full article text content.

        Returns:
            List of extracted reference strings.
        """
        references: list[str] = []

        # Match ISBN patterns
        isbn_matches = re.findall(r"ISBN\s+([\d-]+(?:\s+[Xx])?)", text)
        references.extend(f"ISBN: {m}" for m in isbn_matches)

        # Match DOI patterns
        doi_matches = re.findall(r"10\.\d{4,}/[^\s\]<>\"']+", text)
        references.extend(f"DOI: {m}" for m in doi_matches)

        # Match PMID patterns
        pmid_matches = re.findall(r"PMID\s+(\d+)", text)
        references.extend(f"PMID: {m}" for m in pmid_matches)

        # Match URL patterns
        url_matches = re.findall(r"https?://[^\s\]<>\"'()]+", text)
        references.extend(url_matches)

        return list(set(references))

    def crawl_category(
        self, category: str, max_depth: int = 2, language: str = "en"
    ) -> list[dict[str, Any]]:
        """Crawl all articles within a Wikipedia category tree.

        Traverses the category hierarchy up to the specified depth,
        collecting articles from the target category and all subcategories.

        Args:
            category: Name of the Wikipedia category to crawl.
            max_depth: Maximum recursion depth for subcategories.
            language: Wikipedia language code ('en' or 'zh').

        Returns:
            List of article dictionaries from the category.
        """
        logger.info(
            "Crawling category: '%s' (%s, max_depth=%d)",
            category,
            language,
            max_depth,
        )

        articles: list[dict[str, Any]] = []
        visited_categories: set[str] = set()
        category_key = f"{language}:{category}"

        if category_key in visited_categories:
            return articles

        visited_categories.add(category_key)

        self._rate_limit()

        try:
            wiki = self._get_wiki(language)
            cat_page = wiki.page(f"Category:{category}")

            if not cat_page.exists():
                logger.warning(
                    "Category not found: '%s' (%s)", category, language
                )
                return articles

            # Collect category members (articles)
            self._collect_category_members(
                cat_page, articles, visited_categories, max_depth, language
            )

        except Exception as exc:
            logger.error(
                "Failed to crawl category '%s' (%s): %s", category, language, exc
            )

        logger.info(
            "Category '%s' (%s): %d articles collected",
            category,
            language,
            len(articles),
        )
        return articles

    def _collect_category_members(
        self,
        category_page: Any,
        articles: list[dict[str, Any]],
        visited_categories: set[str],
        remaining_depth: int,
        language: str,
    ) -> None:
        """Recursively collect articles from a category and its subcategories.

        Args:
            category_page: Wikipedia category page object.
            articles: List to append collected article dicts to.
            visited_categories: Set of already-visited category keys.
            remaining_depth: How many more levels of subcategories to traverse.
            language: Wikipedia language code.
        """
        # Process category members (articles)
        for title, page in category_page.categorymembers.items():
            if not title:
                continue

            # Skip category pages (handled recursively below)
            if title.startswith("Category:"):
                if remaining_depth > 0:
                    sub_cat = title.replace("Category:", "")
                    sub_key = f"{language}:{sub_cat}"
                    if sub_key not in visited_categories:
                        visited_categories.add(sub_key)
                        self._rate_limit()
                        try:
                            wiki = self._get_wiki(language)
                            sub_page = wiki.page(title)
                            if sub_page.exists():
                                self._collect_category_members(
                                    sub_page,
                                    articles,
                                    visited_categories,
                                    remaining_depth - 1,
                                    language,
                                )
                        except Exception as exc:
                            logger.error(
                                "Failed to process subcategory '%s': %s",
                                sub_cat,
                                exc,
                            )
                continue

            # Fetch the article
            article = self.get_article(title, language)
            if article is not None:
                articles.append(article.to_dict())

    def _save_articles(
        self,
        articles: list[dict[str, Any]],
        filename: str,
    ) -> int:
        """Append articles to a JSONL output file.

        Args:
            articles: List of article dictionaries to save.
            filename: Name of the JSONL output file.

        Returns:
            Number of new records written.
        """
        if not articles:
            return 0

        output_path = self._output_dir / filename
        written = 0

        with open(output_path, "a", encoding="utf-8") as f:
            for article in articles:
                key = f"{article['language']}:{article['title']}"
                title_set = self._processed_titles[article["language"]]
                if article["title"] in title_set:
                    continue

                f.write(json.dumps(article, ensure_ascii=False) + "\n")
                title_set.add(article["title"])
                written += 1

        logger.info("Wrote %d articles to %s", written, output_path)
        return written

    def crawl_sleep_wikipedia(self) -> list[dict[str, Any]]:
        """Main entry point for crawling sleep-related Wikipedia articles.

        Crawls both English and Chinese Wikipedia, collecting articles from
        sleep-related categories and known individual articles. Results are
        saved as JSONL files.

        Returns:
            List of all collected article dictionaries.
        """
        logger.info("Starting Wikipedia sleep article crawl")
        start_time = time.time()
        all_articles: list[dict[str, Any]] = []

        # Crawl English Wikipedia categories
        logger.info("=== Crawling English Wikipedia ===")
        for category in EN_SLEEP_CATEGORIES:
            cat_articles = self.crawl_category(category, max_depth=2, language="en")
            self._save_articles(cat_articles, "wikipedia_en_sleep.jsonl")
            all_articles.extend(cat_articles)
            self._save_state()

        # Crawl known English articles (some may not be in categories)
        logger.info("Crawling individual English articles")
        for title in EN_SLEEP_ARTICLES:
            article = self.get_article(title, language="en")
            if article is not None:
                article_dict = article.to_dict()
                all_articles.append(article_dict)
                self._save_articles([article_dict], "wikipedia_en_sleep.jsonl")
        self._save_state()

        # Crawl Chinese Wikipedia categories
        logger.info("=== Crawling Chinese Wikipedia ===")
        for category in ZH_SLEEP_CATEGORIES:
            cat_articles = self.crawl_category(category, max_depth=2, language="zh")
            self._save_articles(cat_articles, "wikipedia_zh_sleep.jsonl")
            all_articles.extend(cat_articles)
            self._save_state()

        # Crawl known Chinese articles
        logger.info("Crawling individual Chinese articles")
        for title in ZH_SLEEP_ARTICLES:
            article = self.get_article(title, language="zh")
            if article is not None:
                article_dict = article.to_dict()
                all_articles.append(article_dict)
                self._save_articles([article_dict], "wikipedia_zh_sleep.jsonl")
        self._save_state()

        # Save combined output
        self._save_articles(all_articles, "wikipedia_sleep_all.jsonl")

        elapsed = time.time() - start_time
        en_count = sum(1 for a in all_articles if a["language"] == "en")
        zh_count = sum(1 for a in all_articles if a["language"] == "zh")
        logger.info(
            "Wikipedia crawl complete. %d articles (%d EN, %d ZH) in %.1f seconds",
            len(all_articles),
            en_count,
            zh_count,
            elapsed,
        )

        return all_articles
