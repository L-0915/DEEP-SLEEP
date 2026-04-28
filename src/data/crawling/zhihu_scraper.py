"""
Zhihu sleep topic scraper for collecting Chinese sleep-related Q&A content.

DISCLAIMER: Users of this module must independently review and comply with
Zhihu's Terms of Service (https://www.zhihu.com/terms) and applicable
laws regarding web scraping. This tool is provided for educational and
research purposes only. The authors assume no liability for misuse. Users
should:
  - Review Zhihu's robots.txt (https://www.zhihu.com/robots.txt)
  - Respect rate limits and access restrictions
  - Not use automated access for commercial purposes without authorization
  - Comply with Zhihu's API terms if using official APIs
  - Consider using Zhihu's official API where available

Crawls Zhihu topics related to sleep, collecting questions, answers,
comments, and engagement metrics for Chinese sleep medicine training data.
"""

import json
import logging
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote, urlencode

import requests

logger = logging.getLogger(__name__)

# DISCLAIMER text included in module docstring and logged at startup
STARTUP_DISCLAIMER = (
    "DISCLAIMER: Zhihu scraping must comply with Zhihu's Terms of Service. "
    "Review https://www.zhihu.com/terms before use. Respect robots.txt "
    "and rate limits. This tool is for research purposes only."
)

# Zhihu API endpoints
ZHIHU_BASE_URL = "https://www.zhihu.com"
ZHIHU_API_BASE = "https://www.zhihu.com/api/v4"

# Sleep-related topic IDs on Zhihu (topic IDs are numeric identifiers)
# "睡眠" (Sleep) topic - the primary topic ID may need to be verified
SLEEP_TOPIC_IDS = ["19551181"]  # 睡眠 topic ID

# Search queries for sleep-related content
SLEEP_SEARCH_KEYWORDS = [
    "睡眠",
    "失眠",
    "睡眠障碍",
    "睡眠质量",
    "打呼噜",
    "呼吸暂停",
    "嗜睡",
    "梦游",
    "昼夜节律",
    "褪黑素",
    "入睡困难",
    "早醒",
    "多梦",
    "睡眠不足",
    "熬夜",
    "午休",
    "安眠药",
    "白噪音",
    "睡眠呼吸暂停",
    "不宁腿",
    "发作性睡病",
]

# Rate limiting configuration
ZHIHU_RATE_LIMIT = 3.0  # seconds between requests (be conservative)
DEFAULT_TIMEOUT = 30
MAX_RETRY_ATTEMPTS = 3
RETRY_BACKOFF_BASE = 2.0

# Request headers
DEFAULT_HEADERS = {
    "User-Agent": (
        "DeepSleepLLM/1.0 (Educational sleep medicine research; "
        "+https://github.com/deepsleep)"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Referer": "https://www.zhihu.com/",
}


@dataclass(frozen=True)
class ZhihuQuestion:
    """Structured representation of a Zhihu question."""

    question_id: str
    title: str
    detail: str
    author_name: Optional[str]
    created_time: Optional[str]
    answer_count: int
    follower_count: int
    visit_count: int
    topics: list[str]
    excerpt: str
    url: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ZhihuAnswer:
    """Structured representation of a Zhihu answer."""

    answer_id: str
    question_id: str
    question_title: str
    content: str
    author_name: Optional[str]
    upvotes: int
    comment_count: int
    created_time: Optional[str]
    updated_time: Optional[str]
    excerpt: str
    url: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ZhihuScraper:
    """Scraper for sleep-related content from Zhihu.

    Searches Zhihu for sleep-related questions and answers, collecting
    Chinese language Q&A content for training data.

    DISCLAIMER: Users must comply with Zhihu's Terms of Service. This tool
    is for educational and research purposes only.

    Attributes:
        output_dir: Directory for JSONL output files.
    """

    def __init__(self, output_dir: str = "data/raw/zhihu") -> None:
        # Log disclaimer at initialization
        logger.warning(STARTUP_DISCLAIMER)

        self._output_dir = Path(output_dir)
        self._state_file = self._output_dir / "_zhihu_state.json"

        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Track processed IDs for deduplication and resumability
        self._processed_question_ids: set[str] = set()
        self._processed_answer_ids: set[str] = set()
        self._load_state()

        self._last_request_time: float = 0.0

        # HTTP session
        self._session = requests.Session()
        self._session.headers.update(DEFAULT_HEADERS)
        self._session.max_redirects = 5

    def _load_state(self) -> None:
        """Load previously processed IDs from the state file."""
        if not self._state_file.exists():
            return

        try:
            with open(self._state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
            self._processed_question_ids = set(state.get("processed_questions", []))
            self._processed_answer_ids = set(state.get("processed_answers", []))
            logger.info(
                "Resuming: %d questions, %d answers already processed",
                len(self._processed_question_ids),
                len(self._processed_answer_ids),
            )
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Failed to load state file, starting fresh: %s", exc)

    def _save_state(self) -> None:
        """Persist processed IDs to the state file."""
        state = {
            "processed_questions": sorted(list(self._processed_question_ids)[:100000]),
            "processed_answers": sorted(list(self._processed_answer_ids)[:100000]),
            "question_count": len(self._processed_question_ids),
            "answer_count": len(self._processed_answer_ids),
            "last_updated": datetime.now().isoformat(),
        }
        with open(self._state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def _rate_limit_wait(self) -> None:
        """Wait to satisfy Zhihu's rate limit."""
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < ZHIHU_RATE_LIMIT:
            time.sleep(ZHIHU_RATE_LIMIT - elapsed)
        self._last_request_time = time.monotonic()

    def _api_request(
        self, endpoint: str, params: Optional[dict[str, Any]] = None
    ) -> Optional[dict[str, Any]]:
        """Make a rate-limited API request to Zhihu.

        Args:
            endpoint: API endpoint path (appended to ZHIHU_API_BASE).
            params: Optional query parameters.

        Returns:
            JSON response as dictionary, or None on failure.
        """
        url = f"{ZHIHU_API_BASE}{endpoint}"

        for attempt in range(MAX_RETRY_ATTEMPTS):
            self._rate_limit_wait()
            try:
                response = self._session.get(
                    url,
                    params=params,
                    timeout=DEFAULT_TIMEOUT,
                )
                response.raise_for_status()
                return response.json()

            except requests.exceptions.HTTPError as exc:
                if exc.response is not None:
                    status = exc.response.status_code
                    if status == 429:
                        backoff = RETRY_BACKOFF_BASE ** (attempt + 1)
                        logger.warning(
                            "Rate limited by Zhihu (429), backing off %.1fs",
                            backoff,
                        )
                        time.sleep(backoff)
                    elif status in (401, 403):
                        logger.warning(
                            "Authentication/authorization error (%d) for %s. "
                            "Zhihu may require login for this endpoint.",
                            status,
                            endpoint,
                        )
                        return None
                    elif status == 404:
                        logger.debug("Not found: %s", endpoint)
                        return None
                    else:
                        logger.warning(
                            "HTTP %d for %s (attempt %d): %s",
                            status,
                            endpoint,
                            attempt + 1,
                            exc,
                        )
                else:
                    logger.warning("HTTP error for %s: %s", endpoint, exc)

            except requests.exceptions.RequestException as exc:
                logger.warning(
                    "Request failed for %s (attempt %d): %s",
                    endpoint,
                    attempt + 1,
                    exc,
                )

            if attempt < MAX_RETRY_ATTEMPTS - 1:
                time.sleep(RETRY_BACKOFF_BASE ** (attempt + 1))

        logger.error("Failed to fetch %s after %d attempts", endpoint, MAX_RETRY_ATTEMPTS)
        return None

    def _html_request(self, url: str) -> Optional[str]:
        """Fetch HTML content from a Zhihu page.

        Args:
            url: Full Zhihu URL to fetch.

        Returns:
            HTML content as string, or None on failure.
        """
        for attempt in range(MAX_RETRY_ATTEMPTS):
            self._rate_limit_wait()
            try:
                response = self._session.get(url, timeout=DEFAULT_TIMEOUT)
                response.raise_for_status()
                return response.text

            except requests.exceptions.RequestException as exc:
                logger.warning(
                    "Failed to fetch %s (attempt %d): %s",
                    url,
                    attempt + 1,
                    exc,
                )

            if attempt < MAX_RETRY_ATTEMPTS - 1:
                time.sleep(RETRY_BACKOFF_BASE ** (attempt + 1))

        return None

    def _clean_html(self, html_content: str) -> str:
        """Remove HTML tags and clean up content text.

        Args:
            html_content: Raw HTML string.

        Returns:
            Cleaned plain text.
        """
        # Replace common block elements with newlines
        text = re.sub(r"<br\s*/?>", "\n", html_content, flags=re.I)
        text = re.sub(r"</(p|div|h[1-6]|li|tr)>", "\n", text, flags=re.I)
        # Remove all remaining HTML tags
        text = re.sub(r"<[^>]+>", "", text)
        # Decode HTML entities
        text = (
            text.replace("&nbsp;", " ")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&amp;", "&")
            .replace("&quot;", '"')
            .replace("&#39;", "'")
        )
        # Collapse whitespace
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def search_questions(
        self, keyword: str, max_pages: int = 100
    ) -> list[dict[str, Any]]:
        """Search Zhihu for questions matching a keyword.

        Uses Zhihu's search API to find questions related to the given
        keyword. Paginates through results up to the specified page limit.

        Args:
            keyword: Search keyword (Chinese or English).
            max_pages: Maximum number of result pages to fetch.

        Returns:
            List of question and answer dictionaries.
        """
        logger.info(
            "Searching Zhihu for keyword: '%s' (max_pages=%d)", keyword, max_pages
        )

        results: list[dict[str, Any]] = []
        questions_file = self._output_dir / f"questions_{keyword}.jsonl"
        answers_file = self._output_dir / f"answers_{keyword}.jsonl"

        for page in range(max_pages):
            params = {
                "q": keyword,
                "limit": 20,
                "offset": page * 20,
                "search_source": "Normal",
            }

            data = self._api_request("/search_v3", params=params)
            if data is None:
                # Try alternative search endpoint
                params_v2 = {
                    "q": keyword,
                    "page": page + 1,
                    "limit": 20,
                }
                data = self._api_request("/search", params=params_v2)

            if data is None:
                logger.debug("No more results for '%s' at page %d", keyword, page)
                break

            # Parse search results
            items = []
            if isinstance(data, dict):
                items = data.get("data", [])
                if not items:
                    # Try nested structure
                    items = data.get("hits", [])
                    if isinstance(items, list) and items and isinstance(items[0], dict):
                        items = items[0].get("highlight", {}).get("result", items)

            if not items:
                break

            new_items_found = False
            for item in items:
                if not isinstance(item, dict):
                    continue

                obj_type = item.get("type", item.get("object_type", ""))
                obj = item.get("object", item) or item

                if obj_type == "search_result" and isinstance(obj, dict):
                    # Wrapped result - look for the highlight
                    highlight = obj.get("highlight", {})
                    if highlight:
                        obj = highlight.get("result", obj)

                if not isinstance(obj, dict):
                    continue

                item_type = obj.get("type", "")

                if item_type == "answer":
                    answer = self._parse_answer_obj(obj)
                    if answer and answer.answer_id not in self._processed_answer_ids:
                        results.append(answer.to_dict())
                        self._processed_answer_ids.add(answer.answer_id)
                        new_items_found = True
                        with open(answers_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(answer.to_dict(), ensure_ascii=False) + "\n")

                elif item_type == "question":
                    question = self._parse_question_obj(obj)
                    if question and question.question_id not in self._processed_question_ids:
                        results.append(question.to_dict())
                        self._processed_question_ids.add(question.question_id)
                        new_items_found = True
                        with open(questions_file, "a", encoding="utf-8") as f:
                            f.write(
                                json.dumps(question.to_dict(), ensure_ascii=False) + "\n"
                            )

                        # Fetch answers for this question
                        answers = self._fetch_question_answers(question.question_id)
                        for ans in answers:
                            results.append(ans.to_dict())
                            with open(answers_file, "a", encoding="utf-8") as f:
                                f.write(
                                    json.dumps(ans.to_dict(), ensure_ascii=False) + "\n"
                                )

            if not new_items_found:
                logger.debug(
                    "No new items at page %d for '%s', stopping", page, keyword
                )
                break

            if (page + 1) % 5 == 0:
                logger.info(
                    "Search '%s': page %d/%d, %d results so far",
                    keyword,
                    page + 1,
                    max_pages,
                    len(results),
                )

        self._save_state()
        logger.info(
            "Search '%s' complete: %d results collected", keyword, len(results)
        )
        return results

    def _parse_question_obj(self, obj: dict[str, Any]) -> Optional[ZhihuQuestion]:
        """Parse a question object from Zhihu API response.

        Args:
            obj: Question object from Zhihu API.

        Returns:
            ZhihuQuestion dataclass instance, or None if parsing fails.
        """
        try:
            qid = str(obj.get("id", ""))
            if not qid:
                return None

            # Extract author
            author = obj.get("author", {})
            author_name = author.get("name") if isinstance(author, dict) else None

            # Extract topics
            topics_raw = obj.get("topics", [])
            topics = []
            if isinstance(topics_raw, list):
                topics = [
                    t.get("name", "") for t in topics_raw if isinstance(t, dict)
                ]

            # Extract detail and clean HTML
            detail = obj.get("detail", "") or ""
            detail_text = self._clean_html(detail) if detail else ""

            # Format timestamps
            created = obj.get("created", 0)
            created_str = (
                datetime.fromtimestamp(created).isoformat() if created else None
            )

            return ZhihuQuestion(
                question_id=qid,
                title=obj.get("title", ""),
                detail=detail_text,
                author_name=author_name,
                created_time=created_str,
                answer_count=obj.get("answer_count", 0),
                follower_count=obj.get("follower_count", 0),
                visit_count=obj.get("visit_count", 0),
                topics=topics,
                excerpt=obj.get("excerpt", ""),
                url=f"{ZHIHU_BASE_URL}/question/{qid}",
            )

        except Exception as exc:
            logger.error("Failed to parse question object: %s", exc)
            return None

    def _parse_answer_obj(self, obj: dict[str, Any]) -> Optional[ZhihuAnswer]:
        """Parse an answer object from Zhihu API response.

        Args:
            obj: Answer object from Zhihu API.

        Returns:
            ZhihuAnswer dataclass instance, or None if parsing fails.
        """
        try:
            aid = str(obj.get("id", ""))
            if not aid:
                return None

            # Extract question info
            question = obj.get("question", {})
            if isinstance(question, dict):
                qid = str(question.get("id", ""))
                qtitle = question.get("title", "")
            else:
                qid = ""
                qtitle = obj.get("question_title", "")

            # Extract author
            author = obj.get("author", {})
            author_name = author.get("name") if isinstance(author, dict) else None

            # Clean HTML content
            content_raw = obj.get("content", "") or ""
            content_text = self._clean_html(content_raw) if content_raw else ""

            # Format timestamps
            created = obj.get("created_time", 0) or obj.get("created", 0)
            updated = obj.get("updated_time", 0) or obj.get("updated", 0)
            created_str = (
                datetime.fromtimestamp(created).isoformat() if created else None
            )
            updated_str = (
                datetime.fromtimestamp(updated).isoformat() if updated else None
            )

            return ZhihuAnswer(
                answer_id=aid,
                question_id=qid,
                question_title=qtitle,
                content=content_text,
                author_name=author_name,
                upvotes=obj.get("voteup_count", 0),
                comment_count=obj.get("comment_count", 0),
                created_time=created_str,
                updated_time=updated_str,
                excerpt=obj.get("excerpt", ""),
                url=f"{ZHIHU_BASE_URL}/question/{qid}/answer/{aid}",
            )

        except Exception as exc:
            logger.error("Failed to parse answer object: %s", exc)
            return None

    def _fetch_question_answers(
        self, question_id: str, limit: int = 20
    ) -> list[ZhihuAnswer]:
        """Fetch answers for a specific Zhihu question.

        Args:
            question_id: Zhihu question ID.
            limit: Maximum number of answers to fetch.

        Returns:
            List of ZhihuAnswer instances.
        """
        answers: list[ZhihuAnswer] = []

        params = {
            "limit": min(limit, 20),
            "offset": 0,
            "sort_by": "default",
        }

        data = self._api_request(f"/questions/{question_id}/answers", params=params)
        if data is None or not isinstance(data, list):
            return answers

        for obj in data:
            if not isinstance(obj, dict):
                continue
            answer = self._parse_answer_obj(obj)
            if answer and answer.answer_id not in self._processed_answer_ids:
                answers.append(answer)
                self._processed_answer_ids.add(answer.answer_id)

        return answers

    def _scrape_topic_questions(
        self, topic_id: str, max_pages: int = 50
    ) -> list[dict[str, Any]]:
        """Scrape questions from a Zhihu topic page.

        Args:
            topic_id: Zhihu topic ID.
            max_pages: Maximum number of pages to fetch.

        Returns:
            List of question and answer dictionaries.
        """
        logger.info(
            "Scraping Zhihu topic ID: %s (max_pages=%d)", topic_id, max_pages
        )

        results: list[dict[str, Any]] = []

        for page in range(max_pages):
            params = {
                "limit": 20,
                "offset": page * 20,
            }

            data = self._api_request(f"/topics/{topic_id}/feeds/essence", params=params)
            if data is None or not isinstance(data, list):
                break

            new_items = False
            for item in data:
                if not isinstance(item, dict):
                    continue

                target = item.get("target", item)
                if not isinstance(target, dict):
                    continue

                item_type = target.get("type", "")

                if item_type == "answer":
                    answer = self._parse_answer_obj(target)
                    if answer and answer.answer_id not in self._processed_answer_ids:
                        results.append(answer.to_dict())
                        self._processed_answer_ids.add(answer.answer_id)
                        new_items = True

                elif item_type == "question":
                    question = self._parse_question_obj(target)
                    if question and question.question_id not in self._processed_question_ids:
                        results.append(question.to_dict())
                        self._processed_question_ids.add(question.question_id)
                        new_items = True

            if not new_items:
                break

        return results

    def crawl_sleep_topic(self) -> list[dict[str, Any]]:
        """Main entry point for crawling sleep-related Zhihu content.

        Searches for sleep-related questions using Chinese keywords,
        crawls the sleep topic feed, and collects questions and answers.

        DISCLAIMER: Ensure compliance with Zhihu's Terms of Service
        before using this method.

        Returns:
            List of all collected question and answer dictionaries.
        """
        logger.warning(STARTUP_DISCLAIMER)
        logger.info("Starting Zhihu sleep topic crawl")
        start_time = time.time()
        all_results: list[dict[str, Any]] = []

        # Save combined output
        combined_file = self._output_dir / "zhihu_sleep_all.jsonl"

        # Crawl sleep topic feeds
        for topic_id in SLEEP_TOPIC_IDS:
            try:
                topic_results = self._scrape_topic_questions(topic_id)
                all_results.extend(topic_results)
                self._save_state()
            except Exception as exc:
                logger.error("Failed to scrape topic %s: %s", topic_id, exc)

        # Search for sleep-related keywords
        for keyword in SLEEP_SEARCH_KEYWORDS:
            try:
                keyword_results = self.search_questions(keyword, max_pages=5)
                all_results.extend(keyword_results)
                self._save_state()
            except Exception as exc:
                logger.error("Failed to search keyword '%s': %s", keyword, exc)

        # Write combined output
        with open(combined_file, "w", encoding="utf-8") as f:
            for result in all_results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

        elapsed = time.time() - start_time
        q_count = sum(1 for r in all_results if "question_id" in r and "answer_id" not in r)
        a_count = sum(1 for r in all_results if "answer_id" in r)
        logger.info(
            "Zhihu crawl complete. %d items (%d questions, %d answers) in %.1f seconds",
            len(all_results),
            q_count,
            a_count,
            elapsed,
        )

        return all_results
