"""
Format cleaning module for the DeepSleep LLM data processing pipeline.

Provides text extraction and normalization utilities for cleaning raw content
from HTML pages, PDF files, and plain text.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Common boilerplate patterns to remove (headers, footers, navigation, etc.)
_BOILERPLATE_PATTERNS = [
    # Navigation elements
    re.compile(r"(?:首页|上一页|下一页|尾页|返回目录)\s*", re.DOTALL),
    re.compile(r"(?:Home|Previous|Next|Back|Menu|Table of Contents)\s*", re.IGNORECASE),
    # Page numbers
    re.compile(r"\b第\s*\d+\s*页\b"),
    re.compile(r"\bPage\s+\d+\s*(?:of\s+\d+)?\b", re.IGNORECASE),
    re.compile(r"\b-\s*\d+\s*-\s*$", re.MULTILINE),
    # Copyright notices
    re.compile(
        r"(?:版权所有|著作权|Copyright|All rights reserved).*?(?:\n|$)",
        re.IGNORECASE | re.DOTALL,
    ),
    # URL artifacts
    re.compile(r"https?://[^\s<>\"']+\s*"),
    # Email artifacts (outside of content)
    re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
]

# Encoding artifacts to fix
_ENCODING_FIXES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"&amp;"), "&"),
    (re.compile(r"&lt;"), "<"),
    (re.compile(r"&gt;"), ">"),
    (re.compile(r"&quot;"), '"'),
    (re.compile(r"&apos;"), "'"),
    (re.compile(r"&nbsp;"), " "),
    (re.compile(r"&#(\d+);"), lambda m: _decode_html_entity(m.group(1))),
    (re.compile(r"&#[xX]([0-9a-fA-F]+);"), lambda m: _decode_html_hex(m.group(1))),
    (re.compile(r"\xa0"), " "),
    (re.compile(r"\u3000"), " "),  # Chinese space
    (re.compile(r"\r\n"), "\n"),
    (re.compile(r"\r"), "\n"),
]


def _decode_html_entity(num_str: str) -> str:
    """Decode an HTML numeric entity to a Unicode character."""
    try:
        code_point = int(num_str)
        if code_point > 0x10FFFF:
            return ""
        return chr(code_point)
    except (ValueError, OverflowError):
        return ""


def _decode_html_hex(hex_str: str) -> str:
    """Decode an HTML hexadecimal entity to a Unicode character."""
    try:
        code_point = int(hex_str, 16)
        if code_point > 0x10FFFF:
            return ""
        return chr(code_point)
    except (ValueError, OverflowError):
        return ""


class FormatCleaner:
    """Text format cleaner for normalizing training data.

    Handles extraction of clean text from HTML and PDF sources,
    and normalization of plain text for training data preparation.

    Attributes:
        remove_boilerplate: Whether to remove common boilerplate patterns.
        normalize_unicode: Whether to apply Unicode NFKC normalization.
    """

    def __init__(
        self,
        remove_boilerplate: bool = True,
        normalize_unicode: bool = True,
    ) -> None:
        self.remove_boilerplate = remove_boilerplate
        self.normalize_unicode = normalize_unicode

    def clean_html(self, html_content: str) -> str:
        """Extract clean text from HTML content.

        Uses trafilatura for robust HTML-to-text extraction, handling
        navigation elements, sidebars, and other non-content markup.

        Args:
            html_content: Raw HTML string.

        Returns:
            Clean extracted text.
        """
        if not html_content or not html_content.strip():
            logger.warning("Empty HTML content provided")
            return ""

        try:
            import trafilatura

            extracted = trafilatura.extract(
                html_content,
                include_tables=True,
                include_links=False,
                include_images=False,
                favor_precision=True,
            )

            if extracted:
                result = self.clean_text(extracted)
                logger.debug("Extracted %d chars from HTML", len(result))
                return result

            logger.warning("trafilatura returned empty result, falling back to basic extraction")
        except ImportError:
            logger.error("trafilatura is not installed. Install with: pip install trafilatura")
        except Exception as e:
            logger.warning("trafilatura extraction failed: %s, using fallback", e)

        # Fallback: basic HTML tag removal
        text = re.sub(r"<script[^>]*>.*?</script>", "", html_content, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = self.clean_text(text)
        return text

    def clean_pdf(self, pdf_path: str | Path) -> str:
        """Extract text from a PDF file.

        Uses pymupdf (fitz) for PDF text extraction with layout-aware
        processing.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Extracted text from the PDF.
        """
        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            logger.error("PDF file not found: %s", pdf_file)
            return ""

        try:
            import fitz
        except ImportError:
            logger.error("pymupdf is not installed. Install with: pip install pymupdf")
            raise

        try:
            doc = fitz.open(str(pdf_file))
            pages_text: list[str] = []

            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text")

                if text and text.strip():
                    pages_text.append(text)

            doc.close()

            full_text = "\n\n".join(pages_text)
            cleaned = self.clean_text(full_text)

            logger.info(
                "Extracted %d chars from %s (%d pages)",
                len(cleaned),
                pdf_file,
                len(pages_text),
            )
            return cleaned

        except Exception as e:
            logger.error("Failed to extract text from PDF %s: %s", pdf_file, e)
            return ""

    def clean_text(self, text: str) -> str:
        """Normalize and clean plain text.

        Applies a series of text normalization steps:
        - Unicode normalization (NFKC)
        - Control character removal
        - Encoding artifact fixing
        - Whitespace normalization
        - Boilerplate removal

        Args:
            text: Raw input text.

        Returns:
            Cleaned and normalized text.
        """
        if not text:
            return ""

        result = text

        # Unicode normalization
        if self.normalize_unicode:
            result = unicodedata.normalize("NFKC", result)

        # Fix encoding artifacts
        for pattern, replacement in _ENCODING_FIXES:
            if callable(replacement):
                result = pattern.sub(lambda m: replacement(m), result)
            else:
                result = pattern.sub(replacement, result)

        # Remove control characters except newline and tab
        result = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", result)

        # Remove boilerplate patterns
        if self.remove_boilerplate:
            for pattern in _BOILERPLATE_PATTERNS:
                result = pattern.sub(" ", result)

        # Normalize whitespace
        # Replace multiple spaces with single space
        result = re.sub(r"[^\S\n]+", " ", result)
        # Remove leading/trailing whitespace from lines
        lines = [line.strip() for line in result.split("\n")]
        # Remove empty lines but preserve paragraph breaks (double newline)
        result = "\n".join(line for line in lines if line)
        # Collapse 3+ newlines into 2
        result = re.sub(r"\n{3,}", "\n\n", result)

        # Remove leading/trailing whitespace from entire text
        result = result.strip()

        return result

    def clean_document(
        self,
        doc: dict[str, Any],
        content_field: str = "text",
    ) -> dict[str, Any]:
        """Clean a document dictionary by applying text normalization.

        Creates a new dictionary with the cleaned text field (immutable pattern).
        Other fields are preserved unchanged.

        Args:
            doc: Input document dictionary.
            content_field: Field name containing the text to clean.

        Returns:
            New document dictionary with cleaned text.
        """
        text = str(doc.get(content_field, ""))
        cleaned = self.clean_text(text)

        if len(cleaned) == 0 and len(text) > 0:
            logger.warning(
                "Document cleaning produced empty text from %d chars of input",
                len(text),
            )

        return {**doc, content_field: cleaned}

    def clean_documents(
        self,
        documents: list[dict[str, Any]],
        content_field: str = "text",
    ) -> list[dict[str, Any]]:
        """Clean a batch of documents.

        Args:
            documents: List of document dictionaries.
            content_field: Field name containing the text to clean.

        Returns:
            List of new document dictionaries with cleaned text.
        """
        cleaned_docs: list[dict[str, Any]] = []
        total = len(documents)

        logger.info("Cleaning %d documents", total)

        for idx, doc in enumerate(documents):
            try:
                cleaned = self.clean_document(doc, content_field=content_field)
                cleaned_docs.append(cleaned)
            except Exception as e:
                logger.warning("Failed to clean document at index %d: %s", idx, e)
                # Keep the original document if cleaning fails
                cleaned_docs.append(doc)

            if (idx + 1) % 10000 == 0:
                logger.info("Cleaned %d/%d documents", idx + 1, total)

        logger.info("Document cleaning complete: %d documents processed", len(cleaned_docs))
        return cleaned_docs

    @staticmethod
    def detect_format(text: str) -> str:
        """Detect the format of input text.

        Args:
            text: Input text to analyze.

        Returns:
            Format string: "html", "json", "markdown", or "plain".
        """
        stripped = text.strip()

        if stripped.startswith("<") and stripped.endswith(">"):
            return "html"

        if stripped.startswith("{") or stripped.startswith("["):
            try:
                import json
                json.loads(stripped)
                return "json"
            except (json.JSONDecodeError, ValueError):
                pass

        # Check for markdown patterns
        md_patterns = [
            r"^#{1,6}\s",          # Headers
            r"^\*\s",              # Unordered list
            r"^\d+\.\s",           # Ordered list
            r"\[.+\]\(.+\)",       # Links
            r"```[\s\S]*?```",     # Code blocks
        ]
        for pattern in md_patterns:
            if re.search(pattern, stripped, re.MULTILINE):
                return "markdown"

        return "plain"


if __name__ == "__main__":
    import argparse
    import json as _json

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Format cleaning tools")
    parser.add_argument("--mode", choices=["text", "html", "pdf", "documents"], default="documents")
    parser.add_argument("--input", required=True, help="Input file path (JSONL for documents mode)")
    parser.add_argument("--output", required=True, help="Output file path")
    parser.add_argument("--content-field", default="text", help="Field name for document text")
    parser.add_argument("--no-boilerplate", action="store_true", help="Disable boilerplate removal")
    parser.add_argument("--no-unicode", action="store_true", help="Disable Unicode normalization")
    args = parser.parse_args()

    cleaner = FormatCleaner(
        remove_boilerplate=not args.no_boilerplate,
        normalize_unicode=not args.no_unicode,
    )

    if args.mode == "pdf":
        result = cleaner.clean_pdf(args.input)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(result)
        logger.info("Extracted PDF text written to %s", args.output)

    elif args.mode == "html":
        with open(args.input, "r", encoding="utf-8") as f:
            html = f.read()
        result = cleaner.clean_html(html)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(result)
        logger.info("Cleaned HTML text written to %s", args.output)

    elif args.mode == "text":
        with open(args.input, "r", encoding="utf-8") as f:
            text = f.read()
        result = cleaner.clean_text(text)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(result)
        logger.info("Cleaned text written to %s", args.output)

    elif args.mode == "documents":
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
        cleaned = cleaner.clean_documents(documents, content_field=args.content_field)

        with open(args.output, "w", encoding="utf-8") as f:
            for doc in cleaned:
                f.write(_json.dumps(doc, ensure_ascii=False) + "\n")

        logger.info("Wrote %d cleaned documents to %s", len(cleaned), args.output)

    logger.info("Done")
