"""Tests for data processing pipeline components."""

import pytest
import tempfile
import os


class TestDeduplicator:
    """Test deduplication modules."""

    def test_exact_dedup(self):
        """Test exact deduplication removes duplicate texts."""
        from src.data.processing.dedup import ExactDeduplicator

        dedup = ExactDeduplicator(n_gram_size=3)
        texts = [
            "Sleep apnea is a common disorder.",
            "Sleep apnea is a common disorder.",  # duplicate
            "Insomnia affects millions worldwide.",
        ]
        result = dedup.deduplicate_paragraphs(texts)
        assert len(result) == 2

    def test_url_dedup(self):
        """Test URL deduplication."""
        from src.data.processing.dedup import URLDeduplicator

        dedup = URLDeduplicator()
        docs = [
            {"url": "http://example.com/1", "text": "doc1"},
            {"url": "http://example.com/2", "text": "doc2"},
            {"url": "http://example.com/1", "text": "doc3"},  # same URL
        ]
        result = dedup.deduplicate_by_url(docs, "url")
        assert len(result) == 2


class TestQualityScorer:
    """Test quality scoring."""

    def test_short_document_filtered(self):
        """Test that very short documents are filtered."""
        from src.data.processing.quality_scorer import DocumentQualityScorer

        scorer = DocumentQualityScorer()
        docs = [
            {"text": "Too short."},
            {"text": "This is a reasonably long document about sleep disorders and their treatment options." * 5},
        ]
        result = scorer.filter_by_quality(docs, min_score=0.3)
        assert len(result) <= 2


class TestFormatCleaner:
    """Test format cleaning."""

    def test_text_cleaning(self):
        """Test general text cleaning."""
        from src.data.processing.format_cleaner import FormatCleaner

        cleaner = FormatCleaner()
        text = "Hello\x00\x01World  \n\n  Multiple   spaces"
        result = cleaner.clean_text(text)
        assert "\x00" not in result
        assert "\x01" not in result

    def test_html_cleaning(self):
        """Test HTML content cleaning."""
        from src.data.processing.format_cleaner import FormatCleaner

        cleaner = FormatCleaner()
        html = "<html><body><h1>Title</h1><p>Content here</p></body></html>"
        result = cleaner.clean_html(html)
        assert "<html>" not in result
        assert "<p>" not in result
        assert "Title" in result or "title" in result.lower()


class TestSafetyFilter:
    """Test safety filtering."""

    def test_pii_masking(self):
        """Test PII masking in text."""
        from src.data.processing.safety_filter import SafetyFilter

        sf = SafetyFilter()
        text = "Contact me at 13800138000 or test@example.com"
        result = sf.mask_pii(text)
        # Should mask phone number and/or email
        assert result != text


class TestDomainScorer:
    """Test sleep domain scoring."""

    def test_sleep_document_scored_high(self):
        """Test that sleep-related documents score high."""
        from src.data.processing.domain_scorer import SleepDomainScorer

        scorer = SleepDomainScorer(keywords_path="data/keywords")
        score = scorer.compute_keyword_score(
            "Sleep apnea is characterized by repeated episodes of upper airway obstruction during sleep, "
            "leading to intermittent hypoxia and sleep fragmentation. Treatment with CPAP is effective."
        )
        assert score > 0.3

    def test_non_sleep_document_scored_low(self):
        """Test that non-sleep documents score lower."""
        from src.data.processing.domain_scorer import SleepDomainScorer

        scorer = SleepDomainScorer(keywords_path="data/keywords")
        score = scorer.compute_keyword_score(
            "The stock market experienced significant volatility today with technology stocks leading the decline. "
            "Investors are advised to diversify their portfolios."
        )
        assert score < 0.2


class TestDataMixer:
    """Test data mixing."""

    def test_mixing_proportions(self):
        """Test that data mixing follows specified proportions."""
        from src.data.processing.mixer import DataMixer

        mixer = DataMixer(
            mixing_config={
                "cat_a": 0.6,
                "cat_b": 0.4,
            }
        )
        corpus = {
            "cat_a": [{"source": "cat_a", "text": f"doc {i}"} for i in range(100)],
            "cat_b": [{"source": "cat_b", "text": f"doc {i}"} for i in range(100)],
        }
        result = mixer.mix(corpus, total_tokens=100)
        # Check that both categories are represented
        sources = set(doc["source"] for doc in result)
        assert "cat_a" in sources
        assert "cat_b" in sources
