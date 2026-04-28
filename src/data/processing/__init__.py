"""
Data processing pipeline for the DeepSleep LLM project.

Modules:
    dedup: Document deduplication (MinHash, exact, URL-based)
    lang_detect: Language detection and filtering (fasttext)
    quality_scorer: Quality scoring (perplexity, structural heuristics)
    domain_scorer: Sleep domain relevance scoring (BM25 keyword matching)
    safety_filter: Content safety filtering (PII, harmful content, noise)
    format_cleaner: Format cleaning (HTML, PDF, text normalization)
    mixer: Data mixing and resampling from multiple sources
"""

from src.data.processing.dedup import (
    ExactDeduplicator,
    MinHashDeduplicator,
    URLDeduplicator,
)
from src.data.processing.format_cleaner import FormatCleaner
from src.data.processing.lang_detect import LanguageDetector
from src.data.processing.mixer import DataMixer
from src.data.processing.quality_scorer import (
    DocumentQualityScorer,
    PerplexityScorer,
)
from src.data.processing.safety_filter import SafetyFilter
from src.data.processing.domain_scorer import SleepDomainScorer

__all__ = [
    # Deduplication
    "MinHashDeduplicator",
    "ExactDeduplicator",
    "URLDeduplicator",
    # Language detection
    "LanguageDetector",
    # Quality scoring
    "PerplexityScorer",
    "DocumentQualityScorer",
    # Domain scoring
    "SleepDomainScorer",
    # Safety filtering
    "SafetyFilter",
    # Format cleaning
    "FormatCleaner",
    # Data mixing
    "DataMixer",
]
