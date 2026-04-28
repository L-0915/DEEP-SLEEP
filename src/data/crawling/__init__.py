"""
Data crawling modules for the DeepSleep LLM project.

Collects sleep medicine domain training data from multiple sources:
PubMed, PMC, arXiv, Wikipedia, medical websites, Zhihu, and ClinicalTrials.gov.
"""

from src.data.crawling.pubmed_crawler import PubmedCrawler, PubmedArticle
from src.data.crawling.pmc_crawler import PMCCrawler, PMCArticle
from src.data.crawling.arxiv_crawler import ArxivCrawler, ArxivPaper
from src.data.crawling.wikipedia_crawler import WikipediaCrawler, WikipediaArticle
from src.data.crawling.medical_web_scraper import MedicalWebScraper, ScrapedPage
from src.data.crawling.zhihu_scraper import ZhihuScraper, ZhihuQuestion, ZhihuAnswer
from src.data.crawling.clinical_trials import ClinicalTrialsCrawler, ClinicalTrial

__all__ = [
    "PubmedCrawler",
    "PubmedArticle",
    "PMCCrawler",
    "PMCArticle",
    "ArxivCrawler",
    "ArxivPaper",
    "WikipediaCrawler",
    "WikipediaArticle",
    "MedicalWebScraper",
    "ScrapedPage",
    "ZhihuScraper",
    "ZhihuQuestion",
    "ZhihuAnswer",
    "ClinicalTrialsCrawler",
    "ClinicalTrial",
]
