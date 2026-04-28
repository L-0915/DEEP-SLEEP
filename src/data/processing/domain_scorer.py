"""
Sleep domain relevance scoring for the DeepSleep LLM data processing pipeline.

Provides BM25-style keyword scoring against a comprehensive sleep medicine
keyword dictionary, with optional ML-based fine-grained relevance classification.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Comprehensive built-in sleep medicine keyword lists.
# These serve as the default keyword dictionary when no external files are provided.

_SLEEP_DISORDERS_EN = [
    "insomnia", "sleep apnea", "obstructive sleep apnea", "central sleep apnea",
    "complex sleep apnea", "narcolepsy", "hypersomnia", "idiopathic hypersomnia",
    "restless legs syndrome", "rls", "periodic limb movement disorder", "plmd",
    "rem sleep behavior disorder", "rbd", "sleepwalking", "somnambulism",
    "sleep terror", "night terror", "sleep paralysis", "bruxism",
    "sleep enuresis", "bedwetting", "circadian rhythm disorder",
    "delayed sleep phase", "advanced sleep phase", "shift work disorder",
    "jet lag", "non-24-hour sleep wake disorder", "irregular sleep wake rhythm",
    "fatal familial insomnia", "sleep related eating disorder",
    "sleep related hallucination", "exploding head syndrome",
    "sleep talking", "somniloquy", "cataplexy", "sleep fragmentation",
    "sleep maintenance insomnia", "sleep onset insomnia", "chronic insomnia",
    "acute insomnia", "comorbid insomnia", "sleep state misperception",
    "paradoxical insomnia", "pediatric sleep disorder", "sleep disordered breathing",
    "upper airway resistance syndrome", "sleep hypoventilation",
    "obesity hypoventilation syndrome", "pickwickian syndrome",
    "congenital central hypoventilation", "sleep related laryngospasm",
    "sleep related gastroesophageal reflux", "sleep related asthma",
    "sleep related seizures", "sleep related headache",
    "sleep related painful erections", "sleep related abnormal swallowing",
    "sleep related cardiac arrhythmia", "sleep related ischemic heart disease",
]

_SLEEP_PHYSIOLOGY_EN = [
    "rem sleep", "rapid eye movement", "nrem sleep", "non-rapid eye movement",
    "slow wave sleep", "deep sleep", "stage n1", "stage n2", "stage n3",
    "sleep stage", "sleep architecture", "sleep cycle", "sleep spindle",
    "k-complex", "slow oscillation", "delta wave", "theta wave", "alpha wave",
    "sleep onset", "sleep latency", "sleep duration", "sleep efficiency",
    "sleep continuity", "circadian rhythm", "circadian clock", "suprachiasmatic nucleus",
    "melatonin", "serotonin", "gaba", "adenosine", "orexin", "hypocretin",
    "histamine", "acetylcholine", "norepinephrine", "dopamine",
    "sleep homeostasis", "sleep pressure", "sleep drive", "sleep need",
    "sleep debt", "sleep rebound", "rem rebound", "slow wave rebound",
    "microarousal", "arousal", "awakening", "sleep transition",
    "sleep inertia", "sleep drunkenness", "twilight state",
    "polysomnography", "psg", "electroencephalogram", "eeg",
    "electrooculogram", "eog", "electromyogram", "emg",
    "electrocardiogram", "ecg", "pulse oximetry", "sp02",
    "respiratory effort", "airflow", "snoring", "sleep apnea hypopnea index",
    "ahi", "respiratory disturbance index", "rdi", "oxygen desaturation index",
    "odi", "arousal index", "plm index", "sleep latency", "rem latency",
    "cpap", "continuous positive airway pressure", "bpap", "bipap",
    "adaptive servo ventilation", "asv", "oral appliance", "mandibular advancement",
    "hypoglossal nerve stimulation", "positional therapy", "sleep hygiene",
    "actigraphy", "sleep diary", "sleep log", "sleep questionnaire",
    "epworth sleepiness scale", "ess", "pittsburgh sleep quality index",
    "psqi", "insomnia severity index", "isi", "stop-bang", "berlin questionnaire",
    "stanford sleepiness scale", "sscale", "karolinska sleepiness scale",
    "multiple sleep latency test", "mslt", "maintenance of wakefulness test",
    "mwt", "sleep study", "home sleep test", "hst", "portable sleep monitor",
    "melatonin receptor agonist", "orexin receptor antagonist",
    "suvorexant", "lemborexant", "daridorexant", "ramelteon",
    "eszopiclone", "zolpidem", "zaleplon", "triazolam",
    "temazepam", "doxepin", "mirtazapine", "trazodone",
    "modafinil", "armodafinil", "sodium oxybate", "pitolisant",
    "pramipexole", "ropinirole", "gabapentin", "pregabalin",
    "light therapy", "chronotherapy", "dark therapy", "bright light therapy",
    "cognitive behavioral therapy for insomnia", "cbt-i", "sleep restriction therapy",
    "stimulus control", "sleep compression", "relaxation training",
    "mindfulness based therapy", "sleep medicine", "somnology",
    "board certified sleep medicine", "sleep specialist", "sleep center",
    "sleep laboratory", "aasm", "american academy of sleep medicine",
    "icd", "international classification of sleep disorders", "icsd",
    "dsm", "diagnostic and statistical manual",
]

_SLEEP_DISORDERS_ZH = [
    "失眠", "睡眠呼吸暂停", "阻塞性睡眠呼吸暂停", "中枢性睡眠呼吸暂停",
    "复杂性睡眠呼吸暂停", "发作性睡病", "嗜睡症", "特发性嗜睡",
    "不宁腿综合征", "周期性肢体运动障碍", "快速眼动睡眠行为障碍",
    "rem睡眠行为障碍", "睡行症", "梦游", "夜惊", "睡眠瘫痪",
    "睡眠麻痹", "磨牙症", "遗尿症", "尿床", "昼夜节律障碍",
    "睡眠时相延迟", "睡眠时相前移", "轮班工作障碍", "时差反应",
    "非24小时睡眠觉醒障碍", "不规律睡眠觉醒节律", "致死性家族性失眠",
    "睡眠相关进食障碍", "睡眠相关幻觉", "爆炸性头部综合征",
    "说梦话", "梦语症", "猝倒", "睡眠碎片化", "睡眠维持困难",
    "入睡困难", "慢性失眠", "急性失眠", "共病失眠", "睡眠状态感知不良",
    "矛盾性失眠", "儿童睡眠障碍", "睡眠呼吸障碍", "上气道阻力综合征",
    "睡眠低通气", "肥胖低通气综合征", "先天性中枢性低通气",
    "睡眠相关性喉痉挛", "睡眠相关性胃食管反流", "睡眠相关性哮喘",
    "睡眠相关性癫痫", "睡眠相关性头痛", "睡眠相关性心律失常",
    "睡眠相关性缺血性心脏病", "睡眠呼吸暂停低通气指数",
]

_SLEEP_PHYSIOLOGY_ZH = [
    "快速眼动睡眠", "rem睡眠", "非快速眼动睡眠", "nrem睡眠",
    "慢波睡眠", "深度睡眠", "睡眠分期", "睡眠结构", "睡眠周期",
    "睡眠纺锤波", "k复合波", "慢振荡", "delta波", "delta波",
    "theta波", "alpha波", "入睡", "睡眠潜伏期", "睡眠时长",
    "睡眠效率", "睡眠连续性", "昼夜节律", "生物钟", "视交叉上核",
    "褪黑素", "褪黑激素", "血清素", "5-羟色胺", "gamma氨基丁酸",
    "gaba", "腺苷", "食欲素", "下丘脑分泌素", "组胺",
    "乙酰胆碱", "去甲肾上腺素", "多巴胺", "睡眠稳态",
    "睡眠压力", "睡眠驱力", "睡眠需求", "睡眠债", "睡眠补偿",
    "rem反弹", "慢波反弹", "微觉醒", "觉醒", "入睡潜伏期",
    "睡眠惯性", "睡眠醉态", "多导睡眠图", "多导睡眠监测",
    "脑电图", "眼电图", "肌电图", "心电图",
    "血氧饱和度", "呼吸努力", "气流", "打鼾",
    "呼吸暂停低通气指数", "ahi", "氧减指数", "觉醒指数",
    "肢体运动指数", "持续气道正压通气", "cpap",
    "双水平气道正压通气", "bpap", "自适应伺服通气",
    "口腔矫治器", "下颌前移装置", "舌下神经刺激",
    "体位治疗", "睡眠卫生", "体动记录仪", "睡眠日记",
    "睡眠问卷", "epworth嗜睡量表", "匹兹堡睡眠质量指数",
    "失眠严重程度指数", "stop-bang问卷", "柏林问卷",
    "斯坦福嗜睡量表", "多次睡眠潜伏期试验", "维持觉醒试验",
    "睡眠监测", "居家睡眠监测", "便携式睡眠监测",
    "褪黑素受体激动剂", "食欲素受体拮抗剂",
    "认知行为疗法治疗失眠", "cbt-i", "睡眠限制疗法",
    "刺激控制疗法", "睡眠压缩", "放松训练", "正念疗法",
    "睡眠医学", "睡眠专科", "睡眠中心", "睡眠实验室",
    "美国睡眠医学学会", "国际睡眠障碍分类",
]

# Bucket thresholds for domain relevance classification
_BUCKET_THRESHOLDS = {
    "high": 0.15,
    "medium": 0.05,
    "low": 0.01,
}


class SleepDomainScorer:
    """Sleep medicine domain relevance scoring.

    Scores documents based on keyword density using a BM25-inspired approach.
    Documents are classified into relevance buckets (high/medium/low/irrelevant)
    based on their keyword match scores.

    Attributes:
        keywords_path: Path to directory containing keyword files.
        keywords_en: Set of English sleep medicine keywords.
        keywords_zh: Set of Chinese sleep medicine keywords.
    """

    def __init__(self, keywords_path: str = "data/keywords") -> None:
        self.keywords_path = Path(keywords_path)
        self.keywords_en: set[str] = set()
        self.keywords_zh: set[str] = set()
        self._keyword_idf: dict[str, float] = {}
        self._avg_doc_length: float = 100.0
        self._classifier = None
        self._vectorizer = None

        self._load_keywords()

    def _load_keywords(self) -> None:
        """Load keyword dictionary from files, falling back to built-in defaults."""
        # Load English keywords
        en_file = self.keywords_path / "sleep_keywords_en.txt"
        if en_file.exists():
            try:
                with open(en_file, "r", encoding="utf-8") as f:
                    self.keywords_en = {line.strip().lower() for line in f if line.strip()}
                logger.info("Loaded %d English keywords from %s", len(self.keywords_en), en_file)
            except Exception as e:
                logger.warning("Failed to load English keywords from %s: %s", en_file, e)
                self._use_builtin_keywords_en()
        else:
            logger.info("English keywords file not found at %s, using built-in defaults", en_file)
            self._use_builtin_keywords_en()

        # Load Chinese keywords
        zh_file = self.keywords_path / "sleep_keywords_zh.txt"
        if zh_file.exists():
            try:
                with open(zh_file, "r", encoding="utf-8") as f:
                    self.keywords_zh = {line.strip() for line in f if line.strip()}
                logger.info("Loaded %d Chinese keywords from %s", len(self.keywords_zh), zh_file)
            except Exception as e:
                logger.warning("Failed to load Chinese keywords from %s: %s", zh_file, e)
                self._use_builtin_keywords_zh()
        else:
            logger.info("Chinese keywords file not found at %s, using built-in defaults", zh_file)
            self._use_builtin_keywords_zh()

        # Initialize IDF values (uniform for keyword lists)
        total_keywords = len(self.keywords_en) + len(self.keywords_zh)
        for kw in self.keywords_en | self.keywords_zh:
            self._keyword_idf[kw] = math.log(1.0 + total_keywords / 1.0)

        logger.info(
            "Total keywords loaded: %d English, %d Chinese",
            len(self.keywords_en),
            len(self.keywords_zh),
        )

    def _use_builtin_keywords_en(self) -> None:
        """Load built-in English keyword defaults."""
        self.keywords_en = {kw.lower() for kw in _SLEEP_DISORDERS_EN + _SLEEP_PHYSIOLOGY_EN}

    def _use_builtin_keywords_zh(self) -> None:
        """Load built-in Chinese keyword defaults."""
        self.keywords_zh = set(_SLEEP_DISORDERS_ZH + _SLEEP_PHYSIOLOGY_ZH)

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into keywords for matching.

        Performs character-level matching for Chinese keywords and
        word-level matching for English keywords.

        Args:
            text: Input text.

        Returns:
            List of token strings for matching.
        """
        tokens: list[str] = []
        text_lower = text.lower()

        # Add individual Chinese characters and bigrams for matching
        chinese_chars: list[str] = []
        for char in text_lower:
            if 0x4E00 <= ord(char) <= 0x9FFF:
                chinese_chars.append(char)

        # Add individual chars
        tokens.extend(chinese_chars)

        # Add bigrams for better matching
        for i in range(len(chinese_chars) - 1):
            tokens.append(chinese_chars[i] + chinese_chars[i + 1])

        # Add trigrams
        for i in range(len(chinese_chars) - 2):
            tokens.append(chinese_chars[i] + chinese_chars[i + 1] + chinese_chars[i + 2])

        # Add English word tokens
        english_words = re.findall(r"[a-z][a-z0-9\-]+", text_lower)
        tokens.extend(english_words)

        # Add multi-word English phrases (bigrams and trigrams)
        for i in range(len(english_words) - 1):
            tokens.append(english_words[i] + " " + english_words[i + 1])
        for i in range(len(english_words) - 2):
            tokens.append(
                english_words[i] + " " + english_words[i + 1] + " " + english_words[i + 2]
            )

        return tokens

    def compute_keyword_score(self, text: str) -> float:
        """Compute BM25-style relevance score against sleep keyword dictionary.

        Uses a simplified BM25 scoring approach where each keyword match
        contributes an IDF-weighted score, normalized by document length.

        Args:
            text: Input document text.

        Returns:
            Relevance score between 0.0 (irrelevant) and 1.0 (highly relevant).
            The score is clamped to [0, 1].
        """
        tokens = self._tokenize(text)
        if not tokens:
            return 0.0

        token_counts = Counter(tokens)
        doc_length = len(tokens)
        k1 = 1.5  # BM25 term frequency saturation parameter
        b = 0.75  # BM25 length normalization parameter

        total_score = 0.0

        for keyword, idf in self._keyword_idf.items():
            if keyword in token_counts:
                tf = token_counts[keyword]
                # BM25 scoring formula
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * doc_length / self._avg_doc_length)
                total_score += idf * (numerator / denominator)

        # Normalize to [0, 1] using a sigmoid-like transformation
        # A score of ~10 raw points maps to ~0.8 normalized
        normalized = 1.0 - 1.0 / (1.0 + total_score / 5.0)
        return max(0.0, min(1.0, normalized))

    def score_documents(
        self,
        documents: list[dict[str, Any]],
        text_field: str = "text",
    ) -> list[tuple[dict[str, Any], float]]:
        """Score a list of documents for sleep domain relevance.

        Args:
            documents: List of document dictionaries.
            text_field: Field name containing document text.

        Returns:
            List of (document, score) tuples sorted by score descending.
        """
        scored: list[tuple[dict[str, Any], float]] = []
        total = len(documents)

        logger.info("Scoring %d documents for sleep domain relevance", total)

        for idx, doc in enumerate(documents):
            text = str(doc.get(text_field, ""))
            doc_score = self.compute_keyword_score(text)
            scored.append((doc, doc_score))

            if (idx + 1) % 10000 == 0:
                logger.info("Scored %d/%d documents", idx + 1, total)

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        return scored

    def bucket_documents(
        self,
        scored_docs: list[tuple[dict[str, Any], float]],
    ) -> dict[str, list[dict[str, Any]]]:
        """Bucket scored documents into relevance categories.

        Documents are classified into four buckets based on their
        relevance score:
            - high: score >= 0.15 (strongly sleep-related)
            - medium: score >= 0.05 (moderately sleep-related)
            - low: score >= 0.01 (weakly sleep-related)
            - irrelevant: score < 0.01 (not sleep-related)

        Args:
            scored_docs: List of (document, score) tuples from score_documents.

        Returns:
            Dictionary mapping bucket names to lists of documents.
        """
        buckets: dict[str, list[dict[str, Any]]] = {
            "high": [],
            "medium": [],
            "low": [],
            "irrelevant": [],
        }

        for doc, score in scored_docs:
            if score >= _BUCKET_THRESHOLDS["high"]:
                buckets["high"].append(doc)
            elif score >= _BUCKET_THRESHOLDS["medium"]:
                buckets["medium"].append(doc)
            elif score >= _BUCKET_THRESHOLDS["low"]:
                buckets["low"].append(doc)
            else:
                buckets["irrelevant"].append(doc)

        for bucket_name, docs in buckets.items():
            logger.info(
                "Bucket '%s': %d documents (%.1f%%)",
                bucket_name,
                len(docs),
                (len(docs) / len(scored_docs) * 100) if scored_docs else 0,
            )

        return buckets

    def train_relevance_classifier(
        self,
        train_data: list[dict[str, Any]],
        text_field: str = "text",
        label_field: str = "is_sleep_relevant",
    ) -> None:
        """Train a lightweight relevance classifier for fine-grained scoring.

        Uses scikit-learn LogisticRegression with TF-IDF features and
        keyword density as additional features. The trained classifier
        supplements the keyword-based scoring for borderline cases.

        Args:
            train_data: List of training documents. Each must contain the
                text_field and a binary label_field (1 for sleep-relevant, 0 otherwise).
            text_field: Field name containing document text.
            label_field: Field name containing the binary relevance label.
        """
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import Pipeline
        except ImportError:
            logger.error(
                "scikit-learn is not installed. Install it with: pip install scikit-learn"
            )
            raise

        logger.info("Training relevance classifier on %d documents", len(train_data))

        texts = []
        labels = []
        keyword_densities: list[list[float]] = []

        for doc in train_data:
            text = str(doc.get(text_field, ""))
            label = doc.get(label_field, 0)

            if not text.strip():
                continue

            texts.append(text)
            labels.append(int(label))
            keyword_densities.append([self.compute_keyword_score(text)])

        if len(texts) < 10:
            logger.error("Insufficient training data: need at least 10 samples, got %d", len(texts))
            raise ValueError(f"Insufficient training data: {len(texts)} samples")

        from sklearn.preprocessing import FunctionTransformer
        import numpy as np

        class KeywordDensityTransformer(FunctionTransformer):
            """Custom transformer that adds keyword density as a feature."""

            def __init__(self, densities: list[list[float]]):  # type: ignore[override]
                super().__init__()
                self._densities = densities

            def transform(self, X: Any) -> Any:  # type: ignore[override]
                import numpy as np
                if len(X) != len(self._densities):
                    logger.warning(
                        "Keyword density count mismatch: %d texts vs %d densities",
                        len(X),
                        len(self._densities),
                    )
                return np.array(self._densities[: len(X)])

        try:
            tfidf = TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 3),
                min_df=2,
                max_df=0.95,
                sublinear_tf=True,
            )
            clf = LogisticRegression(
                max_iter=1000,
                C=1.0,
                class_weight="balanced",
            )

            # Combine TF-IDF with keyword density
            X_tfidf = tfidf.fit_transform(texts)
            kw_array = np.array(keyword_densities)

            from scipy.sparse import hstack
            X_combined = hstack([X_tfidf, kw_array])
            y_array = np.array(labels)

            clf.fit(X_combined, y_array)

            self._vectorizer = tfidf
            self._classifier = clf

            train_accuracy = clf.score(X_combined, y_array)
            logger.info("Relevance classifier trained. Training accuracy: %.4f", train_accuracy)

        except Exception as e:
            logger.error("Failed to train relevance classifier: %s", e)
            raise

    def predict_relevance(self, text: str) -> float:
        """Predict sleep relevance using the trained classifier.

        Falls back to keyword-based scoring if no classifier is trained.

        Args:
            text: Input document text.

        Returns:
            Relevance probability between 0.0 and 1.0.
        """
        if self._classifier is None or self._vectorizer is None:
            logger.debug("No classifier trained, falling back to keyword scoring")
            return self.compute_keyword_score(text)

        try:
            import numpy as np
            from scipy.sparse import hstack

            X_tfidf = self._vectorizer.transform([text])
            kw_density = np.array([[self.compute_keyword_score(text)]])
            X_combined = hstack([X_tfidf, kw_density])

            proba = self._classifier.predict_proba(X_combined)
            # Probability of positive class (sleep relevant)
            if proba.shape[1] >= 2:
                return float(proba[0][1])
            return float(proba[0][0])

        except Exception as e:
            logger.warning("Classifier prediction failed, falling back to keyword scoring: %s", e)
            return self.compute_keyword_score(text)

    def get_keyword_coverage(self, text: str) -> dict[str, list[str]]:
        """Identify which sleep keywords are present in the text.

        Args:
            text: Input document text.

        Returns:
            Dictionary with 'en' and 'zh' keys mapping to lists of
            matched keywords found in the text.
        """
        text_lower = text.lower()
        matched_en: list[str] = []
        matched_zh: list[str] = []

        for keyword in self.keywords_en:
            if keyword in text_lower:
                matched_en.append(keyword)

        for keyword in self.keywords_zh:
            if keyword in text:
                matched_zh.append(keyword)

        return {"en": sorted(set(matched_en)), "zh": sorted(set(matched_zh))}


if __name__ == "__main__":
    import argparse
    import json as _json

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Sleep domain relevance scoring")
    parser.add_argument("--mode", choices=["score", "bucket", "train", "coverage"], default="score")
    parser.add_argument("--input", required=True, help="Input JSONL file path")
    parser.add_argument("--output", required=True, help="Output JSONL file path")
    parser.add_argument("--text-field", default="text", help="Field name for document text")
    parser.add_argument("--keywords-path", default="data/keywords", help="Path to keyword files")
    parser.add_argument("--train-label", default="is_sleep_relevant", help="Label field for training")
    args = parser.parse_args()

    scorer = SleepDomainScorer(keywords_path=args.keywords_path)

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

    if args.mode == "score":
        scored = scorer.score_documents(documents, text_field=args.text_field)
        results = [{**doc, "sleep_domain_score": score} for doc, score in scored]

        with open(args.output, "w", encoding="utf-8") as f:
            for doc in results:
                f.write(_json.dumps(doc, ensure_ascii=False) + "\n")

        logger.info("Wrote %d scored documents to %s", len(results), args.output)

    elif args.mode == "bucket":
        scored = scorer.score_documents(documents, text_field=args.text_field)
        buckets = scorer.bucket_documents(scored)

        for bucket_name, docs in buckets.items():
            bucket_file = Path(args.output).with_suffix("") + f"_{bucket_name}.jsonl"
            with open(bucket_file, "w", encoding="utf-8") as f:
                for doc in docs:
                    f.write(_json.dumps(doc, ensure_ascii=False) + "\n")
            logger.info("Wrote %d documents to %s", len(docs), bucket_file)

    elif args.mode == "train":
        scorer.train_relevance_classifier(
            documents,
            text_field=args.text_field,
            label_field=args.train_label,
        )
        logger.info("Classifier trained successfully")

    elif args.mode == "coverage":
        results = []
        for doc in documents:
            text = str(doc.get(args.text_field, ""))
            coverage = scorer.get_keyword_coverage(text)
            result = {**doc, "keyword_coverage": coverage}
            results.append(result)

        with open(args.output, "w", encoding="utf-8") as f:
            for doc in results:
                f.write(_json.dumps(doc, ensure_ascii=False) + "\n")

    logger.info("Done")
