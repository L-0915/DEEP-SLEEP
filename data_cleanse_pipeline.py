"""
Data Cleansing Pipeline for rank_1000.jsonl
============================================
Stages:
  1. Quality Filtering  - drop low-quality records by metrics
  2. Text Cleaning      - strip HTML, URLs, noise characters, normalize
  3. Privacy Desensitization - redact PII (names, phones, IDs, emails, etc.)
  4. Deduplication      - remove near-duplicates via MinHash

Output: data/cleaned/rank_1000_cleaned.jsonl
"""

import json
import re
import os
import sys
import hashlib
import logging
import argparse
from pathlib import Path
from collections import Counter
from typing import Optional

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("cleanse")

# ---------------------------------------------------------------------------
# Stage 1 – Quality Filtering
# ---------------------------------------------------------------------------

class QualityFilter:
    """Drop records that fail basic quality thresholds."""

    # Thresholds (tuneable)
    MIN_TEXT_LEN = 50           # too short → not useful
    MAX_TEXT_LEN = 50000        # too long → likely noise / full book
    MIN_LANG_SCORE = 0.5        # language confidence
    MAX_PERPLEXITY = 5000       # very high → gibberish
    MIN_PERPLEXITY = 10         # very low → boilerplate / lists
    MAX_CHAR_REP_RATIO = 0.3    # excessive character repetition
    MAX_WORD_REP_RATIO = 0.5    # excessive word repetition
    MAX_SPECIAL_CHAR_RATIO = 0.4
    MIN_ALNUM_RATIO = 0.5
    MAX_FLAGGED_WORDS_RATIO = 0.01

    def __init__(self):
        self.stats = Counter()

    def check(self, record: dict) -> bool:
        text = record.get("text", "")
        text_len = len(text)

        # Length
        if text_len < self.MIN_TEXT_LEN:
            self.stats["too_short"] += 1
            return False
        if text_len > self.MAX_TEXT_LEN:
            self.stats["too_long"] += 1
            return False

        # Language
        lang_score = record.get("lang_score", 0)
        if lang_score < self.MIN_LANG_SCORE:
            self.stats["low_lang_score"] += 1
            return False

        # Perplexity
        ppl = record.get("perplexity", 0)
        if ppl < self.MIN_PERPLEXITY or ppl > self.MAX_PERPLEXITY:
            self.stats["bad_perplexity"] += 1
            return False

        # Repetition
        if record.get("char_rep_ratio", 0) > self.MAX_CHAR_REP_RATIO:
            self.stats["high_char_rep"] += 1
            return False
        if record.get("word_rep_ratio", 0) > self.MAX_WORD_REP_RATIO:
            self.stats["high_word_rep"] += 1
            return False

        # Special chars
        if record.get("special_char_ratio", 0) > self.MAX_SPECIAL_CHAR_RATIO:
            self.stats["high_special_chars"] += 1
            return False

        # Alnum
        if record.get("alnum_ratio", 0) < self.MIN_ALNUM_RATIO:
            self.stats["low_alnum"] += 1
            return False

        # Flagged words
        if record.get("flagged_words_ratio", 0) > self.MAX_FLAGGED_WORDS_RATIO:
            self.stats["high_flagged"] += 1
            return False

        self.stats["passed"] += 1
        return True


# ---------------------------------------------------------------------------
# Stage 2 – Text Cleaning
# ---------------------------------------------------------------------------

class TextCleaner:
    """Clean raw text: remove HTML, URLs, noise, normalize whitespace."""

    # Compiled patterns for speed
    RE_HTML_TAG = re.compile(r"<[^>]+>")
    RE_HTML_ENTITY = re.compile(r"&[a-zA-Z]+;|&#\d+;")
    RE_URL = re.compile(r"https?://[^\s<>\"]+|www\.[^\s<>\"]+")
    RE_EMAIL = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
    RE_IP_ADDR = re.compile(
        r"\b(?:(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)\.){3}"
        r"(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)\b"
    )
    RE_MULTI_NEWLINE = re.compile(r"\n{3,}")
    RE_MULTI_SPACE = re.compile(r"[ \t]{4,}")
    RE_NAVIGATION = re.compile(
        r"(?:首页|上一页|下一页|末页|返回|返回列表|更多|查看全部|"
        r"版权所有| All Rights Reserved|Copyright\s*\©?\s*\d*)[^\n]*",
        re.IGNORECASE,
    )
    RE_COOKIE_BANNER = re.compile(
        r"本网站使用.*?cookie|我们使用.*?cookie|继续浏览.*?同意",
        re.IGNORECASE,
    )
    RE_GARBLED = re.compile(r"[^\u4e00-\u9fff\u3000-\u303f\uff00-\uffef"
                            r"a-zA-Z0-9\s.,;:!?()（）、，。；：！？""''\"\'\-"
                            r"<>@#$%^&*+=/\\|{}\[\]~`_—…·\n\r\t]")
    RE_LEADING_QUOTE = re.compile(r'^["""]')
    RE_TRAILING_QUOTE = re.compile(r'["""]$')

    def clean(self, text: str) -> str:
        # Strip HTML
        text = self.RE_HTML_TAG.sub("", text)
        text = self.RE_HTML_ENTITY.sub("", text)

        # Remove URLs
        text = self.RE_URL.sub("[URL]", text)

        # Remove navigation / footer noise
        text = self.RE_NAVIGATION.sub("", text)

        # Remove cookie banners
        text = self.RE_COOKIE_BANNER.sub("", text)

        # Normalize whitespace
        text = self.RE_MULTI_NEWLINE.sub("\n\n", text)
        text = self.RE_MULTI_SPACE.sub("  ", text)

        # Strip leading/trailing quotes
        text = self.RE_LEADING_QUOTE.sub("", text.strip())
        text = self.RE_TRAILING_QUOTE.sub("", text.strip())

        # Remove leading/trailing whitespace per line
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(line for line in lines if line)

        return text.strip()


# ---------------------------------------------------------------------------
# Stage 3 – Privacy Desensitization
# ---------------------------------------------------------------------------

class PrivacyScrubber:
    """Redact personally identifiable information from Chinese medical text."""

    # --- Phone ---
    RE_MOBILE = re.compile(
        r"(?<!\d)(1[3-9]\d)\d{4}(\d{4})(?!\d)"
    )  # 11-digit CN mobile
    RE_LANDLINE = re.compile(
        r"(?<!\d)(0\d{2,3})[-\s]?\d{7,8}(?!\d)"
    )

    # --- ID Card ---
    RE_ID_CARD = re.compile(
        r"(?<!\d)([1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])"
        r"(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx])(?!\d)"
    )

    # --- Email ---
    RE_EMAIL = re.compile(
        r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
    )

    # --- Bank card (16-19 digits) ---
    RE_BANK_CARD = re.compile(
        r"(?<!\d)(\d{4})\s?\d{4}\s?\d{4}\s?\d{2,7}(?!\d)"
    )

    # --- QQ / WeChat ---
    RE_QQ = re.compile(r"(?:QQ|qq)[：:]\s*\d{5,12}")
    RE_WECHAT = re.compile(r"(?:微信|WeChat|wechat)[：:]\s*[a-zA-Z0-9_-]{5,20}")

    # --- IP Address ---
    RE_IP = re.compile(
        r"\b(?:(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)\.){3}"
        r"(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)\b"
    )

    # --- Medical record / hospital numbers ---
    RE_MEDICAL_NO = re.compile(
        r"(?:住院号|病历号|门诊号|就诊号|档案号|床号)[：:\s]*(\d{4,20})"
    )
    RE_HOSPITAL_NO = re.compile(
        r"(?:编号|No\.?)[：:\s]*(\d{6,20})"
    )

    # --- Chinese name patterns (common in medical text) ---
    # "患者XXX", "患儿XXX", "病人XXX" etc.
    RE_PATIENT_NAME = re.compile(
        r"(?:患者|患儿|病人|产妇|孕妇|家属|监护人)[：:，,\s]*"
        r"([\u4e00-\u9fff]{2,4})"
    )
    # "XXX某" pattern (partial names)
    RE_PARTIAL_NAME = re.compile(
        r"([\u4e00-\u9fff]{1,2})某[，,。.、\s]"
    )
    # Doctor names: "XXX医生", "XXX主任", "XXX教授" etc.
    RE_DOCTOR_NAME = re.compile(
        r"([\u4e00-\u9fff]{2,4})"
        r"(?:医生|主任|教授|博士|院长|科长|护士长|主治医师|副主任医师|主任医师)"
    )
    # "姓名：XXX"
    RE_NAME_LABEL = re.compile(
        r"(?:姓名|名字)[：:]\s*([\u4e00-\u9fff]{2,4})"
    )

    # --- Address patterns ---
    RE_ADDRESS = re.compile(
        r"((?:[\u4e00-\u9fff]{1,3}(?:省|市|区|县|镇|乡|村|路|街|道|号|栋|幢|楼|室|单元)){2,})"
    )

    # --- Age + Gender combined (common in case reports) ---
    # e.g. "男, 43岁"  → keep age, it's usually ok
    # but "年龄XX岁" with specific identifiers → redact

    def scrub(self, text: str) -> tuple[str, list[str]]:
        """Return (scrubbed_text, list_of_redaction_types)."""
        redactions = []
        original = text

        # ID Card (highest priority - strict PII)
        text, n = self._redact(text, self.RE_ID_CARD, "[身份证号]")
        if n:
            redactions.append(f"身份证号:{n}")

        # Phone numbers
        text, n = self._redact_mobile(text)
        if n:
            redactions.append(f"手机号:{n}")

        text, n = self._redact(text, self.RE_LANDLINE, "[座机号]")
        if n:
            redactions.append(f"座机号:{n}")

        # Email
        text, n = self._redact(text, self.RE_EMAIL, "[邮箱]")
        if n:
            redactions.append(f"邮箱:{n}")

        # Bank card
        text, n = self._redact(text, self.RE_BANK_CARD, "[银行卡号]")
        if n:
            redactions.append(f"银行卡号:{n}")

        # QQ / WeChat
        text, n = self._redact(text, self.RE_QQ, "[QQ号]")
        if n:
            redactions.append(f"QQ号:{n}")
        text, n = self._redact(text, self.RE_WECHAT, "[微信号]")
        if n:
            redactions.append(f"微信号:{n}")

        # IP Address
        text, n = self._redact(text, self.RE_IP, "[IP地址]")
        if n:
            redactions.append(f"IP地址:{n}")

        # Medical record numbers
        text, n = self._redact(text, self.RE_MEDICAL_NO, "[病历号]")
        if n:
            redactions.append(f"病历号:{n}")

        # Names (patient, doctor, labeled)
        text, n1 = self._redact(text, self.RE_PATIENT_NAME, "[患者姓名]")
        text, n2 = self._redact(text, self.RE_DOCTOR_NAME, "[医生姓名]")
        text, n3 = self._redact(text, self.RE_NAME_LABEL, "[姓名]")
        text, n4 = self._redact(text, self.RE_PARTIAL_NAME, "[姓名]")
        total_names = n1 + n2 + n3 + n4
        if total_names:
            redactions.append(f"姓名:{total_names}")

        # Addresses (only long detailed addresses)
        text, n = self._redact_address(text)
        if n:
            redactions.append(f"详细地址:{n}")

        return text, redactions

    @staticmethod
    def _redact(text: str, pattern: re.Pattern, replacement: str) -> tuple[str, int]:
        count = len(pattern.findall(text))
        return pattern.sub(replacement, text), count

    @staticmethod
    def _redact_mobile(text: str) -> tuple[str, int]:
        """Mask mobile: keep first 3 and last 4 digits."""
        matches = list(PrivacyScrubber.RE_MOBILE.finditer(text))
        if not matches:
            return text, 0
        for m in reversed(matches):
            text = text[: m.start()] + "[手机号]" + text[m.end() :]
        return text, len(matches)

    @staticmethod
    def _redact_address(text: str) -> tuple[str, int]:
        """Only redact addresses with 3+ location components (detailed)."""
        matches = list(PrivacyScrubber.RE_ADDRESS.finditer(text))
        count = 0
        for m in reversed(matches):
            addr = m.group(1)
            # Count location components
            components = len(re.findall(r"(?:省|市|区|县|镇|路|街|道|号|栋|楼|室)", addr))
            if components >= 3:  # Only redact detailed addresses
                text = text[: m.start()] + "[详细地址]" + text[m.end() :]
                count += 1
        return text, count


# ---------------------------------------------------------------------------
# Stage 4 – Deduplication (exact + MinHash)
# ---------------------------------------------------------------------------

class Deduplicator:
    """Remove exact duplicates and near-duplicates via SimHash-style fingerprint."""

    def __init__(self, num_hash_bits: int = 64):
        self.num_hash_bits = num_hash_bits
        self.seen_exact: set[str] = set()
        self.seen_shingle: set[str] = set()
        self.stats = Counter()

    def _shingle_hash(self, text: str) -> str:
        """Fast fingerprint: hash of sorted unique shingles (3-gram chars)."""
        # Use first 200 chars for speed — good enough for near-dup detection
        sample = text[:200]
        shingles = {sample[i:i + 3] for i in range(len(sample) - 2)}
        if not shingles:
            return hashlib.md5(text.encode()).hexdigest()
        key = "|".join(sorted(shingles))
        return hashlib.md5(key.encode()).hexdigest()

    def is_duplicate(self, text: str) -> bool:
        # Exact dedup
        content_hash = hashlib.md5(text.encode()).hexdigest()
        if content_hash in self.seen_exact:
            self.stats["exact_dup"] += 1
            return True

        # Near-dedup via shingle fingerprint
        shingle_hash = self._shingle_hash(text)
        if shingle_hash in self.seen_shingle:
            self.stats["near_dup"] += 1
            return True

        self.seen_exact.add(content_hash)
        self.seen_shingle.add(shingle_hash)
        self.stats["unique"] += 1
        return False


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class Pipeline:
    def __init__(self, input_path: str, output_dir: str):
        self.input_path = input_path
        self.output_dir = output_dir
        self.filter = QualityFilter()
        self.cleaner = TextCleaner()
        self.scrubber = PrivacyScrubber()
        self.dedup = Deduplicator()
        self.total = 0
        self.written = 0
        self.filtered_reasons = Counter()
        self.redaction_summary = Counter()

    def process_record(self, record: dict) -> Optional[dict]:
        """Process a single record through all stages. Returns None if filtered."""
        # Stage 1: Quality filter
        if not self.filter.check(record):
            return None

        # Stage 2: Text cleaning
        text = self.cleaner.clean(record["text"])

        # Post-clean length check
        if len(text) < 50:
            self.filter.stats["too_short_after_clean"] += 1
            return None

        # Stage 3: Privacy desensitization
        text, redactions = self.scrubber.scrub(text)
        for r in redactions:
            self.redaction_summary[r.split(":")[0]] += int(r.split(":")[1])

        # Stage 4: Deduplication
        if self.dedup.is_duplicate(text):
            return None

        # Build cleaned record
        return {
            "id": record["id"],
            "text": text,
            "lang": record["lang"],
            "industry_type": record["industry_type"],
            "text_length": len(text),
            "num_words": record["num_words"],
        }

    def run(self, batch_size: int = 10000):
        output_path = os.path.join(self.output_dir, "rank_1000_cleaned.jsonl")
        stats_path = os.path.join(self.output_dir, "cleaning_stats.json")

        os.makedirs(self.output_dir, exist_ok=True)

        log.info(f"Starting pipeline: {self.input_path} → {output_path}")

        with open(self.input_path, "r", encoding="utf-8") as fin, \
             open(output_path, "w", encoding="utf-8") as fout:

            buffer = []
            for line_num, line in enumerate(fin, 1):
                self.total += 1

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    self.filtered_reasons["json_error"] += 1
                    continue

                result = self.process_record(record)
                if result is not None:
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                    self.written += 1

                # Progress
                if line_num % batch_size == 0:
                    log.info(
                        f"Processed {line_num:,} records | "
                        f"Written {self.written:,} | "
                        f"Filtered {self.total - self.written:,}"
                    )

        # Save stats
        stats = {
            "total_input": self.total,
            "total_output": self.written,
            "filter_rate": f"{(1 - self.written / self.total) * 100:.2f}%",
            "quality_filter_stats": dict(self.filter.stats),
            "dedup_stats": dict(self.dedup.stats),
            "redaction_summary": dict(self.redaction_summary),
        }
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        log.info(f"Pipeline complete. {self.written:,} / {self.total:,} records kept.")
        log.info(f"Stats saved to {stats_path}")
        return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Data cleansing pipeline")
    parser.add_argument(
        "--input", "-i",
        default="data/rank_1000.jsonl",
        help="Input JSONL file path",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="data/cleaned",
        help="Output directory for cleaned data",
    )
    args = parser.parse_args()

    pipeline = Pipeline(
        input_path=args.input,
        output_dir=args.output_dir,
    )
    stats = pipeline.run()

    # Print summary
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    print(f"Input records:  {stats['total_input']:>10,}")
    print(f"Output records: {stats['total_output']:>10,}")
    print(f"Filter rate:    {stats['filter_rate']:>10}")
    print()
    print("Quality filter reasons:")
    for reason, count in sorted(stats["quality_filter_stats"].items()):
        print(f"  {reason}: {count:,}")
    print()
    print("Deduplication:")
    for reason, count in sorted(stats["dedup_stats"].items()):
        print(f"  {reason}: {count:,}")
    print()
    print("Privacy redactions:")
    for pii_type, count in sorted(stats["redaction_summary"].items()):
        print(f"  {pii_type}: {count:,}")
    print("=" * 60)


if __name__ == "__main__":
    main()
