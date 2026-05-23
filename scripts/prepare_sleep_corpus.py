#!/usr/bin/env python3
"""Filter IndustryCorpus to extract sleep-relevant documents.

Reads the unified pretrain JSONL and filters documents that contain
sleep-related keywords, producing a sleep-domain corpus for pretrain mixing.

Usage:
    python scripts/prepare_sleep_corpus.py \
        --input data/cleaned/deepsleep_pretrain_unified.jsonl \
        --output data/cleaned/sleep_domain_corpus.jsonl
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SLEEP_KEYWORDS_EN = [
    "sleep", "insomnia", "apnea", "OSA", "CSA", "narcolepsy", "hypersomnia",
    "circadian", "melatonin", "polysomnography", "PSG", "CPAP", "BiPAP",
    "REM", "NREM", "slow-wave", "arousal", "somnolence", "hypopnea",
    "AHI", "RDI", "actigraphy", "CBT-I", "restless legs", "PLMD",
    "bruxism", "parasomnia", "sleepwalking", "cataplexy",
    "orexin", "hypocretin", "sedation", "hypnotic", "zolpidem", "eszopiclone",
    "polysomnographic", "nocturnal", "dyssomnia", "sleep-onset", "sleep-maintenance",
    "excessive daytime sleepiness", "EDS", "Epworth", "STOP-BANG",
    "obstructive sleep", "central sleep", "upper airway", "adenotonsillectomy",
    "sleep architecture", "sleep efficiency", "sleep latency", "REM latency",
    "sleep spindle", "K-complex", "slow wave sleep", "SWS",
]

SLEEP_KEYWORDS_ZH = [
    "睡眠", "失眠", "呼吸暂停", "打鼾", "嗜睡", "发作性睡病",
    "昼夜节律", "褪黑素", "多导睡眠", "CPAP", "无创通气",
    "慢波", "觉醒", "白天嗜睡", "低通气",
    "体动记录", "不宁腿", "PLMD",
    "磨牙", "异态睡眠", "梦游", "猝倒", "食欲素",
    "镇静", "催眠", "佐匹克隆", "唑吡坦", "右佐匹克隆",
    "夜间", "入睡困难", "睡眠维持", "白天过度嗜睡",
    "阻塞性睡眠", "中枢性睡眠",
    "上气道", "腺样体", "睡眠结构", "睡眠效率",
    "入睡潜伏期", "睡眠纺锤波", "慢波睡眠",
    "快速眼动", "睡眠卫生", "睡眠障碍",
    "睡眠呼吸", "睡眠医学", "睡眠质量", "睡眠监测",
]


def build_pattern(keywords_en, keywords_zh):
    all_keywords = keywords_en + keywords_zh
    escaped = [re.escape(kw) for kw in all_keywords]
    return re.compile("|".join(escaped), re.IGNORECASE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSONL (unified pretrain)")
    parser.add_argument("--output", required=True, help="Output JSONL (sleep domain)")
    parser.add_argument("--min_keywords", type=int, default=2, help="Min keyword matches to include")
    parser.add_argument("--max_docs", type=int, default=None, help="Max docs to output")
    args = parser.parse_args()

    pattern = build_pattern(SLEEP_KEYWORDS_EN, SLEEP_KEYWORDS_ZH)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    matched = 0

    with open(args.input, "r", encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1

            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = doc.get("text", "") or doc.get("content", "")
            matches = pattern.findall(text)

            if len(matches) >= args.min_keywords:
                fout.write(line + "\n")
                matched += 1
                if args.max_docs and matched >= args.max_docs:
                    break

            if total % 100000 == 0:
                logger.info("Processed %d docs, matched %d sleep docs", total, matched)

    logger.info(
        "Done: %d/%d docs matched (%.1f%%) -> %s",
        matched, total, 100 * matched / max(total, 1), args.output,
    )


if __name__ == "__main__":
    main()
