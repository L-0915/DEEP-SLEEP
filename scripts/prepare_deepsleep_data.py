#!/usr/bin/env python3
"""DeepSleep 预训练数据准备 — 从 HuggingFace 加载 CCI4.0-HQ 子集。

策略：直接从 HuggingFace 流式加载 CCI4.0-HQ 高质量中文语料，
随机采样约 12B tokens，保存为本地 JSONL 供 PretrainDataset 使用。

Usage:
    # 默认：采样约12B tokens
    python scripts/prepare_deepsleep_data.py

    # 自定义目标 tokens 数量
    python scripts/prepare_deepsleep_data.py --target_tokens 6_000_000_000

    # 使用本地 tokenizer 精确计数（推荐）
    python scripts/prepare_deepsleep_data.py --tokenizer_path checkpoints/tokenizer

    # 指定输出路径
    python scripts/prepare_deepsleep_data.py --output data/cleaned/pretrain.jsonl
"""

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# 配置
# ============================================================

DEFAULT_DATASET = "CCI-MMC/CCI4.0-HQ"
DEFAULT_SPLIT = "train"
DEFAULT_TARGET_TOKENS = 12_000_000_000  # 12B tokens
DEFAULT_TEXT_FIELD = "text"
# 粗估：中文平均 ~1.5 chars/token (32K BPE vocab)
CHARS_PER_TOKEN_ESTIMATE = 1.5


def estimate_tokens(text: str) -> int:
    """粗估 token 数量。"""
    return int(len(text) / CHARS_PER_TOKEN_ESTIMATE)


def count_tokens_with_tokenizer(text: str, tokenizer) -> int:
    """用 tokenizer 精确计数。"""
    return len(tokenizer.encode(text))


# ============================================================
# 流式采样
# ============================================================

def sample_from_hf(
    dataset_name: str,
    output_path: Path,
    target_tokens: int,
    text_field: str = "text",
    tokenizer_path: str = None,
    seed: int = 42,
    sample_rate: float = 1.0,
):
    """从 HuggingFace 流式加载数据，随机采样到目标 token 数。"""
    from datasets import load_dataset

    count_fn = estimate_tokens
    if tokenizer_path:
        from transformers import AutoTokenizer
        logger.info("加载 tokenizer: %s", tokenizer_path)
        tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        count_fn = lambda t: count_tokens_with_tokenizer(t, tok)
        logger.info("使用 tokenizer 精确计数")

    random.seed(seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("从 HuggingFace 加载 %s (streaming)...", dataset_name)
    ds = load_dataset(dataset_name, split=DEFAULT_SPLIT, streaming=True, trust_remote_code=True)
    ds = ds.shuffle(buffer_size=10_000, seed=seed)

    total_tokens = 0
    total_docs = 0
    skipped_short = 0
    start_time = time.time()

    with open(output_path, "w", encoding="utf-8") as f:
        for item in ds:
            text = item.get(text_field, "")
            if not text or len(text.strip()) < 50:
                skipped_short += 1
                continue

            tokens = count_fn(text)
            record = {"text": text.strip()}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            total_tokens += tokens
            total_docs += 1

            if total_docs % 10000 == 0:
                elapsed = time.time() - start_time
                progress = total_tokens / target_tokens * 100
                docs_per_sec = total_docs / elapsed
                eta_min = (target_tokens - total_tokens) / max(total_tokens / elapsed, 1) / 60
                logger.info(
                    "  进度: %d docs, %.2fB/%.2fB tokens (%.1f%%), "
                    "%.0f docs/s, ETA ~%.0f min",
                    total_docs,
                    total_tokens / 1e9,
                    target_tokens / 1e9,
                    min(progress, 100),
                    docs_per_sec,
                    eta_min,
                )

            if total_tokens >= target_tokens:
                break

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("采样完成！")
    logger.info("  文档数: %d", total_docs)
    logger.info("  Token数: %.2fB", total_tokens / 1e9)
    logger.info("  跳过(过短): %d", skipped_short)
    logger.info("  耗时: %.1f 分钟", elapsed / 60)
    logger.info("  输出: %s", output_path)
    logger.info("=" * 60)

    # 保存统计信息
    stats = {
        "dataset": dataset_name,
        "total_docs": total_docs,
        "total_tokens": total_tokens,
        "total_tokens_B": round(total_tokens / 1e9, 3),
        "target_tokens_B": target_tokens / 1e9,
        "skipped_short": skipped_short,
        "elapsed_sec": round(elapsed, 1),
        "tokenizer": tokenizer_path or "char_estimate",
        "chars_per_token_estimate": CHARS_PER_TOKEN_ESTIMATE,
    }
    stats_path = output_path.parent / "pretrain_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    logger.info("统计信息: %s", stats_path)

    return stats


# ============================================================
# 数据验证
# ============================================================

def validate_data(data_path: Path, tokenizer_path: str = None, sample_n: int = 5):
    """抽样验证数据质量。"""
    import random as _r

    logger.info("验证数据: %s", data_path)
    docs = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))

    total = len(docs)
    avg_chars = sum(len(d["text"]) for d in docs) / total if total else 0

    logger.info("  总文档数: %d", total)
    logger.info("  平均字符数: %.0f", avg_chars)
    logger.info("  估算 tokens: %.2fB", total * avg_chars / CHARS_PER_TOKEN_ESTIMATE / 1e9)

    # 抽样展示
    samples = _r.sample(docs, min(sample_n, total))
    for i, s in enumerate(samples):
        text = s["text"]
        preview = text[:150].replace("\n", " ") + ("..." if len(text) > 150 else "")
        logger.info("  样本 %d: [%d chars] %s", i + 1, len(text), preview)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="DeepSleep 预训练数据准备 (CCI4.0-HQ)")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="HuggingFace dataset name")
    parser.add_argument("--output", default="data/cleaned/pretrain.jsonl", help="Output JSONL path")
    parser.add_argument("--target_tokens", type=int, default=DEFAULT_TARGET_TOKENS,
                        help="Target token count (default: 12B)")
    parser.add_argument("--text_field", default=DEFAULT_TEXT_FIELD, help="Text field name in dataset")
    parser.add_argument("--tokenizer_path", default=None,
                        help="Tokenizer path for accurate token counting (recommended)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validate", action="store_true", help="Only validate existing data")
    args = parser.parse_args()

    if args.validate:
        validate_data(Path(args.output), args.tokenizer_path)
        return

    sample_from_hf(
        dataset_name=args.dataset,
        output_path=Path(args.output),
        target_tokens=args.target_tokens,
        text_field=args.text_field,
        tokenizer_path=args.tokenizer_path,
        seed=args.seed,
    )

    validate_data(Path(args.output), args.tokenizer_path)


if __name__ == "__main__":
    main()
