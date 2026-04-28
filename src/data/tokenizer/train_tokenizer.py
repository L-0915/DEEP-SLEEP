"""BPE tokenizer training for the DeepSleep sleep-medicine LLM.

Trains a byte-level BPE tokenizer on mixed Chinese-English corpus data,
adds special tokens for ChatML formatting, and evaluates compression
performance on held-out data.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Special tokens used by the DeepSleep model (ChatML template).
SPECIAL_TOKENS: list[str] = [
    "<pad>",
    "<unk>",
    "<s>",
    "</s>",
    "<|im_start|>",
    "<|im_end|>",
    "<|assistant|",
    "<|tool_call|>",
    "<|tool_response|>",
    "<|thinking|>",
    "<|summary|>",
]

# Special tokens configuration for the tokenizer_config.json.
_SPECIAL_TOKENS_MAP = {
    "bos_token": "<|im_start|>",
    "eos_token": "<|im_end|>",
    "pad_token": "<pad>",
    "unk_token": "<unk>",
    "additional_special_tokens": [
        "<s>", "</s>",
        "<|assistant|", "<|tool_call|>", "<|tool_response|>",
        "<|thinking|>", "<|summary|>",
    ],
}


def train_bpe_tokenizer(
    input_files: list[str],
    vocab_size: int = 64000,
    output_dir: str = "src/model/",
) -> dict[str, Any]:
    """Train a byte-level BPE tokenizer on mixed Chinese-English corpus.

    Uses the ``tokenizers`` library (HuggingFace) to train a fast BPE tokenizer
    that handles both Chinese characters and English text efficiently.

    Args:
        input_files: List of paths to text files used as training corpus.
        vocab_size: Target vocabulary size.
        output_dir: Directory where tokenizer files will be saved.

    Returns:
        Dictionary with training statistics.
    """
    try:
        from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, normalizers
        from tokenizers.models import BPE
    except ImportError as exc:
        raise ImportError(
            "The 'tokenizers' package is required.  Install it with: "
            "pip install tokenizers"
        ) from exc

    # Validate input files
    valid_paths: list[Path] = []
    for fp in input_files:
        p = Path(fp)
        if p.exists():
            valid_paths.append(p)
        else:
            logger.warning("Training file not found, skipping: %s", fp)

    if not valid_paths:
        raise FileNotFoundError(
            f"No valid training files found among: {input_files}"
        )

    logger.info(
        "Training BPE tokenizer: vocab_size=%d, files=%d",
        vocab_size,
        len(valid_paths),
    )

    # Initialize BPE tokenizer
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))

    # Normalizer: NFC unicode normalization + lowercase for ASCII only
    from tokenizers.normalizers import NFD, Lowercase, StripAccents, Sequence

    tokenizer.normalizer = Sequence([
        NFD(),
        StripAccents(),
    ])

    # Pre-tokenizer: split on whitespace and punctuation
    from tokenizers.pre_tokenizers import ByteLevel, WhitespaceSplit

    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

    # Decoder
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder

    tokenizer.decoder = ByteLevelDecoder()

    # Trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
        show_progress=True,
    )

    # Train
    tokenizer.train(
        files=[str(p) for p in valid_paths],
        trainer=trainer,
    )

    # Enable padding
    tokenizer.enable_padding(pad_id=SPECIAL_TOKENS.index("<pad>"), pad_token="<pad>")

    # Save
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tokenizer.save(str(out / "tokenizer.json"))

    # Also save in legacy HuggingFace format for compatibility
    _save_legacy_format(tokenizer, vocab_size, out)

    stats = {
        "vocab_size": tokenizer.get_vocab_size(),
        "output_dir": str(out),
        "training_files": [str(p) for p in valid_paths],
    }

    logger.info("Tokenizer trained and saved to %s (vocab_size=%d)", out, vocab_size)
    return stats


def test_tokenizer(
    tokenizer_path: str,
    test_files: list[str],
) -> dict[str, Any]:
    """Evaluate tokenizer performance on held-out data.

    Computes vocabulary coverage, compression ratio, and average tokens
    per character for Chinese and English text separately.

    Args:
        tokenizer_path: Path to the tokenizer.json file.
        test_files: List of paths to test text files.

    Returns:
        Dictionary with compression statistics.
    """
    from tokenizers import Tokenizer

    tok_path = Path(tokenizer_path)
    if not tok_path.exists():
        # Try the output directory
        tok_path = Path(tokenizer_path) / "tokenizer.json"

    if not tok_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

    tokenizer = Tokenizer.from_file(str(tok_path))
    vocab = tokenizer.get_vocab()

    zh_stats = {"chars": 0, "tokens": 0, "files": 0}
    en_stats = {"chars": 0, "tokens": 0, "files": 0}

    for fp_str in test_files:
        fp = Path(fp_str)
        if not fp.exists():
            logger.warning("Test file not found: %s", fp_str)
            continue

        with open(fp, "r", encoding="utf-8") as fh:
            text = fh.read()

        encoding = tokenizer.encode(text)
        token_ids = encoding.ids

        # Classify as Chinese or English by dominant script
        zh_count = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
        en_count = sum(1 for c in text if c.isascii() and c.isalpha())
        stats = zh_stats if zh_count > en_count else en_stats
        stats["chars"] += len(text)
        stats["tokens"] += len(token_ids)
        stats["files"] += 1

    def _compute_ratio(s: dict[str, int]) -> dict[str, float]:
        if s["chars"] == 0:
            return {"chars": 0, "tokens": 0, "compression_ratio": 0.0, "tokens_per_char": 0.0}
        cr = s["chars"] / s["tokens"] if s["tokens"] > 0 else 0.0
        tpc = s["tokens"] / s["chars"] if s["chars"] > 0 else 0.0
        return {
            "chars": s["chars"],
            "tokens": s["tokens"],
            "files": s["files"],
            "compression_ratio": round(cr, 3),
            "tokens_per_char": round(tpc, 4),
        }

    results = {
        "vocab_size": len(vocab),
        "chinese": _compute_ratio(zh_stats),
        "english": _compute_ratio(en_stats),
        "overall": _compute_ratio({
            "chars": zh_stats["chars"] + en_stats["chars"],
            "tokens": zh_stats["tokens"] + en_stats["tokens"],
            "files": zh_stats["files"] + en_stats["files"],
        }),
    }

    # Print summary
    logger.info("Tokenizer test results:")
    logger.info("  Vocab size: %d", results["vocab_size"])
    for lang in ("chinese", "english", "overall"):
        s = results[lang]
        logger.info(
            "  %s: %d tokens / %d chars, ratio=%.2f, tokens/char=%.4f",
            lang,
            s["tokens"],
            s["chars"],
            s["compression_ratio"],
            s["tokens_per_char"],
        )

    return results


# ---------------------------------------------------------------------------
# Legacy format helpers
# ---------------------------------------------------------------------------

def _save_legacy_format(tokenizer: Any, vocab_size: int, output_dir: Path) -> None:
    """Save tokenizer in legacy HuggingFace format (vocab.json, merges.txt).

    This is needed for compatibility with PreTrainedTokenizerFast.
    """
    vocab = tokenizer.get_vocab()

    # Build vocab.json: token -> id
    vocab_json = {token: idx for token, idx in sorted(vocab.items(), key=lambda x: x[1])}
    with open(output_dir / "vocab.json", "w", encoding="utf-8") as fh:
        json.dump(vocab_json, fh, ensure_ascii=False, indent=2)

    # Build merges.txt
    with open(output_dir / "merges.txt", "w", encoding="utf-8") as fh:
        fh.write("#version: 0.2\n")
        # Extract merges from the BPE model
        model = tokenizer.get_model()
        if hasattr(model, "merges"):
            for merge in model.merges:
                fh.write(" ".join(merge) + "\n")

    # Save tokenizer_config.json
    config = {
        "tokenizer_class": "DeepSleepTokenizer",
        "model_max_length": 8192,
        "vocab_size": vocab_size,
        **_SPECIAL_TOKENS_MAP,
    }
    with open(output_dir / "tokenizer_config.json", "w", encoding="utf-8") as fh:
        json.dump(config, fh, ensure_ascii=False, indent=2)

    # Save special_tokens_map.json
    with open(output_dir / "special_tokens_map.json", "w", encoding="utf-8") as fh:
        json.dump(_SPECIAL_TOKENS_MAP, fh, ensure_ascii=False, indent=2)

    logger.info("Legacy tokenizer files saved to %s", output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Train or test the DeepSleep BPE tokenizer"
    )
    sub = parser.add_subparsers(dest="command", help="Command to run")

    # train subcommand
    train_parser = sub.add_parser("train", help="Train a new BPE tokenizer")
    train_parser.add_argument(
        "--input-files", nargs="+", required=True,
        help="Paths to training text files",
    )
    train_parser.add_argument(
        "--vocab-size", type=int, default=64000,
        help="Target vocabulary size (default: 64000)",
    )
    train_parser.add_argument(
        "--output-dir", type=str, default="src/model/",
        help="Output directory for tokenizer files (default: src/model/)",
    )

    # test subcommand
    test_parser = sub.add_parser("test", help="Test tokenizer performance")
    test_parser.add_argument(
        "--tokenizer-path", type=str, required=True,
        help="Path to tokenizer.json or directory",
    )
    test_parser.add_argument(
        "--test-files", nargs="+", required=True,
        help="Paths to test text files",
    )

    args = parser.parse_args()

    if args.command == "train":
        stats = train_bpe_tokenizer(
            input_files=args.input_files,
            vocab_size=args.vocab_size,
            output_dir=args.output_dir,
        )
        print(json.dumps(stats, indent=2))

    elif args.command == "test":
        results = test_tokenizer(
            tokenizer_path=args.tokenizer_path,
            test_files=args.test_files,
        )
        print(json.dumps(results, indent=2))

    else:
        parser.print_help()
        sys.exit(1)
