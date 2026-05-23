"""Train BPE tokenizer for DeepSleep.

Supports local data (SFT/DPO/PubMedQA) and CCI4.0-HQ streaming data
for bilingual Chinese+English coverage.

Usage:
    # Default: local data only
    python trainer/train_tokenizer.py

    # With CCI4.0-HQ streaming data (recommended for pretrain)
    python trainer/train_tokenizer.py --use_cci4 --cci4_max_docs 300000

    # Custom vocab size
    python trainer/train_tokenizer.py --use_cci4 --vocab_size 7200
"""

import argparse
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from tokenizers import Tokenizer, trainers, pre_tokenizers, decoders
from tokenizers.models import BPE

SPECIAL_TOKENS = [
    "<pad>",
    "<|im_start|>",
    "<|im_end|>",
    "<unk>",
    "<s>",
    "</s>",
    "<thinking>",
    "</thinking>",
    "<summary>",
]

CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "<|im_start|>assistant\n"
    "{% endif %}"
)


def collect_local_texts():
    """Collect training corpus from local SFT, DPO, and PubMedQA data."""
    texts = []
    data_files = [
        "data/sft/xiaoxi/xiaoxi_sft.jsonl",
        "data/dpo/xiaoxi_dpo.jsonl",
        "data/sft/xiaoxi/all_prompts.jsonl",
    ]
    for path in data_files:
        try:
            with open(path, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    sample = json.loads(line)
                    for key in ["text", "chosen", "rejected", "prompt"]:
                        if key in sample and isinstance(sample[key], str):
                            texts.append(sample[key])
                    for key in ["conversations", "messages"]:
                        if key in sample:
                            for msg in sample[key]:
                                if isinstance(msg, dict) and "content" in msg:
                                    texts.append(msg["content"])
        except FileNotFoundError:
            print(f"  Skip {path}: not found")

    # PubMedQA English medical text
    pqal_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "pubmedqa", "data", "ori_pqal.json"
    )
    try:
        pqal = json.load(open(pqal_path))
        for v in pqal.values():
            texts.append(v.get("QUESTION", ""))
            for ctx in v.get("CONTEXTS", []):
                texts.append(ctx)
            if isinstance(v.get("reasoning"), str):
                texts.append(v["reasoning"])
            texts.append(v.get("answer", ""))
        print(f"  Added PubMedQA: {len(pqal)} entries")
    except FileNotFoundError:
        print(f"  Skip PubMedQA: not found")

    return texts


def collect_cci4_texts(max_docs=300_000, min_length=50, seed=42):
    """Collect bilingual texts from CCI4.0-HQ via HuggingFace streaming.

    CCI4.0-HQ contains high-quality Chinese and English web text.
    We stream and collect up to max_docs documents for tokenizer training.
    """
    from datasets import load_dataset

    print(f"  Loading CCI4.0-HQ (streaming, collecting {max_docs} docs)...")
    ds = load_dataset(
        "CCI-MMC/CCI4.0-HQ", split="train", streaming=True, trust_remote_code=True
    )
    ds = ds.shuffle(buffer_size=10_000, seed=seed)

    texts = []
    skipped = 0
    for i, item in enumerate(ds):
        text = item.get("text", "")
        if not text or len(text.strip()) < min_length:
            skipped += 1
            continue
        texts.append(text.strip())

        if len(texts) % 50_000 == 0 and len(texts) > 0:
            print(f"    Collected {len(texts)}/{max_docs} docs (skipped {skipped})...")

        if len(texts) >= max_docs:
            break

    print(f"  CCI4.0-HQ: {len(texts)} docs collected, {skipped} skipped")
    return texts


def write_tokenizer_config(output_dir):
    """Write tokenizer_config.json for HuggingFace AutoTokenizer compatibility."""
    added_tokens = {}
    for i, tok in enumerate(SPECIAL_TOKENS):
        added_tokens[str(i)] = {
            "content": tok,
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True,
        }

    config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": False,
        "added_tokens_decoder": added_tokens,
        "bos_token": "<|im_start|>",
        "eos_token": "<|im_end|>",
        "pad_token": "<pad>",
        "unk_token": "<unk>",
        "model_max_length": 8192,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "chat_template": CHAT_TEMPLATE,
    }
    path = os.path.join(output_dir, "tokenizer_config.json")
    with open(path, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"  Saved {path}")


def main():
    parser = argparse.ArgumentParser(description="Train DeepSleep BPE tokenizer")
    parser.add_argument("--vocab_size", type=int, default=7200)
    parser.add_argument("--min_frequency", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="checkpoints/tokenizer")
    parser.add_argument("--use_cci4", action="store_true",
                        help="Include CCI4.0-HQ streaming data for bilingual coverage")
    parser.add_argument("--cci4_max_docs", type=int, default=300_000,
                        help="Max docs to collect from CCI4.0-HQ (default: 300K)")
    args = parser.parse_args()

    print("Collecting training corpus...")
    texts = collect_local_texts()
    print(f"  Local data: {len(texts)} segments")

    if args.use_cci4:
        cci4_texts = collect_cci4_texts(max_docs=args.cci4_max_docs)
        texts.extend(cci4_texts)
        print(f"  Total (local + CCI4.0): {len(texts)} segments")
    else:
        print(f"  Total: {len(texts)} segments")

    print(f"\nTraining BPE tokenizer (vocab_size={args.vocab_size})...")
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(
            r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""",
            "isolated",
            invert=False,
        ),
    ])
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
        min_frequency=args.min_frequency,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.decoder = decoders.BPEDecoder()

    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer_path = os.path.join(args.output_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)
    print(f"  Saved {tokenizer_path}")
    print(f"  Vocab size: {tokenizer.get_vocab_size()}")

    write_tokenizer_config(args.output_dir)

    print("\nTest encoding:")
    for text in [
        "睡眠是人类恢复精力的重要过程。",
        "Sleep is essential for health.",
        "<thinking>分析症状</thinking>",
    ]:
        enc = tokenizer.encode(text)
        decoded = tokenizer.decode(enc.ids)
        print(f'  "{text}" -> {len(enc.ids)} tokens -> "{decoded}"')


if __name__ == "__main__":
    main()
