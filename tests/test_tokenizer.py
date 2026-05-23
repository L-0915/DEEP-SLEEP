"""Test tokenizer: load, special tokens, encode/decode, chat template, dataset compatibility.

Uses the real tokenizer at checkpoints/tokenizer.
Run: cd deepsleep && python tests/test_tokenizer.py
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "tokenizer")


def test_load():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    assert tok.vocab_size == 7200
    assert tok.bos_token_id == 1
    assert tok.eos_token_id == 2
    assert tok.pad_token_id == 0
    assert tok.unk_token_id is not None
    print(f"  [PASS] load: vocab={tok.vocab_size}, bos={tok.bos_token_id}, eos={tok.eos_token_id}, pad={tok.pad_token_id}")


def test_special_tokens():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    for name, expected_id in [
        ("<pad>", 0), ("<|im_start|>", 1), ("<|im_end|>", 2), ("<unk>", 3),
        ("<thinking>", 6), ("</thinking>", 7), ("<summary>", 8),
    ]:
        actual = tok.convert_tokens_to_ids(name)
        assert actual == expected_id, f"{name}: expected {expected_id}, got {actual}"
    print("  [PASS] all special tokens have correct ids")


def test_encode_decode():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    for text in [
        "睡眠是人类恢复精力的重要过程。",
        "Sleep is essential for health.",
        "阻塞性睡眠呼吸暂停综合征是一种常见的睡眠呼吸障碍。",
    ]:
        ids = tok.encode(text, add_special_tokens=False)
        decoded = tok.decode(ids, skip_special_tokens=True)
        assert decoded == text, f"roundtrip failed: '{decoded}' != '{text}'"
    # Special tokens are stripped by skip_special_tokens
    ids = tok.encode("<thinking>患者主诉失眠</thinking>", add_special_tokens=False)
    decoded = tok.decode(ids, skip_special_tokens=True)
    assert "患者主诉失眠" in decoded
    print("  [PASS] encode/decode roundtrip (CN/EN/special tokens)")


def test_chat_template():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    msgs = [
        {"role": "system", "content": "你是小曦"},
        {"role": "user", "content": "我失眠了怎么办"},
        {"role": "assistant", "content": "别担心，我来帮你分析"},
    ]
    result = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    assert "<|im_start|>system\n你是小曦<|im_end|>" in result
    assert "<|im_start|>user\n我失眠了怎么办<|im_end|>" in result
    assert "<|im_start|>assistant\n别担心，我来帮你分析<|im_end|>" in result

    result_gen = tok.apply_chat_template(msgs[:2], tokenize=False, add_generation_prompt=True)
    assert result_gen.endswith("<|im_start|>assistant\n")
    print("  [PASS] chat template")


def test_tokenizer_with_model_vocab():
    """Verify tokenizer vocab_size matches model default."""
    from transformers import AutoTokenizer
    from model.model_deepsleep import DeepSleepConfig
    tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    cfg = DeepSleepConfig()
    assert tok.vocab_size <= cfg.vocab_size, f"tokenizer vocab {tok.vocab_size} > model vocab {cfg.vocab_size}"
    print(f"  [PASS] tokenizer vocab ({tok.vocab_size}) <= model vocab ({cfg.vocab_size})")


def test_batch_padding():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    texts = ["短文本", "这是一段比较长的文本用于测试填充功能"]
    encoded = tok(texts, padding=True, return_tensors="pt")
    assert encoded["input_ids"].shape[0] == 2
    assert encoded["input_ids"].shape[1] == encoded["attention_mask"].shape[1]
    print("  [PASS] batch padding")


if __name__ == "__main__":
    print("=" * 60)
    print("Tokenizer Tests (real checkpoints/tokenizer)")
    print("=" * 60)
    test_load()
    test_special_tokens()
    test_encode_decode()
    test_chat_template()
    test_tokenizer_with_model_vocab()
    test_batch_padding()
    print("\n" + "=" * 60)
    print("ALL PASSED")
    print("=" * 60)
