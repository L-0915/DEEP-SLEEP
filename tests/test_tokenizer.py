"""Tests for tokenizer module."""

import pytest


class TestDeepSleepTokenizer:
    """Test DeepSleep tokenizer."""

    def test_special_tokens(self):
        """Test that special tokens are properly registered."""
        from src.data.tokenizer import DeepSleepTokenizer

        tokenizer = DeepSleepTokenizer.from_pretrained("src/model/")
        special_tokens = tokenizer.all_special_tokens
        assert "<|im_start|>" in special_tokens
        assert "<|im_end|>" in special_tokens
        assert "<pad>" in special_tokens

    def test_encode_decode(self):
        """Test encode-decode roundtrip."""
        from src.data.tokenizer import DeepSleepTokenizer

        tokenizer = DeepSleepTokenizer.from_pretrained("src/model/")
        text = "Sleep apnea is a common disorder."
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded, skip_special_tokens=True)
        assert decoded == text or decoded.replace(" ", "") == text.replace(" ", "")

    def test_chinese_text(self):
        """Test Chinese text tokenization."""
        from src.data.tokenizer import DeepSleepTokenizer

        tokenizer = DeepSleepTokenizer.from_pretrained("src/model/")
        text = "睡眠呼吸暂停是最常见的睡眠障碍之一"
        encoded = tokenizer.encode(text)
        assert len(encoded) > 0
        decoded = tokenizer.decode(encoded, skip_special_tokens=True)
        assert "睡眠" in decoded

    def test_chat_template(self):
        """Test chat template formatting."""
        from src.data.tokenizer import DeepSleepTokenizer

        tokenizer = DeepSleepTokenizer.from_pretrained("src/model/")
        messages = [
            {"role": "system", "content": "You are a sleep health consultant."},
            {"role": "user", "content": "I have trouble sleeping."},
        ]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False)
        assert "<|im_start|>" in formatted
        assert "system" in formatted
        assert "user" in formatted

    def test_padding(self):
        """Test padding functionality."""
        from src.data.tokenizer import DeepSleepTokenizer

        tokenizer = DeepSleepTokenizer.from_pretrained("src/model/")
        texts = ["Short text.", "This is a longer piece of text for padding test."]
        encoded = tokenizer(texts, padding=True, return_tensors="pt")
        assert encoded["input_ids"].shape[0] == 2
        assert encoded["input_ids"].shape[1] == max(len(t) for t in texts)
