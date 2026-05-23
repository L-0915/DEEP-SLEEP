"""Test training utilities: LR schedule, init_model, checkpoint, SkipBatchSampler.

Run: cd deepsleep && python tests/test_training.py
"""

import os
import sys
import tempfile

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch

TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "tokenizer")


def test_lr_schedule():
    from trainer.trainer_utils import get_lr
    total = 1000
    lr = 5e-4
    # Warmup: first 10% = 100 steps
    assert get_lr(0, total, lr) < get_lr(50, total, lr) < get_lr(100, total, lr)
    assert abs(get_lr(100, total, lr) - lr) < 1e-6
    # Decay after warmup
    assert get_lr(500, total, lr) < lr
    assert get_lr(999, total, lr) < get_lr(500, total, lr)
    print("  [PASS] LR warmup + cosine decay")


def test_init_model():
    """init_model creates real model with real tokenizer on GPU."""
    from model.model_deepsleep import DeepSleepConfig
    from trainer.trainer_utils import init_model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg = DeepSleepConfig(d_model=768, n_layers=10, vocab_size=7200)
    model, tokenizer = init_model(cfg, "none", TOKENIZER_PATH, device)
    assert model is not None
    assert tokenizer is not None
    assert tokenizer.vocab_size == 7200
    total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  [PASS] init_model: ~{total:.1f}M params, tokenizer OK")


def test_checkpoint_real():
    """Checkpoint save/load with real production model."""
    from model.model_deepsleep import DeepSleepConfig, DeepSleepForCausalLM
    from trainer.trainer_utils import lm_checkpoint
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg = DeepSleepConfig(d_model=768, n_layers=10, vocab_size=7200, use_moe=True,
                          num_experts=8, num_shared_experts=2)
    model = DeepSleepForCausalLM(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    with tempfile.TemporaryDirectory() as tmpdir:
        lm_checkpoint(cfg, model=model, optimizer=optimizer, epoch=0, step=100,
                      save_dir=tmpdir, weight="test")
        ckp = os.path.join(tmpdir, "test_768_moe.pth")
        assert os.path.exists(ckp)
        size = os.path.getsize(ckp) / 1024 / 1024
        data = lm_checkpoint(cfg, save_dir=tmpdir, weight="test")
        assert data["epoch"] == 0
        assert data["step"] == 100
        model2 = DeepSleepForCausalLM(cfg).to(device)
        model2.load_state_dict(data["model"], strict=False)
    print(f"  [PASS] checkpoint: save/load {size:.0f}MB with real model")


def test_skip_batch_sampler():
    from trainer.trainer_utils import SkipBatchSampler
    indices = list(range(100))
    sampler = SkipBatchSampler(indices, batch_size=8, skip_batches=3)
    batches = list(sampler)
    assert len(batches) == 100 // 8 - 3 + 1  # ceil(100/8)=13, skip 3, but 13*8>100 so last batch smaller
    assert batches[0] == [24, 25, 26, 27, 28, 29, 30, 31]
    print(f"  [PASS] SkipBatchSampler: skip=3, bs=8, got {len(batches)} batches")


if __name__ == "__main__":
    print("=" * 60)
    print("Training Utils Tests")
    print("=" * 60)
    test_lr_schedule()
    test_init_model()
    test_checkpoint_real()
    test_skip_batch_sampler()
    print("\n" + "=" * 60)
    print("ALL PASSED")
    print("=" * 60)
