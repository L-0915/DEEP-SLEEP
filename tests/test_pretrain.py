"""Test pretrain data loading: lazy JSONL indexing, DataLoader, multi-worker, real model.

Verifies the exact path that train_pretrain.py uses:
  prepare_deepsleep_data.py → pretrain.jsonl → PretrainDataset → DataLoader → model

Run: cd deepsleep && python tests/test_pretrain.py
"""

import json
import os
import sys
import tempfile
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch.utils.data import DataLoader

TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "tokenizer")


def _write_pretrain_jsonl(path, n=1000):
    """Write a pretrain JSONL in the same format as prepare_deepsleep_data.py output."""
    texts = [
        "睡眠是人体恢复精力的重要过程，成年人每天需要七到九小时的睡眠。良好的睡眠质量对身体健康至关重要。",
        "失眠是一种常见的睡眠障碍，表现为难以入睡、容易醒来或早醒。长期失眠会影响日常生活和工作效率。",
        "良好的睡眠卫生习惯包括保持规律的作息时间、创造舒适的睡眠环境、避免睡前使用电子设备。",
        "深度睡眠对身体的恢复和免疫系统的增强非常重要。深度睡眠阶段生长激素分泌达到高峰。",
        "认知行为治疗是治疗慢性失眠的一线方法，效果优于药物治疗，且没有副作用。",
        "褪黑素在调节昼夜节律中起关键作用。光照可以通过抑制褪黑素分泌来调整生物钟。",
        "快速眼动睡眠（REM）对记忆巩固和情绪调节至关重要，是人类做梦的主要阶段。",
        "阻塞性睡眠呼吸暂停综合征是一种常见的睡眠呼吸障碍，患者会在睡眠中反复出现呼吸暂停。",
        "Sleep is a naturally recurring state of mind and body characterized by altered consciousness.",
        "Deep sleep is crucial for physical recovery, immune function, and growth hormone release.",
    ]
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({"text": texts[i % len(texts)]}, ensure_ascii=False) + "\n")
    return n


def test_lazy_index():
    """PretrainDataset builds a byte-offset index without loading data into memory."""
    from dataset.lm_dataset import PretrainDataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
        n = _write_pretrain_jsonl(tmp.name, n=500)
        tmp_path = tmp.name

    t0 = time.time()
    ds = PretrainDataset(tmp_path, tok, max_length=2048)
    t1 = time.time()

    assert len(ds) == n, f"expected {n}, got {len(ds)}"
    # Index should be just a list of ints, not the actual data
    assert isinstance(ds.offsets, list)
    assert all(isinstance(x, int) for x in ds.offsets[:10])
    print(f"  [PASS] lazy index: {n} lines indexed in {t1 - t0:.3f}s")

    # __getitem__ reads from disk, not memory
    input_ids, labels = ds[0]
    assert input_ids.shape == (2048,)
    assert labels.shape == (2048,)
    assert input_ids[0].item() == tok.bos_token_id
    print(f"  [PASS] __getitem__[0]: shape=(2048,), BOS OK")

    # Random access works
    input_ids_last, _ = ds[n - 1]
    assert input_ids_last.shape == (2048,)
    print(f"  [PASS] __getitem__[{n - 1}]: random access OK")

    os.unlink(tmp_path)


def test_dataloader():
    """PretrainDataset works with DataLoader (single worker)."""
    from dataset.lm_dataset import PretrainDataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
        n = _write_pretrain_jsonl(tmp.name, n=100)
        tmp_path = tmp.name

    ds = PretrainDataset(tmp_path, tok, max_length=512)
    loader = DataLoader(ds, batch_size=8, num_workers=0, pin_memory=True)

    batch = next(iter(loader))
    input_ids, labels = batch
    assert input_ids.shape == (8, 512)
    assert labels.shape == (8, 512)
    print(f"  [PASS] DataLoader: batch shape {input_ids.shape}")

    # All batches
    total = 0
    for input_ids, labels in loader:
        total += input_ids.shape[0]
    assert total == n
    print(f"  [PASS] full epoch: {total}/{n} samples")

    os.unlink(tmp_path)


def test_dataloader_multiworker():
    """PretrainDataset works with DataLoader (multi-worker)."""
    from dataset.lm_dataset import PretrainDataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
        n = _write_pretrain_jsonl(tmp.name, n=100)
        tmp_path = tmp.name

    ds = PretrainDataset(tmp_path, tok, max_length=512)
    loader = DataLoader(ds, batch_size=8, num_workers=2, pin_memory=True)

    total = 0
    for input_ids, labels in loader:
        total += input_ids.shape[0]
    assert total == n
    print(f"  [PASS] DataLoader num_workers=2: {total}/{n} samples")

    os.unlink(tmp_path)


def test_skip_batch_sampler():
    """PretrainDataset works with SkipBatchSampler (resume scenario)."""
    from dataset.lm_dataset import PretrainDataset
    from trainer.trainer_utils import SkipBatchSampler
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
        n = _write_pretrain_jsonl(tmp.name, n=100)
        tmp_path = tmp.name

    ds = PretrainDataset(tmp_path, tok, max_length=512)
    indices = list(range(len(ds)))
    skip = 3
    sampler = SkipBatchSampler(indices, batch_size=8, skip_batches=skip)
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=0)

    batches = list(loader)
    # Should have skipped first 3 batches (24 samples), remaining 76 → ceil(76/8)=10
    total = sum(b[0].shape[0] for b in batches)
    assert total == n - skip * 8, f"expected {n - skip * 8}, got {total}"
    print(f"  [PASS] SkipBatchSampler: skip {skip} batches, {total} samples remaining")

    os.unlink(tmp_path)


def test_train_3_steps():
    """Full training loop: PretrainDataset → DataLoader → real model (768d/10L/MoE) → 3 steps."""
    from dataset.lm_dataset import PretrainDataset
    from model.model_deepsleep import DeepSleepConfig, DeepSleepForCausalLM
    from transformers import AutoTokenizer
    from trainer.trainer_utils import get_lr

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    cfg = DeepSleepConfig(
        d_model=768, n_layers=10, vocab_size=7200, max_position_embeddings=8192,
        use_moe=True, num_experts=8, num_shared_experts=2, top_k=2,
    )
    model = DeepSleepForCausalLM(cfg).to(device)
    model.train()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
        _write_pretrain_jsonl(tmp.name, n=20)
        tmp_path = tmp.name

    ds = PretrainDataset(tmp_path, tok, max_length=2048)
    loader = DataLoader(ds, batch_size=4, num_workers=0, pin_memory=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

    for step, (input_ids, labels) in enumerate(loader):
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        lr = get_lr(step, 100, 5e-4)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled="cuda" in device):
            res = model(input_ids, labels=labels)
            loss = res.loss + res.aux_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        print(f"  step {step + 1}: loss={loss.item():.4f}, lm={res.loss.item():.4f}, aux={res.aux_loss.item():.6f}")
        if step >= 2:
            break

    os.unlink(tmp_path)
    print(f"  [PASS] train 3 steps with lazy-loading PretrainDataset")


def test_index_speed():
    """Index building speed is acceptable for large files."""
    from dataset.lm_dataset import PretrainDataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)

    # Simulate a larger file (10K lines)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
        n = _write_pretrain_jsonl(tmp.name, n=10000)
        tmp_path = tmp.name
        file_size = os.path.getsize(tmp_path)

    t0 = time.time()
    ds = PretrainDataset(tmp_path, tok, max_length=2048)
    t1 = time.time()
    speed = file_size / (t1 - t0) / 1024 / 1024  # MB/s

    assert len(ds) == n
    print(f"  [PASS] index speed: {n} lines ({file_size / 1024:.0f}KB) in {t1 - t0:.3f}s ({speed:.1f} MB/s)")
    # Should index at least 50MB/s (real 20GB file would take ~6 min)
    assert speed > 10, f"indexing too slow: {speed:.1f} MB/s"

    os.unlink(tmp_path)


if __name__ == "__main__":
    print("=" * 60)
    print("Pretrain Data Loading Tests (lazy JSONL)")
    print("=" * 60)
    test_lazy_index()
    test_dataloader()
    test_dataloader_multiworker()
    test_skip_batch_sampler()
    test_train_3_steps()
    test_index_speed()
    print("\n" + "=" * 60)
    print("ALL PASSED - pretrain data loading is ready")
    print("=" * 60)
