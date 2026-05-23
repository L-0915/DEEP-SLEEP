"""Test all dataset classes with REAL data and REAL tokenizer.

PretrainDataset: tests dummy pretrain.jsonl (same format as prepare_deepsleep_data.py output)
SFTDataset: tests real data/sft/xiaoxi/xiaoxi_sft.jsonl
DPODataset: tests real data/dpo/xiaoxi_dpo.jsonl
Run: cd deepsleep && python tests/test_data_pipeline.py
"""

import json
import os
import sys
import tempfile

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
from transformers import AutoTokenizer

TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "tokenizer")
SFT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sft", "xiaoxi", "xiaoxi_sft.jsonl")
DPO_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "dpo", "xiaoxi_dpo.jsonl")


def _tok():
    return AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)


def _dummy_pretrain_jsonl(path, n=20):
    """Write dummy pretrain data in the same format as prepare_deepsleep_data.py output."""
    texts = [
        "睡眠是人体恢复精力的重要过程，成年人每天需要七到九小时的睡眠。",
        "失眠是一种常见的睡眠障碍，表现为难以入睡、容易醒来或早醒。",
        "良好的睡眠卫生习惯包括保持规律的作息时间、创造舒适的睡眠环境。",
        "认知行为治疗是治疗慢性失眠的一线方法。",
        "Sleep is essential for human recovery and health.",
    ] * (n // 5 + 1)
    with open(path, "w") as f:
        for t in texts[:n]:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
    return n


def test_pretrain_dataset_load():
    """PretrainDataset loads the same format prepare_deepsleep_data.py produces."""
    from dataset.lm_dataset import PretrainDataset
    tok = _tok()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
        n = _dummy_pretrain_jsonl(tmp.name)
        tmp_path = tmp.name
    ds = PretrainDataset(tmp_path, tok, max_length=2048)
    assert len(ds) == n, f"expected {n} samples, got {len(ds)}"
    input_ids, labels = ds[0]
    assert input_ids.shape == (2048,)
    assert labels.shape == (2048,)
    # First token should be BOS
    assert input_ids[0].item() == tok.bos_token_id
    # Padded positions should have label -100
    pad_mask = input_ids == tok.pad_token_id
    assert (labels[pad_mask] == -100).all()
    os.unlink(tmp_path)
    print(f"  [PASS] PretrainDataset: {n} samples, shape=(2048,), BOS + padding OK")


def test_pretrain_max_length():
    """PretrainDataset truncates long text to max_length."""
    from dataset.lm_dataset import PretrainDataset
    tok = _tok()
    long_text = "睡眠很重要。" * 500
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
        json.dump({"text": long_text}, tmp)
        tmp.write("\n")
        tmp_path = tmp.name
    ds = PretrainDataset(tmp_path, tok, max_length=512)
    input_ids, _ = ds[0]
    assert input_ids.shape[0] == 512
    os.unlink(tmp_path)
    print("  [PASS] PretrainDataset: truncation to 512 OK")


def test_pretrain_with_real_config():
    """PretrainDataset output can feed into real model."""
    from dataset.lm_dataset import PretrainDataset
    from model.model_deepsleep import DeepSleepConfig, DeepSleepForCausalLM
    tok = _tok()
    cfg = DeepSleepConfig(vocab_size=7200, max_position_embeddings=8192)
    model = DeepSleepForCausalLM(cfg).to("cuda:0" if torch.cuda.is_available() else "cpu")
    device = next(model.parameters()).device
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
        _dummy_pretrain_jsonl(tmp.name, n=4)
        tmp_path = tmp.name
    ds = PretrainDataset(tmp_path, tok, max_length=512)
    input_ids, labels = ds[0]
    input_ids = input_ids.unsqueeze(0).to(device)
    labels = labels.unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        out = model(input_ids, labels=labels)
    assert out.loss is not None and out.loss.item() > 0
    print(f"  [PASS] PretrainDataset → real model: loss={out.loss.item():.4f}")
    os.unlink(tmp_path)


def test_sft_dataset():
    """SFTDataset loads real xiaoxi_sft.jsonl."""
    from dataset.lm_dataset import SFTDataset
    tok = _tok()
    ds = SFTDataset(SFT_PATH, tok, max_length=2048)
    assert len(ds) > 0, "SFT dataset is empty"
    input_ids, labels = ds[0]
    assert input_ids.shape[0] == 2048
    assert labels.shape[0] == 2048
    # Some labels should be non -100 (assistant response part)
    assert (labels != -100).any(), "no trainable labels found"
    print(f"  [PASS] SFTDataset: {len(ds)} samples from real file, labels OK")


def test_sft_with_model():
    """SFTDataset output can feed into real model."""
    from dataset.lm_dataset import SFTDataset
    from model.model_deepsleep import DeepSleepConfig, DeepSleepForCausalLM
    tok = _tok()
    cfg = DeepSleepConfig(vocab_size=7200, max_position_embeddings=8192)
    model = DeepSleepForCausalLM(cfg).to("cuda:0" if torch.cuda.is_available() else "cpu")
    device = next(model.parameters()).device
    ds = SFTDataset(SFT_PATH, tok, max_length=512)
    input_ids, labels = ds[0]
    input_ids = input_ids.unsqueeze(0).to(device)
    labels = labels.unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        out = model(input_ids, labels=labels)
    assert out.loss is not None and out.loss.item() > 0
    print(f"  [PASS] SFTDataset → real model: loss={out.loss.item():.4f}")


def test_dpo_dataset():
    """DPODataset loads real xiaoxi_dpo.jsonl."""
    from dataset.lm_dataset import DPODataset
    tok = _tok()
    ds = DPODataset(DPO_PATH, tok, max_length=4096)
    assert len(ds) > 0, "DPO dataset is empty"
    sample = ds[0]
    for key in ["x_chosen", "y_chosen", "mask_chosen", "x_rejected", "y_rejected", "mask_rejected"]:
        assert key in sample, f"missing key: {key}"
        assert sample[key].dtype == torch.long
    # x and y should differ by one position (shifted)
    assert sample["x_chosen"].shape[0] == sample["y_chosen"].shape[0]
    print(f"  [PASS] DPODataset: {len(ds)} samples from real file, shapes OK")


def test_dpo_with_model():
    """DPODataset chosen/rejected can both feed into real model."""
    from dataset.lm_dataset import DPODataset
    from model.model_deepsleep import DeepSleepConfig, DeepSleepForCausalLM
    tok = _tok()
    cfg = DeepSleepConfig(vocab_size=7200, max_position_embeddings=8192)
    model = DeepSleepForCausalLM(cfg).to("cuda:0" if torch.cuda.is_available() else "cpu")
    device = next(model.parameters()).device
    ds = DPODataset(DPO_PATH, tok, max_length=2048)
    sample = ds[0]
    model.eval()
    with torch.no_grad():
        x_c = sample["x_chosen"].unsqueeze(0).to(device)
        y_c = sample["y_chosen"].unsqueeze(0).to(device)
        out_c = model(x_c, labels=y_c)
        x_r = sample["x_rejected"].unsqueeze(0).to(device)
        y_r = sample["y_rejected"].unsqueeze(0).to(device)
        out_r = model(x_r, labels=y_r)
    assert out_c.loss is not None and out_r.loss is not None
    print(f"  [PASS] DPODataset → real model: chosen_loss={out_c.loss.item():.4f}, rejected_loss={out_r.loss.item():.4f}")


if __name__ == "__main__":
    print("=" * 60)
    print("Data Pipeline Tests (real tokenizer + real data files)")
    print("=" * 60)
    test_pretrain_dataset_load()
    test_pretrain_max_length()
    test_pretrain_with_real_config()
    test_sft_dataset()
    test_sft_with_model()
    test_dpo_dataset()
    test_dpo_with_model()
    print("\n" + "=" * 60)
    print("ALL PASSED")
    print("=" * 60)
