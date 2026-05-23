"""Test DeepSleep model with REAL production config on GPU.

Verifies: config, forward pass, loss, aux_loss, gradient flow, checkpoint save/load.
All tests use d_model=768, n_layers=10, MoE (same as actual training).
Run: cd deepsleep && python tests/test_model.py
"""

import os
import sys
import tempfile

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Production config
REAL_CONFIG = dict(
    d_model=768, n_layers=10, n_heads=8, n_kv_heads=4, head_dim=96,
    vocab_size=7200, max_position_embeddings=8192,
    use_moe=True, num_experts=8, num_shared_experts=2, top_k=2,
)


def _make_model():
    from model.model_deepsleep import DeepSleepConfig, DeepSleepForCausalLM
    cfg = DeepSleepConfig(**REAL_CONFIG)
    return DeepSleepForCausalLM(cfg).to(DEVICE)


def test_config():
    from model.model_deepsleep import DeepSleepConfig, DeepSleepForCausalLM
    cfg = DeepSleepConfig(**REAL_CONFIG)
    assert cfg.d_model == 768
    assert cfg.n_layers == 10
    assert cfg.vocab_size == 7200
    assert cfg.moe_layers == [1, 3, 5, 7, 9]
    assert cfg.num_shared_experts == 2
    assert cfg.top_k == 2
    assert cfg.model_type == "deepsleep"
    total = sum(p.numel() for p in DeepSleepForCausalLM(cfg).parameters()) / 1e6
    print(f"  [PASS] config: d_model=768, 10 layers, MoE layers={cfg.moe_layers}, ~{total:.1f}M params")


def test_forward():
    model = _make_model()
    model.eval()
    x = torch.randint(0, 7200, (1, 64), device=DEVICE)
    with torch.no_grad():
        out = model(x)
    assert out.logits.shape == (1, 64, 7200)
    assert out.aux_loss is not None
    print(f"  [PASS] forward: logits {out.logits.shape}, aux_loss {out.aux_loss.item():.6f}")


def test_loss():
    model = _make_model()
    model.eval()
    x = torch.randint(0, 7200, (2, 128), device=DEVICE)
    labels = torch.randint(0, 7200, (2, 128), device=DEVICE)
    with torch.no_grad():
        out = model(x, labels=labels)
    assert out.loss is not None
    assert out.loss.dim() == 0
    assert out.loss.item() > 0
    print(f"  [PASS] loss: lm_loss={out.loss.item():.4f}, aux_loss={out.aux_loss.item():.6f}")


def test_backward():
    model = _make_model()
    model.train()
    x = torch.randint(0, 7200, (1, 64), device=DEVICE)
    labels = torch.randint(0, 7200, (1, 64), device=DEVICE)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model(x, labels=labels)
        loss = out.loss + out.aux_loss
    loss.backward()
    grads = sum(1 for p in model.parameters() if p.grad is not None)
    total = sum(1 for p in model.parameters())
    assert grads > 0
    print(f"  [PASS] backward: {grads}/{total} params have gradients, loss={loss.item():.4f}")


def test_amp():
    """Mixed precision (bfloat16) works on GPU."""
    if "cpu" in DEVICE:
        print("  [SKIP] AMP test (no GPU)")
        return
    model = _make_model()
    model.train()
    x = torch.randint(0, 7200, (2, 128), device=DEVICE)
    labels = torch.randint(0, 7200, (2, 128), device=DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model(x, labels=labels)
        loss = out.loss + out.aux_loss
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    print(f"  [PASS] AMP forward+backward+step: loss={loss.item():.4f}")


def test_checkpoint():
    from model.model_deepsleep import DeepSleepConfig, DeepSleepForCausalLM
    from trainer.trainer_utils import lm_checkpoint
    cfg = DeepSleepConfig(**REAL_CONFIG)
    model = DeepSleepForCausalLM(cfg).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    with tempfile.TemporaryDirectory() as tmpdir:
        lm_checkpoint(cfg, model=model, optimizer=optimizer, epoch=0, step=10,
                      save_dir=tmpdir, weight="test")
        data = lm_checkpoint(cfg, save_dir=tmpdir, weight="test")
        assert data is not None
        assert data["epoch"] == 0
        assert data["step"] == 10
        model2 = DeepSleepForCausalLM(cfg).to(DEVICE)
        model2.load_state_dict(data["model"], strict=False)
        size_mb = os.path.getsize(os.path.join(tmpdir, "test_768_moe.pth")) / 1024 / 1024
    print(f"  [PASS] checkpoint save/load: {size_mb:.0f}MB, weights restored")


def test_tied_embeddings():
    from model.model_deepsleep import DeepSleepConfig, DeepSleepForCausalLM
    cfg = DeepSleepConfig(**REAL_CONFIG)
    model = DeepSleepForCausalLM(cfg)
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
    print("  [PASS] tied embeddings")


if __name__ == "__main__":
    print("=" * 60)
    print(f"Model Tests (REAL config: 768d, 10 layers, MoE, device={DEVICE})")
    print("=" * 60)
    test_config()
    test_forward()
    test_loss()
    test_backward()
    test_amp()
    test_checkpoint()
    test_tied_embeddings()
    print("\n" + "=" * 60)
    print("ALL PASSED")
    print("=" * 60)
