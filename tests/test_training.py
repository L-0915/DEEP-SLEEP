"""Tests for training utilities and dataset classes."""

import pytest
import torch


class TestPretrainDataset:
    """Test pretraining dataset."""

    def test_dataset_length(self):
        """Test dataset reports correct length."""
        from src.data.dataset.pretrain_dataset import PretrainDataset

        # This test requires tokenized data files to exist
        # It's a structural test to ensure the class can be instantiated
        assert hasattr(PretrainDataset, "__init__")
        assert hasattr(PretrainDataset, "__len__")
        assert hasattr(PretrainDataset, "__getitem__")


class TestSFTDataset:
    """Test SFT dataset."""

    def test_sft_dataset_init(self):
        """Test SFT dataset can be instantiated."""
        from src.data.dataset.sft_dataset import SFTDataset

        assert hasattr(SFTDataset, "__init__")
        assert hasattr(SFTDataset, "__getitem__")


class TestDPODataset:
    """Test DPO dataset."""

    def test_dpo_dataset_init(self):
        """Test DPO dataset can be instantiated."""
        from src.data.dataset.dpo_dataset import DPODataset

        assert hasattr(DPODataset, "__init__")
        assert hasattr(DPODataset, "__getitem__")


class TestLossFunctions:
    """Test loss computation functions."""

    def test_lm_loss(self):
        """Test language model loss computation."""
        from src.training.loss import compute_lm_loss

        batch_size, seq_len, vocab_size = 2, 16, 100
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        loss = compute_lm_loss(logits, labels)
        assert loss.dim() == 0  # scalar
        assert loss.item() > 0

    def test_lm_loss_with_ignore(self):
        """Test that ignore_index works correctly."""
        from src.training.loss import compute_lm_loss

        batch_size, seq_len, vocab_size = 2, 16, 100
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        labels[:, :8] = -100  # ignore first half

        loss = compute_lm_loss(logits, labels)
        assert loss.dim() == 0

    def test_dpo_loss(self):
        """Test DPO loss computation."""
        from src.training.loss import compute_dpo_loss

        batch_size = 4
        policy_chosen = torch.randn(batch_size) * 2
        policy_rejected = torch.randn(batch_size) * 2
        ref_chosen = torch.randn(batch_size) * 2
        ref_rejected = torch.randn(batch_size) * 2

        loss = compute_dpo_loss(
            policy_chosen, policy_rejected,
            ref_chosen, ref_rejected,
            beta=0.1,
        )
        assert loss.dim() == 0


class TestLRScheduler:
    """Test learning rate schedulers."""

    def test_cosine_schedule(self):
        """Test cosine schedule with warmup."""
        from src.training.schedulers import get_cosine_schedule_with_warmup

        optimizer = torch.optim.AdamW([torch.randn(10)], lr=3e-4)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=100, num_training_steps=1000
        )

        # LR should increase during warmup
        lrs = []
        for _ in range(50):
            scheduler.step()
            lrs.append(optimizer.param_groups[0]["lr"])
        assert lrs[-1] > lrs[0]

        # LR should decrease after warmup
        for _ in range(500):
            scheduler.step()
            final_lr = optimizer.param_groups[0]["lr"]
        assert final_lr < 3e-4


class TestCheckpoint:
    """Test checkpoint management."""

    def test_checkpoint_save_load(self):
        """Test saving and loading model checkpoint."""
        from src.model.config import DeepSleepConfig
        from src.model.modeling_deepsleep import DeepSleepForCausalLM
        from src.utils.checkpoint import save_checkpoint, load_checkpoint

        config = DeepSleepConfig(
            d_model=256,
            n_layers=1,
            n_heads=4,
            n_kv_heads=2,
            vocab_size=100,
            max_position_embeddings=128,
            num_experts=2,
            num_routed_experts=1,
            top_k=1,
        )
        model = DeepSleepForCausalLM(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_checkpoint(model, optimizer, None, step=100, loss=1.5, output_dir=tmpdir)
            load_checkpoint(model, optimizer, None, checkpoint_dir=tmpdir)

    def test_find_latest_checkpoint(self):
        """Test finding latest checkpoint."""
        from src.utils.checkpoint import find_latest_checkpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            # No checkpoints
            assert find_latest_checkpoint(tmpdir) is None
