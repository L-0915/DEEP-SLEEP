"""Tests for DeepSleep model configuration and basic model functionality."""

import pytest
import torch


class TestDeepSleepConfig:
    """Test model configuration."""

    def test_config_defaults(self):
        """Test default configuration values."""
        from src.model.config import DeepSleepConfig

        config = DeepSleepConfig()
        assert config.d_model == 2048
        assert config.n_layers == 24
        assert config.n_heads == 16
        assert config.n_kv_heads == 4
        assert config.vocab_size == 64000
        assert config.max_position_embeddings == 8192
        assert config.hidden_act == "silu"
        assert config.tie_word_embeddings is True
        assert config.num_experts == 6
        assert config.num_routed_experts == 5
        assert config.top_k == 2

    def test_config_custom(self):
        """Test custom configuration."""
        from src.model.config import DeepSleepConfig

        config = DeepSleepConfig(d_model=1024, n_layers=12, n_heads=8)
        assert config.d_model == 1024
        assert config.n_layers == 12
        assert config.n_heads == 8

    def test_config_model_type(self):
        """Test model type is correctly set."""
        from src.model.config import DeepSleepConfig

        config = DeepSleepConfig()
        assert config.model_type == "deepsleep"


class TestDeepSleepRMSNorm:
    """Test RMSNorm layer."""

    def test_output_shape(self):
        """Test RMSNorm preserves input shape."""
        from src.model.layers import DeepSleepRMSNorm

        norm = DeepSleepRMSNorm(2048)
        x = torch.randn(2, 8, 2048)
        out = norm(x)
        assert out.shape == x.shape

    def test_norm_behavior(self):
        """Test that RMSNorm normalizes correctly."""
        from src.model.layers import DeepSleepRMSNorm

        norm = DeepSleepRMSNorm(64, eps=1e-6)
        x = torch.randn(1, 4, 64) * 10
        out = norm(x)
        # Output should have roughly unit variance (scaled by weight)
        assert out.std() < x.std() * 0.5


class TestDeepSleepRotaryEmbedding:
    """Test RoPE implementation."""

    def test_output_shape(self):
        """Test rotary embedding output shape."""
        from src.model.layers import DeepSleepRotaryEmbedding

        rope = DeepSleepRotaryEmbedding(128, 8192)
        q = torch.randn(2, 8, 16, 128)
        k = torch.randn(2, 8, 16, 128)
        q_out, k_out = rope(q, k)
        assert q_out.shape == q.shape
        assert k_out.shape == k.shape

    def test_rotary_properties(self):
        """Test that rotary embeddings are applied correctly."""
        from src.model.layers import DeepSleepRotaryEmbedding

        rope = DeepSleepRotaryEmbedding(64, 1024)
        q = torch.randn(1, 1, 8, 64)
        k = torch.randn(1, 1, 8, 64)
        q_rot, k_rot = rope(q, k)
        # Rotated embeddings should differ from original
        assert not torch.allclose(q, q_rot)
        assert not torch.allclose(k, k_rot)


class TestDeepSleepMLP:
    """Test SwiGLU feed-forward network."""

    def test_output_shape(self):
        """Test MLP output shape matches input."""
        from src.model.feedforward import DeepSleepMLP

        mlp = DeepSleepMLP(2048, 5632)
        x = torch.randn(2, 8, 2048)
        out = mlp(x)
        assert out.shape == x.shape

    def test_no_bias(self):
        """Test that all linear layers have no bias."""
        from src.model.feedforward import DeepSleepMLP

        mlp = DeepSleepMLP(2048, 5632)
        for param_name, param in mlp.named_parameters():
            if "weight" not in param_name:
                pytest.fail(f"Unexpected parameter without 'weight': {param_name}")


class TestDeepSleepAttention:
    """Test grouped-query attention."""

    def test_output_shape(self):
        """Test attention output shape."""
        from src.model.attention import DeepSleepAttention

        attn = DeepSleepAttention(
            config=None,
            d_model=2048,
            n_heads=16,
            n_kv_heads=4,
            head_dim=128,
        )
        hidden_states = torch.randn(2, 8, 2048)
        out = attn(hidden_states)
        assert out[0].shape == hidden_states.shape  # hidden_states output

    def test_gqa_groups(self):
        """Test GQA has correct number of KV heads."""
        from src.model.attention import DeepSleepAttention

        attn = DeepSleepAttention(
            config=None,
            d_model=2048,
            n_heads=16,
            n_kv_heads=4,
            head_dim=128,
        )
        assert attn.num_kv_heads == 4
        assert attn.num_key_value_groups == 4  # 16 / 4


class TestDeepSleepMoE:
    """Test Mixture of Experts layer."""

    def test_output_shape(self):
        """Test MoE output shape matches input."""
        from src.model.moe import DeepSleepMoE

        moe = DeepSleepMoE(
            d_model=2048,
            num_routed_experts=5,
            intermediate_size=1408,
            shared_expert_intermediate_size=5632,
        )
        x = torch.randn(2, 8, 2048)
        out, aux_loss = moe(x)
        assert out.shape == x.shape
        assert "aux_loss" in aux_loss

    def test_aux_loss_scalar(self):
        """Test that auxiliary loss is a scalar."""
        from src.model.moe import DeepSleepMoE

        moe = DeepSleepMoE(
            d_model=2048,
            num_routed_experts=5,
            intermediate_size=1408,
            shared_expert_intermediate_size=5632,
        )
        x = torch.randn(2, 8, 2048)
        _, aux_loss = moe(x)
        assert aux_loss["aux_loss"].dim() == 0  # scalar


class TestDeepSleepModel:
    """Test full model forward pass."""

    def test_forward_pass(self):
        """Test basic forward pass produces correct output shape."""
        from src.model.modeling_deepsleep import DeepSleepForCausalLM
        from src.model.config import DeepSleepConfig

        config = DeepSleepConfig(
            d_model=512,
            n_layers=2,
            n_heads=8,
            n_kv_heads=2,
            vocab_size=1000,
            max_position_embeddings=512,
            num_experts=4,
            num_routed_experts=3,
            top_k=2,
        )
        model = DeepSleepForCausalLM(config)
        model.eval()

        input_ids = torch.randint(0, 1000, (1, 32))
        with torch.no_grad():
            outputs = model(input_ids)

        assert outputs.logits.shape == (1, 32, 1000)

    def test_loss_computation(self):
        """Test that loss is computed when labels are provided."""
        from src.model.modeling_deepsleep import DeepSleepForCausalLM
        from src.model.config import DeepSleepConfig

        config = DeepSleepConfig(
            d_model=512,
            n_layers=2,
            n_heads=8,
            n_kv_heads=2,
            vocab_size=1000,
            max_position_embeddings=512,
            num_experts=4,
            num_routed_experts=3,
            top_k=2,
        )
        model = DeepSleepForCausalLM(config)
        model.eval()

        input_ids = torch.randint(0, 1000, (1, 32))
        labels = torch.randint(0, 1000, (1, 32))
        with torch.no_grad():
            outputs = model(input_ids, labels=labels)

        assert outputs.loss is not None
        assert outputs.loss.dim() == 0  # scalar

    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        from src.model.modeling_deepsleep import DeepSleepForCausalLM
        from src.model.config import DeepSleepConfig

        config = DeepSleepConfig(
            d_model=512,
            n_layers=2,
            n_heads=8,
            n_kv_heads=2,
            vocab_size=1000,
            max_position_embeddings=512,
            num_experts=4,
            num_routed_experts=3,
            top_k=2,
        )
        model = DeepSleepForCausalLM(config)
        model.train()

        input_ids = torch.randint(0, 1000, (1, 32))
        labels = torch.randint(0, 1000, (1, 32))
        outputs = model(input_ids, labels=labels)
        outputs.loss.backward()

        # Check gradients exist for at least some parameters
        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad, "No gradients found in model parameters"
