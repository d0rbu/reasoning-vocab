"""
Tests for Qwen3 reasoning model.

Tests include:
- Reasoning vocabulary initialization
- Forward pass with and without reasoning tokens
- Shape consistency checks
- Embedding/unembedding behavior
"""

import pytest
import torch as th

from core.modeling_qwen3_reasoning import Qwen3ReasoningForCausalLM


@pytest.fixture
def tiny_config():
    """Create a tiny model config for testing."""
    from transformers import Qwen2Config

    config = Qwen2Config(
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=512,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
    )
    return config


@pytest.fixture
def reasoning_token_ids():
    """Sample reasoning token IDs for testing."""
    # Use token IDs within the small vocab size (< 1000)
    return [10, 20, 30, 40, 50]


def test_model_initialization(tiny_config):
    """Test that model initializes correctly with reasoning vocabulary."""
    # Test initialization with default reasoning vocab size
    model = Qwen3ReasoningForCausalLM(tiny_config)

    assert hasattr(model, "reasoning_embed")
    assert hasattr(model, "reasoning_unembed")
    assert model.num_reasoning_tokens == tiny_config.vocab_size
    assert model.standard_vocab_size == tiny_config.vocab_size

    # Test reasoning embedding shape
    assert model.reasoning_embed.weight.shape == (tiny_config.vocab_size, tiny_config.hidden_size)

    # Test reasoning unembedding shape
    assert model.reasoning_unembed.weight.shape == (tiny_config.vocab_size, tiny_config.hidden_size)


def test_model_initialization_custom_size(tiny_config, reasoning_token_ids):
    """Test model initialization with custom reasoning vocab size."""
    num_reasoning = 5

    model = Qwen3ReasoningForCausalLM(
        tiny_config, num_reasoning_tokens=num_reasoning, reasoning_token_ids=reasoning_token_ids
    )

    assert model.num_reasoning_tokens == num_reasoning
    assert model.reasoning_embed.weight.shape == (num_reasoning, tiny_config.hidden_size)
    assert model.reasoning_unembed.weight.shape == (num_reasoning, tiny_config.hidden_size)

    # Verify reasoning token IDs are stored
    retrieved_ids = model.get_reasoning_token_ids()
    assert len(retrieved_ids) == num_reasoning
    assert retrieved_ids == reasoning_token_ids


def test_model_initialization_from_specific_tokens(tiny_config, reasoning_token_ids):
    """Test that reasoning embeddings are initialized from specific tokens."""
    model = Qwen3ReasoningForCausalLM(
        tiny_config,
        num_reasoning_tokens=len(reasoning_token_ids),
        reasoning_token_ids=reasoning_token_ids,
    )

    # Check that reasoning embeddings match the specified standard tokens
    for i, token_id in enumerate(reasoning_token_ids):
        # Embeddings should be initialized from standard embeddings
        assert th.allclose(
            model.reasoning_embed.weight[i], model.model.embed_tokens.weight[token_id], atol=1e-6
        )

        # Unembeddings should be initialized from standard unembeddings
        assert th.allclose(
            model.reasoning_unembed.weight[i], model.lm_head.weight[token_id], atol=1e-6
        )


def test_forward_pass_standard(tiny_config):
    """Test forward pass with standard vocabulary only."""
    model = Qwen3ReasoningForCausalLM(tiny_config)
    model.eval()

    batch_size = 2
    seq_len = 10

    # Create dummy input
    input_ids = th.randint(0, tiny_config.vocab_size, (batch_size, seq_len))

    # Forward pass without reasoning vocab
    with th.no_grad():
        outputs = model(input_ids, use_reasoning_vocab=False)

    # Check output shape (should be standard vocab size)
    assert outputs.logits.shape == (batch_size, seq_len, tiny_config.vocab_size)


def test_forward_pass_reasoning(tiny_config, reasoning_token_ids):
    """Test forward pass with reasoning vocabulary enabled."""
    num_reasoning = len(reasoning_token_ids)
    model = Qwen3ReasoningForCausalLM(
        tiny_config, num_reasoning_tokens=num_reasoning, reasoning_token_ids=reasoning_token_ids
    )
    model.eval()

    batch_size = 2
    seq_len = 10

    # Create dummy input (standard tokens only)
    input_ids = th.randint(0, tiny_config.vocab_size, (batch_size, seq_len))

    # Forward pass with reasoning vocab
    with th.no_grad():
        outputs = model(input_ids, use_reasoning_vocab=True)

    # Check output shape (should be standard + reasoning vocab size)
    expected_vocab_size = tiny_config.vocab_size + num_reasoning
    assert outputs.logits.shape == (batch_size, seq_len, expected_vocab_size)


def test_forward_pass_with_reasoning_input_ids(tiny_config, reasoning_token_ids):
    """Test forward pass with reasoning token IDs in input."""
    num_reasoning = len(reasoning_token_ids)
    model = Qwen3ReasoningForCausalLM(
        tiny_config, num_reasoning_tokens=num_reasoning, reasoning_token_ids=reasoning_token_ids
    )
    model.eval()

    batch_size = 2
    seq_len = 10

    # Create input with mix of standard and reasoning tokens
    input_ids = th.randint(0, tiny_config.vocab_size, (batch_size, seq_len))
    # Make some tokens reasoning tokens
    input_ids[:, 5:] = tiny_config.vocab_size + th.randint(
        0, num_reasoning, (batch_size, seq_len - 5)
    )

    # Forward pass with reasoning vocab
    with th.no_grad():
        outputs = model(input_ids, use_reasoning_vocab=True)

    # Check output shape
    expected_vocab_size = tiny_config.vocab_size + num_reasoning
    assert outputs.logits.shape == (batch_size, seq_len, expected_vocab_size)


def test_logits_shape(tiny_config):
    """Test that output logits have correct shape."""
    model = Qwen3ReasoningForCausalLM(tiny_config)
    model.eval()

    batch_sizes = [1, 2, 4]
    seq_lens = [5, 10, 20]

    for batch_size in batch_sizes:
        for seq_len in seq_lens:
            input_ids = th.randint(0, tiny_config.vocab_size, (batch_size, seq_len))

            # Test standard vocab
            with th.no_grad():
                outputs_standard = model(input_ids, use_reasoning_vocab=False)
            assert outputs_standard.logits.shape == (batch_size, seq_len, tiny_config.vocab_size)

            # Test with reasoning vocab
            with th.no_grad():
                outputs_reasoning = model(input_ids, use_reasoning_vocab=True)
            assert outputs_reasoning.logits.shape == (
                batch_size,
                seq_len,
                tiny_config.vocab_size * 2,  # Default reasoning size equals standard size
            )


def test_loss_computation(tiny_config):
    """Test that loss is computed correctly."""
    model = Qwen3ReasoningForCausalLM(tiny_config)
    model.train()

    batch_size = 2
    seq_len = 10

    input_ids = th.randint(0, tiny_config.vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()

    # Test with standard vocab
    outputs_standard = model(input_ids, labels=labels, use_reasoning_vocab=False)
    assert outputs_standard.loss is not None
    assert outputs_standard.loss.ndim == 0  # Scalar loss

    # Test with reasoning vocab
    outputs_reasoning = model(input_ids, labels=labels, use_reasoning_vocab=True)
    assert outputs_reasoning.loss is not None
    assert outputs_reasoning.loss.ndim == 0  # Scalar loss


def test_gradient_flow(tiny_config):
    """Test that gradients flow through reasoning embeddings."""
    num_reasoning = 10
    model = Qwen3ReasoningForCausalLM(tiny_config, num_reasoning_tokens=num_reasoning)
    model.train()

    batch_size = 2
    seq_len = 5

    # Create input with some reasoning tokens to ensure they're used
    input_ids = th.randint(0, tiny_config.vocab_size, (batch_size, seq_len))
    # Replace some tokens with reasoning tokens
    input_ids[:, 2:4] = tiny_config.vocab_size + th.randint(0, num_reasoning, (batch_size, 2))

    labels = input_ids.clone()

    # Forward pass with reasoning vocab
    outputs = model(input_ids, labels=labels, use_reasoning_vocab=True)
    loss = outputs.loss

    # Backward pass
    loss.backward()

    # Check that reasoning unembeddings have gradients (they always get gradients from logits)
    assert model.reasoning_unembed.weight.grad is not None
    assert model.reasoning_unembed.weight.grad.abs().sum() > 0

    # Reasoning embeddings should have gradients because we used reasoning tokens in input
    assert model.reasoning_embed.weight.grad is not None
    assert model.reasoning_embed.weight.grad.abs().sum() > 0


def test_generate_compatibility(tiny_config):
    """Test that generate method works."""
    model = Qwen3ReasoningForCausalLM(tiny_config, num_reasoning_tokens=10)
    model.eval()

    input_ids = th.tensor([[1, 2, 3]])

    # Test basic generation
    with th.no_grad():
        output = model.generate(input_ids, max_new_tokens=5, do_sample=False)

    assert output.shape[0] == 1
    assert output.shape[1] > input_ids.shape[1]
