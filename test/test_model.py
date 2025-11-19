"""
Tests for Qwen3ReasoningVocabForCausalLM model.

This module tests:
- Model initialization with and without reasoning tokens
- Token embedding resizing
- LogitsProcessor functionality
- Forward pass compatibility
- Generation compatibility
"""

import pytest
import torch as th
from transformers import AutoTokenizer
from transformers.generation.logits_process import LogitsProcessorList

from core.reasoning_vocab_model import (
    Qwen3ReasoningVocabForCausalLM,
    ReasoningVocabLogitsProcessor,
)
from core.reasoning_vocab_utils import get_reasoning_token_ids
from core.tokenizer_utils import ReasoningTokenizer


@pytest.fixture
def tiny_config():
    """Create a tiny model config for testing."""
    from transformers import Qwen3Config

    config = Qwen3Config(
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=512,
        max_position_embeddings=32768,
        rms_norm_eps=1e-6,
    )
    return config


@pytest.fixture
def gpt2_tokenizer():
    """Create a GPT-2 tokenizer for testing."""
    return AutoTokenizer.from_pretrained("gpt2")


def test_model_initialization_no_reasoning(tiny_config):
    """Test that model initializes correctly without reasoning vocabulary."""
    model = Qwen3ReasoningVocabForCausalLM(tiny_config)

    assert model.standard_vocab_size == tiny_config.vocab_size
    assert model.num_reasoning_tokens() == 0
    assert len(model.reasoning_token_ids) == 0
    # Vocab size should remain unchanged
    assert model.get_input_embeddings().num_embeddings == tiny_config.vocab_size


def test_model_initialization_with_reasoning(tiny_config):
    """Test that model initializes correctly with reasoning vocabulary."""
    reasoning_token_ids = [10, 20, 30, 40, 50]
    original_vocab_size = tiny_config.vocab_size
    model = Qwen3ReasoningVocabForCausalLM(tiny_config, reasoning_token_ids)

    assert model.standard_vocab_size == original_vocab_size
    assert model.num_reasoning_tokens() == 5
    assert model.get_reasoning_token_ids() == tuple(reasoning_token_ids)
    # Vocab size should be extended
    assert model.get_input_embeddings().num_embeddings == original_vocab_size + len(
        reasoning_token_ids
    )


def test_model_initialization_with_all_tokens(tiny_config):
    """Test initialization with all standard tokens as reasoning tokens."""
    original_vocab_size = tiny_config.vocab_size
    reasoning_token_ids = get_reasoning_token_ids(original_vocab_size)
    model = Qwen3ReasoningVocabForCausalLM(tiny_config, reasoning_token_ids)

    assert model.standard_vocab_size == original_vocab_size
    assert model.num_reasoning_tokens() == original_vocab_size
    # Vocab size should be doubled
    assert model.get_input_embeddings().num_embeddings == original_vocab_size * 2


def test_reasoning_embeddings_initialized_correctly(tiny_config):
    """Test that reasoning embeddings are initialized from standard tokens."""
    reasoning_token_ids = [10, 20, 30]
    original_vocab_size = tiny_config.vocab_size
    model = Qwen3ReasoningVocabForCausalLM(tiny_config, reasoning_token_ids)

    embed_weights = model.model.embed_tokens.weight
    lm_head_weights = model.lm_head.weight

    # Check that reasoning embeddings match their source standard embeddings
    for i, token_id in enumerate(reasoning_token_ids):
        reasoning_idx = original_vocab_size + i
        # Embeddings should be equal
        assert th.allclose(embed_weights[reasoning_idx], embed_weights[token_id])
        assert th.allclose(lm_head_weights[reasoning_idx], lm_head_weights[token_id])


def test_forward_pass_no_reasoning(tiny_config):
    """Test forward pass with no reasoning tokens."""
    model = Qwen3ReasoningVocabForCausalLM(tiny_config)

    batch_size = 2
    seq_len = 10
    input_ids = th.randint(0, tiny_config.vocab_size, (batch_size, seq_len))

    outputs = model(input_ids)

    assert outputs.logits.shape == (batch_size, seq_len, tiny_config.vocab_size)


def test_forward_pass_with_reasoning(tiny_config):
    """Test forward pass with reasoning tokens."""
    reasoning_token_ids = [10, 20, 30, 40, 50]
    original_vocab_size = tiny_config.vocab_size
    model = Qwen3ReasoningVocabForCausalLM(tiny_config, reasoning_token_ids)

    batch_size = 2
    seq_len = 10
    input_ids = th.randint(0, original_vocab_size, (batch_size, seq_len))

    outputs = model(input_ids)

    # Logits should include both standard and reasoning vocab
    expected_vocab_size = original_vocab_size + len(reasoning_token_ids)
    assert outputs.logits.shape == (batch_size, seq_len, expected_vocab_size)


def test_forward_pass_with_reasoning_input_tokens(tiny_config):
    """Test forward pass when input contains reasoning token IDs."""
    reasoning_token_ids = [10, 20, 30, 40, 50]
    original_vocab_size = tiny_config.vocab_size
    model = Qwen3ReasoningVocabForCausalLM(tiny_config, reasoning_token_ids)

    batch_size = 2
    seq_len = 10
    # Create input with some reasoning tokens
    input_ids = th.randint(0, original_vocab_size, (batch_size, seq_len))
    # Add some reasoning tokens
    input_ids[:, 5:8] = original_vocab_size + th.randint(
        0, len(reasoning_token_ids), (batch_size, 3)
    )

    outputs = model(input_ids)

    expected_vocab_size = original_vocab_size + len(reasoning_token_ids)
    assert outputs.logits.shape == (batch_size, seq_len, expected_vocab_size)


def test_loss_computation(tiny_config):
    """Test that loss is computed correctly with labels."""
    reasoning_token_ids = [10, 20, 30]
    model = Qwen3ReasoningVocabForCausalLM(tiny_config, reasoning_token_ids)

    batch_size = 2
    seq_len = 10
    input_ids = th.randint(0, tiny_config.vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()

    outputs = model(input_ids, labels=labels)

    assert outputs.loss is not None
    assert outputs.loss.item() > 0
    # Check that loss is a scalar
    assert outputs.loss.shape == ()


def test_gradient_flow(tiny_config):
    """Test that gradients flow through reasoning embeddings."""
    reasoning_token_ids = [10, 20, 30]
    original_vocab_size = tiny_config.vocab_size
    model = Qwen3ReasoningVocabForCausalLM(tiny_config, reasoning_token_ids)
    model.train()

    batch_size = 2
    seq_len = 5

    # Create input with reasoning tokens to ensure they're used
    input_ids = th.randint(0, original_vocab_size, (batch_size, seq_len))
    input_ids[:, 2:4] = original_vocab_size + th.randint(
        0, len(reasoning_token_ids), (batch_size, 2)
    )

    labels = input_ids.clone()

    # Forward and backward pass
    outputs = model(input_ids, labels=labels)
    loss = outputs.loss
    loss.backward()

    # Check that embeddings have gradients
    assert model.model.embed_tokens.weight.grad is not None
    assert model.lm_head.weight.grad is not None

    # Check that gradients are non-zero
    assert model.model.embed_tokens.weight.grad.abs().sum() > 0
    assert model.lm_head.weight.grad.abs().sum() > 0


def test_logits_processor_standard_mode(tiny_config, gpt2_tokenizer):
    """Test LogitsProcessor masks reasoning vocab when no think tags present."""
    reasoning_token_ids = [10, 20, 30]
    original_vocab_size = tiny_config.vocab_size
    model = Qwen3ReasoningVocabForCausalLM(tiny_config, reasoning_token_ids)

    # Create reasoning tokenizer wrapper
    reasoning_tokenizer = ReasoningTokenizer(gpt2_tokenizer, reasoning_token_ids)

    processor = ReasoningVocabLogitsProcessor(model.standard_vocab_size, reasoning_tokenizer)

    batch_size = 2
    seq_len = 5
    extended_vocab_size = original_vocab_size + len(reasoning_token_ids)

    # Create input without think tags - just use GPT-2 vocab tokens
    input_ids_tensor = th.randint(
        0, min(gpt2_tokenizer.vocab_size, original_vocab_size), (batch_size, seq_len)
    )
    assert isinstance(input_ids_tensor, th.LongTensor), "randint should create LongTensor"
    input_ids: th.LongTensor = input_ids_tensor

    logits_tensor = th.randn(batch_size, extended_vocab_size)
    assert isinstance(logits_tensor, th.FloatTensor), "randn should create FloatTensor"
    logits: th.FloatTensor = logits_tensor

    processed_logits = processor(input_ids, logits)

    # Standard vocab should be unchanged
    assert th.allclose(processed_logits[:, :original_vocab_size], logits[:, :original_vocab_size])

    # Reasoning vocab should be -inf
    assert th.all(processed_logits[:, original_vocab_size:] == float("-inf"))


def test_logits_processor_reasoning_mode(tiny_config, gpt2_tokenizer):
    """Test LogitsProcessor allows reasoning vocab when think tag is open."""
    reasoning_token_ids = [10, 20, 30]
    original_vocab_size = tiny_config.vocab_size
    model = Qwen3ReasoningVocabForCausalLM(tiny_config, reasoning_token_ids)

    # Create reasoning tokenizer wrapper
    reasoning_tokenizer = ReasoningTokenizer(gpt2_tokenizer, reasoning_token_ids)

    processor = ReasoningVocabLogitsProcessor(
        model.standard_vocab_size, reasoning_tokenizer, "<think>", "</think>"
    )

    batch_size = 2
    extended_vocab_size = original_vocab_size + len(reasoning_token_ids)

    # First sequence: encode "<think>" to get actual tokens
    think_text = "<think> some reasoning"
    think_ids = gpt2_tokenizer.encode(think_text, return_tensors="pt")[0]

    # Second sequence: regular text without tags
    normal_text = "just regular text"
    normal_ids = gpt2_tokenizer.encode(normal_text, return_tensors="pt")[0]

    # Pad to same length
    max_len = max(len(think_ids), len(normal_ids))
    think_ids = th.nn.functional.pad(
        think_ids, (0, max_len - len(think_ids)), value=gpt2_tokenizer.eos_token_id
    )
    normal_ids = th.nn.functional.pad(
        normal_ids, (0, max_len - len(normal_ids)), value=gpt2_tokenizer.eos_token_id
    )

    input_ids_tensor = th.stack([think_ids, normal_ids])
    assert isinstance(input_ids_tensor, th.LongTensor), "stacked tensor should be LongTensor"
    input_ids: th.LongTensor = input_ids_tensor

    logits_tensor = th.randn(batch_size, extended_vocab_size)
    assert isinstance(logits_tensor, th.FloatTensor), "randn should create FloatTensor"
    logits: th.FloatTensor = logits_tensor

    processed_logits = processor(input_ids, logits)

    # First sequence (with think tag) should have all logits available
    assert th.allclose(processed_logits[0, :original_vocab_size], logits[0, :original_vocab_size])
    assert th.allclose(processed_logits[0, original_vocab_size:], logits[0, original_vocab_size:])

    # Second sequence (no think tag) should have reasoning masked
    assert th.allclose(processed_logits[1, :original_vocab_size], logits[1, :original_vocab_size])
    assert th.all(processed_logits[1, original_vocab_size:] == float("-inf"))


def test_generate_compatibility(tiny_config):
    """Test that model is compatible with generate() method."""
    reasoning_token_ids = [10, 20, 30]
    model = Qwen3ReasoningVocabForCausalLM(tiny_config, reasoning_token_ids)
    model.eval()

    batch_size = 1
    seq_len = 5
    input_ids = th.randint(0, tiny_config.vocab_size, (batch_size, seq_len))

    # Test basic generation without reasoning vocab
    with th.no_grad():
        output = model.generate(input_ids, max_new_tokens=5, do_sample=False)

    assert output.shape[0] == batch_size
    assert output.shape[1] > seq_len


def test_generate_with_logits_processor(tiny_config, gpt2_tokenizer):
    """Test generation with LogitsProcessor to control reasoning vocab."""
    reasoning_token_ids = [10, 20, 30]
    original_vocab_size = tiny_config.vocab_size
    model = Qwen3ReasoningVocabForCausalLM(tiny_config, reasoning_token_ids)
    model.eval()

    batch_size = 1
    seq_len = 5
    input_ids_tensor = th.randint(
        0, min(gpt2_tokenizer.vocab_size, original_vocab_size), (batch_size, seq_len)
    )
    assert isinstance(input_ids_tensor, th.LongTensor), "randint should create LongTensor"
    input_ids: th.LongTensor = input_ids_tensor

    # Create reasoning tokenizer wrapper
    reasoning_tokenizer = ReasoningTokenizer(gpt2_tokenizer, reasoning_token_ids)

    # Get logits processor (no thinking tags, so reasoning vocab will be masked)
    processor = ReasoningVocabLogitsProcessor(model.standard_vocab_size, reasoning_tokenizer)

    # Test generation with processor - use LogitsProcessorList
    processor_list = LogitsProcessorList([processor])

    # Test generation with processor
    with th.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=5,
            do_sample=False,
            logits_processor=processor_list,
        )

    assert output.shape[0] == batch_size
    assert output.shape[1] > seq_len

    # All generated tokens should be from standard vocab
    generated_tokens = output[:, seq_len:]
    assert th.all(generated_tokens < original_vocab_size)
