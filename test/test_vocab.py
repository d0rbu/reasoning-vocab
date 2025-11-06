"""
Tests for tokenizer utilities.

Tests include:
- Standard tokenization/detokenization
- Reasoning token translation
- Multiplicity tracking
- Edge cases and error handling
"""

import pytest
import torch as th
from transformers import AutoTokenizer

from core.tokenizer_utils import ReasoningTokenizer


@pytest.fixture
def base_tokenizer():
    """Create a base tokenizer for testing."""
    tokenizer = AutoTokenizer.from_pretrained(
        "hf-internal-testing/tiny-random-GPTNeoXForCausalLM", trust_remote_code=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@pytest.fixture
def reasoning_token_ids():
    """Sample reasoning token IDs for testing."""
    return [10, 20, 30, 40, 50]


@pytest.fixture
def reasoning_token_ids_with_repeats():
    """Sample reasoning token IDs with repeated tokens for testing multiplicity."""
    return [10, 20, 10, 30, 20, 10]  # 10 appears 3 times, 20 appears 2 times


@pytest.fixture
def reasoning_tokenizer(base_tokenizer, reasoning_token_ids):
    """Create a reasoning tokenizer for testing."""
    return ReasoningTokenizer(
        tokenizer=base_tokenizer,
        reasoning_token_ids=reasoning_token_ids,
        vocab_size=base_tokenizer.vocab_size,
    )


def test_initialization(base_tokenizer, reasoning_token_ids):
    """Test ReasoningTokenizer initialization."""
    vocab_size = base_tokenizer.vocab_size
    rt = ReasoningTokenizer(
        tokenizer=base_tokenizer, reasoning_token_ids=reasoning_token_ids, vocab_size=vocab_size
    )

    assert rt.vocab_size == vocab_size
    assert rt.reasoning_token_ids == reasoning_token_ids
    assert len(rt.reasoning_to_standard) == len(reasoning_token_ids)
    assert len(rt.reasoning_to_multiplicity) == len(reasoning_token_ids)


def test_standard_tokenization(reasoning_tokenizer):
    """Test standard tokenization works correctly."""
    text = "Hello world"

    # Encode should work like normal tokenizer
    encoded = reasoning_tokenizer.encode(text)
    assert isinstance(encoded, list)
    assert len(encoded) > 0

    # All tokens should be in standard vocabulary range
    assert all(token_id < reasoning_tokenizer.vocab_size for token_id in encoded)


def test_standard_decoding(reasoning_tokenizer):
    """Test standard decoding works correctly."""
    text = "Hello world"

    # Encode and decode should roundtrip
    encoded = reasoning_tokenizer.encode(text)
    decoded = reasoning_tokenizer.decode(encoded)

    # The decoded text should be similar (may have extra spaces/tokens)
    assert isinstance(decoded, str)
    assert len(decoded) > 0


def test_reasoning_token_translation(base_tokenizer, reasoning_token_ids):
    """Test reasoning token translation to standard tokens."""
    vocab_size = base_tokenizer.vocab_size
    rt = ReasoningTokenizer(
        tokenizer=base_tokenizer, reasoning_token_ids=reasoning_token_ids, vocab_size=vocab_size
    )

    # Create token IDs that include reasoning tokens
    token_ids = th.tensor(
        [
            10,  # Standard token
            vocab_size + 0,  # First reasoning token (maps to reasoning_token_ids[0])
            vocab_size + 1,  # Second reasoning token (maps to reasoning_token_ids[1])
            20,  # Standard token
        ]
    )

    # Convert to standard IDs
    standard_ids = rt._convert_to_standard_ids(token_ids)

    # First and last should be unchanged
    assert standard_ids[0].item() == 10
    assert standard_ids[3].item() == 20

    # Reasoning tokens should be converted
    assert standard_ids[1].item() == reasoning_token_ids[0]
    assert standard_ids[2].item() == reasoning_token_ids[1]


def test_multiplicity_tracking(base_tokenizer, reasoning_token_ids_with_repeats):
    """Test that multiplicity is correctly tracked for reasoning tokens."""
    vocab_size = base_tokenizer.vocab_size
    rt = ReasoningTokenizer(
        tokenizer=base_tokenizer,
        reasoning_token_ids=reasoning_token_ids_with_repeats,
        vocab_size=vocab_size,
    )

    # Check multiplicity mapping
    # reasoning_token_ids_with_repeats = [10, 20, 10, 30, 20, 10]
    # Token 10 appears at indices 0, 2, 4 with multiplicities 1, 2, 3
    # Token 20 appears at indices 1, 4 with multiplicities 1, 2
    # Token 30 appears at index 3 with multiplicity 1

    expected_multiplicities = {
        0: 1,  # First 10
        1: 1,  # First 20
        2: 2,  # Second 10
        3: 1,  # First 30
        4: 2,  # Second 20
        5: 3,  # Third 10
    }

    for idx, expected_mult in expected_multiplicities.items():
        actual_mult = rt.reasoning_to_multiplicity[idx].item()
        assert actual_mult == expected_mult, (
            f"Token at index {idx} (token_id={reasoning_token_ids_with_repeats[idx]}) "
            f"has multiplicity {actual_mult}, expected {expected_mult}"
        )


def test_get_multiplicities(base_tokenizer, reasoning_token_ids_with_repeats):
    """Test getting multiplicities for a sequence of tokens."""
    vocab_size = base_tokenizer.vocab_size
    rt = ReasoningTokenizer(
        tokenizer=base_tokenizer,
        reasoning_token_ids=reasoning_token_ids_with_repeats,
        vocab_size=vocab_size,
    )

    # Create mixed standard and reasoning tokens
    token_ids = th.tensor(
        [
            10,  # Standard token -> multiplicity 0
            vocab_size + 0,  # First reasoning token (10) -> multiplicity 1
            vocab_size + 2,  # Third reasoning token (10) -> multiplicity 2
            20,  # Standard token -> multiplicity 0
        ]
    )

    multiplicities = rt._get_multiplicities(token_ids)

    assert multiplicities[0].item() == 0  # Standard token
    assert multiplicities[1].item() == 1  # First occurrence of 10
    assert multiplicities[2].item() == 2  # Second occurrence of 10
    assert multiplicities[3].item() == 0  # Standard token


def test_detokenize_with_multiplicity(base_tokenizer, reasoning_token_ids):
    """Test detokenization with multiplicity information."""
    vocab_size = base_tokenizer.vocab_size
    rt = ReasoningTokenizer(
        tokenizer=base_tokenizer, reasoning_token_ids=reasoning_token_ids, vocab_size=vocab_size
    )

    # Create mixed standard and reasoning tokens
    token_ids = [
        10,
        vocab_size + 0,  # Reasoning token
        vocab_size + 1,  # Reasoning token
        20,
    ]

    text, multiplicities = rt.decode_with_multiplicity(token_ids)

    # Check that we get text and multiplicities
    assert isinstance(text, str)
    assert isinstance(multiplicities, th.Tensor)
    assert len(multiplicities) == len(token_ids)

    # Check multiplicity values
    assert multiplicities[0].item() == 0  # Standard
    assert multiplicities[1].item() >= 1  # Reasoning (at least 1)
    assert multiplicities[2].item() >= 1  # Reasoning (at least 1)
    assert multiplicities[3].item() == 0  # Standard


def test_get_token_string_and_multiplicity(base_tokenizer, reasoning_token_ids):
    """Test getting token string and multiplicity for a single token."""
    vocab_size = base_tokenizer.vocab_size
    rt = ReasoningTokenizer(
        tokenizer=base_tokenizer, reasoning_token_ids=reasoning_token_ids, vocab_size=vocab_size
    )

    # Test standard token
    token_str, mult = rt.get_token_string_and_multiplicity(10)
    assert isinstance(token_str, str)
    assert mult == 0

    # Test reasoning token
    reasoning_id = vocab_size + 0
    token_str, mult = rt.get_token_string_and_multiplicity(reasoning_id)
    assert isinstance(token_str, str)
    assert mult >= 1  # Should have multiplicity >= 1


def test_get_token_strings_and_multiplicities(base_tokenizer, reasoning_token_ids):
    """Test getting token strings and multiplicities for a sequence."""
    vocab_size = base_tokenizer.vocab_size
    rt = ReasoningTokenizer(
        tokenizer=base_tokenizer, reasoning_token_ids=reasoning_token_ids, vocab_size=vocab_size
    )

    token_ids = [10, vocab_size + 0, vocab_size + 1, 20]

    token_strings, multiplicities = rt.get_token_strings_and_multiplicities(token_ids)

    # Check return types
    assert isinstance(token_strings, list)
    assert isinstance(multiplicities, th.Tensor)
    assert len(token_strings) == len(token_ids)
    assert len(multiplicities) == len(token_ids)

    # Check that all strings are strings
    assert all(isinstance(s, str) for s in token_strings)


def test_batch_decode(base_tokenizer, reasoning_token_ids):
    """Test batch decoding."""
    vocab_size = base_tokenizer.vocab_size
    rt = ReasoningTokenizer(
        tokenizer=base_tokenizer, reasoning_token_ids=reasoning_token_ids, vocab_size=vocab_size
    )

    batch_ids = [
        [10, 20, 30],
        [vocab_size + 0, vocab_size + 1, 40],
        [15, 25, vocab_size + 2],
    ]

    decoded_batch = rt.batch_decode(batch_ids)

    assert isinstance(decoded_batch, list)
    assert len(decoded_batch) == len(batch_ids)
    assert all(isinstance(text, str) for text in decoded_batch)


def test_reasoning_token_consistency(base_tokenizer, reasoning_token_ids):
    """Test that reasoning tokens decode to the same text as their base tokens."""
    vocab_size = base_tokenizer.vocab_size
    rt = ReasoningTokenizer(
        tokenizer=base_tokenizer, reasoning_token_ids=reasoning_token_ids, vocab_size=vocab_size
    )

    for i, base_token_id in enumerate(reasoning_token_ids):
        reasoning_token_id = vocab_size + i

        # Decode base token
        base_text = base_tokenizer.decode([base_token_id])

        # Decode reasoning token
        reasoning_text = rt.decode([reasoning_token_id])

        # They should be the same
        assert base_text == reasoning_text


def test_properties(reasoning_tokenizer, base_tokenizer):
    """Test that tokenizer properties are accessible."""
    assert reasoning_tokenizer.pad_token_id == base_tokenizer.pad_token_id
    assert reasoning_tokenizer.eos_token_id == base_tokenizer.eos_token_id
    assert reasoning_tokenizer.bos_token_id == base_tokenizer.bos_token_id


def test_edge_case_empty_sequence(reasoning_tokenizer):
    """Test handling of empty sequences."""
    # Empty list
    decoded = reasoning_tokenizer.decode([])
    assert isinstance(decoded, str)

    # Empty tensor
    decoded = reasoning_tokenizer.decode(th.tensor([]))
    assert isinstance(decoded, str)


def test_edge_case_all_reasoning_tokens(base_tokenizer, reasoning_token_ids):
    """Test sequence with all reasoning tokens."""
    vocab_size = base_tokenizer.vocab_size
    rt = ReasoningTokenizer(
        tokenizer=base_tokenizer, reasoning_token_ids=reasoning_token_ids, vocab_size=vocab_size
    )

    # Create sequence of all reasoning tokens
    token_ids = [vocab_size + i for i in range(len(reasoning_token_ids))]

    decoded = rt.decode(token_ids)
    assert isinstance(decoded, str)

    text, mults = rt.decode_with_multiplicity(token_ids)
    assert isinstance(text, str)
    assert all(m >= 1 for m in mults.tolist())


def test_edge_case_all_standard_tokens(reasoning_tokenizer):
    """Test sequence with all standard tokens."""
    # Create sequence of all standard tokens
    token_ids = [10, 20, 30, 40, 50]

    decoded = reasoning_tokenizer.decode(token_ids)
    assert isinstance(decoded, str)

    text, mults = reasoning_tokenizer.decode_with_multiplicity(token_ids)
    assert isinstance(text, str)
    assert all(m == 0 for m in mults.tolist())
