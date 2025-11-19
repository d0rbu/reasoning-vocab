"""
Tests for ReasoningTokenizer.

This module tests:
- Tokenizer initialization and mapping
- Token conversion and multiplicity tracking
- Decoding with and without multiplicity information
- Batch operations
- Edge cases
"""

import pytest
from transformers import AutoTokenizer

from core.tokenizer_utils import ReasoningTokenizer, TokenMultiplicityInfo


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    # Use a real tokenizer but with a small vocab for testing
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    return tokenizer


def test_initialization(mock_tokenizer):
    """Test ReasoningTokenizer initialization."""
    reasoning_token_ids = [10, 20, 30, 40, 50]
    tokenizer = ReasoningTokenizer(mock_tokenizer, reasoning_token_ids)

    assert tokenizer.standard_vocab_size == mock_tokenizer.vocab_size
    assert tokenizer.vocab_size == mock_tokenizer.vocab_size + len(reasoning_token_ids)
    assert len(tokenizer.reasoning_token_ids) == 5


def test_multiplicity_mapping(mock_tokenizer):
    """Test that multiplicity mapping is built correctly."""
    # Include repeated tokens
    reasoning_token_ids = [10, 20, 10, 30, 20, 10]
    tokenizer = ReasoningTokenizer(mock_tokenizer, reasoning_token_ids)

    # Check multiplicities (stored as count - 1, so add 1 when retrieving)
    # First 10: multiplicity 0 (will be 1 after +1)
    assert tokenizer.reasoning_to_multiplicity[0].item() == 0
    # First 20: multiplicity 0
    assert tokenizer.reasoning_to_multiplicity[1].item() == 0
    # Second 10: multiplicity 1
    assert tokenizer.reasoning_to_multiplicity[2].item() == 1
    # First 30: multiplicity 0
    assert tokenizer.reasoning_to_multiplicity[3].item() == 0
    # Second 20: multiplicity 1
    assert tokenizer.reasoning_to_multiplicity[4].item() == 1
    # Third 10: multiplicity 2
    assert tokenizer.reasoning_to_multiplicity[5].item() == 2


def test_standard_tokenization(mock_tokenizer):
    """Test that standard tokenization works."""
    tokenizer = ReasoningTokenizer(mock_tokenizer, [10, 20, 30])

    text = "Hello world"
    encoded = tokenizer.encode(text)

    # Should match base tokenizer
    assert encoded == mock_tokenizer.encode(text)


def test_standard_decoding(mock_tokenizer):
    """Test decoding of standard tokens."""
    tokenizer = ReasoningTokenizer(mock_tokenizer, [10, 20, 30])

    token_ids = [100, 200, 300]
    decoded = tokenizer.decode(token_ids)

    # Should match base tokenizer
    assert decoded == mock_tokenizer.decode(token_ids)


def test_reasoning_token_translation(mock_tokenizer):
    """Test that reasoning tokens are translated to standard tokens."""
    reasoning_token_ids = [10, 20, 30]
    tokenizer = ReasoningTokenizer(mock_tokenizer, reasoning_token_ids)

    # Mix of standard and reasoning tokens
    vocab_size = mock_tokenizer.vocab_size
    token_ids = [
        100,  # standard
        vocab_size + 0,  # reasoning (maps to 10)
        vocab_size + 1,  # reasoning (maps to 20)
        200,  # standard
    ]

    decoded = tokenizer.decode(token_ids)

    # Should decode as if it were [100, 10, 20, 200]
    expected = mock_tokenizer.decode([100, 10, 20, 200])
    assert decoded == expected


def test_decode_with_multiplicity(mock_tokenizer):
    """Test decode_with_multiplicity returns correct multiplicities."""
    reasoning_token_ids = [10, 20, 10]  # 10 appears twice
    tokenizer = ReasoningTokenizer(mock_tokenizer, reasoning_token_ids)

    vocab_size = mock_tokenizer.vocab_size
    token_ids = [
        100,  # standard (mult=0)
        vocab_size + 0,  # first 10 (mult=1)
        vocab_size + 2,  # second 10 (mult=2)
        vocab_size + 1,  # 20 (mult=1)
    ]

    text, infos = tokenizer.decode_with_multiplicity(token_ids)

    # Check multiplicities
    assert len(infos) == 4
    assert infos[0].multiplicity == 0  # standard token
    assert infos[1].multiplicity == 1  # first reasoning occurrence of 10
    assert infos[2].multiplicity == 2  # second reasoning occurrence of 10
    assert infos[3].multiplicity == 1  # first reasoning occurrence of 20


def test_convert_ids_to_tokens_and_multiplicity(mock_tokenizer):
    """Test converting IDs to tokens with multiplicity."""
    reasoning_token_ids = [10, 20, 30]
    tokenizer = ReasoningTokenizer(mock_tokenizer, reasoning_token_ids)

    vocab_size = mock_tokenizer.vocab_size
    token_ids = [100, vocab_size + 0, vocab_size + 1, 200]

    tokens, multiplicities = tokenizer.convert_ids_to_tokens_and_multiplicity(token_ids)

    assert len(tokens) == 4
    assert len(multiplicities) == 4
    assert multiplicities[0] == 0  # standard
    assert multiplicities[1] == 1  # reasoning
    assert multiplicities[2] == 1  # reasoning
    assert multiplicities[3] == 0  # standard


def test_get_token_string_and_multiplicity_standard(mock_tokenizer):
    """Test getting token string and multiplicity for standard token."""
    reasoning_token_ids = [10, 20, 30]
    tokenizer = ReasoningTokenizer(mock_tokenizer, reasoning_token_ids)

    token_string, multiplicity = tokenizer.get_token_string_and_multiplicity(100)

    assert multiplicity == 0
    assert token_string == mock_tokenizer.decode([100])


def test_get_token_string_and_multiplicity_reasoning(mock_tokenizer):
    """Test getting token string and multiplicity for reasoning token."""
    reasoning_token_ids = [10, 20, 30]
    tokenizer = ReasoningTokenizer(mock_tokenizer, reasoning_token_ids)

    vocab_size = mock_tokenizer.vocab_size
    token_id = vocab_size + 1  # Maps to token 20

    token_string, multiplicity = tokenizer.get_token_string_and_multiplicity(token_id)

    assert multiplicity == 1
    assert token_string == mock_tokenizer.decode([20])


def test_batch_decode(mock_tokenizer):
    """Test batch decoding."""
    reasoning_token_ids = [10, 20, 30]
    tokenizer = ReasoningTokenizer(mock_tokenizer, reasoning_token_ids)

    vocab_size = mock_tokenizer.vocab_size
    token_ids_batch = [
        [100, 200, 300],
        [vocab_size + 0, vocab_size + 1, 400],
    ]

    decoded_batch = tokenizer.batch_decode(token_ids_batch)

    assert len(decoded_batch) == 2
    assert decoded_batch[0] == mock_tokenizer.decode([100, 200, 300])
    assert decoded_batch[1] == mock_tokenizer.decode([10, 20, 400])


def test_batch_decode_with_multiplicity(mock_tokenizer):
    """Test batch decoding with multiplicity."""
    reasoning_token_ids = [10, 20, 30]
    tokenizer = ReasoningTokenizer(mock_tokenizer, reasoning_token_ids)

    vocab_size = mock_tokenizer.vocab_size
    token_ids_batch = [
        [100, 200],
        [vocab_size + 0, vocab_size + 1],
    ]

    results = tokenizer.batch_decode_with_multiplicity(token_ids_batch)

    assert len(results) == 2
    text1, infos1 = results[0]
    text2, infos2 = results[1]

    assert len(infos1) == 2
    assert len(infos2) == 2
    assert infos1[0].multiplicity == 0
    assert infos2[0].multiplicity == 1


def test_properties(mock_tokenizer):
    """Test property accessors."""
    tokenizer = ReasoningTokenizer(mock_tokenizer, [10, 20, 30])

    assert tokenizer.pad_token_id == mock_tokenizer.pad_token_id
    assert tokenizer.eos_token_id == mock_tokenizer.eos_token_id
    assert tokenizer.bos_token_id == mock_tokenizer.bos_token_id


def test_edge_case_empty_reasoning_vocab(mock_tokenizer):
    """Test tokenizer with empty reasoning vocabulary."""
    tokenizer = ReasoningTokenizer(mock_tokenizer, [])

    assert tokenizer.vocab_size == mock_tokenizer.vocab_size
    assert len(tokenizer.reasoning_token_ids) == 0

    # Should work like standard tokenizer
    token_ids = [100, 200, 300]
    decoded = tokenizer.decode(token_ids)
    assert decoded == mock_tokenizer.decode(token_ids)


def test_edge_case_all_reasoning_tokens(mock_tokenizer):
    """Test decoding when all tokens are reasoning tokens."""
    reasoning_token_ids = [10, 20, 30]
    tokenizer = ReasoningTokenizer(mock_tokenizer, reasoning_token_ids)

    vocab_size = mock_tokenizer.vocab_size
    token_ids = [vocab_size + 0, vocab_size + 1, vocab_size + 2]

    text, infos = tokenizer.decode_with_multiplicity(token_ids)

    assert len(infos) == 3
    assert all(info.multiplicity > 0 for info in infos)


def test_edge_case_all_standard_tokens(mock_tokenizer):
    """Test decoding when all tokens are standard tokens."""
    reasoning_token_ids = [10, 20, 30]
    tokenizer = ReasoningTokenizer(mock_tokenizer, reasoning_token_ids)

    token_ids = [100, 200, 300]

    text, infos = tokenizer.decode_with_multiplicity(token_ids)

    assert len(infos) == 3
    assert all(info.multiplicity == 0 for info in infos)


def test_token_multiplicity_info():
    """Test TokenMultiplicityInfo dataclass."""
    info = TokenMultiplicityInfo(multiplicity=2, start_index=10, length=5)

    assert info.multiplicity == 2
    assert info.start_index == 10
    assert info.length == 5


def test_reasoning_token_consistency(mock_tokenizer):
    """Test that reasoning tokens decode to same text as their base tokens."""
    reasoning_token_ids = [10, 20, 30]
    tokenizer = ReasoningTokenizer(mock_tokenizer, reasoning_token_ids)

    vocab_size = mock_tokenizer.vocab_size

    for i, token_id in enumerate(reasoning_token_ids):
        reasoning_token = vocab_size + i
        standard_text = mock_tokenizer.decode([token_id])
        reasoning_text = tokenizer.decode([reasoning_token])

        # Should decode to same text
        assert reasoning_text == standard_text
