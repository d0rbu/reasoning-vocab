"""
Tokenizer utilities for handling standard and reasoning vocabularies.

This module contains:
- ReasoningTokenizer class for managing reasoning token translation
- Methods for tokenizing/detokenizing with reasoning tokens
- Multiplicity tracking for reasoning tokens
- TokenMultiplicityInfo dataclass for rich token information
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import torch as th
from transformers import PreTrainedTokenizer


@dataclass
class TokenMultiplicityInfo:
    """
    Information about a token's multiplicity and position in decoded text.

    Attributes:
        multiplicity: Token multiplicity (0 for standard tokens, >=1 for reasoning tokens)
        start_index: Starting character index in the decoded string
        length: Length of the token string in characters
    """

    multiplicity: int
    start_index: int
    length: int


class ReasoningTokenizer:
    """
    Tokenizer wrapper that handles both standard and reasoning vocabularies.

    The reasoning tokenizer maps reasoning token IDs (>= standard_vocab_size) to their
    corresponding standard token IDs for detokenization, while tracking
    multiplicities for tokens that appear multiple times in the reasoning vocab.

    Args:
        tokenizer: Base HuggingFace tokenizer for standard vocabulary
        reasoning_token_ids: Sequence of standard token IDs used to initialize reasoning vocab

    Example:
        >>> from transformers import AutoTokenizer
        >>> base_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        >>> reasoning_tokenizer = ReasoningTokenizer(
        ...     tokenizer=base_tokenizer,
        ...     reasoning_token_ids=[10, 20, 30, 10, 20]  # Note: 10 and 20 repeat
        ... )
        >>> # Decode with multiplicity tracking
        >>> token_ids = [10, base_tokenizer.vocab_size + 0, base_tokenizer.vocab_size + 3]
        >>> text, infos = reasoning_tokenizer.decode_with_multiplicity(token_ids)
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, reasoning_token_ids: Sequence[int]):
        self.tokenizer = tokenizer
        self.standard_vocab_size = tokenizer.vocab_size

        self.reasoning_token_ids = th.tensor(reasoning_token_ids, dtype=th.long)
        self.reasoning_vocab_size = len(reasoning_token_ids)

        # Create mapping for multiplicity tracking
        self._build_reasoning_mapping()

    def _build_reasoning_mapping(self):
        """
        Build internal multiplicity mapping for reasoning tokens.

        Creates reasoning_to_multiplicity tensor that tracks how many times
        each token appears in the reasoning vocabulary.
        """
        token_counter = Counter()

        self.reasoning_to_multiplicity = th.empty(self.reasoning_vocab_size, dtype=th.long)

        for idx, token_id in enumerate(self.reasoning_token_ids.tolist()):
            self.reasoning_to_multiplicity[idx] = token_counter[token_id]
            token_counter[token_id] += 1

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size (standard + reasoning tokens)."""
        return self.standard_vocab_size + self.reasoning_vocab_size

    def encode(self, text: str | list[str], **kwargs) -> list[int] | th.Tensor:
        """
        Encode text using standard tokenizer.

        Reasoning tokens are only used during generation, not encoding.

        Args:
            text: Text or list of texts to encode
            **kwargs: Additional arguments passed to tokenizer

        Returns:
            Encoded token IDs
        """
        return self.tokenizer.encode(text, **kwargs)

    def _convert_to_standard_ids_and_multiplicities(
        self, token_ids: th.Tensor
    ) -> tuple[th.Tensor, th.Tensor]:
        """
        Convert token IDs to standard vocabulary IDs and get multiplicities.

        Efficiently processes both standard and reasoning tokens in one pass.

        Args:
            token_ids: Token IDs to convert

        Returns:
            Tuple of (standard_ids, multiplicities)
        """
        standard_ids = token_ids.clone()
        multiplicities = th.zeros_like(token_ids, dtype=th.long)

        reasoning_mask = token_ids >= self.standard_vocab_size

        if reasoning_mask.any():
            reasoning_indices = token_ids[reasoning_mask] - self.standard_vocab_size

            standard_ids[reasoning_mask] = self.reasoning_token_ids[reasoning_indices]

            multiplicities[reasoning_mask] = self.reasoning_to_multiplicity[reasoning_indices] + 1

        return standard_ids, multiplicities

    def convert_ids_to_tokens_and_multiplicity(
        self, token_ids: list[int] | th.Tensor, **kwargs
    ) -> tuple[list[str], list[int]]:
        """
        Convert token IDs to token strings and multiplicities.

        Args:
            token_ids: Sequence of token IDs
            **kwargs: Additional arguments to pass to tokenizer.convert_ids_to_tokens

        Returns:
            Tuple of (token_strings, multiplicities)
        """
        # Convert to tensor if needed
        if isinstance(token_ids, list):
            token_ids = th.tensor(token_ids, dtype=th.long)

        # Get standard IDs and multiplicities
        standard_ids, multiplicities = self._convert_to_standard_ids_and_multiplicities(token_ids)

        # Convert IDs to tokens using base tokenizer
        tokens = self.tokenizer.convert_ids_to_tokens(standard_ids.tolist(), **kwargs)

        return tokens, multiplicities.tolist()

    def convert_tokens_and_multiplicity_to_string_and_multiplicity(
        self, tokens: list[str], multiplicities: list[int]
    ) -> tuple[str, list[TokenMultiplicityInfo]]:
        """
        Convert tokens and multiplicities to a string with positional information.

        Args:
            tokens: List of token strings
            multiplicities: List of token multiplicities

        Returns:
            Tuple of (decoded_string, multiplicity_info_list)
        """
        # Convert tokens to string
        decoded_string = self.tokenizer.convert_tokens_to_string(tokens)

        # Build multiplicity info with positions
        multiplicity_infos = []
        current_pos = 0

        for token, mult in zip(tokens, multiplicities, strict=True):
            # Decode individual token to get its string representation
            token_str = self.tokenizer.convert_tokens_to_string([token])
            token_len = len(token_str)

            # Find token in decoded string starting from current position
            # This handles cases where tokenizer adds/removes spaces
            idx = decoded_string.find(token_str, current_pos)
            if idx == -1:
                # Token not found as-is, use current position
                idx = current_pos

            multiplicity_infos.append(
                TokenMultiplicityInfo(multiplicity=mult, start_index=idx, length=token_len)
            )

            current_pos = idx + token_len

        return decoded_string, multiplicity_infos

    def decode(self, token_ids: list[int] | th.Tensor, **kwargs) -> str:
        """
        Decode token IDs to text, handling both standard and reasoning tokens.

        Reasoning tokens (>= standard_vocab_size) are converted to their corresponding
        standard tokens before decoding.

        Args:
            token_ids: Token IDs to decode
            **kwargs: Additional arguments passed to tokenizer (e.g., skip_special_tokens)

        Returns:
            Decoded text string
        """
        text, _ = self.decode_with_multiplicity(token_ids, **kwargs)
        return text

    def decode_with_multiplicity(
        self, token_ids: list[int] | th.Tensor, **kwargs
    ) -> tuple[str, list[TokenMultiplicityInfo]]:
        """
        Decode token IDs to text with multiplicity information.

        Returns both the decoded text and detailed information about each token's
        multiplicity and position in the decoded string.

        Args:
            token_ids: Token IDs to decode
            **kwargs: Additional arguments passed to tokenizer (e.g., skip_special_tokens)

        Returns:
            Tuple of (decoded_text, multiplicity_info_list)
        """
        # Convert IDs to tokens and multiplicities
        tokens, multiplicities = self.convert_ids_to_tokens_and_multiplicity(token_ids)

        # Convert tokens to string with positional info
        text, infos = self.convert_tokens_and_multiplicity_to_string_and_multiplicity(
            tokens, multiplicities
        )

        return text, infos

    def batch_decode(self, token_ids: list[list[int]] | th.Tensor, **kwargs) -> list[str]:
        """
        Decode batch of token ID sequences.

        Args:
            token_ids: Batch of token ID sequences
            **kwargs: Additional arguments passed to tokenizer (e.g., skip_special_tokens)

        Returns:
            List of decoded text strings
        """
        if isinstance(token_ids, th.Tensor):
            token_ids = token_ids.tolist()

        return [self.decode(seq, **kwargs) for seq in token_ids]

    def batch_decode_with_multiplicity(
        self, token_ids: list[list[int]] | th.Tensor, **kwargs
    ) -> list[tuple[str, list[TokenMultiplicityInfo]]]:
        """
        Decode batch of token ID sequences with multiplicity information.

        Args:
            token_ids: Batch of token ID sequences
            **kwargs: Additional arguments passed to tokenizer (e.g., skip_special_tokens)

        Returns:
            List of (decoded_text, multiplicity_info_list) tuples
        """
        if isinstance(token_ids, th.Tensor):
            token_ids = token_ids.tolist()

        return [self.decode_with_multiplicity(seq, **kwargs) for seq in token_ids]

    def get_token_string_and_multiplicity(self, token_id: int) -> tuple[str, int]:
        """
        Get the string representation and multiplicity for a single token ID.

        Args:
            token_id: Token ID to decode

        Returns:
            Tuple of (token_string, multiplicity)
        """
        if token_id < self.standard_vocab_size:
            return self.tokenizer.decode([token_id]), 0

        reasoning_idx = token_id - self.standard_vocab_size
        standard_id = self.reasoning_token_ids[reasoning_idx].item()
        multiplicity = self.reasoning_to_multiplicity[reasoning_idx].item() + 1
        token_string = self.tokenizer.decode([standard_id])

        return token_string, multiplicity

    # Property accessors for convenience
    @property
    def pad_token_id(self):
        """Get pad token ID from base tokenizer."""
        return self.tokenizer.pad_token_id

    @property
    def eos_token_id(self):
        """Get EOS token ID from base tokenizer."""
        return self.tokenizer.eos_token_id

    @property
    def bos_token_id(self):
        """Get BOS token ID from base tokenizer."""
        return self.tokenizer.bos_token_id
