"""
Tokenizer utilities for handling standard and reasoning vocabularies.

This module contains:
- ReasoningTokenizer class for managing reasoning token translation
- Methods for tokenizing/detokenizing with reasoning tokens
- Multiplicity tracking for reasoning tokens
"""

import torch as th
from transformers import PreTrainedTokenizer


class ReasoningTokenizer:
    """
    Tokenizer wrapper that handles both standard and reasoning vocabularies.

    The reasoning tokenizer maps reasoning token IDs (>= vocab_size) to their
    corresponding standard token IDs for detokenization, while tracking
    multiplicities for tokens that appear multiple times in the reasoning vocab.

    Args:
        tokenizer: Base HuggingFace tokenizer for standard vocabulary
        reasoning_token_ids: List of standard token IDs used to initialize reasoning vocab
        vocab_size: Size of the standard vocabulary (reasoning tokens start at vocab_size)
    """

    def __init__(
        self, tokenizer: PreTrainedTokenizer, reasoning_token_ids: list[int], vocab_size: int
    ):
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.reasoning_token_ids = reasoning_token_ids

        # Create mapping from reasoning token ID to (standard token ID, multiplicity)
        self._build_reasoning_mapping()

    def _build_reasoning_mapping(self):
        """
        Build internal mappings for reasoning tokens.

        Creates:
        - reasoning_to_standard: Maps reasoning indices to standard token IDs
        - reasoning_to_multiplicity: Maps reasoning indices to their multiplicity
        """
        # Track how many times each standard token appears in reasoning vocab
        token_count = {}

        self.reasoning_to_standard = th.zeros(len(self.reasoning_token_ids), dtype=th.long)
        self.reasoning_to_multiplicity = th.zeros(len(self.reasoning_token_ids), dtype=th.long)

        for idx, token_id in enumerate(self.reasoning_token_ids):
            self.reasoning_to_standard[idx] = token_id

            # Track multiplicity (how many times this token appears)
            if token_id not in token_count:
                token_count[token_id] = 0
            else:
                token_count[token_id] += 1

            # Multiplicity starts at 1 for reasoning tokens (0 is for standard vocab)
            self.reasoning_to_multiplicity[idx] = token_count[token_id] + 1

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

    def decode(
        self, token_ids: list[int] | th.Tensor, skip_special_tokens: bool = False, **kwargs
    ) -> str:
        """
        Decode token IDs to text, handling both standard and reasoning tokens.

        Reasoning tokens (>= vocab_size) are converted to their corresponding
        standard tokens before decoding.

        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens
            **kwargs: Additional arguments passed to tokenizer

        Returns:
            Decoded text string
        """
        # Convert to tensor if needed
        if isinstance(token_ids, list):
            token_ids = th.tensor(token_ids)

        # Convert reasoning tokens to standard tokens
        standard_ids = self._convert_to_standard_ids(token_ids)

        # Decode using standard tokenizer
        return self.tokenizer.decode(
            standard_ids.tolist(), skip_special_tokens=skip_special_tokens, **kwargs
        )

    def decode_with_multiplicity(
        self, token_ids: list[int] | th.Tensor, skip_special_tokens: bool = False, **kwargs
    ) -> tuple[str, th.Tensor]:
        """
        Decode token IDs to text with multiplicity information.

        Returns both the decoded text and a tensor of multiplicities for each token.
        Standard tokens have multiplicity 0, reasoning tokens have multiplicity >= 1.

        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens
            **kwargs: Additional arguments passed to tokenizer

        Returns:
            Tuple of (decoded_text, multiplicities_tensor)
        """
        # Convert to tensor if needed
        if isinstance(token_ids, list):
            token_ids = th.tensor(token_ids)

        # Get multiplicities
        multiplicities = self._get_multiplicities(token_ids)

        # Convert reasoning tokens to standard tokens
        standard_ids = self._convert_to_standard_ids(token_ids)

        # Decode using standard tokenizer
        text = self.tokenizer.decode(
            standard_ids.tolist(), skip_special_tokens=skip_special_tokens, **kwargs
        )

        return text, multiplicities

    def batch_decode(
        self, token_ids: list[list[int]] | th.Tensor, skip_special_tokens: bool = False, **kwargs
    ) -> list[str]:
        """
        Decode batch of token ID sequences.

        Args:
            token_ids: Batch of token ID sequences
            skip_special_tokens: Whether to skip special tokens
            **kwargs: Additional arguments passed to tokenizer

        Returns:
            List of decoded text strings
        """
        # Convert to tensor if needed
        if isinstance(token_ids, list):
            token_ids = th.tensor(token_ids)

        # Convert reasoning tokens to standard tokens
        standard_ids = self._convert_to_standard_ids(token_ids)

        # Decode using standard tokenizer
        return self.tokenizer.batch_decode(
            standard_ids.tolist(), skip_special_tokens=skip_special_tokens, **kwargs
        )

    def get_token_string_and_multiplicity(self, token_id: int) -> tuple[str, int]:
        """
        Get the string representation and multiplicity for a single token ID.

        Args:
            token_id: Token ID to decode

        Returns:
            Tuple of (token_string, multiplicity)
        """
        if token_id >= self.vocab_size:
            # Reasoning token
            reasoning_idx = token_id - self.vocab_size
            standard_id = self.reasoning_to_standard[reasoning_idx].item()
            multiplicity = self.reasoning_to_multiplicity[reasoning_idx].item()
            token_string = self.tokenizer.decode([standard_id])
        else:
            # Standard token
            token_string = self.tokenizer.decode([token_id])
            multiplicity = 0

        return token_string, multiplicity

    def get_token_strings_and_multiplicities(
        self, token_ids: list[int] | th.Tensor
    ) -> tuple[list[str], th.Tensor]:
        """
        Get token strings and multiplicities for a sequence of token IDs.

        Args:
            token_ids: Sequence of token IDs

        Returns:
            Tuple of (token_strings_list, multiplicities_tensor)
        """
        # Convert to tensor if needed
        if isinstance(token_ids, list):
            token_ids = th.tensor(token_ids)

        # Get multiplicities
        multiplicities = self._get_multiplicities(token_ids)

        # Convert to standard IDs
        standard_ids = self._convert_to_standard_ids(token_ids)

        # Get individual token strings
        token_strings = [self.tokenizer.decode([tid]) for tid in standard_ids.tolist()]

        return token_strings, multiplicities

    def _convert_to_standard_ids(self, token_ids: th.Tensor) -> th.Tensor:
        """
        Convert token IDs to standard vocabulary IDs.

        Tokens < vocab_size are kept as-is.
        Tokens >= vocab_size are converted to their corresponding standard token IDs.

        Args:
            token_ids: Tensor of token IDs

        Returns:
            Tensor of standard vocabulary token IDs
        """
        standard_ids = token_ids.clone()

        # Find reasoning tokens
        reasoning_mask = token_ids >= self.vocab_size

        if reasoning_mask.any():
            # Get reasoning indices
            reasoning_indices = token_ids[reasoning_mask] - self.vocab_size

            # Map to standard tokens
            standard_ids[reasoning_mask] = self.reasoning_to_standard[reasoning_indices]

        return standard_ids

    def _get_multiplicities(self, token_ids: th.Tensor) -> th.Tensor:
        """
        Get multiplicity values for token IDs.

        Standard tokens (< vocab_size) have multiplicity 0.
        Reasoning tokens (>= vocab_size) have multiplicity >= 1.

        Args:
            token_ids: Tensor of token IDs

        Returns:
            Tensor of multiplicity values
        """
        multiplicities = th.zeros_like(token_ids)

        # Find reasoning tokens
        reasoning_mask = token_ids >= self.vocab_size

        if reasoning_mask.any():
            # Get reasoning indices
            reasoning_indices = token_ids[reasoning_mask] - self.vocab_size

            # Get multiplicities
            multiplicities[reasoning_mask] = self.reasoning_to_multiplicity[reasoning_indices]

        return multiplicities

    @property
    def pad_token_id(self) -> int | None:
        """Get pad token ID from base tokenizer."""
        return self.tokenizer.pad_token_id

    @property
    def eos_token_id(self) -> int | None:
        """Get EOS token ID from base tokenizer."""
        return self.tokenizer.eos_token_id

    @property
    def bos_token_id(self) -> int | None:
        """Get BOS token ID from base tokenizer."""
        return self.tokenizer.bos_token_id
