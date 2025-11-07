"""
Utility functions for reasoning vocabulary management.

This module contains helper functions for generating and managing reasoning token IDs.
"""


def get_reasoning_token_ids(vocab_size: int) -> tuple[int, ...]:
    """
    Generate reasoning token IDs from 0 to vocab_size-1.

    This helper function creates a tuple of token IDs that can be used to initialize
    the reasoning vocabulary. By default, it includes all tokens from the standard
    vocabulary.

    Args:
        vocab_size: Size of the standard vocabulary

    Returns:
        Tuple of token IDs from 0 to vocab_size-1

    Example:
        >>> token_ids = get_reasoning_token_ids(1000)
        >>> len(token_ids)
        1000
        >>> token_ids[0]
        0
        >>> token_ids[-1]
        999
    """
    return tuple(range(vocab_size))
