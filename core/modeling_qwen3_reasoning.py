"""
Qwen3 model with reasoning vocabulary extension.

This module contains the extended Qwen3ForCausalLM class with:
- Reasoning vocabulary embeddings and unembeddings via resize_token_embeddings
- LogitsProcessor for controlling reasoning vocabulary activation
- Generation logic with <reasoning> tag detection
"""

from collections.abc import Sequence

import torch as th
from transformers import Qwen3ForCausalLM
from transformers.generation import LogitsProcessor


class ReasoningVocabLogitsProcessor(LogitsProcessor):
    """
    LogitsProcessor that masks reasoning vocabulary logits when not in reasoning mode.

    This processor ensures that reasoning tokens (those beyond the standard vocabulary)
    are only available when the model is in reasoning mode.

    Args:
        standard_vocab_size: Size of the standard vocabulary
        use_reasoning_vocab: Whether to enable reasoning vocabulary
    """

    def __init__(self, standard_vocab_size: int, use_reasoning_vocab: bool = False):
        self.standard_vocab_size = standard_vocab_size
        self.use_reasoning_vocab = use_reasoning_vocab

    def __call__(self, input_ids: th.LongTensor, scores: th.FloatTensor) -> th.FloatTensor:
        """
        Mask reasoning vocabulary logits when not in reasoning mode.

        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            scores: Logits from the model (batch_size, vocab_size)

        Returns:
            Modified logits with reasoning tokens masked if not in reasoning mode
        """
        if not self.use_reasoning_vocab:
            # Mask out reasoning vocabulary by setting logits to -inf
            scores[:, self.standard_vocab_size :] = float("-inf")

        return scores


class Qwen3ReasoningVocabForCausalLM(Qwen3ForCausalLM):
    """
    Extended Qwen3 model with reasoning vocabulary support.

    This model extends the standard Qwen3ForCausalLM by resizing the token embeddings
    to include additional reasoning tokens. A LogitsProcessor controls when these
    reasoning tokens are available during generation.

    Args:
        config: Model configuration
        reasoning_token_ids: Sequence of token IDs to initialize reasoning vocab from.
                           Defaults to empty tuple (no reasoning tokens).

    Example:
        >>> from transformers import AutoConfig
        >>> from core.reasoning_vocab_utils import get_reasoning_token_ids
        >>>
        >>> config = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")
        >>> # Initialize with all tokens as reasoning tokens
        >>> token_ids = get_reasoning_token_ids(config.vocab_size)
        >>> model = Qwen3ReasoningVocabForCausalLM(config, reasoning_token_ids=token_ids)
    """

    def __init__(
        self,
        config,
        reasoning_token_ids: Sequence[int] = tuple(),
    ):
        # Store the original vocab size before any modifications
        original_vocab_size = config.vocab_size

        super().__init__(config)

        # Store original vocab size as an attribute (do this after super().__init__)
        self.standard_vocab_size = original_vocab_size
        self.reasoning_token_ids = tuple(reasoning_token_ids)

        # Extend embeddings if reasoning tokens are provided
        if len(self.reasoning_token_ids) > 0:
            new_vocab_size = self.standard_vocab_size + len(self.reasoning_token_ids)
            self.resize_token_embeddings(new_vocab_size)

            # Initialize reasoning embeddings from specified standard tokens
            self._initialize_reasoning_vocab()

    def _initialize_reasoning_vocab(self):
        """
        Initialize reasoning vocabulary embeddings from standard vocabulary.

        Uses efficient tensor indexing to copy embeddings from the specified
        standard tokens to the new reasoning token positions.
        """
        if len(self.reasoning_token_ids) == 0:
            return

        with th.no_grad():
            # Convert to tensor for efficient indexing
            token_ids_tensor = th.tensor(
                self.reasoning_token_ids, dtype=th.long, device=self.device
            )

            # Get the reasoning token indices (after standard vocab)
            reasoning_start = self.standard_vocab_size
            reasoning_end = reasoning_start + len(self.reasoning_token_ids)

            # Copy embeddings efficiently using tensor indexing
            self.model.embed_tokens.weight[reasoning_start:reasoning_end] = (
                self.model.embed_tokens.weight[token_ids_tensor]
            )
            self.lm_head.weight[reasoning_start:reasoning_end] = self.lm_head.weight[
                token_ids_tensor
            ]

    def get_reasoning_token_ids(self) -> tuple[int, ...]:
        """
        Get the token IDs used to initialize reasoning embeddings.

        Returns:
            Tuple of token IDs
        """
        return self.reasoning_token_ids

    @property
    def num_reasoning_tokens(self) -> int:
        """Get the number of reasoning tokens."""
        return len(self.reasoning_token_ids)

    def get_logits_processor(
        self, use_reasoning_vocab: bool = False
    ) -> ReasoningVocabLogitsProcessor:
        """
        Get a LogitsProcessor for controlling reasoning vocabulary.

        Args:
            use_reasoning_vocab: Whether to enable reasoning vocabulary

        Returns:
            LogitsProcessor instance
        """
        return ReasoningVocabLogitsProcessor(self.standard_vocab_size, use_reasoning_vocab)
