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
    LogitsProcessor that dynamically masks reasoning vocabulary based on thinking tags.

    This processor checks if there's an opening thinking tag without a corresponding
    closing tag in the input sequence. If so, reasoning tokens are enabled.
    If there's both an opening and closing tag, only standard tokens are allowed.

    Args:
        standard_vocab_size: Size of the standard vocabulary
        think_token_id: Token ID for the opening thinking tag (e.g., <think>)
        end_think_token_id: Token ID for the closing thinking tag (e.g., </think>)
    """

    def __init__(
        self,
        standard_vocab_size: int,
        think_token_id: int | None = None,
        end_think_token_id: int | None = None,
    ):
        self.standard_vocab_size = standard_vocab_size
        self.think_token_id = think_token_id
        self.end_think_token_id = end_think_token_id

    def __call__(self, input_ids: th.LongTensor, scores: th.FloatTensor) -> th.FloatTensor:
        """
        Dynamically mask reasoning vocabulary based on thinking tags in input_ids.

        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            scores: Logits from the model (batch_size, vocab_size)

        Returns:
            Modified logits with reasoning tokens masked when appropriate
        """
        # If no thinking tags are configured, disable reasoning vocab
        if self.think_token_id is None or self.end_think_token_id is None:
            scores[:, self.standard_vocab_size :] = float("-inf")
            return scores

        # Check each sequence in the batch
        for batch_idx in range(input_ids.shape[0]):
            sequence = input_ids[batch_idx]

            # Count opening and closing tags
            has_think = (sequence == self.think_token_id).any().item()
            has_end_think = (sequence == self.end_think_token_id).any().item()

            # If we have an opening tag without a closing tag, allow reasoning vocab
            # Otherwise, mask reasoning vocab
            if not (has_think and not has_end_think):
                scores[batch_idx, self.standard_vocab_size :] = float("-inf")

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

        # Convert reasoning_token_ids to tensor and store
        if len(reasoning_token_ids) > 0:
            self.reasoning_token_ids = th.tensor(reasoning_token_ids, dtype=th.long)
        else:
            self.reasoning_token_ids = th.tensor([], dtype=th.long)

        self.reasoning_vocab_size = len(reasoning_token_ids)

        # Extend embeddings if reasoning tokens are provided
        if self.reasoning_vocab_size > 0:
            new_vocab_size = self.standard_vocab_size + self.reasoning_vocab_size
            self.resize_token_embeddings(new_vocab_size)

            # Initialize reasoning embeddings from specified standard tokens
            self._initialize_reasoning_vocab()

    def _initialize_reasoning_vocab(self):
        """
        Initialize reasoning vocabulary embeddings from standard vocabulary.

        Uses efficient tensor indexing to copy embeddings from the specified
        standard tokens to the new reasoning token positions.
        """
        if self.reasoning_vocab_size == 0:
            return

        with th.no_grad():
            # Move tensor to correct device if needed
            token_ids_tensor = self.reasoning_token_ids.to(self.device)

            # Get the reasoning token indices (after standard vocab)
            reasoning_start = self.standard_vocab_size

            # Copy embeddings efficiently using tensor indexing
            self.model.embed_tokens.weight[reasoning_start:] = self.model.embed_tokens.weight[
                token_ids_tensor
            ]
            self.lm_head.weight[reasoning_start:] = self.lm_head.weight[token_ids_tensor]

    def get_reasoning_token_ids(self) -> tuple[int, ...]:
        """
        Get the token IDs used to initialize reasoning embeddings.

        Returns:
            Tuple of token IDs
        """
        return tuple(self.reasoning_token_ids.tolist())

    @property
    def num_reasoning_tokens(self) -> int:
        """Get the number of reasoning tokens."""
        return self.reasoning_vocab_size

    def get_logits_processor(
        self,
        think_token_id: int | None = None,
        end_think_token_id: int | None = None,
    ) -> ReasoningVocabLogitsProcessor:
        """
        Get a LogitsProcessor for dynamic reasoning vocabulary control.

        Args:
            think_token_id: Token ID for opening thinking tag (e.g., <think>)
            end_think_token_id: Token ID for closing thinking tag (e.g., </think>)

        Returns:
            LogitsProcessor instance
        """
        return ReasoningVocabLogitsProcessor(
            self.standard_vocab_size, think_token_id, end_think_token_id
        )
