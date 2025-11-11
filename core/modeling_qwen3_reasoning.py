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
from transformers.generation.logits_process import LogitsProcessor

from core.tokenizer_utils import ReasoningTokenizer


class ReasoningVocabLogitsProcessor(LogitsProcessor):
    """
    LogitsProcessor that dynamically masks reasoning vocabulary based on thinking tags.

    This processor decodes the input sequence and checks for string patterns
    like "<think>" and "</think>" to determine whether reasoning tokens should
    be available. This handles multi-token tag sequences correctly.

    Args:
        standard_vocab_size: Size of the standard vocabulary
        tokenizer: ReasoningTokenizer instance for decoding sequences with
            support for tokens outside the normal vocab range (reasoning tokens).
        think_tag: String pattern for opening thinking tag (default: "<think>")
        end_think_tag: String pattern for closing thinking tag (default: "</think>")
    """

    def __init__(
        self,
        standard_vocab_size: int,
        tokenizer: ReasoningTokenizer,
        think_tag: str = "<think>",
        end_think_tag: str = "</think>",
    ):
        self.standard_vocab_size = standard_vocab_size
        self.tokenizer = tokenizer
        self.think_tag = think_tag
        self.end_think_tag = end_think_tag

    def __call__(self, input_ids: th.LongTensor, scores: th.FloatTensor) -> th.FloatTensor:
        """
        Dynamically mask reasoning vocabulary based on thinking tags in input_ids.

        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            scores: Logits from the model (batch_size, vocab_size)

        Returns:
            Modified logits with reasoning tokens masked when appropriate
        """
        # Check each sequence in the batch
        for batch_idx, sequence in enumerate(input_ids):
            # Decode sequence to text
            text = self.tokenizer.decode(sequence, skip_special_tokens=False)

            # Check for think tags in the decoded text
            has_think = self.think_tag in text
            has_end_think = self.end_think_tag in text

            # If we don't have an opening tag, or we have a closing tag, mask reasoning vocab
            if not has_think or has_end_think:
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
        self.standard_vocab_size: int = original_vocab_size

        # Convert reasoning_token_ids to tensor and store
        self.reasoning_token_ids = th.tensor(reasoning_token_ids, dtype=th.long)
        reasoning_vocab_size: int = len(reasoning_token_ids)
        self.reasoning_vocab_size: int = reasoning_vocab_size

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
