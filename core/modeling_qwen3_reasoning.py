"""
Qwen3 model with reasoning vocabulary extension.

This module contains the extended Qwen3ForCausalLM class with:
- Reasoning vocabulary embeddings and unembeddings via resize_token_embeddings
- LogitsProcessor for controlling reasoning vocabulary activation
- Generation logic with <reasoning> tag detection
"""

from collections.abc import Sequence
from typing import Any

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
        super().__init__(config)

        self._init_reasoning_vocab(config.vocab_size, reasoning_token_ids)

    def _init_reasoning_vocab(self, original_vocab_size: int, reasoning_token_ids: Sequence[int] = tuple()) -> None:
        """
        Initialize reasoning vocabulary embeddings from standard vocabulary.

        Uses efficient tensor indexing to copy embeddings from the specified
        standard tokens to the new reasoning token positions.
        """
        self.standard_vocab_size: int = original_vocab_size

        self.reasoning_token_ids: th.Tensor = th.tensor(reasoning_token_ids, dtype=th.long)
        self.reasoning_vocab_size: int = len(reasoning_token_ids)

        if self.reasoning_vocab_size == 0:
            return

        self.resize_token_embeddings(self.standard_vocab_size + self.reasoning_vocab_size)

        with th.no_grad():
            token_ids_tensor = self.reasoning_token_ids.to(self.device)
            reasoning_start = self.standard_vocab_size

            # copy embeddings from target tokens to reasoning tokens
            self.model.embed_tokens.weight[reasoning_start:] = self.model.embed_tokens.weight[
                token_ids_tensor
            ]
            self.lm_head.weight[reasoning_start:] = self.lm_head.weight[token_ids_tensor]

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        reasoning_token_ids: Sequence[int] = tuple(),
        **kwargs: dict[str, Any],
    ) -> "Qwen3ReasoningVocabForCausalLM":
        """
        Load a pretrained model with reasoning vocabulary support.
        This method loads a pretrained Qwen3 model and extends it with reasoning
        vocabulary by:
        1. Loading the config
        2. Creating an instance with reasoning vocabulary
        3. Loading pretrained weights (strict=False to handle new reasoning tokens)
        Args:
            pretrained_model_name_or_path: Model name or path
            reasoning_token_ids: Sequence of token IDs to initialize reasoning vocab from
            **kwargs: Additional arguments passed to from_pretrained
        Returns:
            Qwen3ReasoningVocabForCausalLM instance with pretrained weights
        """
        model = super().from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=kwargs.get("trust_remote_code", False),
        )
        model._init_reasoning_vocab(model.config.vocab_size, reasoning_token_ids)

        return model

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
