"""
Qwen3 model with reasoning vocabulary extension.

This module contains the extended Qwen3ForCausalLM class with:
- Reasoning vocabulary embeddings and unembeddings via resize_token_embeddings
- LogitsProcessor for controlling reasoning vocabulary activation
- Generation logic with <reasoning> tag detection
"""

from abc import abstractmethod
from collections.abc import Sequence

import torch as th
from transformers import LlamaForCausalLM, PretrainedConfig, PreTrainedModel, Qwen3ForCausalLM
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import GenerationMixin

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


class ReasoningVocabModel(PreTrainedModel, GenerationMixin):
    """
    Base class for reasoning vocabulary models.
    """

    @abstractmethod
    def init_reasoning_vocab(
        self, original_vocab_size: int, reasoning_token_ids: Sequence[int] = tuple()
    ) -> None:
        """
        Initialize reasoning vocabulary embeddings from standard vocabulary.
        """
        pass

    @abstractmethod
    def get_reasoning_token_ids(self) -> tuple[int, ...]:
        """
        Get the token IDs used to initialize reasoning embeddings.
        """
        pass

    @abstractmethod
    def num_reasoning_tokens(self) -> int:
        """Get the number of reasoning tokens."""
        pass

    @abstractmethod
    def num_standard_tokens(self) -> int:
        """Get the number of standard tokens."""
        pass


def get_reasoning_class(model_class) -> type[ReasoningVocabModel]:
    assert issubclass(model_class, PreTrainedModel), (
        f"Model class {model_class} must be a subclass of PreTrainedModel"
    )
    assert issubclass(model_class, GenerationMixin), (
        f"Model class {model_class} must be a subclass of GenerationMixin"
    )

    """
    Get a reasoning-enabled class for a given model class.

    Args:
        model_class: The base model class to extend
    Returns:
        Extended model class with reasoning vocabulary support
    """

    assert issubclass(model_class, PreTrainedModel), (
        f"Model class {model_class} must be a subclass of PreTrainedModel"
    )

    class ReasoningModel(ReasoningVocabModel, model_class):
        def __init__(self, config: PretrainedConfig, reasoning_token_ids: Sequence[int] = tuple()):
            super().__init__(config)
            self.init_reasoning_vocab(config.vocab_size, reasoning_token_ids)

        def init_reasoning_vocab(
            self, original_vocab_size: int, reasoning_token_ids: Sequence[int] = tuple()
        ) -> None:
            """
            Initialize reasoning vocabulary embeddings from standard vocabulary.
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
                self.model.embed_tokens.weight[reasoning_start:] = self.model.embed_tokens.weight[
                    token_ids_tensor
                ]
                self.lm_head.weight[reasoning_start:] = self.lm_head.weight[token_ids_tensor]

        def get_reasoning_token_ids(self) -> tuple[int, ...]:
            """
            Get the token IDs used to initialize reasoning embeddings.
            """
            return tuple(self.reasoning_token_ids.tolist())

        def num_reasoning_tokens(self) -> int:
            """Get the number of reasoning tokens."""
            return self.reasoning_vocab_size

        def num_standard_tokens(self) -> int:
            """Get the number of standard tokens."""
            return self.standard_vocab_size

    return ReasoningModel


Qwen3ReasoningVocabForCausalLM = get_reasoning_class(Qwen3ForCausalLM)
LlamaReasoningVocabForCausalLM = get_reasoning_class(LlamaForCausalLM)
