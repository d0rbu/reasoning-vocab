"""
Qwen3 model with reasoning vocabulary extension.

This module contains the extended Qwen3ForCausalLM class with:
- Reasoning vocabulary embeddings and unembeddings
- Modified forward pass to support reasoning tokens
- Generation logic with <reasoning> tag detection
"""

import torch as th
import torch.nn as nn
from transformers import Qwen2ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast


class Qwen3ReasoningForCausalLM(Qwen2ForCausalLM):
    """
    Extended Qwen3 model with reasoning vocabulary support.

    This model extends the standard Qwen2ForCausalLM (Qwen3 uses Qwen2 architecture)
    with additional reasoning embeddings and unembeddings that can be activated
    during generation when the model enters <reasoning> blocks.

    Args:
        config: Model configuration
        num_reasoning_tokens: Number of reasoning vocabulary tokens (default: None, will match standard vocab)
        reasoning_token_ids: List of token IDs used to initialize reasoning embeddings (default: None, will use random tokens)
    """

    def __init__(
        self,
        config,
        num_reasoning_tokens: int | None = None,
        reasoning_token_ids: list[int] | None = None,
    ):
        super().__init__(config)

        # Determine number of reasoning tokens
        if num_reasoning_tokens is None:
            num_reasoning_tokens = config.vocab_size

        self.num_reasoning_tokens = num_reasoning_tokens
        self.standard_vocab_size = config.vocab_size

        # Store reasoning token IDs for initialization
        self.reasoning_token_ids = reasoning_token_ids

        # Initialize reasoning vocabulary layers
        self.reasoning_embed = nn.Embedding(num_reasoning_tokens, config.hidden_size)
        self.reasoning_unembed = nn.Linear(config.hidden_size, num_reasoning_tokens, bias=False)

        # Initialize reasoning embeddings from standard embeddings
        self._initialize_reasoning_vocab()

    def _initialize_reasoning_vocab(self):
        """
        Initialize reasoning vocabulary from standard vocabulary.

        If reasoning_token_ids is provided, use those specific tokens.
        Otherwise, randomly sample tokens from the standard vocabulary.
        """
        with th.no_grad():
            if self.reasoning_token_ids is not None:
                # Use provided token IDs (repeating if necessary)
                n_repeats = (self.num_reasoning_tokens + len(self.reasoning_token_ids) - 1) // len(
                    self.reasoning_token_ids
                )
                expanded_ids = (self.reasoning_token_ids * n_repeats)[: self.num_reasoning_tokens]

                # Initialize reasoning embeddings from specified tokens
                for i, token_id in enumerate(expanded_ids):
                    self.reasoning_embed.weight[i].copy_(self.model.embed_tokens.weight[token_id])
                    self.reasoning_unembed.weight[i].copy_(self.lm_head.weight[token_id])
            else:
                # Randomly sample from standard vocabulary
                random_indices = th.randperm(self.standard_vocab_size)[: self.num_reasoning_tokens]

                for i, idx in enumerate(random_indices):
                    self.reasoning_embed.weight[i].copy_(self.model.embed_tokens.weight[idx])
                    self.reasoning_unembed.weight[i].copy_(self.lm_head.weight[idx])

    def get_reasoning_token_ids(self) -> list[int]:
        """
        Get the token IDs used to initialize reasoning embeddings.

        Returns:
            List of token IDs
        """
        if self.reasoning_token_ids is not None:
            n_repeats = (self.num_reasoning_tokens + len(self.reasoning_token_ids) - 1) // len(
                self.reasoning_token_ids
            )
            return (self.reasoning_token_ids * n_repeats)[: self.num_reasoning_tokens]
        return []

    def forward(
        self,
        input_ids: th.LongTensor | None = None,
        attention_mask: th.Tensor | None = None,
        position_ids: th.LongTensor | None = None,
        past_key_values: list[th.FloatTensor] | None = None,
        inputs_embeds: th.FloatTensor | None = None,
        labels: th.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: th.LongTensor | None = None,
        use_reasoning_vocab: bool = False,
        **kwargs,  # Accept additional kwargs for compatibility
    ) -> tuple | CausalLMOutputWithPast:
        """
        Forward pass with optional reasoning vocabulary.

        Args:
            use_reasoning_vocab: If True, concatenates reasoning logits to standard logits

        Returns:
            CausalLMOutputWithPast with potentially expanded vocabulary logits
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Handle inputs_embeds for reasoning tokens if needed
        if inputs_embeds is None and input_ids is not None:
            # Check if any input_ids are in reasoning vocab range
            if use_reasoning_vocab and (input_ids >= self.standard_vocab_size).any():
                # Create embeddings, handling both standard and reasoning tokens
                inputs_embeds = th.zeros(
                    (*input_ids.shape, self.config.hidden_size),
                    dtype=self.model.embed_tokens.weight.dtype,
                    device=input_ids.device,
                )

                # Standard tokens
                standard_mask = input_ids < self.standard_vocab_size
                if standard_mask.any():
                    inputs_embeds[standard_mask] = self.model.embed_tokens(input_ids[standard_mask])

                # Reasoning tokens
                reasoning_mask = input_ids >= self.standard_vocab_size
                if reasoning_mask.any():
                    reasoning_ids = input_ids[reasoning_mask] - self.standard_vocab_size
                    inputs_embeds[reasoning_mask] = self.reasoning_embed(reasoning_ids)

        # Call parent forward
        outputs = super().forward(
            input_ids=input_ids if inputs_embeds is None else None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=None,  # We'll handle labels ourselves
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,  # Pass through additional kwargs
        )

        hidden_states = outputs.hidden_states[-1] if output_hidden_states else outputs.logits

        # Get standard logits (already computed by parent)
        standard_logits = outputs.logits

        # Compute reasoning logits if requested
        if use_reasoning_vocab:
            # Get hidden states from the model output
            # We need to re-extract hidden states if they weren't output
            if not output_hidden_states:
                # Run through model again to get hidden states
                model_outputs = self.model(
                    input_ids=input_ids if inputs_embeds is None else None,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=True,
                    return_dict=True,
                    cache_position=cache_position,
                )
                hidden_states = model_outputs.last_hidden_state
            else:
                hidden_states = outputs.hidden_states[-1]

            reasoning_logits = self.reasoning_unembed(hidden_states)

            # Concatenate standard and reasoning logits
            logits = th.cat([standard_logits, reasoning_logits], dim=-1)
        else:
            logits = standard_logits

        # Handle labels if provided
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @th.no_grad()
    def generate(
        self,
        input_ids: th.LongTensor | None = None,
        attention_mask: th.Tensor | None = None,
        reasoning_start_token_id: int | None = None,
        reasoning_end_token_id: int | None = None,
        **kwargs,
    ):
        """
        Generate text with automatic reasoning vocabulary activation.

        This method wraps the standard generate() and automatically activates
        the reasoning vocabulary when <reasoning> tokens are detected.

        Args:
            reasoning_start_token_id: Token ID for <reasoning>
            reasoning_end_token_id: Token ID for </reasoning>
            **kwargs: Additional arguments passed to parent generate()

        Returns:
            Generated token IDs
        """
        # For now, use standard generation
        # TODO: Implement tag-based reasoning vocabulary switching
        # This requires a custom generation loop or logits processor

        return super().generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
