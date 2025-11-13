"""
Training utilities for RLVR experiments.

This module contains:
- Checkpoint management with reasoning token map saving
- Training callbacks for TRL trainers
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import cast

from loguru import logger
from transformers import PreTrainedModel, TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments


def save_reasoning_token_map(checkpoint_path: Path, model: PreTrainedModel) -> None:
    """
    Save reasoning token map alongside model checkpoint.

    This function should be called whenever a model checkpoint is saved.
    It creates a reasoning_token_map.json file that tracks which standard
    tokens were used to initialize each reasoning token.

    For baseline models (no reasoning vocabulary), it saves empty lists.

    Args:
        checkpoint_path: Path to checkpoint directory
        model: The model being saved

    File format:
        {
            "standard_token_ids": [3, 34, 940, 3, 3],  # Which standard token initialized each reasoning token
            "multiplicities": [1, 1, 1, 2, 3]           # Multiplicity for each reasoning token
        }

        Index i corresponds to reasoning token vocab_size + i
    """
    map_path = checkpoint_path / "reasoning_token_map.json"

    # Assert that model has get_reasoning_token_ids method
    assert hasattr(
        model, "get_reasoning_token_ids"
    ), "Model must have get_reasoning_token_ids() method"

    # Get reasoning token IDs (will be empty tuple for baseline models)
    get_ids = cast(Callable[[], tuple[int, ...]], model.get_reasoning_token_ids)
    standard_token_ids = list(get_ids())

    # Compute multiplicities by counting occurrences of each token
    multiplicities = []
    token_counts: dict[int, int] = {}

    for token_id in standard_token_ids:
        # Get current count for this token (0 if first occurrence)
        count = token_counts.get(token_id, 0)
        # Multiplicity is count + 1 (first occurrence has multiplicity 1)
        multiplicities.append(count + 1)
        # Update count
        token_counts[token_id] = count + 1

    # Save the map
    data = {"standard_token_ids": standard_token_ids, "multiplicities": multiplicities}

    # Ensure checkpoint directory exists
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Write the map
    with open(map_path, "w") as f:
        json.dump(data, f, indent=2)

    if standard_token_ids:
        logger.debug(
            f"Saved reasoning token map with {len(standard_token_ids)} reasoning tokens at {map_path}"
        )
    else:
        logger.debug(f"Saved empty reasoning token map for baseline model at {map_path}")


class ReasoningTokenMapCallback(TrainerCallback):
    """
    Trainer callback that saves reasoning token map alongside checkpoints.

    This callback hooks into the checkpoint saving process and ensures
    that reasoning_token_map.json is created for every checkpoint.

    Usage:
        trainer = GRPOTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            callbacks=[ReasoningTokenMapCallback()],
            ...
        )
    """

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: PreTrainedModel | None = None,
        **kwargs,
    ) -> TrainerControl:
        """
        Called when a checkpoint is being saved.

        Args:
            args: Training arguments
            state: Current trainer state
            control: Trainer control flow
            model: The model being saved
            **kwargs: Additional arguments

        Returns:
            TrainerControl (unchanged)
        """
        if model is None:
            logger.warning("No model provided to ReasoningTokenMapCallback.on_save()")
            return control

        # Get checkpoint path from args
        checkpoint_path = Path(args.output_dir)

        # If we're in the middle of training, get the actual checkpoint-{step} directory
        if state.global_step > 0:
            checkpoint_path = checkpoint_path / f"checkpoint-{state.global_step}"

        # Save the reasoning token map
        save_reasoning_token_map(checkpoint_path, model)

        return control
