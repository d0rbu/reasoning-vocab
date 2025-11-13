"""
Training utilities for RLVR experiments.

This module contains:
- Checkpoint management with reasoning token map saving
- Training callbacks for TRL trainers
"""

import json
from pathlib import Path

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

    # Check if model has reasoning vocabulary extension
    has_reasoning_vocab = hasattr(model, "reasoning_embed") or hasattr(
        model, "reasoning_std_token_ids"
    )

    if not has_reasoning_vocab:
        # Baseline model - save empty lists
        data = {"standard_token_ids": [], "multiplicities": []}
        logger.debug(f"Saving empty reasoning token map for baseline model at {map_path}")
    else:
        # Model with reasoning vocabulary - extract initialization info
        if hasattr(model, "reasoning_std_token_ids") and hasattr(model, "reasoning_multiplicities"):
            # Model stores these explicitly
            std_ids = model.reasoning_std_token_ids
            mults = model.reasoning_multiplicities

            # Convert torch tensors to lists if needed
            if hasattr(std_ids, "tolist") and callable(getattr(std_ids, "tolist", None)):
                std_ids = std_ids.tolist()  # type: ignore[union-attr]
            if hasattr(mults, "tolist") and callable(getattr(mults, "tolist", None)):
                mults = mults.tolist()  # type: ignore[union-attr]

            data = {
                "standard_token_ids": list(std_ids),  # type: ignore[arg-type]
                "multiplicities": list(mults),  # type: ignore[arg-type]
            }
            logger.debug(
                f"Saving reasoning token map with {len(std_ids)} reasoning tokens at {map_path}"  # type: ignore[arg-type]
            )
        else:
            # Model has reasoning vocab but doesn't expose the mapping
            # This shouldn't happen if the model is implemented correctly
            logger.warning(
                f"Model has reasoning_embed but no reasoning_std_token_ids/reasoning_multiplicities. "
                f"Saving empty map at {map_path}"
            )
            data = {"standard_token_ids": [], "multiplicities": []}

    # Ensure checkpoint directory exists
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Write the map
    with open(map_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.debug(f"Reasoning token map saved to {map_path}")


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
