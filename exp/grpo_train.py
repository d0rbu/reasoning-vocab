"""
GRPO training script for reasoning vocabulary experiments.

This script uses Hydra for configuration management and WandB for logging.

Usage:
    python grpo_train.py                           # Use default configs
    python grpo_train.py training.learning_rate=1e-5  # Override specific params
    python grpo_train.py exp_name=my_experiment    # Change experiment name
"""

import os
from pathlib import Path
from typing import Any, cast

import hydra
import torch as th
import wandb
from datasets import Dataset, load_dataset
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from trl import GRPOConfig, GRPOTrainer
from trl.rewards import accuracy_reward, think_format_reward

from core.modeling_qwen3_reasoning import Qwen3ReasoningVocabForCausalLM
from core.train_utils import save_reasoning_token_map


def setup_wandb(cfg: DictConfig) -> None:
    """
    Initialize Weights & Biases logging.

    Args:
        cfg: Hydra configuration
    """
    if not cfg.logging.enabled:
        os.environ["WANDB_DISABLED"] = "true"
        return

    # Build wandb config with proper types
    wandb_config: dict[str, Any] = {
        "project": str(cfg.logging.project),
        "name": str(cfg.logging.run_name) if cfg.logging.run_name else str(cfg.exp_name),
        "config": cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True)),
        "tags": list(cfg.logging.tags) if cfg.logging.tags else [],
        "notes": str(cfg.logging.notes) if cfg.logging.notes else "",
        "mode": str(cfg.logging.mode),
    }

    if cfg.logging.entity:
        wandb_config["entity"] = str(cfg.logging.entity)

    wandb.init(**wandb_config)


def load_model_and_tokenizer(cfg: DictConfig) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load model and tokenizer from HuggingFace.

    Args:
        cfg: Hydra configuration

    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading model: {cfg.model.name}")

    # Determine dtype
    dtype_map = {
        "fp32": th.float32,
        "float32": th.float32,
        "fp16": th.float16,
        "float16": th.float16,
        "bf16": th.bfloat16,
        "bfloat16": th.bfloat16,
    }
    torch_dtype = dtype_map.get(cfg.model.model_kwargs.torch_dtype, th.bfloat16)

    # Load model with kwargs from config (unpacking model_kwargs)
    model_kwargs_raw = OmegaConf.to_container(cfg.model.model_kwargs, resolve=True)
    model_kwargs: dict[str, Any] = cast(dict[str, Any], model_kwargs_raw)
    model_kwargs["torch_dtype"] = torch_dtype  # Override with mapped dtype

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        **model_kwargs,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.name,
        trust_remote_code=cfg.model.model_kwargs.trust_remote_code,
    )

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    # Enable gradient checkpointing if specified
    if cfg.training.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    num_params = sum(p.numel() for p in model.parameters()) / 1e9
    logger.debug(f"Model loaded: {model.__class__.__name__}")
    logger.debug(f"Model size: {num_params:.2f}B parameters")

    return model, tokenizer


def preprocess_dataset(dataset: Dataset, tokenizer: PreTrainedTokenizer) -> Dataset:
    """
    Preprocess dataset into conversational format using chat template.

    Args:
        dataset: HuggingFace dataset
        tokenizer: Tokenizer with chat template support

    Returns:
        Preprocessed dataset with 'prompt' and 'answer' fields
    """

    def format_example(example: dict[str, Any]) -> dict[str, str]:
        # Create messages in chat format (no system prompt)
        messages = [
            {"role": "user", "content": str(example["problem"])},
        ]

        # Apply chat template to get the formatted prompt
        # add_generation_prompt=True adds the assistant prompt at the end
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        return {"prompt": str(prompt), "answer": str(example["answer"])}

    return dataset.map(format_example, desc="Formatting dataset with chat template")


def load_and_prepare_dataset(cfg: DictConfig, tokenizer: PreTrainedTokenizer) -> Dataset:
    """
    Load and preprocess dataset.

    Args:
        cfg: Hydra configuration
        tokenizer: Tokenizer for chat template formatting

    Returns:
        Preprocessed dataset
    """
    logger.info(f"Loading dataset: {cfg.dataset.name}")

    # Load dataset - cast to Dataset type since we know we're loading a specific split
    dataset_raw = load_dataset(
        cfg.dataset.name, split=cfg.dataset.train_split, streaming=cfg.dataset.streaming
    )
    dataset = cast(Dataset, dataset_raw)

    # Subsample if requested
    if cfg.dataset.max_train_samples is not None:
        dataset = cast(
            Dataset,
            dataset.select(range(min(int(cfg.dataset.max_train_samples), len(dataset)))),
        )
        logger.debug(f"Subsampled to {len(dataset)} examples")

    # Preprocess with chat template (no system prompt)
    dataset = preprocess_dataset(dataset, tokenizer)

    logger.debug(f"Dataset prepared with {len(dataset)} examples")

    return dataset


def create_grpo_config(cfg: DictConfig) -> GRPOConfig:
    """
    Create GRPOConfig from Hydra configuration.

    Args:
        cfg: Hydra configuration

    Returns:
        GRPOConfig instance
    """
    return GRPOConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.training.num_train_epochs,
        learning_rate=cfg.training.learning_rate,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        max_grad_norm=cfg.training.max_grad_norm,
        warmup_ratio=cfg.training.warmup_ratio,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        logging_steps=cfg.training.logging_steps,
        save_steps=cfg.training.save_steps,
        save_total_limit=cfg.training.save_total_limit,
        bf16=cfg.training.bf16,
        fp16=cfg.training.fp16,
        remove_unused_columns=cfg.training.remove_unused_columns,
        seed=cfg.seed,
        # GRPO-specific
        num_generations=cfg.training.num_generations,
        max_prompt_length=cfg.training.max_prompt_length,
        max_completion_length=cfg.training.max_completion_length,
        # WandB
        report_to="wandb" if cfg.logging.enabled else "none",
        run_name=cfg.exp_name,
    )


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """
    Main training function.

    Args:
        cfg: Hydra configuration
    """
    # Print configuration
    logger.debug("=" * 80)
    logger.debug("Configuration:")
    logger.debug("=" * 80)
    logger.debug(OmegaConf.to_yaml(cfg))
    logger.debug("=" * 80)

    # Set random seed
    th.manual_seed(cfg.seed)

    # Setup WandB
    setup_wandb(cfg)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(cfg)

    # Load and prepare dataset
    train_dataset = load_and_prepare_dataset(cfg, tokenizer)

    # Create GRPO config
    training_args = create_grpo_config(cfg)

    # Initialize trainer with TRL's reward functions
    logger.info("Initializing GRPOTrainer with accuracy_reward and think_format_reward...")
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,  # GRPOTrainer uses processing_class, not tokenizer
        reward_funcs=[accuracy_reward, think_format_reward],  # Multiple TRL rewards
    )

    # Save initial checkpoint (checkpoint-0) before training
    # This serves as the baseline for visualization, showing the initialized model
    # with reasoning vocabulary but no training
    checkpoint_0_path = Path(cfg.output_dir) / "checkpoint-0"
    logger.info(f"Saving initial checkpoint to {checkpoint_0_path}")
    trainer.save_model(str(checkpoint_0_path))

    # Save reasoning token map for checkpoint-0
    # Model must be Qwen3ReasoningVocabForCausalLM for this training pipeline
    assert isinstance(model, Qwen3ReasoningVocabForCausalLM), (
        f"Model must be Qwen3ReasoningVocabForCausalLM, got {type(model)}"
    )
    save_reasoning_token_map(checkpoint_0_path, model)
    logger.info("Saved reasoning token map for checkpoint-0")

    # Train
    logger.info("=" * 80)
    logger.info("Starting training...")
    logger.info("=" * 80)
    trainer.train()

    # Save final model
    final_model_path = Path(cfg.output_dir) / "final_model"
    logger.info(f"Saving final model to {final_model_path}")
    trainer.save_model(str(final_model_path))

    # Finish WandB run
    if cfg.logging.enabled:
        wandb.finish()

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
