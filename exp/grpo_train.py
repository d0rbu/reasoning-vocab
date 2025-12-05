"""
GRPO training script for reasoning vocabulary experiments.

This script uses Hydra for configuration management and WandB for logging.

Usage:
    python grpo_train.py                           # Use default configs
    python grpo_train.py training.learning_rate=1e-5  # Override specific params
    python grpo_train.py exp_name=my_experiment    # Change experiment name
"""

import os
import time
from pathlib import Path
from typing import Any, cast

import datasets.builder
import hydra
import torch as th
import torch.distributed as dist
import transformers
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from transformers import (
    AutoConfig,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerBase,
)
from trl import GRPOConfig, GRPOTrainer
from trl.rewards import accuracy_reward
from trl.trainer.grpo_trainer import RewardFunc

import wandb
from core.reasoning_vocab_model import (
    ReasoningVocabModel,
    get_reasoning_class,
)
from core.reasoning_vocab_utils import get_reasoning_token_ids
from core.tokenizer_utils import add_chat_template_if_needed
from core.train_utils import save_reasoning_token_map

DatasetType = Dataset | IterableDataset | DatasetDict | IterableDatasetDict
ListDatasetType = Dataset | IterableDataset
SizedDatasetType = Dataset | DatasetDict


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

    # Prepare reasoning token IDs based on reasoning_vocab_size
    reasoning_vocab_size = int(cfg.model.reasoning_vocab_size)
    reasoning_token_ids = get_reasoning_token_ids(reasoning_vocab_size)
    logger.info(f"Initializing model with reasoning vocab size: {reasoning_vocab_size}")

    # Load model with pretrained weights and reasoning vocabulary
    model_kwargs_raw = OmegaConf.to_container(cfg.model.model_kwargs, resolve=True)
    model_kwargs: dict[str, Any] = cast(dict[str, Any], model_kwargs_raw)
    model_kwargs["torch_dtype"] = torch_dtype

    model_config = AutoConfig.from_pretrained(cfg.model.name)

    assert len(model_config.architectures) == 1, (
        f"Model {cfg.model.name} must have exactly one architecture, got {model_config.architectures}"
    )

    model_class = getattr(transformers, model_config.architectures[0])
    reasoning_model_class = get_reasoning_class(model_class)

    model = reasoning_model_class.from_pretrained(
        cfg.model.name,
        reasoning_token_ids=reasoning_token_ids,
        **model_kwargs,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.name,
        trust_remote_code=cfg.model.model_kwargs.trust_remote_code,
    )
    add_chat_template_if_needed(tokenizer, model.config._name_or_path)

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Always ensure model config has pad_token_id
    if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # Enable gradient checkpointing if specified
    if cfg.training.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    num_params = sum(p.numel() for p in model.parameters()) / 1e9
    logger.debug(f"Model loaded: {model.__class__.__name__}")
    logger.debug(f"Model size: {num_params:.2f}B parameters")
    logger.debug(f"Reasoning vocab size: {model.reasoning_vocab_size}")
    logger.debug(f"Standard vocab size: {model.standard_vocab_size}")

    return model, tokenizer


PROBLEM_KEY_CANDIDATES = [
    "problem",
    "question",
    "prompt",
    "messages",
]
ANSWER_KEY_CANDIDATES = [
    "answer",
    "solution",
    "response",
    "output",
    "ground_truth",
]


def get_format_example_fn(sample: dict[str, Any]):
    """
    Analyze a dataset sample and return the appropriate format function.

    Args:
        sample: A single example from the dataset

    Returns:
        A format function that converts dataset examples to the required format
        with 'prompt' (conversational format) and 'solution' (string) fields

    Raises:
        AssertionError: If the sample format is invalid
        ValueError: If the problem key type is not supported
    """
    problem_key = next((key for key in PROBLEM_KEY_CANDIDATES if key in sample), None)
    answer_key = next((key for key in ANSWER_KEY_CANDIDATES if key in sample), None)
    assert problem_key is not None and answer_key is not None, (
        f"Could not find problem or answer key in sample: {sample}"
    )
    assert isinstance(sample[answer_key], str), (
        f"{answer_key} must be a string, got {type(sample[answer_key])}: {sample[answer_key]}"
    )

    if isinstance(sample[problem_key], list):
        # Conversational format - validate structure
        assert len(sample[problem_key]) == 1, (
            f"Got conversational format for {problem_key}, but found multiple messages: {sample[problem_key]}"
        )
        sample_message = sample[problem_key][0]
        assert "role" in sample_message and "content" in sample_message, (
            f"Got conversational format for {problem_key}, but found invalid message: {sample_message}"
        )
        assert sample_message["role"] == "user", (
            f"Got conversational format for {problem_key}, but found invalid role: {sample_message}"
        )

        def format_example(
            example: dict[str, list[dict[str, str]] | str],
        ) -> dict[str, list[dict[str, str]] | str]:
            return {"prompt": example[problem_key], "solution": example[answer_key]}

    elif isinstance(sample[problem_key], str):
        # String format - convert to conversational format
        def format_example(example: dict[str, str]) -> dict[str, list[dict[str, str]] | str]:
            prompt = [
                {"role": "user", "content": example[problem_key]},
            ]
            return {"prompt": prompt, "solution": example[answer_key]}

    else:
        raise ValueError(
            f"Got invalid format for {problem_key} ({type(sample[problem_key])}): {sample[problem_key]}"
        )

    return format_example


def preprocess_dataset(dataset: DatasetType, tokenizer: PreTrainedTokenizer) -> DatasetType:
    """
    Preprocess dataset into conversational format using chat template.

    Args:
        dataset: HuggingFace dataset
        tokenizer: Tokenizer with chat template support

    Returns:
        Preprocessed dataset with 'prompt' and 'answer' fields
    """
    if isinstance(dataset, Dataset | DatasetDict):
        assert len(dataset) > 0, "Dataset is empty"
        sample = dataset[0]
    else:
        try:
            sample = next(iter(dataset))
        except StopIteration as e:
            raise ValueError("Dataset is empty") from e

    assert isinstance(sample, dict), "Sample must be a dictionary"

    format_example = get_format_example_fn(cast(dict[str, Any], sample))

    if isinstance(dataset, IterableDataset | IterableDatasetDict):
        return dataset.map(format_example)
    else:
        return dataset.map(format_example, desc="Formatting dataset with chat template")


def load_and_prepare_dataset(cfg: DictConfig, tokenizer: PreTrainedTokenizer) -> DatasetType:
    """
    Load and preprocess dataset.

    Args:
        cfg: Hydra configuration
        tokenizer: Tokenizer for chat template formatting

    Returns:
        Preprocessed dataset
    """
    logger.info(f"Loading dataset: {cfg.dataset.name}")

    ignore_sufficient_disk_space_check = cfg.dataset.ignore_sufficient_disk_space_check
    if ignore_sufficient_disk_space_check:
        logger.warning(
            "Ignoring sufficient disk space check. Please make sure you actually have enough disk space to load the dataset!"
        )
        # i dont like this solution either, but its to get around cluster nfs shenanigans
        datasets.builder.has_sufficient_disk_space = lambda needed_bytes, directory=".": True  # type: ignore

    # Load dataset - cast to Dataset type since we know we're loading a specific split
    dataset = load_dataset(
        cfg.dataset.name, split=cfg.dataset.train_split, streaming=cfg.dataset.streaming
    )

    # Subsample if requested
    if cfg.dataset.max_train_samples is not None:
        assert isinstance(dataset, ListDatasetType), (
            f"Dataset must be a Dataset or IterableDataset object, got {type(dataset)}"
        )
        dataset = dataset.take(int(cfg.dataset.max_train_samples))
        logger.debug(f"Subsampled to {cfg.dataset.max_train_samples} examples")

    # Preprocess with chat template (no system prompt)
    dataset = preprocess_dataset(dataset, tokenizer)

    if isinstance(dataset, SizedDatasetType):
        logger.debug(f"Dataset prepared with {len(dataset)} examples")
    else:
        logger.debug("Streaming dataset prepared")

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


TOKEN_TYPE_IDS_NAME = "token_type_ids"
TRAINER_STATE_NAME = "trainer_state.json"


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
    assert isinstance(train_dataset, ListDatasetType), (
        f"Train dataset must be a Dataset or IterableDataset object, got {type(train_dataset)}"
    )

    checkpoint_exists = False

    training_dir_exists = os.path.exists(cfg.output_dir)
    if training_dir_exists:
        checkpoint_candidates = os.listdir(cfg.output_dir)
        for checkpoint_candidate in checkpoint_candidates:
            if not os.path.isdir(os.path.join(cfg.output_dir, checkpoint_candidate)):
                continue

            if os.path.exists(
                os.path.join(cfg.output_dir, checkpoint_candidate, TRAINER_STATE_NAME)
            ):
                checkpoint_exists = True
                break

    if checkpoint_exists:
        logger.info(f"Checkpoint exists at {cfg.output_dir}. Resuming training...")
    else:
        logger.info(f"No checkpoint found at {cfg.output_dir}. Starting training from scratch...")

    time.sleep(60)

    # Create GRPO config
    training_args = create_grpo_config(cfg)

    # Initialize trainer with TRL's reward functions
    logger.info("Initializing GRPOTrainer with accuracy_reward...")
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        reward_funcs=cast(list[RewardFunc], [accuracy_reward]),
    )

    assert isinstance(trainer.processing_class, PreTrainedTokenizerBase), (
        f"Processing class must be a PreTrainedTokenizerBase, got {type(trainer.processing_class)}"
    )

    if TOKEN_TYPE_IDS_NAME in trainer.processing_class.model_input_names:
        token_type_ids_idx = trainer.processing_class.model_input_names.index(TOKEN_TYPE_IDS_NAME)
        trainer.processing_class.model_input_names.pop(token_type_ids_idx)
        logger.trace(f"Removed token_type_ids from model input names: {token_type_ids_idx}")
    else:
        logger.trace("No token type IDs found in model input names")

    # Save initial checkpoint (checkpoint-0) before training
    # This serves as the baseline for visualization, showing the initialized model
    # with reasoning vocabulary but no training
    checkpoint_0_path = Path(cfg.output_dir) / "checkpoint-0"
    logger.debug(f"Saving initial checkpoint to {checkpoint_0_path}")
    trainer.save_model(str(checkpoint_0_path))

    # Save reasoning token map for checkpoint-0
    assert isinstance(model, ReasoningVocabModel), (
        f"Model must be ReasoningVocabModel, got {type(model)}"
    )
    save_reasoning_token_map(checkpoint_0_path, model)
    logger.debug("Saved reasoning token map for checkpoint-0")

    # Sanity check distributed training
    logger.info("=" * 80)
    logger.info("Sanity checking distributed training...")
    logger.info("=" * 80)
    
    if dist.is_initialized():
        logger.info(f"Distributed training pre-initialized with {dist.get_world_size()} devices")
    else:
        dist.init_process_group(backend="nccl")
        logger.info(f"Distributed training initialized with {dist.get_world_size()} devices")

    # Train
    logger.info("=" * 80)
    logger.info("Starting training...")
    logger.info("=" * 80)
    trainer.train(resume_from_checkpoint=checkpoint_exists)

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
