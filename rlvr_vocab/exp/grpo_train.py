"""
GRPO training script for reasoning vocabulary experiments.

This script uses Hydra for configuration management and WandB for logging.

Usage:
    python grpo_train.py                           # Use default configs
    python grpo_train.py training.learning_rate=1e-5  # Override specific params
    python grpo_train.py exp_name=my_experiment    # Change experiment name
"""

import os
import sys
from pathlib import Path

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from rlvr_vocab.core.dataset import MathDataset
from rlvr_vocab.core.reward import create_reward_function


def setup_wandb(cfg: DictConfig):
    """
    Initialize Weights & Biases logging.

    Args:
        cfg: Hydra configuration
    """
    if not cfg.logging.enabled:
        os.environ["WANDB_DISABLED"] = "true"
        return

    wandb_config = {
        "project": cfg.logging.project,
        "name": cfg.logging.run_name or cfg.exp_name,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "tags": cfg.logging.tags,
        "notes": cfg.logging.notes,
        "mode": cfg.logging.mode,
    }

    if cfg.logging.entity:
        wandb_config["entity"] = cfg.logging.entity

    wandb.init(**wandb_config)


def load_model_and_tokenizer(cfg: DictConfig):
    """
    Load model and tokenizer from HuggingFace.

    Args:
        cfg: Hydra configuration

    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model: {cfg.model.name}")

    # Determine dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map.get(cfg.model.torch_dtype, torch.bfloat16)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        torch_dtype=torch_dtype,
        trust_remote_code=cfg.model.trust_remote_code,
        load_in_8bit=cfg.model.load_in_8bit,
        load_in_4bit=cfg.model.load_in_4bit,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.name,
        trust_remote_code=cfg.model.trust_remote_code,
    )

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    # Enable gradient checkpointing if specified
    if cfg.training.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    print(f"Model loaded: {model.__class__.__name__}")
    print(f"Model size: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")

    return model, tokenizer


def load_dataset(cfg: DictConfig):
    """
    Load and preprocess dataset.

    Args:
        cfg: Hydra configuration

    Returns:
        MathDataset instance
    """
    dataset = MathDataset(
        dataset_name=cfg.dataset.name,
        train_split=cfg.dataset.train_split,
        val_split=cfg.dataset.val_split,
        prompt_template=cfg.dataset.prompt_template,
        max_train_samples=cfg.dataset.max_train_samples,
        max_val_samples=cfg.dataset.max_val_samples,
    )

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
    print("=" * 80)
    print("Configuration:")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    # Set random seed
    torch.manual_seed(cfg.seed)

    # Setup WandB
    setup_wandb(cfg)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(cfg)

    # Load dataset
    math_dataset = load_dataset(cfg)
    train_dataset = math_dataset.get_train_dataset()

    # Create reward function
    print("Creating reward function...")
    reward_fn = create_reward_function(train_dataset)

    # Create GRPO config
    training_args = create_grpo_config(cfg)

    # Initialize trainer
    print("Initializing GRPOTrainer...")
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        reward_funcs=reward_fn,
    )

    # Train
    print("=" * 80)
    print("Starting training...")
    print("=" * 80)
    trainer.train()

    # Save final model
    final_model_path = Path(cfg.output_dir) / "final_model"
    print(f"Saving final model to {final_model_path}")
    trainer.save_model(str(final_model_path))

    # Finish WandB run
    if cfg.logging.enabled:
        wandb.finish()

    print("Training complete!")


if __name__ == "__main__":
    main()
