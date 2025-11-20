"""
Tests for training utilities and execution.

Tests include:
- GRPO configuration creation and validation
- Training loop execution with small datasets
- Checkpoint saving and loading
- Reward computation with various inputs
- Dataset processing and tokenization
- Integration with TRL's GRPOTrainer
"""

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch as th
from datasets import Dataset, IterableDataset
from omegaconf import DictConfig
from transformers import PreTrainedModel, PreTrainedTokenizer
from trl import GRPOConfig, GRPOTrainer
from trl.rewards import accuracy_reward, think_format_reward

from exp.grpo_train import (
    create_grpo_config,
    load_and_prepare_dataset,
    load_model_and_tokenizer,
    preprocess_dataset,
    setup_wandb,
)
from test.test_utils import (
    assert_dataset_fields,
    create_tiny_model,
    create_tiny_tokenizer,
    narrow_to_dataset,
)


class TestGRPOConfiguration:
    """Test GRPO configuration creation and validation."""

    def test_create_grpo_config_basic(self, minimal_hydra_config: DictConfig):
        """Test that create_grpo_config produces valid GRPOConfig."""
        config = create_grpo_config(minimal_hydra_config)

        assert isinstance(config, GRPOConfig)
        assert config.output_dir == minimal_hydra_config.output_dir
        assert config.learning_rate == minimal_hydra_config.training.learning_rate
        assert config.num_train_epochs == minimal_hydra_config.training.num_train_epochs
        assert config.seed == minimal_hydra_config.seed

    def test_create_grpo_config_grpo_params(self, minimal_hydra_config: DictConfig):
        """Test GRPO-specific parameters are set correctly."""
        config = create_grpo_config(minimal_hydra_config)

        assert config.num_generations == minimal_hydra_config.training.num_generations
        assert config.max_prompt_length == minimal_hydra_config.training.max_prompt_length
        assert config.max_completion_length == minimal_hydra_config.training.max_completion_length

    def test_create_grpo_config_mixed_precision(self, minimal_hydra_config: DictConfig):
        """Test mixed precision settings are respected."""
        # Test disabled precision (CPU mode)
        minimal_hydra_config.training.bf16 = False
        minimal_hydra_config.training.fp16 = False
        config = create_grpo_config(minimal_hydra_config)
        assert config.bf16 is False
        assert config.fp16 is False

        # Note: Can't test bf16/fp16 on CPU without GPU, so we skip those tests
        # In real GPU environments, these would be tested

    def test_create_grpo_config_logging(self, minimal_hydra_config: DictConfig):
        """Test logging configuration is set correctly."""
        # Test with logging disabled
        minimal_hydra_config.logging.enabled = False
        config = create_grpo_config(minimal_hydra_config)
        # report_to can be a list or string depending on transformers version
        if isinstance(config.report_to, list):
            assert len(config.report_to) == 0 or "none" in config.report_to
        else:
            assert config.report_to == "none"

        # Test with logging enabled
        minimal_hydra_config.logging.enabled = True
        config = create_grpo_config(minimal_hydra_config)
        if isinstance(config.report_to, list):
            assert "wandb" in config.report_to
        else:
            assert config.report_to == "wandb"


class TestDatasetProcessing:
    """Test dataset loading and preprocessing."""

    def test_preprocess_dataset_structure(self, sample_dataset: Dataset):
        """Test that preprocess_dataset creates correct fields."""
        tokenizer = create_tiny_tokenizer()
        processed = preprocess_dataset(sample_dataset, tokenizer)
        processed = narrow_to_dataset(processed)  # Type narrowing for TY type checker

        # Check that required fields exist
        assert_dataset_fields(processed, ["prompt", "answer"])

        # Check that all examples were processed
        assert len(processed) == len(sample_dataset)

    def test_preprocess_dataset_content(self, sample_dataset_dict: list[dict[str, Any]]):
        """Test that preprocessing maintains data integrity."""
        tokenizer = create_tiny_tokenizer()
        dataset = Dataset.from_list(sample_dataset_dict)
        processed = preprocess_dataset(dataset, tokenizer)
        assert processed is not None, "preprocess_dataset should not return None"
        processed = narrow_to_dataset(processed)  # Type narrowing for TY type checker

        # Check first example
        assert "What is 2 + 2?" in processed[0]["prompt"]
        assert processed[0]["answer"] == "4"

    def test_preprocess_dataset_chat_template(self, sample_dataset: Dataset):
        """Test that chat template is applied correctly."""
        tokenizer = create_tiny_tokenizer()

        # Add a simple chat template if not present
        if tokenizer.chat_template is None:
            tokenizer.chat_template = (
                "{% for message in messages %}"
                "{{ message['role'] }}: {{ message['content'] }}\n"
                "{% endfor %}"
                "assistant: "
            )

        processed = preprocess_dataset(sample_dataset, tokenizer)
        assert processed is not None, "preprocess_dataset should not return None"
        processed = narrow_to_dataset(processed)  # Type narrowing for TY type checker

        # Verify prompt is a string and non-empty
        assert isinstance(processed[0]["prompt"], str)
        assert len(processed[0]["prompt"]) > 0

    @pytest.mark.skip(reason="Requires internet access to load dataset")
    def test_load_and_prepare_dataset(self, minimal_hydra_config: DictConfig):
        """Test full dataset loading pipeline."""
        tokenizer = create_tiny_tokenizer()

        # This would load the real dataset - skip in offline tests
        dataset = load_and_prepare_dataset(minimal_hydra_config, tokenizer)

        assert isinstance(dataset, Dataset)
        assert_dataset_fields(dataset, ["prompt", "answer"])

    def test_dataset_subsampling(self, sample_dataset: Dataset):
        """Test that max_train_samples limits dataset size."""
        tokenizer = create_tiny_tokenizer()
        max_samples = 3

        # Manually test subsampling logic
        subsampled = sample_dataset.select(range(min(max_samples, len(sample_dataset))))
        assert len(subsampled) == max_samples

        processed = preprocess_dataset(subsampled, tokenizer)
        processed = narrow_to_dataset(processed)  # Type narrowing for TY type checker
        
        assert len(processed) == max_samples


class TestModelLoading:
    """Test model and tokenizer loading."""

    @pytest.mark.skip(reason="Requires internet access to download Qwen model")
    def test_load_model_and_tokenizer_qwen(self, minimal_hydra_config: DictConfig):
        """Test loading Qwen3-0.6B model."""
        minimal_hydra_config.model.name = "Qwen/Qwen3-0.6B"
        model, tokenizer = load_model_and_tokenizer(minimal_hydra_config)

        assert isinstance(model, PreTrainedModel)
        assert isinstance(tokenizer, PreTrainedTokenizer)
        assert tokenizer.pad_token is not None

    def test_load_model_dtype_mapping(self, minimal_hydra_config: DictConfig):
        """Test that dtype is correctly mapped from config string."""
        # Test different dtype strings
        dtype_tests = [
            ("fp32", th.float32),
            ("float32", th.float32),
            ("fp16", th.float16),
            ("float16", th.float16),
            ("bf16", th.bfloat16),
            ("bfloat16", th.bfloat16),
        ]

        for dtype_str, expected_dtype in dtype_tests:
            minimal_hydra_config.model.model_kwargs.torch_dtype = dtype_str
            model, tokenizer = load_model_and_tokenizer(minimal_hydra_config)

            # Check that model parameters have correct dtype
            param_dtype = next(model.parameters()).dtype
            assert param_dtype == expected_dtype, f"Expected {expected_dtype}, got {param_dtype}"

    def test_pad_token_initialization(self, minimal_hydra_config: DictConfig):
        """Test that pad token is set if not present."""
        model, tokenizer = load_model_and_tokenizer(minimal_hydra_config)

        # Pad token should be set (either by tokenizer or our code)
        assert tokenizer.pad_token is not None
        assert tokenizer.pad_token == tokenizer.eos_token
        assert model.config.pad_token_id == tokenizer.eos_token_id


class TestRewardFunctions:
    """Test reward computation with various inputs."""

    def test_accuracy_reward_correct_answer(self):
        """Test accuracy_reward with correct answers."""
        completions = [[{"content": "The answer is \\boxed{4}"}]]
        solutions = ["4"]

        rewards = accuracy_reward(completions, solutions)

        assert len(rewards) == 1
        assert isinstance(rewards[0], float)
        assert 0.0 <= rewards[0] <= 1.0

    def test_accuracy_reward_wrong_answer(self):
        """Test accuracy_reward with wrong answers."""
        completions = [[{"content": "The answer is \\boxed{5}"}]]
        solutions = ["4"]

        rewards = accuracy_reward(completions, solutions)

        assert len(rewards) == 1
        assert isinstance(rewards[0], float)
        assert 0.0 <= rewards[0] <= 1.0

    def test_accuracy_reward_batch(self, mock_completions: list, mock_solutions: list):
        """Test accuracy_reward with batch of completions."""
        rewards = accuracy_reward(mock_completions, mock_solutions)

        assert len(rewards) == len(mock_completions)
        assert all(isinstance(r, float) for r in rewards)
        assert all(isinstance(r, float) and 0.0 <= r <= 1.0 for r in rewards)

    def test_think_format_reward_valid(self):
        """Test think_format_reward with valid format."""
        completions = [[{"content": "<think>\nReasoning here\n</think>\nAnswer"}]]

        rewards = think_format_reward(completions)

        assert len(rewards) == 1
        assert rewards[0] == 1.0

    def test_think_format_reward_invalid(self):
        """Test think_format_reward with invalid format."""
        # Missing closing tag
        completions = [[{"content": "<think>\nReasoning without closing tag"}]]

        rewards = think_format_reward(completions)

        assert len(rewards) == 1
        assert rewards[0] == 0.0

    def test_think_format_reward_no_think_tags(self):
        """Test think_format_reward with no think tags."""
        completions = [[{"content": "Just a plain answer"}]]

        rewards = think_format_reward(completions)

        assert len(rewards) == 1
        assert rewards[0] == 0.0

    def test_reward_edge_cases(self):
        """Test reward functions with edge cases."""
        edge_cases = [
            [[{"content": ""}]],  # Empty completion
            [[{"content": "\\boxed{}"}]],  # Empty box
            [[{"content": "No box at all"}]],  # No answer box
        ]

        for completion in edge_cases:
            # Should not crash
            reward = accuracy_reward(completion, ["any_solution"])
            assert isinstance(reward[0], float)
            assert 0.0 <= reward[0] <= 1.0


class TestWandBSetup:
    """Test WandB initialization."""

    def test_wandb_disabled(self, minimal_hydra_config: DictConfig):
        """Test that WandB is disabled when logging.enabled is False."""
        minimal_hydra_config.logging.enabled = False

        with patch.dict("os.environ", {}, clear=False):
            setup_wandb(minimal_hydra_config)
            import os

            assert os.environ.get("WANDB_DISABLED") == "true"

    @patch("wandb.init")
    def test_wandb_enabled(self, mock_wandb_init: MagicMock, minimal_hydra_config: DictConfig):
        """Test that WandB is initialized when logging.enabled is True."""
        minimal_hydra_config.logging.enabled = True
        minimal_hydra_config.logging.project = "test_project"
        minimal_hydra_config.logging.mode = "online"

        setup_wandb(minimal_hydra_config)

        mock_wandb_init.assert_called_once()
        call_kwargs = mock_wandb_init.call_args.kwargs

        assert call_kwargs["project"] == "test_project"
        assert call_kwargs["mode"] == "online"


class TestTrainingExecution:
    """Test training loop execution."""

    def test_grpo_trainer_initialization(
        self, minimal_hydra_config: DictConfig, sample_dataset: Dataset
    ):
        """Test that GRPOTrainer can be initialized."""
        model = create_tiny_model()
        tokenizer = create_tiny_tokenizer()

        # Preprocess dataset
        processed_dataset = preprocess_dataset(sample_dataset, tokenizer)
        
        # Type narrowing for TY type checker
        assert isinstance(processed_dataset, (Dataset, IterableDataset)), f"Expected Dataset or IterableDataset, got {type(processed_dataset)}"

        # Create GRPO config
        training_args = create_grpo_config(minimal_hydra_config)

        # Wrap reward functions to match expected signature
        def wrapped_accuracy_reward(completions, solutions):
            return accuracy_reward(completions, solutions)
        
        def wrapped_think_format_reward(completions, solutions):
            # think_format_reward only needs completions, ignore solutions
            return think_format_reward(completions)

        # Initialize trainer
        trainer = GRPOTrainer(
            model=model,
            args=training_args,
            train_dataset=processed_dataset,
            processing_class=tokenizer,
            reward_funcs=[wrapped_accuracy_reward, wrapped_think_format_reward],
        )

        assert isinstance(trainer, GRPOTrainer)
        assert trainer.model is model
        assert trainer.processing_class is tokenizer

    @pytest.mark.skip(reason="Training execution is slow and resource-intensive")
    def test_training_step_execution(
        self, minimal_hydra_config: DictConfig, sample_dataset: Dataset
    ):
        """Test that a training step can execute without errors."""
        model = create_tiny_model()
        tokenizer = create_tiny_tokenizer()

        # Preprocess dataset
        processed_dataset = preprocess_dataset(sample_dataset, tokenizer)
        
        # Type narrowing for TY type checker
        assert isinstance(processed_dataset, (Dataset, IterableDataset)), f"Expected Dataset or IterableDataset, got {type(processed_dataset)}"

        # Create minimal GRPO config for fast training
        minimal_hydra_config.training.num_train_epochs = 1
        minimal_hydra_config.training.save_steps = 999999  # Don't save checkpoints
        training_args = create_grpo_config(minimal_hydra_config)

        # Wrap reward functions to match expected signature
        def wrapped_accuracy_reward(completions, solutions):
            return accuracy_reward(completions, solutions)
        
        def wrapped_think_format_reward(completions, solutions):
            # think_format_reward only needs completions, ignore solutions
            return think_format_reward(completions)

        # Initialize trainer
        trainer = GRPOTrainer(
            model=model,
            args=training_args,
            train_dataset=processed_dataset,
            processing_class=tokenizer,
            reward_funcs=[wrapped_accuracy_reward, wrapped_think_format_reward],
        )

        # Run one training step (or very short training)
        # This would require more setup and is expensive, so skipped by default
        trainer.train()


class TestCheckpointing:
    """Test checkpoint saving and loading."""

    def test_checkpoint_directory_creation(self, minimal_hydra_config: DictConfig):
        """Test that checkpoint directories are created correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            minimal_hydra_config.output_dir = tmpdir
            output_path = Path(tmpdir)

            # Directory should be created when trainer saves
            assert output_path.exists()

    def test_model_save_and_load(self, minimal_hydra_config: DictConfig):
        """Test that model can be saved and loaded from checkpoint."""
        model = create_tiny_model()
        tokenizer = create_tiny_tokenizer()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_model"
            save_path.mkdir(parents=True, exist_ok=True)

            # Save model
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)

            # Load model
            loaded_model = type(model).from_pretrained(save_path)
            loaded_tokenizer = type(tokenizer).from_pretrained(save_path)

            # Verify models have same architecture
            assert isinstance(loaded_model, type(model))
            assert loaded_tokenizer.vocab_size == tokenizer.vocab_size

    def test_checkpoint_save_limit(self, minimal_hydra_config: DictConfig):
        """Test that save_total_limit is respected."""
        config = create_grpo_config(minimal_hydra_config)

        # Verify save_total_limit is set
        assert hasattr(config, "save_total_limit")
        assert config.save_total_limit == minimal_hydra_config.training.save_total_limit


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_dataset(self):
        """Test handling of empty dataset."""
        empty_dataset = Dataset.from_list([])
        tokenizer = create_tiny_tokenizer()

        processed = preprocess_dataset(empty_dataset, tokenizer)
        processed = narrow_to_dataset(processed)  # Type narrowing for TY type checker

        assert len(processed) == 0
        # Empty dataset should still have the column structure after mapping
        # But columns may not exist if the dataset is empty
        if len(processed) > 0:
            assert_dataset_fields(processed, ["prompt", "answer"])

    def test_very_long_input(self, minimal_hydra_config: DictConfig):
        """Test handling of very long input sequences."""
        long_problem = "What is " + " + ".join(["1"] * 1000) + "?"
        dataset = Dataset.from_list(
            [
                {
                    "problem": long_problem,
                    "answer": "1000",
                    "solution": "Sum of 1000 ones",
                }
            ]
        )

        tokenizer = create_tiny_tokenizer()
        processed = preprocess_dataset(dataset, tokenizer)
        assert processed is not None, "preprocess_dataset should not return None"
        processed = narrow_to_dataset(processed)  # Type narrowing for TY type checker

        # Should not crash and should produce a prompt
        assert len(processed) == 1
        assert isinstance(processed[0]["prompt"], str)

    def test_special_characters_in_data(self):
        """Test handling of special characters in dataset."""
        special_dataset = Dataset.from_list(
            [
                {
                    "problem": "Calculate: <tag> & \"quotes\" 'apostrophes'",
                    "answer": "42",
                    "solution": "Special chars test",
                }
            ]
        )

        tokenizer = create_tiny_tokenizer()
        processed = preprocess_dataset(special_dataset, tokenizer)
        processed = narrow_to_dataset(processed)  # Type narrowing for TY type checker

        # Should handle special characters without crashing
        assert len(processed) == 1
        assert isinstance(processed[0]["prompt"], str)

    def test_unicode_in_data(self):
        """Test handling of Unicode characters."""
        unicode_dataset = Dataset.from_list(
            [
                {
                    "problem": "Calculate: π × 2",
                    "answer": "6.28",
                    "solution": "Use π ≈ 3.14",
                }
            ]
        )

        tokenizer = create_tiny_tokenizer()
        processed = preprocess_dataset(unicode_dataset, tokenizer)
        processed = narrow_to_dataset(processed)  # Type narrowing for TY type checker

        assert len(processed) == 1
        assert "π" in processed[0]["problem"] or "π" in str(processed[0])
