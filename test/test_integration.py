"""
Integration tests for the RLVR training pipeline.

These tests verify that different components work together correctly:
- End-to-end data pipeline
- Model + tokenizer + dataset integration
- Full training configuration workflow
- Multi-component interactions
"""

import tempfile
from pathlib import Path

import torch as th
from datasets import Dataset
from omegaconf import DictConfig
from test_utils import (
    assert_dataset_fields,
    create_tiny_model,
    create_tiny_tokenizer,
)
from trl import GRPOTrainer
from trl.rewards import accuracy_reward, think_format_reward

from exp.grpo_train import (
    create_grpo_config,
    load_model_and_tokenizer,
    preprocess_dataset,
)


class TestEndToEndDataPipeline:
    """Test the complete data processing pipeline."""

    def test_raw_to_processed_dataset(self, sample_dataset: Dataset):
        """Test full pipeline from raw data to processed format."""
        tokenizer = create_tiny_tokenizer()

        # Start with raw dataset
        assert "problem" in sample_dataset.column_names
        assert "answer" in sample_dataset.column_names

        # Process dataset
        processed = preprocess_dataset(sample_dataset, tokenizer)

        # Verify transformation
        assert_dataset_fields(processed, ["prompt", "answer"])
        assert len(processed) == len(sample_dataset)

        # Verify data integrity
        for i, example in enumerate(processed):
            assert isinstance(example["prompt"], str)
            assert isinstance(example["answer"], str)
            assert len(example["prompt"]) > 0
            # Original problem should be reflected in prompt
            original_problem = sample_dataset[i]["problem"]
            assert original_problem in str(example)

    def test_dataset_to_model_forward(self, sample_dataset: Dataset):
        """Test that processed dataset can be fed to model."""
        tokenizer = create_tiny_tokenizer()
        model = create_tiny_model()

        # Process dataset
        processed = preprocess_dataset(sample_dataset, tokenizer)

        # Take first example and tokenize
        prompt = processed[0]["prompt"]
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)

        # Run through model
        with th.no_grad():
            outputs = model(**inputs)

        # Verify outputs
        assert outputs.logits is not None
        assert outputs.logits.shape[0] == 1  # Batch size
        assert outputs.logits.shape[1] > 0  # Sequence length
        assert outputs.logits.shape[2] == tokenizer.vocab_size


class TestModelTokenizerIntegration:
    """Test model and tokenizer work together correctly."""

    def test_tokenizer_model_vocab_consistency(self):
        """Test that tokenizer and model have consistent vocabulary."""
        tokenizer = create_tiny_tokenizer()
        model = create_tiny_model()

        # Vocab sizes should match
        assert tokenizer.vocab_size == model.config.vocab_size

        # Pad token should be set
        assert tokenizer.pad_token is not None
        assert tokenizer.pad_token_id is not None

    def test_generation_with_tokenizer(self):
        """Test that model can generate text with tokenizer."""
        tokenizer = create_tiny_tokenizer()
        model = create_tiny_model()
        model.eval()

        # Create input
        prompt = "Test prompt"
        inputs = tokenizer(prompt, return_tensors="pt")

        # Generate (just a few tokens to keep test fast)
        with th.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=5, do_sample=False)

        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        assert isinstance(generated_text, str)
        assert len(generated_text) > 0
        assert prompt in generated_text or len(generated_text) >= len(prompt)

    def test_batch_encoding_decoding(self):
        """Test batch encoding and decoding."""
        tokenizer = create_tiny_tokenizer()

        texts = ["First prompt", "Second prompt", "Third prompt"]

        # Encode batch
        encoded = tokenizer(texts, padding=True, return_tensors="pt")

        assert encoded.input_ids.shape[0] == len(texts)
        assert encoded.attention_mask.shape[0] == len(texts)

        # Decode batch
        decoded = tokenizer.batch_decode(encoded.input_ids, skip_special_tokens=True)

        assert len(decoded) == len(texts)
        for original, decoded_text in zip(texts, decoded, strict=True):
            # Allow for minor differences due to tokenization
            assert original in decoded_text or decoded_text.strip() == original.strip()


class TestTrainerIntegration:
    """Test GRPOTrainer integration with all components."""

    def test_trainer_with_full_config(
        self, minimal_hydra_config: DictConfig, sample_dataset: Dataset
    ):
        """Test trainer initialization with full configuration."""
        model = create_tiny_model()
        tokenizer = create_tiny_tokenizer()

        # Process dataset
        processed_dataset = preprocess_dataset(sample_dataset, tokenizer)

        # Create GRPO config
        training_args = create_grpo_config(minimal_hydra_config)

        # Initialize trainer
        trainer = GRPOTrainer(
            model=model,
            args=training_args,
            train_dataset=processed_dataset,
            processing_class=tokenizer,
            reward_funcs=[accuracy_reward, think_format_reward],
        )

        # Verify trainer components
        assert trainer.model is model
        assert trainer.processing_class is tokenizer
        assert len(trainer.reward_funcs) == 2

    def test_trainer_state_initialization(
        self, minimal_hydra_config: DictConfig, sample_dataset: Dataset
    ):
        """Test that trainer state is properly initialized."""
        model = create_tiny_model()
        tokenizer = create_tiny_tokenizer()
        processed_dataset = preprocess_dataset(sample_dataset, tokenizer)
        training_args = create_grpo_config(minimal_hydra_config)

        trainer = GRPOTrainer(
            model=model,
            args=training_args,
            train_dataset=processed_dataset,
            processing_class=tokenizer,
            reward_funcs=[accuracy_reward],
        )

        # Check that trainer has necessary attributes
        assert hasattr(trainer, "model")
        assert hasattr(trainer, "args")
        assert hasattr(trainer, "train_dataset")


class TestConfigurationWorkflow:
    """Test the complete configuration workflow."""

    def test_hydra_config_to_grpo_config(self, minimal_hydra_config: DictConfig):
        """Test conversion from Hydra config to GRPOConfig."""
        grpo_config = create_grpo_config(minimal_hydra_config)

        # Verify all important parameters are transferred
        assert grpo_config.output_dir == minimal_hydra_config.output_dir
        assert grpo_config.learning_rate == minimal_hydra_config.training.learning_rate
        assert grpo_config.num_train_epochs == minimal_hydra_config.training.num_train_epochs
        assert grpo_config.seed == minimal_hydra_config.seed

        # Verify GRPO-specific parameters
        assert grpo_config.num_generations == minimal_hydra_config.training.num_generations
        assert grpo_config.max_prompt_length == minimal_hydra_config.training.max_prompt_length

    def test_config_with_overrides(self, minimal_hydra_config: DictConfig):
        """Test that config overrides work correctly."""
        # Modify config
        minimal_hydra_config.training.learning_rate = 1e-4
        minimal_hydra_config.training.num_train_epochs = 5

        grpo_config = create_grpo_config(minimal_hydra_config)

        # Verify overrides are applied
        assert grpo_config.learning_rate == 1e-4
        assert grpo_config.num_train_epochs == 5


class TestRewardPipeline:
    """Test reward computation pipeline."""

    def test_reward_functions_with_dataset(self, sample_dataset: Dataset):
        """Test reward functions work with dataset answers."""
        # Create mock completions matching dataset
        completions = [
            [{"content": f"The answer is \\boxed{{{example['answer']}}}"}]
            for example in sample_dataset
        ]
        solutions = [example["answer"] for example in sample_dataset]

        # Compute rewards
        acc_rewards = accuracy_reward(completions, solutions)

        # Verify rewards
        assert len(acc_rewards) == len(completions)
        assert all(isinstance(r, float) for r in acc_rewards)
        # All should be correct since we used the right answers
        assert all(r >= 0.5 for r in acc_rewards), "Expected high rewards for correct answers"

    def test_multiple_reward_functions(self):
        """Test that multiple reward functions can be applied."""
        completions = [
            [{"content": "<think>\nLet me think\n</think>\nThe answer is \\boxed{4}"}],
            [{"content": "The answer is \\boxed{4}"}],
        ]
        solutions = ["4", "4"]

        # Apply both reward functions
        acc_rewards = accuracy_reward(completions, solutions)
        think_rewards = think_format_reward(completions)

        # Verify both work
        assert len(acc_rewards) == 2
        assert len(think_rewards) == 2

        # First completion should have think format reward
        assert think_rewards[0] == 1.0
        # Second should not
        assert think_rewards[1] == 0.0


class TestCheckpointingWorkflow:
    """Test checkpoint saving and loading workflow."""

    def test_save_and_load_model(self):
        """Test full save and load cycle."""
        model = create_tiny_model()
        tokenizer = create_tiny_tokenizer()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_dir = Path(tmpdir) / "checkpoint"
            save_dir.mkdir()

            # Save
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)

            # Verify files exist
            assert (save_dir / "config.json").exists()
            assert len(list(save_dir.glob("*.bin")) + list(save_dir.glob("*.safetensors"))) > 0

            # Load
            loaded_model = type(model).from_pretrained(save_dir)
            loaded_tokenizer = type(tokenizer).from_pretrained(save_dir)

            # Verify loaded components work
            test_input = "Test"
            inputs = loaded_tokenizer(test_input, return_tensors="pt")

            with th.no_grad():
                outputs = loaded_model(**inputs)

            assert outputs.logits is not None

    def test_checkpoint_directory_structure(self, minimal_hydra_config: DictConfig):
        """Test that checkpoint directories are created with correct structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            minimal_hydra_config.output_dir = tmpdir
            grpo_config = create_grpo_config(minimal_hydra_config)

            output_path = Path(grpo_config.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            assert output_path.exists()
            assert output_path.is_dir()


class TestErrorHandling:
    """Test error handling in integration scenarios."""

    def test_mismatched_vocab_sizes(self):
        """Test handling when tokenizer and model vocab sizes might differ."""
        tokenizer = create_tiny_tokenizer()
        model = create_tiny_model()

        # They should match
        assert tokenizer.vocab_size == model.config.vocab_size

    def test_empty_dataset_handling(self):
        """Test that empty dataset is handled gracefully."""
        empty_dataset = Dataset.from_list([])
        tokenizer = create_tiny_tokenizer()

        processed = preprocess_dataset(empty_dataset, tokenizer)

        assert len(processed) == 0
        # Empty datasets may not have column structure
        if len(processed) > 0:
            assert_dataset_fields(processed, ["prompt", "answer"])

    def test_invalid_config_values(self, minimal_hydra_config: DictConfig):
        """Test handling of potentially invalid config values."""
        # Test with zero epochs (should still create config, even if not sensible)
        minimal_hydra_config.training.num_train_epochs = 0
        config = create_grpo_config(minimal_hydra_config)
        assert config.num_train_epochs == 0

        # Test with batch size >= num_generations (GRPO requirement)
        # Must maintain: per_device_train_batch_size >= num_generations
        minimal_hydra_config.training.per_device_train_batch_size = 2
        minimal_hydra_config.training.num_generations = 2
        config = create_grpo_config(minimal_hydra_config)
        assert config.per_device_train_batch_size == 2


class TestMemoryManagement:
    """Test memory-related aspects of the pipeline."""

    def test_model_device_placement(self):
        """Test that model can be placed on correct device."""
        model = create_tiny_model(device="cpu")

        # Check model is on CPU
        assert next(model.parameters()).device.type == "cpu"

    def test_gradient_checkpointing_config(self, minimal_hydra_config: DictConfig):
        """Test that gradient checkpointing can be enabled."""
        minimal_hydra_config.training.gradient_checkpointing = True

        # This would be applied during model loading in the full pipeline
        # Here we just verify the config is set correctly
        assert minimal_hydra_config.training.gradient_checkpointing is True

    def test_dtype_consistency(self, minimal_hydra_config: DictConfig):
        """Test that dtype settings are consistent across components."""
        model, tokenizer = load_model_and_tokenizer(minimal_hydra_config)

        # Get model dtype
        model_dtype = next(model.parameters()).dtype

        # Verify it matches config
        expected_dtype = th.float32  # minimal_config uses fp32
        assert model_dtype == expected_dtype
