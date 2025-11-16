"""
Shared test utilities and fixtures.

This module provides reusable test components:
- Fixtures for models, tokenizers, datasets
- Mock creators for external services
- Helper functions for test data generation
"""

from typing import Any

import pytest
import torch as th
from datasets import Dataset
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer


@pytest.fixture
def tiny_model_name() -> str:
    """Return the name of a tiny model for fast testing."""
    # Using the smallest GPT-2 model for fast test execution
    return "hf-internal-testing/tiny-random-GPTNeoXForCausalLM"


@pytest.fixture
def sample_dataset_dict() -> list[dict[str, Any]]:
    """Create a small sample dataset for testing."""
    return [
        {
            "problem": "What is 2 + 2?",
            "answer": "4",
            "solution": "Simply add 2 and 2 to get 4.",
        },
        {
            "problem": "Calculate 5 * 3",
            "answer": "15",
            "solution": "Multiply 5 by 3 to get 15.",
        },
        {
            "problem": "What is 10 - 7?",
            "answer": "3",
            "solution": "Subtract 7 from 10 to get 3.",
        },
        {
            "problem": "Solve: 8 / 4",
            "answer": "2",
            "solution": "Divide 8 by 4 to get 2.",
        },
        {
            "problem": "What is 3^2?",
            "answer": "9",
            "solution": "Calculate 3 to the power of 2 to get 9.",
        },
    ]


@pytest.fixture
def sample_dataset(sample_dataset_dict: list[dict[str, Any]]) -> Dataset:
    """Create a HuggingFace Dataset from sample data."""
    return Dataset.from_list(sample_dataset_dict)


@pytest.fixture
def minimal_hydra_config() -> DictConfig:
    """Create a minimal Hydra configuration for testing."""
    config_dict = {
        "exp_name": "test_experiment",
        "seed": 42,
        "output_dir": "/tmp/test_output",
        "model": {
            "name": "tiny-random/qwen3",
            "reasoning_vocab_size": 0,  # Baseline model for tests
            "model_kwargs": {
                "torch_dtype": "fp32",
                "trust_remote_code": True,
                "device_map": "cpu",
            },
        },
        "dataset": {
            "name": "test_dataset",
            "train_split": "train[:10]",
            "max_train_samples": 5,
        },
        "training": {
            "num_train_epochs": 1,
            "learning_rate": 1e-5,
            "per_device_train_batch_size": 2,  # Must be >= num_generations for GRPO
            "gradient_accumulation_steps": 1,
            "max_grad_norm": 1.0,
            "warmup_ratio": 0.1,
            "lr_scheduler_type": "cosine",
            "logging_steps": 1,
            "save_steps": 10,
            "save_total_limit": 2,
            "bf16": False,
            "fp16": False,
            "remove_unused_columns": False,
            "gradient_checkpointing": False,
            "num_generations": 2,
            "max_prompt_length": 128,
            "max_completion_length": 64,
        },
        "logging": {
            "enabled": False,
            "project": "test_project",
            "run_name": None,
            "entity": None,
            "tags": [],
            "notes": "",
            "mode": "disabled",
        },
    }
    return OmegaConf.create(config_dict)


@pytest.fixture
def mock_completions() -> list[list[dict[str, str]]]:
    """Create mock model completions for reward function testing."""
    return [
        [{"content": "The answer is \\boxed{4}"}],
        [{"content": "<think>\nLet me calculate: 5 * 3 = 15\n</think>\nThe answer is \\boxed{15}"}],
        [{"content": "\\boxed{3}"}],
        [{"content": "I don't know"}],
        [{"content": "<think>\nReasoning about the problem\n</think>\n\\boxed{9}"}],
    ]


@pytest.fixture
def mock_solutions() -> list[str]:
    """Create mock solutions corresponding to mock_completions."""
    return ["4", "15", "3", "2", "9"]


@pytest.fixture
def cpu_device() -> th.device:
    """Return CPU device for testing."""
    return th.device("cpu")


def create_tiny_tokenizer(model_name: str | None = None):
    """
    Create a tokenizer for testing.

    Args:
        model_name: Model name to load tokenizer from. Defaults to tiny test model.

    Returns:
        Tokenizer instance
    """
    if model_name is None:
        model_name = "hf-internal-testing/tiny-random-GPTNeoXForCausalLM"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)

    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Add simple chat template if not present
    if tokenizer.chat_template is None:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{ message['role'] }}: {{ message['content'] }}\n"
            "{% endfor %}"
            "assistant: "
        )

    return tokenizer


def create_tiny_model(model_name: str | None = None, device: str = "cpu"):
    """
    Create a tiny model for testing.

    Args:
        model_name: Model name to load. Defaults to tiny test model.
        device: Device to load model on.

    Returns:
        Model instance
    """
    if model_name is None:
        model_name = "hf-internal-testing/tiny-random-GPTNeoXForCausalLM"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=th.float32,
        trust_remote_code=False,
    )
    model = model.to(device)
    model.eval()

    return model


def assert_model_trainable(model):
    """
    Assert that a model has trainable parameters.

    Args:
        model: PyTorch model

    Raises:
        AssertionError: If model has no trainable parameters
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable_params > 0, "Model has no trainable parameters"


def assert_tensor_shape(tensor: th.Tensor, expected_shape: tuple[int, ...], name: str = "tensor"):
    """
    Assert that a tensor has the expected shape.

    Args:
        tensor: Tensor to check
        expected_shape: Expected shape tuple
        name: Name of tensor for error message

    Raises:
        AssertionError: If shapes don't match
    """
    assert tensor.shape == expected_shape, (
        f"{name} shape mismatch: expected {expected_shape}, got {tensor.shape}"
    )


def assert_dataset_fields(dataset: Dataset, required_fields: list[str]):
    """
    Assert that a dataset has all required fields.

    Args:
        dataset: HuggingFace dataset
        required_fields: List of required field names

    Raises:
        AssertionError: If any required fields are missing
    """
    dataset_fields = set(dataset.column_names)
    missing_fields = set(required_fields) - dataset_fields

    assert not missing_fields, f"Dataset missing required fields: {missing_fields}"
