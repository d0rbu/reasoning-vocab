"""
Pytest configuration and shared fixtures.

This file makes fixtures from test_utils.py available to all test files.
"""

# Import all fixtures from test_utils so they're available across all test files
from test.test_utils import (
    cpu_device,
    minimal_hydra_config,
    mock_completions,
    mock_solutions,
    sample_dataset,
    sample_dataset_dict,
    tiny_model_name,
)

__all__ = [
    "tiny_model_name",
    "sample_dataset_dict",
    "sample_dataset",
    "minimal_hydra_config",
    "mock_completions",
    "mock_solutions",
    "cpu_device",
]
