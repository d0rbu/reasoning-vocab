"""
Tests for the sampling script.

Since the sampling script requires a trained model, these tests primarily check:
- Function interfaces and data structures
- Prompt formatting for different datasets
- Answer extraction logic
- Sample saving/loading
"""

# Import functions from the sampling script - use importlib to load it
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest

script_path = Path(__file__).parent.parent / "script" / "sample_reasoning.py"
spec = importlib.util.spec_from_file_location("sample_reasoning", script_path)
sample_reasoning = importlib.util.module_from_spec(spec)
sys.modules["sample_reasoning"] = sample_reasoning
spec.loader.exec_module(sample_reasoning)

extract_answer = sample_reasoning.extract_answer
format_prompt = sample_reasoning.format_prompt
save_sample = sample_reasoning.save_sample


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    tokenizer = Mock()
    tokenizer.tokenizer = Mock()
    tokenizer.tokenizer.apply_chat_template = Mock(return_value="<|im_start|>user\nTest question<|im_end|>\n<|im_start|>assistant\n")
    return tokenizer


class TestPromptFormatting:
    def test_gsm8k_format(self, mock_tokenizer):
        example = {"question": "What is 2+2?", "answer": "#### 4"}
        prompt = format_prompt(example, "openai/gsm8k", mock_tokenizer)

        # Verify the tokenizer's apply_chat_template was called with correct messages
        mock_tokenizer.tokenizer.apply_chat_template.assert_called_once_with(
            [{"role": "user", "content": "What is 2+2?"}], 
            tokenize=False, 
            add_generation_prompt=True
        )
        assert prompt == "<|im_start|>user\nTest question<|im_end|>\n<|im_start|>assistant\n"

    def test_math_format(self, mock_tokenizer):
        example = {"problem": "Solve x^2 = 4", "solution": "x = ±2"}
        prompt = format_prompt(example, "hendrycks/math", mock_tokenizer)

        # Verify the tokenizer's apply_chat_template was called with correct messages
        mock_tokenizer.tokenizer.apply_chat_template.assert_called_once_with(
            [{"role": "user", "content": "Solve x^2 = 4"}], 
            tokenize=False, 
            add_generation_prompt=True
        )
        assert prompt == "<|im_start|>user\nTest question<|im_end|>\n<|im_start|>assistant\n"

    def test_generic_format(self, mock_tokenizer):
        example = {"question": "How are you?"}
        prompt = format_prompt(example, "custom/dataset", mock_tokenizer)

        # Verify the tokenizer's apply_chat_template was called with correct messages
        mock_tokenizer.tokenizer.apply_chat_template.assert_called_once_with(
            [{"role": "user", "content": "How are you?"}], 
            tokenize=False, 
            add_generation_prompt=True
        )
        assert prompt == "<|im_start|>user\nTest question<|im_end|>\n<|im_start|>assistant\n"

    def test_unknown_format_raises(self, mock_tokenizer):
        example = {"unknown_key": "value"}
        with pytest.raises(ValueError, match="Unknown dataset format"):
            format_prompt(example, "unknown/dataset", mock_tokenizer)


class TestAnswerExtraction:
    def test_gsm8k_answer_extraction(self):
        example = {"answer": "Let me think...\n#### 42"}
        answer = extract_answer(example, "openai/gsm8k")

        assert answer == "42"

    def test_gsm8k_answer_without_separator(self):
        example = {"answer": "42"}
        answer = extract_answer(example, "openai/gsm8k")

        assert answer == "42"

    def test_math_answer_extraction(self):
        example = {"solution": "The solution is x = 5"}
        answer = extract_answer(example, "hendrycks/math")

        assert answer == "The solution is x = 5"

    def test_generic_answer_extraction(self):
        example = {"answer": "Generic answer"}
        answer = extract_answer(example, "custom/dataset")

        assert answer == "Generic answer"

    def test_unknown_answer_format_raises(self):
        example = {"unknown_key": "value"}
        with pytest.raises(ValueError, match="Unknown answer format"):
            extract_answer(example, "unknown/dataset")


class TestSampleSaving:
    def test_save_and_load_sample(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Save a sample
            save_sample(
                output_dir=output_dir,
                sample_idx=0,
                prompt="Test prompt",
                token_ids=[1, 2, 3],
                decoded_text="Test output",
                multiplicities=[0, 1, 2],
                ground_truth="Test answer",
                is_correct=True,
                was_truncated=False,
            )

            # Load and verify
            sample_file = output_dir / "sample_0000.json"
            assert sample_file.exists()

            with open(sample_file) as f:
                loaded_data = json.load(f)

            assert loaded_data["sample_idx"] == 0
            assert loaded_data["prompt"] == "Test prompt"
            assert loaded_data["token_ids"] == [1, 2, 3]
            assert loaded_data["decoded_text"] == "Test output"
            assert loaded_data["multiplicities"] == [0, 1, 2]
            assert loaded_data["ground_truth"] == "Test answer"
            assert loaded_data["is_correct"] is True
            assert loaded_data["was_truncated"] is False

    def test_save_multiple_samples(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Save multiple samples
            for i in range(3):
                save_sample(
                    output_dir=output_dir,
                    sample_idx=i,
                    prompt=f"Prompt {i}",
                    token_ids=[i],
                    decoded_text=f"Output {i}",
                    multiplicities=[0],
                    ground_truth=f"Answer {i}",
                    is_correct=i % 2 == 0,
                    was_truncated=False,
                )

            # Verify all files exist
            assert (output_dir / "sample_0000.json").exists()
            assert (output_dir / "sample_0001.json").exists()
            assert (output_dir / "sample_0002.json").exists()

            # Verify content
            with open(output_dir / "sample_0001.json") as f:
                data = json.load(f)
            assert data["prompt"] == "Prompt 1"
            assert data["is_correct"] is False


class TestSampleDataStructure:
    def test_sample_has_required_fields(self):
        """Verify saved samples have all required fields for visualization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            save_sample(
                output_dir=output_dir,
                sample_idx=0,
                prompt="Test",
                token_ids=[1, 2, 3],
                decoded_text="Test",
                multiplicities=[0, 1, 2],
                ground_truth="Answer",
                is_correct=True,
                was_truncated=False,
            )

            with open(output_dir / "sample_0000.json") as f:
                data = json.load(f)

            # Required fields for visualize_reasoning.py
            assert "token_ids" in data
            assert "multiplicities" in data
            assert "decoded_text" in data

            # Metadata fields
            assert "sample_idx" in data
            assert "is_correct" in data
            assert "was_truncated" in data

            # Verify multiplicities match token_ids length
            assert len(data["multiplicities"]) == len(data["token_ids"])
