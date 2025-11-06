"""
Tests for vocabulary usage visualization script.

This module tests the functionality of visualize_vocab_usage.py:
- Loading checkpoint data
- Analyzing token usage statistics
- Creating visualization plots
- End-to-end integration
"""

from pathlib import Path

import matplotlib
import pytest
import torch as th
from matplotlib.figure import Figure

from viz.visualize_vocab_usage import (
    analyze_token_usage,
    create_stacked_area_plot,
    create_usage_plot,
    load_checkpoint_outputs,
    visualize_vocab_usage,
)

# Use non-interactive backend for testing
matplotlib.use("Agg")


@pytest.fixture
def temp_checkpoint_dir(tmp_path: Path) -> Path:
    """Create a temporary checkpoint directory with mock data."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    # Create mock checkpoints at different steps
    for step in [100, 200, 300]:
        ckpt_dir = checkpoint_dir / f"checkpoint-{step}"
        ckpt_dir.mkdir()

        # Create mock token IDs with increasing reasoning token usage
        # Standard vocab size: 100, reasoning tokens: >= 100
        batch_size = 4
        seq_len = 20

        # Gradually increase reasoning token usage
        num_reasoning = step // 50  # 2, 4, 6 reasoning tokens per sequence

        token_ids = th.randint(0, 100, (batch_size, seq_len))
        # Replace some tokens with reasoning tokens
        for i in range(min(num_reasoning, seq_len)):
            token_ids[:, i] = th.randint(100, 150, (batch_size,))

        # Save to file
        generations_file = ckpt_dir / "generations.pt"
        th.save(token_ids, generations_file)

    return checkpoint_dir


@pytest.fixture
def sample_token_ids() -> th.Tensor:
    """Create sample token IDs for testing."""
    # Create a tensor with known distribution:
    # 80 standard tokens (< 100), 20 reasoning tokens (>= 100)
    standard_tokens = th.randint(0, 100, (80,))
    reasoning_tokens = th.randint(100, 200, (20,))
    return th.cat([standard_tokens, reasoning_tokens])


class TestLoadCheckpointOutputs:
    """Tests for loading checkpoint data."""

    def test_load_valid_checkpoints(self, temp_checkpoint_dir: Path):
        """Test loading checkpoints from a valid directory."""
        checkpoints = load_checkpoint_outputs(temp_checkpoint_dir)

        # Should have loaded 3 checkpoints
        assert len(checkpoints) == 3
        assert set(checkpoints.keys()) == {100, 200, 300}

        # Check structure of loaded data
        for _step, data in checkpoints.items():
            assert isinstance(data, th.Tensor)

    def test_load_nonexistent_directory(self):
        """Test error handling for nonexistent directory."""
        with pytest.raises(FileNotFoundError):
            load_checkpoint_outputs(Path("/nonexistent/path"))

    def test_load_empty_directory(self, tmp_path: Path):
        """Test loading from directory with no checkpoints."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        checkpoints = load_checkpoint_outputs(empty_dir)
        assert len(checkpoints) == 0

    def test_load_malformed_checkpoint_name(self, tmp_path: Path):
        """Test handling of checkpoint directories with invalid names."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        # Create directory with invalid name
        invalid_dir = checkpoint_dir / "checkpoint-invalid"
        invalid_dir.mkdir()

        checkpoints = load_checkpoint_outputs(checkpoint_dir)
        assert len(checkpoints) == 0

    def test_load_missing_generations_file(self, tmp_path: Path):
        """Test handling of checkpoint directories without generations.pt."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        # Create checkpoint directory but no generations.pt
        ckpt_dir = checkpoint_dir / "checkpoint-100"
        ckpt_dir.mkdir()

        checkpoints = load_checkpoint_outputs(checkpoint_dir)
        assert len(checkpoints) == 0


class TestAnalyzeTokenUsage:
    """Tests for token usage analysis."""

    def test_analyze_known_distribution(self, sample_token_ids: th.Tensor):
        """Test analysis with known token distribution."""
        vocab_size = 100

        num_std, num_reas, pct_std, pct_reas = analyze_token_usage(sample_token_ids, vocab_size)

        # Should have 80 standard and 20 reasoning tokens
        assert num_std == 80
        assert num_reas == 20
        assert pct_std == pytest.approx(80.0, abs=0.1)
        assert pct_reas == pytest.approx(20.0, abs=0.1)

    def test_analyze_all_standard_tokens(self):
        """Test analysis when all tokens are standard."""
        vocab_size = 100
        token_ids = th.randint(0, 100, (50,))

        num_std, num_reas, pct_std, pct_reas = analyze_token_usage(token_ids, vocab_size)

        assert num_std == 50
        assert num_reas == 0
        assert pct_std == 100.0
        assert pct_reas == 0.0

    def test_analyze_all_reasoning_tokens(self):
        """Test analysis when all tokens are reasoning."""
        vocab_size = 100
        token_ids = th.randint(100, 200, (50,))

        num_std, num_reas, pct_std, pct_reas = analyze_token_usage(token_ids, vocab_size)

        assert num_std == 0
        assert num_reas == 50
        assert pct_std == 0.0
        assert pct_reas == 100.0

    def test_analyze_empty_tensor(self):
        """Test analysis with empty tensor."""
        vocab_size = 100
        token_ids = th.tensor([])

        num_std, num_reas, pct_std, pct_reas = analyze_token_usage(token_ids, vocab_size)

        assert num_std == 0
        assert num_reas == 0
        assert pct_std == 0.0
        assert pct_reas == 0.0

    def test_analyze_2d_tensor(self):
        """Test analysis with 2D tensor (batch_size, seq_len)."""
        vocab_size = 100
        # Create 2D tensor: 3 batches, 10 tokens each
        # First 5 tokens standard, last 5 reasoning in each batch
        token_ids = th.cat([th.randint(0, 100, (3, 5)), th.randint(100, 200, (3, 5))], dim=1)

        num_std, num_reas, pct_std, pct_reas = analyze_token_usage(token_ids, vocab_size)

        # Total: 30 tokens (3 * 10), 15 standard, 15 reasoning
        assert num_std == 15
        assert num_reas == 15
        assert pct_std == pytest.approx(50.0, abs=0.1)
        assert pct_reas == pytest.approx(50.0, abs=0.1)

    def test_analyze_boundary_cases(self):
        """Test tokens at vocabulary boundary."""
        vocab_size = 100

        # Token at boundary (99 is standard, 100 is reasoning)
        token_ids = th.tensor([99, 100, 101])

        num_std, num_reas, pct_std, pct_reas = analyze_token_usage(token_ids, vocab_size)

        assert num_std == 1  # Only 99
        assert num_reas == 2  # 100 and 101

    def test_analyze_list_of_tensors(self):
        """Test analysis with list of tensors (ragged case)."""
        vocab_size = 100

        # Create a list of tensors with different lengths
        tensor1 = th.tensor([10, 20, 100, 110])  # 2 standard, 2 reasoning
        tensor2 = th.tensor([30, 40, 50])  # 3 standard
        tensor3 = th.tensor([120, 130, 140, 150, 160])  # 5 reasoning
        token_ids = [tensor1, tensor2, tensor3]

        num_std, num_reas, pct_std, pct_reas = analyze_token_usage(token_ids, vocab_size)

        # Total: 12 tokens (4 + 3 + 5), 5 standard, 7 reasoning
        assert num_std == 5
        assert num_reas == 7
        assert pct_std == pytest.approx(41.67, abs=0.1)
        assert pct_reas == pytest.approx(58.33, abs=0.1)

    def test_analyze_empty_list_of_tensors(self):
        """Test analysis with empty list of tensors."""
        vocab_size = 100
        token_ids: list[th.Tensor] = []

        num_std, num_reas, pct_std, pct_reas = analyze_token_usage(token_ids, vocab_size)

        assert num_std == 0
        assert num_reas == 0
        assert pct_std == 0.0
        assert pct_reas == 0.0


class TestCreateUsagePlot:
    """Tests for usage plot creation."""

    def test_create_basic_plot(self):
        """Test creating a basic usage plot."""
        steps = [100, 200, 300]
        standard_pcts = [80.0, 70.0, 60.0]
        reasoning_pcts = [20.0, 30.0, 40.0]

        fig = create_usage_plot(steps, standard_pcts, reasoning_pcts)

        assert isinstance(fig, Figure)
        assert len(fig.axes) == 1

        ax = fig.axes[0]
        assert len(ax.lines) == 2  # Two lines (standard and reasoning)
        assert ax.get_xlabel() == "Training Step"
        assert ax.get_ylabel() == "Token Usage (%)"

    def test_create_plot_with_custom_title(self):
        """Test creating plot with custom title."""
        steps = [1, 2, 3]
        standard_pcts = [50.0, 50.0, 50.0]
        reasoning_pcts = [50.0, 50.0, 50.0]
        custom_title = "Custom Test Title"

        fig = create_usage_plot(steps, standard_pcts, reasoning_pcts, title=custom_title)

        assert isinstance(fig, Figure)
        assert fig.axes[0].get_title() == custom_title

    def test_create_plot_single_datapoint(self):
        """Test creating plot with single data point."""
        steps = [100]
        standard_pcts = [75.0]
        reasoning_pcts = [25.0]

        fig = create_usage_plot(steps, standard_pcts, reasoning_pcts)

        assert isinstance(fig, Figure)
        assert len(fig.axes[0].lines) == 2

    def test_create_plot_empty_data(self):
        """Test creating plot with empty data."""
        steps: list[int] = []
        standard_pcts: list[float] = []
        reasoning_pcts: list[float] = []

        fig = create_usage_plot(steps, standard_pcts, reasoning_pcts)

        # Should still create figure, just with no data
        assert isinstance(fig, Figure)


class TestCreateStackedAreaPlot:
    """Tests for stacked area plot creation."""

    def test_create_basic_stacked_plot(self):
        """Test creating a basic stacked area plot."""
        steps = [100, 200, 300]
        standard_counts = [800, 700, 600]
        reasoning_counts = [200, 300, 400]

        fig = create_stacked_area_plot(steps, standard_counts, reasoning_counts)

        assert isinstance(fig, Figure)
        assert len(fig.axes) == 1

        ax = fig.axes[0]
        assert ax.get_xlabel() == "Training Step"
        assert ax.get_ylabel() == "Token Count"

    def test_create_stacked_plot_with_custom_title(self):
        """Test creating stacked plot with custom title."""
        steps = [1, 2]
        standard_counts = [100, 200]
        reasoning_counts = [50, 100]
        custom_title = "Custom Stacked Title"

        fig = create_stacked_area_plot(steps, standard_counts, reasoning_counts, title=custom_title)

        assert isinstance(fig, Figure)
        assert fig.axes[0].get_title() == custom_title

    def test_create_stacked_plot_single_datapoint(self):
        """Test creating stacked plot with single data point."""
        steps = [100]
        standard_counts = [1000]
        reasoning_counts = [500]

        fig = create_stacked_area_plot(steps, standard_counts, reasoning_counts)

        assert isinstance(fig, Figure)

    def test_create_stacked_plot_zero_counts(self):
        """Test creating stacked plot with zero counts."""
        steps = [100, 200]
        standard_counts = [1000, 0]
        reasoning_counts = [0, 1000]

        fig = create_stacked_area_plot(steps, standard_counts, reasoning_counts)

        assert isinstance(fig, Figure)


class TestVisualizeVocabUsageIntegration:
    """Integration tests for the full visualization pipeline."""

    def test_full_pipeline(self, temp_checkpoint_dir: Path, tmp_path: Path):
        """Test the complete visualization pipeline."""
        output_dir = tmp_path / "output"
        vocab_size = 100

        # Run visualization
        visualize_vocab_usage(temp_checkpoint_dir, vocab_size, output_dir)

        # Check that output files were created
        assert output_dir.exists()
        assert (output_dir / "vocab_usage_percentage.png").exists()
        assert (output_dir / "vocab_usage_stacked.png").exists()

    def test_pipeline_with_empty_checkpoints(self, tmp_path: Path):
        """Test pipeline with no checkpoints."""
        checkpoint_dir = tmp_path / "empty_checkpoints"
        checkpoint_dir.mkdir()
        output_dir = tmp_path / "output"
        vocab_size = 100

        # Should handle gracefully without errors
        visualize_vocab_usage(checkpoint_dir, vocab_size, output_dir)

        # Output directory should be created but no plots
        assert output_dir.exists()
        assert not (output_dir / "vocab_usage_percentage.png").exists()

    def test_pipeline_default_output_dir(
        self, temp_checkpoint_dir: Path, tmp_path: Path, monkeypatch
    ):
        """Test pipeline with default output directory."""
        vocab_size = 100

        # Change to temp directory to avoid polluting real ./fig
        monkeypatch.chdir(tmp_path)

        # Run with default output_dir (should create ./fig)
        visualize_vocab_usage(temp_checkpoint_dir, vocab_size)

        # Check that ./fig was created in temp directory
        assert (tmp_path / "fig").exists()
        assert (tmp_path / "fig" / "vocab_usage_percentage.png").exists()

    def test_pipeline_creates_output_directory(self, temp_checkpoint_dir: Path, tmp_path: Path):
        """Test that pipeline creates output directory if it doesn't exist."""
        output_dir = tmp_path / "nested" / "output" / "dir"
        vocab_size = 100

        # Output dir doesn't exist yet
        assert not output_dir.exists()

        visualize_vocab_usage(temp_checkpoint_dir, vocab_size, output_dir)

        # Should have been created
        assert output_dir.exists()
        assert (output_dir / "vocab_usage_percentage.png").exists()


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_analyze_very_large_vocab_size(self):
        """Test with very large vocabulary size."""
        vocab_size = 1_000_000
        token_ids = th.randint(0, 100, (100,))

        num_std, num_reas, pct_std, pct_reas = analyze_token_usage(token_ids, vocab_size)

        # All tokens should be standard since vocab_size is huge
        assert num_std == 100
        assert num_reas == 0

    def test_analyze_zero_vocab_size(self):
        """Test with zero vocabulary size."""
        vocab_size = 0
        token_ids = th.randint(0, 100, (50,))

        num_std, num_reas, pct_std, pct_reas = analyze_token_usage(token_ids, vocab_size)

        # All tokens should be reasoning since vocab_size is 0
        assert num_std == 0
        assert num_reas == 50

    def test_load_corrupted_checkpoint_file(self, tmp_path: Path):
        """Test handling of corrupted checkpoint files."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        ckpt_dir = checkpoint_dir / "checkpoint-100"
        ckpt_dir.mkdir()

        # Create corrupted file
        generations_file = ckpt_dir / "generations.pt"
        with open(generations_file, "w") as f:
            f.write("This is not a valid PyTorch file")

        # Should handle gracefully
        checkpoints = load_checkpoint_outputs(checkpoint_dir)
        assert len(checkpoints) == 0

    def test_plot_with_mismatched_lengths(self):
        """Test plot creation with mismatched list lengths."""
        steps = [100, 200, 300]
        standard_pcts = [80.0, 70.0]  # Missing one element
        reasoning_pcts = [20.0, 30.0, 40.0]

        # matplotlib raises ValueError for mismatched dimensions
        with pytest.raises(ValueError, match="must have same first dimension"):
            create_usage_plot(steps, standard_pcts, reasoning_pcts)
