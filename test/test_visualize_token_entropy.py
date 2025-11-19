"""
Tests for token entropy visualization.

This module tests:
- Output distribution entropy computation
- Processing validation data through checkpoints
- Grouping entropies by token
- Trajectory computation across checkpoints
- Plot generation
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch as th

from viz.visualize_token_entropy import (
    TokenEntropyTrajectory,
    compute_output_entropy,
    compute_token_entropy_trajectory,
    plot_token_entropy_comparison,
    visualize_token_entropy,
)


class TestEntropyComputation:
    """Tests for entropy computation from output logits."""

    def test_compute_output_entropy_uniform(self):
        """Test entropy computation for uniform distribution."""
        # Uniform logits (all zeros) -> uniform probs after softmax
        vocab_size = 100
        logits = th.zeros(vocab_size)
        entropy = compute_output_entropy(logits, vocab_size)

        # For uniform distribution over n items: H = log(n)
        expected_entropy = np.log(vocab_size)
        assert np.isclose(entropy, expected_entropy, rtol=0.01)

    def test_compute_output_entropy_peaked(self):
        """Test entropy computation for peaked distribution."""
        # Create peaked distribution (one logit much higher)
        vocab_size = 100
        logits = th.zeros(vocab_size)
        logits[0] = 10.0

        entropy = compute_output_entropy(logits, vocab_size)

        # Peaked distribution should have low entropy
        assert entropy < 1.0

    def test_compute_output_entropy_deterministic(self):
        """Test that entropy computation is deterministic."""
        vocab_size = 50
        logits = th.randn(vocab_size)

        entropy1 = compute_output_entropy(logits, vocab_size)
        entropy2 = compute_output_entropy(logits, vocab_size)

        assert entropy1 == entropy2

    def test_compute_output_entropy_positive(self):
        """Test that entropy is always non-negative."""
        vocab_size = 100
        for _ in range(10):
            logits = th.randn(vocab_size)
            entropy = compute_output_entropy(logits, vocab_size)
            assert entropy >= 0.0

    def test_compute_output_entropy_different_vocab_sizes(self):
        """Test entropy with different vocabulary sizes."""
        for vocab_size in [100, 1000, 10000, 50000]:
            logits = th.randn(vocab_size)
            entropy = compute_output_entropy(logits, vocab_size)
            assert entropy >= 0.0
            # Entropy bounded by log(vocab_size)
            assert entropy <= np.log(vocab_size) + 1.0


class TestTrajectoryComputation:
    """Tests for computing trajectories across checkpoints."""

    @pytest.fixture
    def mock_checkpoints(self, tmp_path):
        """Create mock checkpoint directories."""
        checkpoints = []
        for step in [100, 200, 300]:
            checkpoint_dir = tmp_path / f"checkpoint-{step}"
            checkpoint_dir.mkdir()
            checkpoints.append(checkpoint_dir)
        return checkpoints

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = MagicMock()
        tokenizer.decode = lambda ids: f"token_{ids[0]}"
        return tokenizer

    @patch("viz.visualize_token_entropy.AutoModelForCausalLM.from_pretrained")
    @patch("viz.visualize_token_entropy.compute_token_entropies_from_model")
    def test_compute_trajectory_basic(
        self, mock_compute_entropies, mock_from_pretrained, mock_checkpoints, mock_tokenizer
    ):
        """Test basic trajectory computation."""
        # Mock model loading
        mock_model = MagicMock()
        mock_from_pretrained.return_value = mock_model

        # Mock compute_token_entropies_from_model to return entropy tensor
        vocab_size = 1000
        mock_compute_entropies.side_effect = [
            # Step 100
            th.tensor([2.0 if i == 42 else 3.0 if i == 100 else 0.0 for i in range(vocab_size)]),
            # Step 200
            th.tensor([2.5 if i == 42 else 3.2 if i == 100 else 0.0 for i in range(vocab_size)]),
            # Step 300
            th.tensor([3.0 if i == 42 else 3.5 if i == 100 else 0.0 for i in range(vocab_size)]),
        ]

        dataset = ["test"]
        token_ids = [42, 100]

        result = compute_token_entropy_trajectory(
            mock_checkpoints, mock_tokenizer, dataset, vocab_size, token_ids, max_samples=10
        )

        # Should have results for both tokens
        assert 42 in result
        assert 100 in result

        # Should have 3 steps for each
        assert len(result[42].steps) == 3
        assert len(result[42].entropies) == 3
        assert result[42].steps == [100, 200, 300]

        # Check entropy values
        assert np.isclose(result[42].entropies[0], 2.0, atol=0.01)
        assert np.isclose(result[42].entropies[1], 2.5, atol=0.01)
        assert np.isclose(result[42].entropies[2], 3.0, atol=0.01)

    @patch("viz.visualize_token_entropy.AutoModelForCausalLM.from_pretrained")
    @patch("viz.visualize_token_entropy.compute_token_entropies_from_model")
    def test_compute_trajectory_missing_token(
        self, mock_compute_entropies, mock_from_pretrained, mock_checkpoints, mock_tokenizer
    ):
        """Test handling when token is missing from checkpoint."""
        # Mock model loading
        mock_model = MagicMock()
        mock_from_pretrained.return_value = mock_model

        # Token 42 present in all, token 100 present in all (can't actually be "missing" in tensor)
        vocab_size = 1000
        mock_compute_entropies.side_effect = [
            th.tensor([2.0 if i == 42 else 3.0 if i == 100 else 0.0 for i in range(vocab_size)]),
            th.tensor([2.5 if i == 42 else 0.0 for i in range(vocab_size)]),  # Token 100 has 0
            th.tensor([3.0 if i == 42 else 3.5 if i == 100 else 0.0 for i in range(vocab_size)]),
        ]

        dataset = ["test"]
        token_ids = [42, 100]

        result = compute_token_entropy_trajectory(
            mock_checkpoints, mock_tokenizer, dataset, vocab_size, token_ids
        )

        # Token 42 should have all 3 steps
        assert len(result[42].steps) == 3

        # Token 100 should have all 3 steps (entropy may be zero for step 200)
        assert len(result[100].steps) == 3
        assert result[100].steps == [100, 200, 300]

    @patch("viz.visualize_token_entropy.AutoModelForCausalLM.from_pretrained")
    @patch("viz.visualize_token_entropy.compute_token_entropies_from_model")
    def test_compute_trajectory_error_handling(
        self, mock_compute_entropies, mock_from_pretrained, mock_checkpoints, mock_tokenizer
    ):
        """Test that errors in processing are handled gracefully."""
        # Mock model loading
        mock_model = MagicMock()
        mock_from_pretrained.return_value = mock_model

        # Make middle checkpoint fail
        vocab_size = 1000
        mock_compute_entropies.side_effect = [
            th.tensor([2.0 if i == 42 else 0.0 for i in range(vocab_size)]),
            RuntimeError("Failed to load"),
            th.tensor([3.0 if i == 42 else 0.0 for i in range(vocab_size)]),
        ]

        dataset = ["test"]
        token_ids = [42]

        result = compute_token_entropy_trajectory(
            mock_checkpoints, mock_tokenizer, dataset, vocab_size, token_ids
        )

        # Should have 2 successful checkpoints
        assert len(result[42].steps) == 2
        assert 200 not in result[42].steps

    def test_compute_trajectory_invalid_checkpoint_names(self, tmp_path, mock_tokenizer):
        """Test handling of invalid checkpoint names."""
        checkpoints = [
            tmp_path / "checkpoint-100",
            tmp_path / "checkpoint-abc",  # Invalid
            tmp_path / "checkpoint-200",
        ]

        for cp in checkpoints:
            cp.mkdir()

        vocab_size = 1000
        with (
            patch(
                "viz.visualize_token_entropy.AutoModelForCausalLM.from_pretrained"
            ) as mock_from_pretrained,
            patch(
                "viz.visualize_token_entropy.compute_token_entropies_from_model"
            ) as mock_compute_entropies,
        ):
            mock_model = MagicMock()
            mock_from_pretrained.return_value = mock_model

            mock_compute_entropies.return_value = th.tensor(
                [2.0 if i == 42 else 0.0 for i in range(vocab_size)]
            )

            result = compute_token_entropy_trajectory(
                checkpoints, mock_tokenizer, ["test"], vocab_size, [42]
            )

            # Should only process valid checkpoints
            assert len(result[42].steps) == 2
            assert result[42].steps == [100, 200]


class TestPlotting:
    """Tests for plotting functions."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = MagicMock()
        tokenizer.decode = lambda ids: f"token_{ids[0]}"
        return tokenizer

    def test_plot_basic(self, tmp_path, mock_tokenizer):
        """Test basic plot generation."""
        baseline_data = {
            42: TokenEntropyTrajectory(steps=[100, 200, 300], entropies=[2.0, 2.5, 3.0]),
            100: TokenEntropyTrajectory(steps=[100, 200, 300], entropies=[3.0, 3.2, 3.5]),
        }

        reasoning_data = {
            42: TokenEntropyTrajectory(steps=[100, 200, 300], entropies=[2.1, 2.4, 2.9]),
            100: TokenEntropyTrajectory(steps=[100, 200, 300], entropies=[3.1, 3.3, 3.6]),
        }

        output_path = tmp_path / "test_plot.png"

        plot_token_entropy_comparison(baseline_data, reasoning_data, mock_tokenizer, output_path)

        assert output_path.exists()

    def test_plot_empty_baseline(self, tmp_path, mock_tokenizer):
        """Test plot with no baseline data."""
        baseline_data = {}

        reasoning_data = {
            42: TokenEntropyTrajectory(steps=[100, 200], entropies=[2.0, 2.5]),
        }

        output_path = tmp_path / "test_plot_no_baseline.png"

        plot_token_entropy_comparison(baseline_data, reasoning_data, mock_tokenizer, output_path)

        assert output_path.exists()

    def test_plot_creates_output_directory(self, tmp_path, mock_tokenizer):
        """Test that plot creates output directory if needed."""
        output_path = tmp_path / "nested" / "dir" / "plot.png"

        data = {42: TokenEntropyTrajectory(steps=[100], entropies=[2.0])}

        plot_token_entropy_comparison(data, data, mock_tokenizer, output_path)

        assert output_path.exists()
        assert output_path.parent.exists()


class TestVisualizationPipeline:
    """Tests for the full visualization pipeline."""

    @pytest.fixture
    def mock_checkpoint_dirs(self, tmp_path):
        """Create mock checkpoint directories."""
        baseline_dir = tmp_path / "baseline"
        reasoning_dir = tmp_path / "reasoning"

        for step in [100, 200]:
            (baseline_dir / f"checkpoint-{step}").mkdir(parents=True)
            (reasoning_dir / f"checkpoint-{step}").mkdir(parents=True)

        return baseline_dir, reasoning_dir

    @pytest.fixture
    def mock_dataset(self):
        """Create a mock dataset."""
        dataset = MagicMock()
        dataset.column_names = ["question", "answer"]
        dataset.__iter__ = lambda self: iter(
            [
                {"question": "What is 2+2?", "answer": "4"},
                {"question": "What is 3+3?", "answer": "6"},
            ]
        )
        return dataset

    @patch("viz.visualize_token_entropy.load_dataset")
    @patch("viz.visualize_token_entropy.AutoTokenizer.from_pretrained")
    @patch("viz.visualize_token_entropy.compute_token_entropy_trajectory")
    @patch("viz.visualize_token_entropy.plot_token_entropy_comparison")
    def test_full_pipeline(
        self,
        mock_plot,
        mock_compute,
        mock_tokenizer,
        mock_load_dataset,
        mock_checkpoint_dirs,
        mock_dataset,
        tmp_path,
    ):
        """Test full visualization pipeline."""
        baseline_dir, reasoning_dir = mock_checkpoint_dirs

        # Setup mocks
        mock_load_dataset.return_value = mock_dataset
        mock_tokenizer.return_value = MagicMock()

        mock_compute.side_effect = [
            {42: TokenEntropyTrajectory(steps=[100, 200], entropies=[2.0, 2.5])},  # Baseline
            {42: TokenEntropyTrajectory(steps=[100, 200], entropies=[2.1, 2.4])},  # Reasoning
        ]

        output_dir = tmp_path / "output"

        result = visualize_token_entropy(
            baseline_dir=baseline_dir,
            reasoning_dir=reasoning_dir,
            token_ids=[42],
            dataset_name="openai/gsm8k",
            max_samples=10,
            output_dir=output_dir,
        )

        # Check results structure
        assert "baseline" in result
        assert "reasoning" in result
        assert "token_ids" in result
        assert result["token_ids"] == [42]

        # Check that plot was called
        assert mock_plot.called

    @patch("viz.visualize_token_entropy.load_dataset")
    @patch("viz.visualize_token_entropy.AutoTokenizer.from_pretrained")
    @patch("viz.visualize_token_entropy.compute_token_entropy_trajectory")
    @patch("viz.visualize_token_entropy.plot_token_entropy_comparison")
    def test_pipeline_no_baseline(
        self,
        mock_plot,
        mock_compute,
        mock_tokenizer,
        mock_load_dataset,
        mock_checkpoint_dirs,
        mock_dataset,
        tmp_path,
    ):
        """Test pipeline without baseline directory."""
        _, reasoning_dir = mock_checkpoint_dirs

        mock_load_dataset.return_value = mock_dataset
        mock_tokenizer.return_value = MagicMock()
        mock_compute.return_value = {42: TokenEntropyTrajectory(steps=[100], entropies=[2.0])}

        result = visualize_token_entropy(
            baseline_dir=None,
            reasoning_dir=reasoning_dir,
            token_ids=[42],
            max_samples=10,
            output_dir=tmp_path,
        )

        # Should work without baseline
        assert "baseline" in result
        assert result["baseline"] == {}

    @patch("viz.visualize_token_entropy.load_dataset")
    @patch("viz.visualize_token_entropy.AutoTokenizer.from_pretrained")
    def test_pipeline_unknown_dataset_structure(self, mock_tokenizer, mock_load_dataset, tmp_path):
        """Test handling of unknown dataset structure."""
        reasoning_dir = tmp_path / "reasoning"
        reasoning_dir.mkdir()

        # Mock tokenizer to avoid loading issues
        mock_tokenizer.return_value = MagicMock()

        dataset = MagicMock()
        dataset.column_names = ["unknown_field"]
        mock_load_dataset.return_value = dataset

        with pytest.raises(ValueError, match="Unknown dataset structure"):
            visualize_token_entropy(
                baseline_dir=None,
                reasoning_dir=reasoning_dir,
                token_ids=[42],
                output_dir=tmp_path,
            )


class TestEdgeCases:
    """Tests for edge cases."""

    def test_compute_output_entropy_single_vocab(self):
        """Test entropy with vocabulary size of 1."""
        vocab_size = 1
        logits = th.tensor([1.0])
        entropy = compute_output_entropy(logits, vocab_size)

        # Single item should have zero entropy
        assert np.isclose(entropy, 0.0, atol=1e-6)

    def test_compute_output_entropy_extreme_logits(self):
        """Test entropy with extreme logit values."""
        # Very large logits
        vocab_size = 4
        logits = th.tensor([1000.0, -1000.0, 500.0, -500.0])
        entropy = compute_output_entropy(logits, vocab_size)
        assert entropy >= 0.0
        assert not np.isnan(entropy)
        assert not np.isinf(entropy)

    def test_compute_output_entropy_very_large_vocab(self):
        """Test entropy with very large vocabulary."""
        vocab_size = 50000  # Large vocab like GPT
        logits = th.randn(vocab_size)
        entropy = compute_output_entropy(logits, vocab_size)
        assert entropy >= 0.0
        assert not np.isnan(entropy)
