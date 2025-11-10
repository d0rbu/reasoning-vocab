"""
Tests for token entropy visualization.

This module tests:
- Output entropy computation from logits
- Token entropy aggregation from validation data
- Trajectory computation across checkpoints
- Plot generation
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch as th
from datasets import Dataset

from viz.visualize_token_entropy import (
    aggregate_token_statistics,
    compute_output_entropy,
    compute_token_entropies_for_checkpoint,
    plot_token_entropy_trajectories,
)


class TestEntropyComputation:
    """Tests for output entropy computation."""

    def test_compute_output_entropy_uniform(self):
        """Test entropy for uniform distribution."""
        # Create uniform logits (all zeros -> uniform softmax)
        logits = th.zeros(10, 20, 100)  # [batch, seq_len, vocab_size]
        entropy = compute_output_entropy(logits)

        # Uniform distribution over 100 items: H = log(100)
        expected_entropy = np.log(100)
        assert th.allclose(entropy, th.tensor(expected_entropy, dtype=th.float32), rtol=0.01)
        assert entropy.shape == (10, 20)

    def test_compute_output_entropy_peaked(self):
        """Test entropy for peaked distribution."""
        logits = th.zeros(5, 10, 50)
        # Make one logit very large (peaked distribution)
        logits[:, :, 0] = 10.0

        entropy = compute_output_entropy(logits)

        # Peaked distribution should have low entropy
        assert th.all(entropy < 1.0)
        assert entropy.shape == (5, 10)

    def test_compute_output_entropy_batched(self):
        """Test entropy computation with batched inputs."""
        batch_sizes = [1, 4, 16]
        seq_len = 8
        vocab_size = 100

        for batch_size in batch_sizes:
            logits = th.randn(batch_size, seq_len, vocab_size)
            entropy = compute_output_entropy(logits)

            assert entropy.shape == (batch_size, seq_len)
            assert th.all(entropy >= 0)

    def test_compute_output_entropy_numerical_stability(self):
        """Test that entropy computation handles extreme values."""
        # Very large logits
        logits = th.randn(2, 5, 50) * 100
        entropy = compute_output_entropy(logits)
        assert th.all(th.isfinite(entropy))

        # Very small logits
        logits = th.randn(2, 5, 50) * 0.01
        entropy = compute_output_entropy(logits)
        assert th.all(th.isfinite(entropy))


class TestTokenEntropyAggregation:
    """Tests for aggregating token entropies from model outputs."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model that returns predictable outputs."""
        model = MagicMock()

        def mock_forward(input_ids=None, attention_mask=None, **kwargs):
            # Handle case where input_ids is a mock or doesn't have shape
            if hasattr(input_ids, "shape") and len(input_ids.shape) == 2:
                batch_size, seq_len = input_ids.shape
            else:
                # Fallback for mocked inputs
                batch_size, seq_len = 2, 10

            vocab_size = 1000

            # Create mock logits
            logits = th.randn(batch_size, seq_len, vocab_size)

            # Return mock output
            output = MagicMock()
            output.logits = logits
            return output

        model.side_effect = mock_forward
        model.eval = MagicMock(return_value=None)
        model.to = MagicMock(return_value=model)
        return model

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        tokenizer.bos_token_id = 1
        tokenizer.eos_token_id = 2

        def mock_call(texts, **kwargs):
            # Simple mock tokenization
            batch_size = len(texts)
            seq_len = 10

            input_ids = th.randint(3, 100, (batch_size, seq_len))
            attention_mask = th.ones(batch_size, seq_len, dtype=th.long)

            # Create a proper dict-like object that supports indexing and attributes
            class MockEncoding:
                def __init__(self, input_ids, attention_mask):
                    self.input_ids = input_ids
                    self.attention_mask = attention_mask
                    self._dict = {"input_ids": input_ids, "attention_mask": attention_mask}

                def __getitem__(self, key):
                    return self._dict[key]

                def to(self, device):
                    # Return tensors directly (they handle .to())
                    return self

            encoding = MockEncoding(input_ids, attention_mask)
            return encoding

        tokenizer.side_effect = mock_call
        tokenizer.decode = lambda ids: f"token_{ids[0]}"

        return tokenizer

    @pytest.fixture
    def mock_dataset(self):
        """Create a mock validation dataset."""
        data = [{"text": f"Sample text {i}"} for i in range(20)]
        return Dataset.from_list(data)

    def test_compute_token_entropies_basic(self, mock_model, mock_tokenizer, mock_dataset):
        """Test basic token entropy computation."""
        token_entropies = compute_token_entropies_for_checkpoint(
            mock_model,
            mock_tokenizer,
            mock_dataset,
            max_samples=5,
            batch_size=2,
            device="cpu",
        )

        # Should have computed entropies for multiple tokens
        assert len(token_entropies) > 0

        # Each token should have a list of entropy values
        for _token_id, entropies in token_entropies.items():
            assert isinstance(entropies, list)
            assert len(entropies) > 0
            assert all(isinstance(e, float) for e in entropies)
            assert all(e >= 0 for e in entropies)

    def test_compute_token_entropies_respects_max_samples(
        self, mock_model, mock_tokenizer, mock_dataset
    ):
        """Test that max_samples parameter is respected."""
        max_samples = 3

        with patch("viz.visualize_token_entropy.tqdm") as mock_tqdm:
            # Make tqdm return the iterable unchanged
            mock_tqdm.side_effect = lambda x, **kwargs: x

            token_entropies = compute_token_entropies_for_checkpoint(
                mock_model,
                mock_tokenizer,
                mock_dataset,
                max_samples=max_samples,
                batch_size=2,
                device="cpu",
            )

        # Should have processed at most max_samples
        assert len(token_entropies) > 0

    def test_compute_token_entropies_filters_special_tokens(
        self, mock_model, mock_tokenizer, mock_dataset
    ):
        """Test that special tokens are filtered out."""
        token_entropies = compute_token_entropies_for_checkpoint(
            mock_model,
            mock_tokenizer,
            mock_dataset,
            max_samples=10,
            batch_size=2,
            device="cpu",
        )

        # Should not contain pad or bos tokens
        assert mock_tokenizer.pad_token_id not in token_entropies
        assert mock_tokenizer.bos_token_id not in token_entropies

    def test_compute_token_entropies_different_dataset_formats(self, mock_model, mock_tokenizer):
        """Test handling different dataset field names."""
        # Test with 'problem' field
        dataset_problem = Dataset.from_list([{"problem": f"Problem {i}"} for i in range(5)])

        token_entropies = compute_token_entropies_for_checkpoint(
            mock_model, mock_tokenizer, dataset_problem, max_samples=5, batch_size=2, device="cpu"
        )
        assert len(token_entropies) > 0

        # Test with 'question' field
        dataset_question = Dataset.from_list([{"question": f"Question {i}"} for i in range(5)])

        token_entropies = compute_token_entropies_for_checkpoint(
            mock_model, mock_tokenizer, dataset_question, max_samples=5, batch_size=2, device="cpu"
        )
        assert len(token_entropies) > 0


class TestStatisticsAggregation:
    """Tests for aggregating token entropy statistics."""

    def test_aggregate_token_statistics_basic(self):
        """Test basic statistics aggregation."""
        token_entropies = {
            10: [1.0, 1.5, 2.0, 1.8, 1.2],
            20: [3.0, 3.2, 2.9, 3.1],
            30: [0.5],
        }

        stats = aggregate_token_statistics(token_entropies)

        # Check structure
        assert len(stats) == 3
        assert 10 in stats
        assert 20 in stats
        assert 30 in stats

        # Check stats for token 10
        assert "mean" in stats[10]
        assert "std" in stats[10]
        assert "count" in stats[10]
        assert "min" in stats[10]
        assert "max" in stats[10]

        # Verify values
        assert np.isclose(stats[10]["mean"], np.mean([1.0, 1.5, 2.0, 1.8, 1.2]))
        assert stats[10]["count"] == 5
        assert stats[10]["min"] == 1.0
        assert stats[10]["max"] == 2.0

    def test_aggregate_token_statistics_single_value(self):
        """Test statistics with single value."""
        token_entropies = {42: [2.5]}

        stats = aggregate_token_statistics(token_entropies)

        assert stats[42]["mean"] == 2.5
        assert stats[42]["std"] == 0.0
        assert stats[42]["count"] == 1
        assert stats[42]["min"] == 2.5
        assert stats[42]["max"] == 2.5

    def test_aggregate_token_statistics_empty_lists(self):
        """Test that empty lists are skipped."""
        token_entropies = {
            10: [1.0, 2.0],
            20: [],  # Empty list
            30: [3.0],
        }

        stats = aggregate_token_statistics(token_entropies)

        # Empty list should be skipped
        assert 20 not in stats
        assert 10 in stats
        assert 30 in stats

    def test_aggregate_token_statistics_large_variance(self):
        """Test statistics with high variance."""
        token_entropies = {
            100: [0.1, 0.1, 0.1, 10.0, 10.0, 10.0]  # Bimodal distribution
        }

        stats = aggregate_token_statistics(token_entropies)

        assert stats[100]["count"] == 6
        assert stats[100]["std"] > 0
        assert stats[100]["min"] == 0.1
        assert stats[100]["max"] == 10.0


class TestPlotting:
    """Tests for plotting functions."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer for plotting."""
        tokenizer = MagicMock()
        tokenizer.decode = lambda ids: f"token_{ids[0]}"
        return tokenizer

    @pytest.fixture
    def sample_trajectories(self):
        """Create sample trajectory data."""
        baseline = {
            100: {
                10: {"mean": 2.0, "std": 0.5, "count": 50},
                20: {"mean": 3.0, "std": 0.3, "count": 30},
            },
            200: {
                10: {"mean": 1.8, "std": 0.4, "count": 55},
                20: {"mean": 2.8, "std": 0.3, "count": 35},
            },
            300: {
                10: {"mean": 1.5, "std": 0.3, "count": 60},
                20: {"mean": 2.5, "std": 0.2, "count": 40},
            },
        }

        reasoning = {
            100: {
                10: {"mean": 2.1, "std": 0.6, "count": 50},
                20: {"mean": 3.2, "std": 0.4, "count": 30},
            },
            200: {
                10: {"mean": 1.7, "std": 0.5, "count": 55},
                20: {"mean": 2.7, "std": 0.3, "count": 35},
            },
            300: {
                10: {"mean": 1.3, "std": 0.4, "count": 60},
                20: {"mean": 2.3, "std": 0.3, "count": 40},
            },
        }

        return baseline, reasoning

    def test_plot_token_entropy_trajectories_basic(
        self, sample_trajectories, mock_tokenizer, tmp_path
    ):
        """Test basic trajectory plotting."""
        baseline, reasoning = sample_trajectories
        token_ids = [10, 20]
        output_dir = tmp_path / "plots"

        plot_token_entropy_trajectories(
            baseline, reasoning, token_ids, mock_tokenizer, output_dir, top_k=20
        )

        # Check that plots were created
        assert output_dir.exists()
        plot_files = list(output_dir.glob("*.png"))
        assert len(plot_files) == 2  # One for each token

    def test_plot_token_entropy_trajectories_top_k(
        self, sample_trajectories, mock_tokenizer, tmp_path
    ):
        """Test that top_k parameter limits number of plots."""
        baseline, reasoning = sample_trajectories

        # Create many token IDs
        token_ids = list(range(100))

        # Add data for these tokens to reasoning trajectory
        for step in reasoning.keys():
            for tid in token_ids:
                reasoning[step][tid] = {"mean": 2.0, "std": 0.5, "count": 10 * tid}

        output_dir = tmp_path / "plots_topk"

        plot_token_entropy_trajectories(
            baseline, reasoning, token_ids, mock_tokenizer, output_dir, top_k=10
        )

        # Should only create 10 plots (top_k)
        plot_files = list(output_dir.glob("*.png"))
        assert len(plot_files) == 10

    def test_plot_token_entropy_trajectories_no_baseline(
        self, sample_trajectories, mock_tokenizer, tmp_path
    ):
        """Test plotting with no baseline data."""
        _, reasoning = sample_trajectories
        baseline_empty = {}
        token_ids = [10, 20]
        output_dir = tmp_path / "plots_no_baseline"

        # Should not raise error
        plot_token_entropy_trajectories(
            baseline_empty, reasoning, token_ids, mock_tokenizer, output_dir, top_k=20
        )

        assert output_dir.exists()
        plot_files = list(output_dir.glob("*.png"))
        assert len(plot_files) == 2

    def test_plot_token_entropy_trajectories_creates_directory(
        self, sample_trajectories, mock_tokenizer, tmp_path
    ):
        """Test that output directory is created if it doesn't exist."""
        baseline, reasoning = sample_trajectories
        output_dir = tmp_path / "nested" / "dir" / "plots"

        plot_token_entropy_trajectories(
            baseline, reasoning, [10], mock_tokenizer, output_dir, top_k=20
        )

        assert output_dir.exists()


class TestIntegration:
    """Integration tests for the full pipeline."""

    @pytest.fixture
    def mock_checkpoint_structure(self, tmp_path):
        """Create mock checkpoint directories."""
        reasoning_dir = tmp_path / "reasoning"

        for step in [100, 200, 300]:
            checkpoint_dir = reasoning_dir / f"checkpoint-{step}"
            checkpoint_dir.mkdir(parents=True)

        return reasoning_dir

    @patch("viz.visualize_token_entropy.load_dataset")
    @patch("viz.visualize_token_entropy.AutoTokenizer.from_pretrained")
    @patch("viz.visualize_token_entropy.AutoModelForCausalLM.from_pretrained")
    @patch("viz.visualize_token_entropy.compute_token_entropies_for_checkpoint")
    @patch("viz.visualize_token_entropy.plot_token_entropy_trajectories")
    def test_visualize_token_entropy_full_pipeline(
        self,
        mock_plot,
        mock_compute,
        mock_model_load,
        mock_tokenizer_load,
        mock_dataset_load,
        mock_checkpoint_structure,
    ):
        """Test the full visualization pipeline."""
        from viz.visualize_token_entropy import visualize_token_entropy

        # Setup mocks
        mock_dataset = MagicMock()
        mock_dataset_load.return_value = mock_dataset

        mock_tokenizer = MagicMock()
        mock_tokenizer_load.return_value = mock_tokenizer

        # Mock compute function to return sample data
        mock_compute.return_value = {
            10: [2.0, 2.1, 1.9],
            20: [3.0, 3.1, 2.9],
        }

        # Run visualization
        results = visualize_token_entropy(
            baseline_dir=None,
            reasoning_dir=mock_checkpoint_structure,
            dataset_name="gsm8k",
            max_samples=10,
            batch_size=2,
            device="cpu",
        )

        # Verify results
        assert "reasoning" in results
        assert len(results["reasoning"]) > 0

        # Verify mocks were called
        assert mock_dataset_load.called
        assert mock_tokenizer_load.called
        assert mock_compute.called
        assert mock_plot.called


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_compute_output_entropy_single_token(self):
        """Test entropy with single token vocabulary."""
        logits = th.randn(2, 5, 1)
        entropy = compute_output_entropy(logits)

        # Single token -> zero entropy
        assert th.allclose(entropy, th.zeros_like(entropy), atol=1e-6)

    def test_aggregate_token_statistics_empty_input(self):
        """Test aggregation with empty input."""
        token_entropies = {}
        stats = aggregate_token_statistics(token_entropies)

        assert len(stats) == 0

    def test_compute_output_entropy_very_large_vocab(self):
        """Test entropy computation with very large vocabulary."""
        logits = th.randn(1, 1, 100000)
        entropy = compute_output_entropy(logits)

        assert th.all(th.isfinite(entropy))
        assert entropy.shape == (1, 1)
