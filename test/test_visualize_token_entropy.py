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
    compute_output_entropy,
    compute_token_entropy_trajectory,
    plot_token_entropy_comparison,
    process_checkpoint_for_token_entropies,
    visualize_token_entropy,
)


class TestEntropyComputation:
    """Tests for entropy computation from output logits."""

    def test_compute_output_entropy_uniform(self):
        """Test entropy computation for uniform distribution."""
        # Uniform logits (all zeros) -> uniform probs after softmax
        logits = th.zeros(100)
        entropy = compute_output_entropy(logits)

        # For uniform distribution over n items: H = log(n)
        expected_entropy = np.log(100)
        assert np.isclose(entropy, expected_entropy, rtol=0.01)

    def test_compute_output_entropy_peaked(self):
        """Test entropy computation for peaked distribution."""
        # Create peaked distribution (one logit much higher)
        logits = th.zeros(100)
        logits[0] = 10.0

        entropy = compute_output_entropy(logits)

        # Peaked distribution should have low entropy
        assert entropy < 1.0

    def test_compute_output_entropy_deterministic(self):
        """Test that entropy computation is deterministic."""
        logits = th.randn(50)

        entropy1 = compute_output_entropy(logits)
        entropy2 = compute_output_entropy(logits)

        assert entropy1 == entropy2

    def test_compute_output_entropy_positive(self):
        """Test that entropy is always non-negative."""
        for _ in range(10):
            logits = th.randn(100)
            entropy = compute_output_entropy(logits)
            assert entropy >= 0.0

    def test_compute_output_entropy_different_vocab_sizes(self):
        """Test entropy with different vocabulary sizes."""
        for vocab_size in [100, 1000, 10000, 50000]:
            logits = th.randn(vocab_size)
            entropy = compute_output_entropy(logits)
            assert entropy >= 0.0
            # Entropy bounded by log(vocab_size)
            assert entropy <= np.log(vocab_size) + 1.0


class TestCheckpointProcessing:
    """Tests for processing checkpoints with validation data."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model that returns logits."""
        model = MagicMock()
        model.eval = MagicMock(return_value=None)

        # Mock forward pass to return logits
        def forward(**kwargs):
            input_ids = kwargs["input_ids"]
            seq_len = input_ids.shape[1]
            vocab_size = 1000

            # Return mock logits
            outputs = MagicMock()
            outputs.logits = th.randn(1, seq_len, vocab_size)
            return outputs

        model.side_effect = forward

        return model

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = MagicMock()

        def tokenize(text, **kwargs):
            # Return fake token IDs
            fake_ids = [1, 2, 3, 4, 5]  # Simple sequence
            return {"input_ids": th.tensor([fake_ids])}

        tokenizer.side_effect = tokenize
        tokenizer.decode = lambda ids: f"token_{ids[0]}"

        return tokenizer

    @patch("viz.visualize_token_entropy.AutoModelForCausalLM.from_pretrained")
    def test_process_checkpoint_basic(
        self, mock_from_pretrained, mock_model, mock_tokenizer, tmp_path
    ):
        """Test basic checkpoint processing."""
        mock_from_pretrained.return_value = mock_model

        checkpoint_path = tmp_path / "checkpoint-100"
        checkpoint_path.mkdir()

        dataset = ["hello world", "test sentence", "another example"]

        result = process_checkpoint_for_token_entropies(
            checkpoint_path, mock_tokenizer, dataset, max_samples=None
        )

        # Should return dictionary of token_id -> list of entropies
        assert isinstance(result, dict)
        assert len(result) > 0

        # Each token should have a list of entropy values
        for _token_id, entropies in result.items():
            assert isinstance(entropies, list)
            assert len(entropies) > 0
            assert all(e >= 0.0 for e in entropies)

    @patch("viz.visualize_token_entropy.AutoModelForCausalLM.from_pretrained")
    def test_process_checkpoint_max_samples(
        self, mock_from_pretrained, mock_model, mock_tokenizer, tmp_path
    ):
        """Test that max_samples limits processing."""
        mock_from_pretrained.return_value = mock_model

        checkpoint_path = tmp_path / "checkpoint-100"
        checkpoint_path.mkdir()

        dataset = ["sample"] * 100

        # Process with limit
        result = process_checkpoint_for_token_entropies(
            checkpoint_path, mock_tokenizer, dataset, max_samples=10
        )

        # Should still return results
        assert isinstance(result, dict)
        assert len(result) > 0

    @patch("viz.visualize_token_entropy.AutoModelForCausalLM.from_pretrained")
    def test_process_checkpoint_groups_by_token(
        self, mock_from_pretrained, mock_tokenizer, tmp_path
    ):
        """Test that entropies are correctly grouped by token."""
        # Create a model that returns consistent logits
        model = MagicMock()
        model.eval.return_value = None

        def forward(**kwargs):
            input_ids = kwargs["input_ids"]
            seq_len = input_ids.shape[1]

            outputs = MagicMock()
            # Return specific logits pattern
            outputs.logits = th.ones(1, seq_len, 100) * 2.0
            return outputs

        model.side_effect = forward
        mock_from_pretrained.return_value = model

        checkpoint_path = tmp_path / "checkpoint-100"
        checkpoint_path.mkdir()

        dataset = ["test"]

        result = process_checkpoint_for_token_entropies(checkpoint_path, mock_tokenizer, dataset)

        # Check that results are grouped by token ID
        assert isinstance(result, dict)
        for _token_id, entropies in result.items():
            # All entropies for same token should be similar (same logits pattern)
            if len(entropies) > 1:
                assert np.std(entropies) < 0.1  # Low variance


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

    @patch("viz.visualize_token_entropy.process_checkpoint_for_token_entropies")
    def test_compute_trajectory_basic(self, mock_process, mock_checkpoints, mock_tokenizer):
        """Test basic trajectory computation."""
        # Mock process_checkpoint to return entropy data
        mock_process.side_effect = [
            {42: [2.0, 2.1, 2.0], 100: [3.0, 3.1]},  # Step 100
            {42: [2.5, 2.4, 2.6], 100: [3.2, 3.3]},  # Step 200
            {42: [3.0, 2.9, 3.1], 100: [3.5, 3.4]},  # Step 300
        ]

        dataset = ["test"]
        token_ids = [42, 100]

        result = compute_token_entropy_trajectory(
            mock_checkpoints, mock_tokenizer, dataset, token_ids, max_samples=10
        )

        # Should have results for both tokens
        assert 42 in result
        assert 100 in result

        # Should have 3 steps for each
        assert len(result[42]["steps"]) == 3
        assert len(result[42]["entropies"]) == 3
        assert result[42]["steps"] == [100, 200, 300]

        # Entropies should be averages
        assert np.isclose(result[42]["entropies"][0], 2.0333, atol=0.01)
        assert np.isclose(result[42]["entropies"][1], 2.5, atol=0.01)
        assert np.isclose(result[42]["entropies"][2], 3.0, atol=0.01)

    @patch("viz.visualize_token_entropy.process_checkpoint_for_token_entropies")
    def test_compute_trajectory_missing_token(self, mock_process, mock_checkpoints, mock_tokenizer):
        """Test handling when token is missing from checkpoint."""
        # Token 42 present in all, token 100 missing from step 200
        mock_process.side_effect = [
            {42: [2.0], 100: [3.0]},  # Step 100
            {42: [2.5]},  # Step 200 - token 100 missing
            {42: [3.0], 100: [3.5]},  # Step 300
        ]

        dataset = ["test"]
        token_ids = [42, 100]

        result = compute_token_entropy_trajectory(
            mock_checkpoints, mock_tokenizer, dataset, token_ids
        )

        # Token 42 should have all 3 steps
        assert len(result[42]["steps"]) == 3

        # Token 100 should only have 2 steps (missing step 200)
        assert len(result[100]["steps"]) == 2
        assert result[100]["steps"] == [100, 300]

    @patch("viz.visualize_token_entropy.process_checkpoint_for_token_entropies")
    def test_compute_trajectory_error_handling(
        self, mock_process, mock_checkpoints, mock_tokenizer
    ):
        """Test that errors in processing are handled gracefully."""
        # Make middle checkpoint fail
        mock_process.side_effect = [
            {42: [2.0]},
            RuntimeError("Failed to load"),
            {42: [3.0]},
        ]

        dataset = ["test"]
        token_ids = [42]

        result = compute_token_entropy_trajectory(
            mock_checkpoints, mock_tokenizer, dataset, token_ids
        )

        # Should have 2 successful checkpoints
        assert len(result[42]["steps"]) == 2
        assert 200 not in result[42]["steps"]

    def test_compute_trajectory_invalid_checkpoint_names(self, tmp_path, mock_tokenizer):
        """Test handling of invalid checkpoint names."""
        checkpoints = [
            tmp_path / "checkpoint-100",
            tmp_path / "checkpoint-abc",  # Invalid
            tmp_path / "checkpoint-200",
        ]

        for cp in checkpoints:
            cp.mkdir()

        with patch(
            "viz.visualize_token_entropy.process_checkpoint_for_token_entropies"
        ) as mock_process:
            mock_process.return_value = {42: [2.0]}

            result = compute_token_entropy_trajectory(checkpoints, mock_tokenizer, ["test"], [42])

            # Should only process valid checkpoints
            assert len(result[42]["steps"]) == 2
            assert result[42]["steps"] == [100, 200]


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
            42: {"steps": [100, 200, 300], "entropies": [2.0, 2.5, 3.0]},
            100: {"steps": [100, 200, 300], "entropies": [3.0, 3.2, 3.5]},
        }

        reasoning_data = {
            42: {"steps": [100, 200, 300], "entropies": [2.1, 2.4, 2.9]},
            100: {"steps": [100, 200, 300], "entropies": [3.1, 3.3, 3.6]},
        }

        output_path = tmp_path / "test_plot.png"

        plot_token_entropy_comparison(baseline_data, reasoning_data, mock_tokenizer, output_path)

        assert output_path.exists()

    def test_plot_empty_baseline(self, tmp_path, mock_tokenizer):
        """Test plot with no baseline data."""
        baseline_data = {}

        reasoning_data = {
            42: {"steps": [100, 200], "entropies": [2.0, 2.5]},
        }

        output_path = tmp_path / "test_plot_no_baseline.png"

        plot_token_entropy_comparison(baseline_data, reasoning_data, mock_tokenizer, output_path)

        assert output_path.exists()

    def test_plot_creates_output_directory(self, tmp_path, mock_tokenizer):
        """Test that plot creates output directory if needed."""
        output_path = tmp_path / "nested" / "dir" / "plot.png"

        data = {42: {"steps": [100], "entropies": [2.0]}}

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
            {42: {"steps": [100, 200], "entropies": [2.0, 2.5]}},  # Baseline
            {42: {"steps": [100, 200], "entropies": [2.1, 2.4]}},  # Reasoning
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
        mock_compute.return_value = {42: {"steps": [100], "entropies": [2.0]}}

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
        logits = th.tensor([1.0])
        entropy = compute_output_entropy(logits)

        # Single item should have zero entropy
        assert np.isclose(entropy, 0.0, atol=1e-6)

    def test_compute_output_entropy_extreme_logits(self):
        """Test entropy with extreme logit values."""
        # Very large logits
        logits = th.tensor([1000.0, -1000.0, 500.0, -500.0])
        entropy = compute_output_entropy(logits)
        assert entropy >= 0.0
        assert not np.isnan(entropy)
        assert not np.isinf(entropy)

    def test_compute_output_entropy_very_large_vocab(self):
        """Test entropy with very large vocabulary."""
        logits = th.randn(50000)  # Large vocab like GPT
        entropy = compute_output_entropy(logits)
        assert entropy >= 0.0
        assert not np.isnan(entropy)
