"""
Tests for token entropy visualization.

This module tests:
- Entropy computation from embeddings
- Checkpoint loading and embedding extraction
- Trajectory computation across checkpoints
- Plot generation and file output
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch as th
from transformers import AutoModelForCausalLM

from viz.visualize_token_entropy import (
    compute_embedding_entropy,
    compute_entropy_trajectory,
    load_checkpoint_embeddings,
    plot_entropy_comparison,
    visualize_token_entropy,
)


class TestEntropyComputation:
    """Tests for entropy computation functions."""

    def test_compute_embedding_entropy_uniform(self):
        """Test entropy computation for uniform distribution."""
        # Uniform distribution should have maximum entropy
        embedding = th.zeros(100)  # Softmax will make this uniform
        entropy = compute_embedding_entropy(embedding)

        # For uniform distribution over n items: H = log(n)
        expected_entropy = np.log(100)
        assert np.isclose(entropy, expected_entropy, rtol=0.01)

    def test_compute_embedding_entropy_peaked(self):
        """Test entropy computation for peaked distribution."""
        # Create a strongly peaked distribution
        embedding = th.zeros(100)
        embedding[0] = 10.0  # One very high value

        entropy = compute_embedding_entropy(embedding)

        # Peaked distribution should have low entropy
        assert entropy < 1.0

    def test_compute_embedding_entropy_deterministic(self):
        """Test that entropy computation is deterministic."""
        embedding = th.randn(50)

        entropy1 = compute_embedding_entropy(embedding)
        entropy2 = compute_embedding_entropy(embedding)

        assert entropy1 == entropy2

    def test_compute_embedding_entropy_positive(self):
        """Test that entropy is always non-negative."""
        for _ in range(10):
            embedding = th.randn(100)
            entropy = compute_embedding_entropy(embedding)
            assert entropy >= 0.0

    def test_compute_embedding_entropy_different_sizes(self):
        """Test entropy computation with different embedding sizes."""
        for size in [10, 50, 100, 500]:
            embedding = th.randn(size)
            entropy = compute_embedding_entropy(embedding)
            assert entropy >= 0.0
            # Entropy should be bounded by log(size)
            assert entropy <= np.log(size) + 1.0  # Small tolerance


class TestCheckpointLoading:
    """Tests for checkpoint loading and embedding extraction."""

    @pytest.fixture
    def mock_model_standard(self):
        """Create a mock model with standard embeddings."""
        model = MagicMock(spec=AutoModelForCausalLM)

        # Create mock embedding layers
        embed_tokens = MagicMock()
        embed_tokens.weight = th.randn(1000, 128)

        lm_head = MagicMock()
        lm_head.weight = th.randn(1000, 128)

        # Set up model structure (GPT-2 style)
        model.transformer = MagicMock()
        model.transformer.wte = embed_tokens
        model.lm_head = lm_head

        return model

    @pytest.fixture
    def mock_model_reasoning(self):
        """Create a mock model with reasoning vocab."""
        model = MagicMock(spec=AutoModelForCausalLM)

        # Standard embeddings
        embed_tokens = MagicMock()
        embed_tokens.weight = th.randn(1000, 128)

        lm_head = MagicMock()
        lm_head.weight = th.randn(1000, 128)

        # Reasoning embeddings
        reasoning_embed = MagicMock()
        reasoning_embed.weight = th.randn(500, 128)

        reasoning_unembed = MagicMock()
        reasoning_unembed.weight = th.randn(128, 500)

        # Set up model structure
        model.model = MagicMock()
        model.model.embed_tokens = embed_tokens
        model.lm_head = lm_head
        model.reasoning_embed = reasoning_embed
        model.reasoning_unembed = reasoning_unembed

        return model

    @patch("viz.visualize_token_entropy.AutoModelForCausalLM.from_pretrained")
    def test_load_checkpoint_embeddings_standard(
        self, mock_from_pretrained, mock_model_standard, tmp_path
    ):
        """Test loading standard token embeddings from checkpoint."""
        mock_from_pretrained.return_value = mock_model_standard

        checkpoint_path = tmp_path / "checkpoint-100"
        checkpoint_path.mkdir()

        token_id = 42

        embedding_vec, unembedding_vec = load_checkpoint_embeddings(
            checkpoint_path, token_id, is_reasoning_token=False
        )

        # Verify shapes
        assert embedding_vec.shape == (128,)
        assert unembedding_vec.shape == (128,)

        # Verify we got the right token
        expected_embedding = mock_model_standard.transformer.wte.weight[token_id]
        expected_unembedding = mock_model_standard.lm_head.weight[token_id]

        assert th.allclose(embedding_vec, expected_embedding)
        assert th.allclose(unembedding_vec, expected_unembedding)

    @patch("viz.visualize_token_entropy.AutoModelForCausalLM.from_pretrained")
    def test_load_checkpoint_embeddings_reasoning(
        self, mock_from_pretrained, mock_model_reasoning, tmp_path
    ):
        """Test loading reasoning token embeddings from checkpoint."""
        mock_from_pretrained.return_value = mock_model_reasoning

        checkpoint_path = tmp_path / "checkpoint-200"
        checkpoint_path.mkdir()

        token_id = 10

        embedding_vec, unembedding_vec = load_checkpoint_embeddings(
            checkpoint_path, token_id, is_reasoning_token=True
        )

        # Verify shapes
        assert embedding_vec.shape == (128,)
        assert unembedding_vec.shape == (128,)

        # Verify we got the right reasoning token
        expected_embedding = mock_model_reasoning.reasoning_embed.weight[token_id]
        expected_unembedding = mock_model_reasoning.reasoning_unembed.weight[:, token_id]

        assert th.allclose(embedding_vec, expected_embedding)
        assert th.allclose(unembedding_vec, expected_unembedding)

    @patch("viz.visualize_token_entropy.AutoModelForCausalLM.from_pretrained")
    def test_load_checkpoint_embeddings_no_reasoning_vocab(
        self, mock_from_pretrained, mock_model_standard, tmp_path
    ):
        """Test error when trying to load reasoning token from non-reasoning model."""
        mock_from_pretrained.return_value = mock_model_standard

        checkpoint_path = tmp_path / "checkpoint-100"
        checkpoint_path.mkdir()

        with pytest.raises(ValueError, match="does not have reasoning_embed"):
            load_checkpoint_embeddings(checkpoint_path, 10, is_reasoning_token=True)


class TestTrajectoryComputation:
    """Tests for entropy trajectory computation."""

    @pytest.fixture
    def mock_checkpoints(self, tmp_path):
        """Create mock checkpoint directories."""
        checkpoints = []
        for step in [100, 200, 300, 400, 500]:
            checkpoint_dir = tmp_path / f"checkpoint-{step}"
            checkpoint_dir.mkdir()
            checkpoints.append(checkpoint_dir)
        return checkpoints

    @patch("viz.visualize_token_entropy.load_checkpoint_embeddings")
    @patch("viz.visualize_token_entropy.compute_embedding_entropy")
    def test_compute_entropy_trajectory_basic(self, mock_entropy, mock_load, mock_checkpoints):
        """Test basic entropy trajectory computation."""
        # Mock embedding loading to return simple vectors
        mock_load.return_value = (th.randn(128), th.randn(128))

        # Mock entropy computation to return sequential values
        mock_entropy.side_effect = [1.0, 1.5, 2.0, 2.5, 3.0]

        steps, entropies = compute_entropy_trajectory(
            mock_checkpoints, token_id=42, is_reasoning_token=False, use_unembedding=False
        )

        # Verify we got all checkpoints
        assert steps == [100, 200, 300, 400, 500]
        assert entropies == [1.0, 1.5, 2.0, 2.5, 3.0]

        # Verify load_checkpoint_embeddings was called correctly
        assert mock_load.call_count == 5

    @patch("viz.visualize_token_entropy.load_checkpoint_embeddings")
    def test_compute_entropy_trajectory_use_unembedding(self, mock_load, mock_checkpoints):
        """Test that unembedding is used when specified."""
        embedding_vec = th.zeros(128)
        unembedding_vec = th.ones(128)
        mock_load.return_value = (embedding_vec, unembedding_vec)

        # Use unembedding
        steps, entropies = compute_entropy_trajectory(
            mock_checkpoints, token_id=42, is_reasoning_token=False, use_unembedding=True
        )

        # Entropy from unembedding (all ones) should be different from embedding (all zeros)
        # Both will be log(128) after softmax makes them uniform, but good to check computation happens
        assert len(entropies) == 5
        assert all(e > 0 for e in entropies)

    @patch("viz.visualize_token_entropy.load_checkpoint_embeddings")
    def test_compute_entropy_trajectory_error_handling(self, mock_load, mock_checkpoints):
        """Test that errors in loading checkpoints are handled gracefully."""
        # Make one checkpoint fail
        mock_load.side_effect = [
            (th.randn(128), th.randn(128)),
            (th.randn(128), th.randn(128)),
            RuntimeError("Failed to load"),
            (th.randn(128), th.randn(128)),
            (th.randn(128), th.randn(128)),
        ]

        steps, entropies = compute_entropy_trajectory(
            mock_checkpoints, token_id=42, is_reasoning_token=False, use_unembedding=False
        )

        # Should have 4 successful checkpoints
        assert len(steps) == 4
        assert len(entropies) == 4
        # Should skip step 300
        assert 300 not in steps

    def test_compute_entropy_trajectory_invalid_checkpoint_names(self, tmp_path):
        """Test handling of checkpoints with invalid naming."""
        # Create checkpoints with various naming issues
        invalid_checkpoints = [
            tmp_path / "checkpoint-100",  # Valid
            tmp_path / "checkpoint-abc",  # Invalid: not a number
            tmp_path / "checkpoint-200",  # Valid
            tmp_path / "not-a-checkpoint",  # Invalid: wrong format
        ]

        for checkpoint in invalid_checkpoints:
            checkpoint.mkdir()

        with patch("viz.visualize_token_entropy.load_checkpoint_embeddings") as mock_load:
            mock_load.return_value = (th.randn(128), th.randn(128))

            steps, entropies = compute_entropy_trajectory(
                invalid_checkpoints, token_id=42, is_reasoning_token=False, use_unembedding=False
            )

            # Should only process valid checkpoints
            assert steps == [100, 200]


class TestPlotting:
    """Tests for plotting functions."""

    def test_plot_entropy_comparison_basic(self, tmp_path):
        """Test basic plot generation."""
        baseline_data = ([100, 200, 300], [2.0, 2.5, 3.0])
        standard_data = ([100, 200, 300], [2.1, 2.4, 2.8])
        reasoning_data = ([100, 200, 300], [3.0, 3.5, 4.0])

        output_path = tmp_path / "test_plot.png"

        plot_entropy_comparison(
            baseline_data,
            standard_data,
            reasoning_data,
            token_str="test_token",
            output_path=output_path,
            use_unembedding=False,
        )

        # Verify plot was saved
        assert output_path.exists()

    def test_plot_entropy_comparison_empty_data(self, tmp_path):
        """Test plot generation with some empty datasets."""
        baseline_data = ([], [])  # No baseline data
        standard_data = ([100, 200], [2.0, 2.5])
        reasoning_data = ([100, 200], [3.0, 3.5])

        output_path = tmp_path / "test_plot_partial.png"

        # Should not raise an error
        plot_entropy_comparison(
            baseline_data,
            standard_data,
            reasoning_data,
            token_str="test_token",
            output_path=output_path,
            use_unembedding=False,
        )

        assert output_path.exists()

    def test_plot_entropy_comparison_unembedding(self, tmp_path):
        """Test plot generation for unembedding layer."""
        data = ([100, 200], [2.0, 2.5])

        output_path = tmp_path / "test_plot_unembed.png"

        plot_entropy_comparison(
            data,
            data,
            data,
            token_str="test_token",
            output_path=output_path,
            use_unembedding=True,
        )

        assert output_path.exists()

    def test_plot_creates_output_directory(self, tmp_path):
        """Test that plot function creates output directory if needed."""
        output_path = tmp_path / "nested" / "dir" / "plot.png"

        data = ([100, 200], [2.0, 2.5])
        plot_entropy_comparison(data, data, data, "token", output_path, False)

        assert output_path.exists()
        assert output_path.parent.exists()


class TestVisualizationPipeline:
    """Tests for the main visualization pipeline."""

    @pytest.fixture
    def mock_checkpoint_structure(self, tmp_path):
        """Create a complete mock checkpoint structure."""
        baseline_dir = tmp_path / "baseline"
        reasoning_dir = tmp_path / "reasoning"

        for step in [100, 200, 300]:
            (baseline_dir / f"checkpoint-{step}").mkdir(parents=True)
            (reasoning_dir / f"checkpoint-{step}").mkdir(parents=True)

        return baseline_dir, reasoning_dir

    @patch("viz.visualize_token_entropy.compute_entropy_trajectory")
    @patch("viz.visualize_token_entropy.plot_entropy_comparison")
    def test_visualize_token_entropy_full_pipeline(
        self, mock_plot, mock_trajectory, mock_checkpoint_structure, tmp_path
    ):
        """Test full visualization pipeline."""
        baseline_dir, reasoning_dir = mock_checkpoint_structure
        output_dir = tmp_path / "output"

        # Mock trajectory computation
        mock_trajectory.side_effect = [
            ([100, 200, 300], [2.0, 2.5, 3.0]),  # Baseline embedding
            ([100, 200, 300], [2.1, 2.4, 2.8]),  # Standard embedding
            ([100, 200, 300], [3.0, 3.5, 4.0]),  # Reasoning embedding
            ([100, 200, 300], [1.8, 2.3, 2.9]),  # Baseline unembedding
            ([100, 200, 300], [1.9, 2.2, 2.7]),  # Standard unembedding
            ([100, 200, 300], [2.9, 3.4, 3.9]),  # Reasoning unembedding
        ]

        results = visualize_token_entropy(
            baseline_dir=baseline_dir,
            reasoning_dir=reasoning_dir,
            token_id=42,
            reasoning_token_id=10,
            token_str="test_token",
            output_dir=output_dir,
            plot_both_layers=True,
        )

        # Verify results structure
        assert "embedding" in results
        assert "unembedding" in results

        assert "baseline" in results["embedding"]
        assert "standard" in results["embedding"]
        assert "reasoning" in results["embedding"]

        # Verify trajectory calls
        assert mock_trajectory.call_count == 6  # 3 for embedding, 3 for unembedding

        # Verify plot calls
        assert mock_plot.call_count == 2  # One for embedding, one for unembedding

    @patch("viz.visualize_token_entropy.compute_entropy_trajectory")
    @patch("viz.visualize_token_entropy.plot_entropy_comparison")
    def test_visualize_token_entropy_no_baseline(
        self, mock_plot, mock_trajectory, mock_checkpoint_structure, tmp_path
    ):
        """Test visualization without baseline directory."""
        _, reasoning_dir = mock_checkpoint_structure
        output_dir = tmp_path / "output"

        mock_trajectory.return_value = ([100, 200], [2.0, 2.5])

        results = visualize_token_entropy(
            baseline_dir=None,  # No baseline
            reasoning_dir=reasoning_dir,
            token_id=42,
            output_dir=output_dir,
            plot_both_layers=False,
        )

        # Should still work without baseline
        assert "embedding" in results
        assert results["embedding"]["baseline"]["steps"] == []
        assert results["embedding"]["baseline"]["entropies"] == []

    @patch("viz.visualize_token_entropy.compute_entropy_trajectory")
    @patch("viz.visualize_token_entropy.plot_entropy_comparison")
    def test_visualize_token_entropy_default_params(
        self, mock_plot, mock_trajectory, mock_checkpoint_structure, tmp_path
    ):
        """Test visualization with default parameters."""
        _, reasoning_dir = mock_checkpoint_structure

        mock_trajectory.return_value = ([100], [2.0])

        # Use defaults for reasoning_token_id, token_str, output_dir
        with patch("viz.visualize_token_entropy.Path") as mock_path:
            mock_path.return_value = tmp_path / "fig"

            results = visualize_token_entropy(
                baseline_dir=None,
                reasoning_dir=reasoning_dir,
                token_id=42,
            )

            # Should use token_id for reasoning_token_id
            # Should generate token_str as "Token_42"
            # Should use "fig" as output_dir
            assert results is not None

    @patch("viz.visualize_token_entropy.compute_entropy_trajectory")
    @patch("viz.visualize_token_entropy.plot_entropy_comparison")
    def test_visualize_token_entropy_embedding_only(
        self, mock_plot, mock_trajectory, mock_checkpoint_structure, tmp_path
    ):
        """Test visualization with only embedding layer."""
        _, reasoning_dir = mock_checkpoint_structure
        output_dir = tmp_path / "output"

        mock_trajectory.return_value = ([100, 200], [2.0, 2.5])

        results = visualize_token_entropy(
            baseline_dir=None,
            reasoning_dir=reasoning_dir,
            token_id=42,
            output_dir=output_dir,
            plot_both_layers=False,
        )

        # Should only have embedding results
        assert "embedding" in results
        assert "unembedding" not in results

        # Should call trajectory 2 times (standard + reasoning for embedding only)
        assert mock_trajectory.call_count == 2

        # Should call plot once
        assert mock_plot.call_count == 1


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_compute_embedding_entropy_single_element(self):
        """Test entropy with single-element embedding."""
        embedding = th.tensor([1.0])
        entropy = compute_embedding_entropy(embedding)

        # Single element should have zero entropy
        assert np.isclose(entropy, 0.0, atol=1e-6)

    def test_compute_embedding_entropy_very_large(self):
        """Test entropy with very large embedding."""
        embedding = th.randn(10000)
        entropy = compute_embedding_entropy(embedding)

        # Should handle large embeddings
        assert entropy >= 0.0
        assert not np.isnan(entropy)
        assert not np.isinf(entropy)

    def test_compute_embedding_entropy_extreme_values(self):
        """Test entropy with extreme embedding values."""
        # Very large values
        embedding = th.tensor([1000.0, -1000.0, 500.0, -500.0])
        entropy = compute_embedding_entropy(embedding)
        assert entropy >= 0.0
        assert not np.isnan(entropy)

        # Very small values
        embedding = th.tensor([1e-10, 1e-10, 1e-10, 1e-10])
        entropy = compute_embedding_entropy(embedding)
        assert entropy >= 0.0
        assert not np.isnan(entropy)

    @patch("viz.visualize_token_entropy.compute_entropy_trajectory")
    def test_visualize_token_entropy_no_checkpoints(self, mock_trajectory, tmp_path):
        """Test behavior when no checkpoints are found."""
        reasoning_dir = tmp_path / "reasoning"
        reasoning_dir.mkdir()
        output_dir = tmp_path / "output"

        # No checkpoints found
        mock_trajectory.return_value = ([], [])

        results = visualize_token_entropy(
            baseline_dir=None,
            reasoning_dir=reasoning_dir,
            token_id=42,
            output_dir=output_dir,
            plot_both_layers=False,
        )

        # Should handle empty results gracefully
        assert results["embedding"]["standard"]["steps"] == []
        assert results["embedding"]["standard"]["entropies"] == []
