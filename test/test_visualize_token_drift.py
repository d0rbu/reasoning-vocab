"""
Tests for token drift visualization.

This module tests:
- Loading embeddings from checkpoints
- Computing PCA trajectories
- Creating visualizations
- Error handling for missing checkpoints
"""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch as th
from matplotlib.figure import Figure

from viz.visualize_token_drift import (
    EmbeddingType,
    collect_embedding_trajectories,
    compute_pca_trajectories,
    load_checkpoint_embeddings,
    plot_2d_drift,
    plot_3d_drift,
    visualize_token_drift,
)


def create_reasoning_token_map(
    checkpoint_path: Path, standard_token_ids: list[int], vocab_size: int = 100
) -> None:
    """Helper to create reasoning_token_map.json and config.json in a checkpoint directory."""
    # Compute multiplicities
    multiplicities = []
    token_counts: dict[int, int] = {}

    for token_id in standard_token_ids:
        count = token_counts.get(token_id, 0)
        multiplicities.append(count + 1)
        token_counts[token_id] = count + 1

    data = {"standard_token_ids": standard_token_ids, "multiplicities": multiplicities}

    with open(checkpoint_path / "reasoning_token_map.json", "w") as f:
        json.dump(data, f)

    # Also create minimal config.json with vocab_size
    config_data = {"vocab_size": vocab_size}
    with open(checkpoint_path / "config.json", "w") as f:
        json.dump(config_data, f)


@pytest.fixture
def mock_model_with_embeddings():
    """Create a mock model with embedding layers."""
    model = Mock()

    # Create embedding layers with realistic shapes
    hidden_size = 128
    vocab_size = 100

    # Standard embeddings
    input_embed = Mock()
    input_embed.weight = Mock()
    input_embed.weight.data = th.randn(vocab_size, hidden_size)

    output_embed = Mock()
    output_embed.weight = Mock()
    output_embed.weight.data = th.randn(vocab_size, hidden_size)

    model.get_input_embeddings = Mock(return_value=input_embed)
    model.get_output_embeddings = Mock(return_value=output_embed)

    return model


@pytest.fixture
def mock_model_with_reasoning_vocab():
    """Create a mock model with reasoning vocabulary layers."""
    model = Mock()

    # Create embedding layers
    hidden_size = 128
    vocab_size = 100
    reasoning_vocab_size = 50

    # Standard embeddings
    input_embed = Mock()
    input_embed.weight = Mock()
    input_embed.weight.data = th.randn(vocab_size, hidden_size)

    output_embed = Mock()
    output_embed.weight = Mock()
    output_embed.weight.data = th.randn(vocab_size, hidden_size)

    model.get_input_embeddings = Mock(return_value=input_embed)
    model.get_output_embeddings = Mock(return_value=output_embed)

    # Reasoning embeddings
    model.reasoning_embed = Mock()
    model.reasoning_embed.weight = Mock()
    model.reasoning_embed.weight.data = th.randn(reasoning_vocab_size, hidden_size)

    model.reasoning_unembed = Mock()
    model.reasoning_unembed.weight = Mock()
    model.reasoning_unembed.weight.data = th.randn(reasoning_vocab_size, hidden_size)

    return model


@pytest.fixture
def sample_trajectories():
    """Create sample trajectory data for testing."""
    # Shape: (num_checkpoints=5, num_tokens=3, hidden_size=128)
    return np.random.randn(5, 3, 128)


@pytest.fixture
def sample_pca_trajectories_2d():
    """Create sample 2D PCA trajectories for testing."""
    # Shape: (num_checkpoints=5, num_tokens=3, n_components=2)
    return np.random.randn(5, 3, 2)


@pytest.fixture
def sample_pca_trajectories_3d():
    """Create sample 3D PCA trajectories for testing."""
    # Shape: (num_checkpoints=5, num_tokens=3, n_components=3)
    return np.random.randn(5, 3, 3)


class TestLoadCheckpointEmbeddings:
    """Tests for load_checkpoint_embeddings function."""

    def test_load_input_embeddings_standard(self, tmp_path, mock_model_with_embeddings):
        """Test loading standard input embeddings."""
        checkpoint_path = tmp_path / "checkpoint"
        checkpoint_path.mkdir()

        with patch(
            "viz.visualize_token_drift.AutoModelForCausalLM.from_pretrained",
            return_value=mock_model_with_embeddings,
        ):
            token_ids = [0, 1, 2]
            embeddings = load_checkpoint_embeddings(
                checkpoint_path, token_ids, embedding_type=EmbeddingType.INPUT
            )

            assert embeddings.shape == (3, 128)
            assert embeddings.dtype == th.float32

    def test_load_output_embeddings_standard(self, tmp_path, mock_model_with_embeddings):
        """Test loading standard output embeddings."""
        checkpoint_path = tmp_path / "checkpoint"
        checkpoint_path.mkdir()

        with patch(
            "viz.visualize_token_drift.AutoModelForCausalLM.from_pretrained",
            return_value=mock_model_with_embeddings,
        ):
            token_ids = [0, 1, 2]
            embeddings = load_checkpoint_embeddings(
                checkpoint_path, token_ids, embedding_type=EmbeddingType.OUTPUT
            )

            assert embeddings.shape == (3, 128)
            assert embeddings.dtype == th.float32

    def test_load_reasoning_embeddings(self, tmp_path, mock_model_with_reasoning_vocab):
        """Test loading reasoning vocabulary embeddings."""
        checkpoint_path = tmp_path / "checkpoint"
        checkpoint_path.mkdir()

        with patch(
            "viz.visualize_token_drift.AutoModelForCausalLM.from_pretrained",
            return_value=mock_model_with_reasoning_vocab,
        ):
            token_ids = [0, 1, 2]
            embeddings = load_checkpoint_embeddings(
                checkpoint_path, token_ids, embedding_type=EmbeddingType.INPUT
            )

            assert embeddings.shape == (3, 128)
            assert embeddings.dtype == th.float32

    def test_checkpoint_not_found(self, tmp_path):
        """Test error handling when checkpoint doesn't exist."""
        checkpoint_path = tmp_path / "nonexistent"

        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            load_checkpoint_embeddings(checkpoint_path, [0, 1], embedding_type=EmbeddingType.INPUT)


class TestCollectEmbeddingTrajectories:
    """Tests for collect_embedding_trajectories function."""

    def test_collect_single_checkpoint(self, tmp_path, mock_model_with_embeddings):
        """Test collecting trajectories from a single checkpoint."""
        checkpoint_path = tmp_path / "checkpoint"
        checkpoint_path.mkdir()

        with patch(
            "viz.visualize_token_drift.AutoModelForCausalLM.from_pretrained",
            return_value=mock_model_with_embeddings,
        ):
            token_ids = [0, 1, 2]
            trajectories = collect_embedding_trajectories(
                [checkpoint_path], token_ids, embedding_type=EmbeddingType.INPUT
            )

            assert trajectories.shape == (1, 3, 128)
            assert isinstance(trajectories, np.ndarray)

    def test_collect_multiple_checkpoints(self, tmp_path, mock_model_with_embeddings):
        """Test collecting trajectories from multiple checkpoints."""
        # Create multiple checkpoints
        checkpoints = []
        for i in range(3):
            ckpt = tmp_path / f"checkpoint_{i}"
            ckpt.mkdir()
            checkpoints.append(ckpt)

        with patch(
            "viz.visualize_token_drift.AutoModelForCausalLM.from_pretrained",
            return_value=mock_model_with_embeddings,
        ):
            token_ids = [0, 1, 2]
            trajectories = collect_embedding_trajectories(
                checkpoints, token_ids, embedding_type=EmbeddingType.INPUT
            )

            assert trajectories.shape == (3, 3, 128)
            assert isinstance(trajectories, np.ndarray)

    def test_collect_reasoning_trajectories(self, tmp_path, mock_model_with_reasoning_vocab):
        """Test collecting reasoning vocabulary trajectories."""
        checkpoint_path = tmp_path / "checkpoint"
        checkpoint_path.mkdir()

        with patch(
            "viz.visualize_token_drift.AutoModelForCausalLM.from_pretrained",
            return_value=mock_model_with_reasoning_vocab,
        ):
            token_ids = [0, 1, 2]
            trajectories = collect_embedding_trajectories(
                [checkpoint_path], token_ids, embedding_type=EmbeddingType.INPUT
            )

            assert trajectories.shape == (1, 3, 128)
            assert isinstance(trajectories, np.ndarray)


class TestComputePCATrajectories:
    """Tests for compute_pca_trajectories function."""

    def test_pca_2d(self, sample_trajectories):
        """Test PCA with 2 components."""
        pca_traj, pca_model = compute_pca_trajectories(sample_trajectories, n_components=2)

        assert pca_traj.shape == (5, 3, 2)
        assert isinstance(pca_traj, np.ndarray)
        assert pca_model.n_components == 2

    def test_pca_3d(self, sample_trajectories):
        """Test PCA with 3 components."""
        pca_traj, pca_model = compute_pca_trajectories(sample_trajectories, n_components=3)

        assert pca_traj.shape == (5, 3, 3)
        assert isinstance(pca_traj, np.ndarray)
        assert pca_model.n_components == 3

    def test_pca_explained_variance(self, sample_trajectories):
        """Test that PCA explains non-zero variance."""
        pca_traj, pca_model = compute_pca_trajectories(sample_trajectories, n_components=2)

        explained_var = pca_model.explained_variance_ratio_.sum()
        assert 0 < explained_var <= 1.0

    def test_pca_deterministic(self, sample_trajectories):
        """Test that PCA produces deterministic results."""
        pca_traj1, _ = compute_pca_trajectories(sample_trajectories, n_components=2)
        pca_traj2, _ = compute_pca_trajectories(sample_trajectories, n_components=2)

        # Results should be identical (up to sign flip)
        # Check that absolute values are close
        np.testing.assert_allclose(np.abs(pca_traj1), np.abs(pca_traj2), rtol=1e-5)


class TestPlot2DDrift:
    """Tests for plot_2d_drift function."""

    def test_basic_2d_plot(self, sample_pca_trajectories_2d, tmp_path):
        """Test basic 2D plotting functionality."""
        trajectories_dict = {
            "baseline": sample_pca_trajectories_2d[:1],
            "standard": sample_pca_trajectories_2d,
            "reasoning": sample_pca_trajectories_2d,
        }

        fig = plot_2d_drift(trajectories_dict)

        assert isinstance(fig, Figure)
        assert len(fig.axes) == 1

    def test_2d_plot_with_labels(self, sample_pca_trajectories_2d):
        """Test 2D plotting with token labels."""
        trajectories_dict = {
            "baseline": sample_pca_trajectories_2d[:1],
            "standard": sample_pca_trajectories_2d,
        }
        token_labels = ["token_a", "token_b", "token_c"]

        fig = plot_2d_drift(trajectories_dict, token_labels=token_labels)

        assert isinstance(fig, Figure)

    def test_2d_plot_save(self, sample_pca_trajectories_2d, tmp_path):
        """Test saving 2D plot to file."""
        trajectories_dict = {
            "baseline": sample_pca_trajectories_2d[:1],
            "standard": sample_pca_trajectories_2d,
        }
        save_path = tmp_path / "test_plot.png"

        fig = plot_2d_drift(trajectories_dict, save_path=save_path)

        assert isinstance(fig, Figure)
        assert save_path.exists()

    def test_2d_plot_custom_title(self, sample_pca_trajectories_2d):
        """Test 2D plotting with custom title."""
        trajectories_dict = {
            "baseline": sample_pca_trajectories_2d[:1],
        }
        custom_title = "My Custom Title"

        fig = plot_2d_drift(trajectories_dict, title=custom_title)

        assert isinstance(fig, Figure)
        assert fig.axes[0].get_title() == custom_title


class TestPlot3DDrift:
    """Tests for plot_3d_drift function."""

    def test_basic_3d_plot(self, sample_pca_trajectories_3d):
        """Test basic 3D plotting functionality."""
        trajectories_dict = {
            "baseline": sample_pca_trajectories_3d[:1],
            "standard": sample_pca_trajectories_3d,
            "reasoning": sample_pca_trajectories_3d,
        }

        fig = plot_3d_drift(trajectories_dict)

        assert isinstance(fig, Figure)
        assert len(fig.axes) == 1
        # Check that it's a 3D axis
        assert hasattr(fig.axes[0], "zaxis")

    def test_3d_plot_with_labels(self, sample_pca_trajectories_3d):
        """Test 3D plotting with token labels."""
        trajectories_dict = {
            "baseline": sample_pca_trajectories_3d[:1],
            "standard": sample_pca_trajectories_3d,
        }
        token_labels = ["token_a", "token_b", "token_c"]

        fig = plot_3d_drift(trajectories_dict, token_labels=token_labels)

        assert isinstance(fig, Figure)

    def test_3d_plot_save(self, sample_pca_trajectories_3d, tmp_path):
        """Test saving 3D plot to file."""
        trajectories_dict = {
            "baseline": sample_pca_trajectories_3d[:1],
            "standard": sample_pca_trajectories_3d,
        }
        save_path = tmp_path / "test_plot_3d.png"

        fig = plot_3d_drift(trajectories_dict, save_path=save_path)

        assert isinstance(fig, Figure)
        assert save_path.exists()


class TestVisualizeTokenDrift:
    """Tests for the main visualize_token_drift function."""

    def test_visualize_2d(self, tmp_path, mock_model_with_reasoning_vocab):
        """Test end-to-end 2D visualization."""
        # Create checkpoint directories
        baseline = tmp_path / "baseline"
        baseline.mkdir()
        create_reasoning_token_map(baseline, [])

        checkpoints = []
        for i in range(3):
            ckpt = tmp_path / f"checkpoint_{i}"
            ckpt.mkdir()
            create_reasoning_token_map(ckpt, list(range(50)))  # 50 reasoning tokens
            checkpoints.append(ckpt)

        output_dir = tmp_path / "figures"

        with patch(
            "viz.visualize_token_drift.AutoModelForCausalLM.from_pretrained",
            return_value=mock_model_with_reasoning_vocab,
        ):
            figures = visualize_token_drift(
                baseline_checkpoint=baseline,
                reasoning_checkpoints=checkpoints,
                token_ids=[0, 1, 2],
                token_labels=["a", "b", "c"],
                embedding_type=EmbeddingType.INPUT,
                n_components=2,
                output_dir=output_dir,
                experiment_name="test",
            )

            assert "2d" in figures
            assert isinstance(figures["2d"], Figure)

            # Check that output file was created
            expected_file = output_dir / "test_input_2d.png"
            assert expected_file.exists()

    def test_visualize_3d(self, tmp_path, mock_model_with_reasoning_vocab):
        """Test end-to-end 3D visualization."""
        baseline = tmp_path / "baseline"
        baseline.mkdir()
        create_reasoning_token_map(baseline, [])

        checkpoints = []
        for i in range(3):
            ckpt = tmp_path / f"checkpoint_{i}"
            ckpt.mkdir()
            create_reasoning_token_map(ckpt, list(range(50)))  # 50 reasoning tokens
            checkpoints.append(ckpt)

        output_dir = tmp_path / "figures"

        with patch(
            "viz.visualize_token_drift.AutoModelForCausalLM.from_pretrained",
            return_value=mock_model_with_reasoning_vocab,
        ):
            figures = visualize_token_drift(
                baseline_checkpoint=baseline,
                reasoning_checkpoints=checkpoints,
                token_ids=[0, 1, 2],
                embedding_type=EmbeddingType.OUTPUT,
                n_components=3,
                output_dir=output_dir,
                experiment_name="test_3d",
            )

            assert "3d" in figures
            assert isinstance(figures["3d"], Figure)

            # Check that output file was created
            expected_file = output_dir / "test_3d_output_3d.png"
            assert expected_file.exists()

    def test_visualize_invalid_components(self, tmp_path, mock_model_with_reasoning_vocab):
        """Test error handling for invalid number of components."""
        baseline = tmp_path / "baseline"
        baseline.mkdir()
        create_reasoning_token_map(baseline, [])

        checkpoints = [tmp_path / "checkpoint"]
        checkpoints[0].mkdir()
        create_reasoning_token_map(checkpoints[0], list(range(50)))

        with patch(
            "viz.visualize_token_drift.AutoModelForCausalLM.from_pretrained",
            return_value=mock_model_with_reasoning_vocab,
        ):
            with pytest.raises(ValueError, match="n_components must be 2 or 3"):
                visualize_token_drift(
                    baseline_checkpoint=baseline,
                    reasoning_checkpoints=checkpoints,
                    token_ids=[0, 1],
                    n_components=4,  # Invalid
                )

    def test_visualize_output_embeddings(self, tmp_path, mock_model_with_reasoning_vocab):
        """Test visualization of output embeddings."""
        baseline = tmp_path / "baseline"
        baseline.mkdir()
        create_reasoning_token_map(baseline, [])

        checkpoints = [tmp_path / "checkpoint"]
        checkpoints[0].mkdir()
        create_reasoning_token_map(checkpoints[0], list(range(50)))

        output_dir = tmp_path / "figures"

        with patch(
            "viz.visualize_token_drift.AutoModelForCausalLM.from_pretrained",
            return_value=mock_model_with_reasoning_vocab,
        ):
            figures = visualize_token_drift(
                baseline_checkpoint=baseline,
                reasoning_checkpoints=checkpoints,
                token_ids=[0, 1],
                embedding_type=EmbeddingType.OUTPUT,
                n_components=2,
                output_dir=output_dir,
            )

            assert "2d" in figures
            expected_file = output_dir / "token_drift_output_2d.png"
            assert expected_file.exists()


class TestIntegration:
    """Integration tests with more realistic scenarios."""

    def test_multiple_tokens_multiple_checkpoints(self, tmp_path, mock_model_with_reasoning_vocab):
        """Test with multiple tokens tracked across multiple checkpoints."""
        baseline = tmp_path / "baseline"
        baseline.mkdir()
        create_reasoning_token_map(baseline, [])

        num_checkpoints = 5
        checkpoints = []
        for i in range(num_checkpoints):
            ckpt = tmp_path / f"checkpoint_{i}"
            ckpt.mkdir()
            create_reasoning_token_map(ckpt, list(range(50)))
            checkpoints.append(ckpt)

        output_dir = tmp_path / "figures"
        num_tokens = 10
        token_ids = list(range(num_tokens))
        token_labels = [f"token_{i}" for i in range(num_tokens)]

        with patch(
            "viz.visualize_token_drift.AutoModelForCausalLM.from_pretrained",
            return_value=mock_model_with_reasoning_vocab,
        ):
            figures = visualize_token_drift(
                baseline_checkpoint=baseline,
                reasoning_checkpoints=checkpoints,
                token_ids=token_ids,
                token_labels=token_labels,
                embedding_type=EmbeddingType.INPUT,
                n_components=2,
                output_dir=output_dir,
                experiment_name="multi_token",
            )

            assert "2d" in figures
            assert isinstance(figures["2d"], Figure)

    def test_consistent_trajectory_shapes(self, tmp_path):
        """Test that trajectory shapes are consistent throughout pipeline."""
        num_checkpoints = 4
        num_tokens = 5
        hidden_size = 128

        # Create synthetic trajectories
        trajectories = np.random.randn(num_checkpoints, num_tokens, hidden_size)

        # Apply PCA
        pca_traj, _ = compute_pca_trajectories(trajectories, n_components=2)

        # Check shapes
        assert pca_traj.shape == (num_checkpoints, num_tokens, 2)

        # Create plot
        trajectories_dict = {"test": pca_traj}
        fig = plot_2d_drift(trajectories_dict)

        assert isinstance(fig, Figure)
