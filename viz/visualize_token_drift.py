"""
Visualize token embedding drift during training using PCA.

This script tracks how token embeddings evolve over training, comparing:
- Baseline model (pretrained, no training)
- Standard vocabulary tokens during reasoning vocab training
- Reasoning vocabulary tokens during reasoning vocab training

The visualization uses PCA to project high-dimensional embeddings into 2D/3D space,
showing trajectories of how embeddings drift over training checkpoints.
"""

import argparse
from enum import Enum
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import torch as th
from loguru import logger
from matplotlib.figure import Figure
from sklearn.decomposition import PCA
from torch import nn
from transformers import AutoModelForCausalLM


class EmbeddingType(str, Enum):
    """Type of embeddings to load from model."""

    INPUT = "input"
    OUTPUT = "output"


def load_checkpoint_embeddings(
    checkpoint_path: Path,
    token_ids: list[int],
    embedding_type: EmbeddingType = EmbeddingType.INPUT,
) -> th.Tensor:
    """
    Load embeddings for specific tokens from a model checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint directory
        token_ids: List of token IDs to extract embeddings for
        embedding_type: Whether to load input embeddings or output unembeddings

    Returns:
        Tensor of shape (num_tokens, hidden_size) containing embeddings

    Raises:
        FileNotFoundError: If checkpoint doesn't exist
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.debug(f"Loading {embedding_type.value} embeddings from {checkpoint_path}")

    # Load model checkpoint
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=th.float32,
        device_map="cpu",
    )

    # Get embeddings based on type
    if embedding_type == EmbeddingType.INPUT:
        embed_layer = cast(nn.Embedding, model.get_input_embeddings())
    else:
        embed_layer = cast(nn.Linear, model.get_output_embeddings())

    embeddings = cast(th.Tensor, embed_layer.weight.data[token_ids])

    logger.debug(f"Loaded embeddings with shape: {embeddings.shape}")
    return embeddings


def collect_embedding_trajectories(
    checkpoint_dirs: list[Path],
    token_ids: list[int],
    embedding_type: EmbeddingType = EmbeddingType.INPUT,
) -> np.ndarray:
    """
    Collect embedding trajectories across multiple checkpoints.

    Args:
        checkpoint_dirs: List of checkpoint directories in chronological order
        token_ids: List of token IDs to track
        embedding_type: Whether to load input embeddings or output unembeddings

    Returns:
        Array of shape (num_checkpoints, num_tokens, hidden_size)
    """
    return np.array(
        [
            load_checkpoint_embeddings(ckpt_dir, token_ids, embedding_type).numpy()
            for ckpt_dir in checkpoint_dirs
        ]
    )


def compute_pca_trajectories(
    trajectories: np.ndarray,
    n_components: int = 2,
) -> tuple[np.ndarray, PCA]:
    """
    Apply PCA to embedding trajectories.

    Args:
        trajectories: Array of shape (num_checkpoints, num_tokens, hidden_size)
        n_components: Number of PCA components (2 or 3)

    Returns:
        Tuple of (pca_trajectories, pca_model) where:
            - pca_trajectories has shape (num_checkpoints, num_tokens, n_components)
            - pca_model is the fitted PCA transformer
    """
    num_checkpoints, num_tokens, hidden_size = trajectories.shape

    # Reshape to (num_checkpoints * num_tokens, hidden_size) for PCA fitting
    flat_embeddings = trajectories.reshape(-1, hidden_size)

    # Fit PCA
    pca = PCA(n_components=n_components)
    pca_embeddings = pca.fit_transform(flat_embeddings)

    # Reshape back to (num_checkpoints, num_tokens, n_components)
    pca_trajectories = pca_embeddings.reshape(num_checkpoints, num_tokens, n_components)

    explained_var = pca.explained_variance_ratio_.sum()
    logger.debug(f"PCA with {n_components} components explains {explained_var:.2%} of variance")

    return pca_trajectories, pca


def plot_2d_drift(
    trajectories_dict: dict[str, np.ndarray],
    token_labels: list[str] | None = None,
    title: str = "Token Embedding Drift (PCA)",
    save_path: Path | None = None,
) -> Figure:
    """
    Plot 2D PCA trajectories for token embeddings.

    Args:
        trajectories_dict: Dict mapping trajectory names to arrays of shape
                          (num_checkpoints, num_tokens, 2)
        token_labels: Optional labels for each token
        title: Plot title
        save_path: Optional path to save figure

    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Color palette for different trajectory types
    colors = {
        "baseline": "#1f77b4",
        "standard": "#ff7f0e",
        "reasoning": "#2ca02c",
    }

    # Plot each trajectory type
    for traj_name, trajectories in trajectories_dict.items():
        num_checkpoints, num_tokens, _ = trajectories.shape
        color = colors.get(traj_name, "#333333")

        for token_idx in range(num_tokens):
            trajectory = trajectories[:, token_idx, :]

            # Plot trajectory line
            ax.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                color=color,
                alpha=0.6,
                linewidth=2,
                label=traj_name if token_idx == 0 else None,
            )

            # Mark start point
            ax.scatter(
                trajectory[0, 0],
                trajectory[0, 1],
                color=color,
                marker="o",
                s=100,
                alpha=0.8,
                edgecolors="black",
                linewidths=1,
            )

            # Mark end point
            ax.scatter(
                trajectory[-1, 0],
                trajectory[-1, 1],
                color=color,
                marker="*",
                s=200,
                alpha=0.8,
                edgecolors="black",
                linewidths=1,
            )

            # Add token label if provided
            if token_labels and token_idx < len(token_labels):
                ax.annotate(
                    token_labels[token_idx],
                    xy=(float(trajectory[-1, 0]), float(trajectory[-1, 1])),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                    alpha=0.7,
                )

    ax.set_xlabel("PC1", fontsize=12)
    ax.set_ylabel("PC2", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved figure to {save_path}")

    return fig


def plot_3d_drift(
    trajectories_dict: dict[str, np.ndarray],
    token_labels: list[str] | None = None,
    title: str = "Token Embedding Drift (PCA 3D)",
    save_path: Path | None = None,
) -> Figure:
    """
    Plot 3D PCA trajectories for token embeddings.

    Args:
        trajectories_dict: Dict mapping trajectory names to arrays of shape
                          (num_checkpoints, num_tokens, 3)
        token_labels: Optional labels for each token
        title: Plot title
        save_path: Optional path to save figure

    Returns:
        Matplotlib Figure object
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Color palette for different trajectory types
    colors = {
        "baseline": "#1f77b4",
        "standard": "#ff7f0e",
        "reasoning": "#2ca02c",
    }

    # Plot each trajectory type
    for traj_name, trajectories in trajectories_dict.items():
        num_checkpoints, num_tokens, _ = trajectories.shape
        color = colors.get(traj_name, "#333333")

        for token_idx in range(num_tokens):
            trajectory = trajectories[:, token_idx, :]

            # Plot trajectory line
            ax.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                trajectory[:, 2],
                color=color,
                alpha=0.6,
                linewidth=2,
                label=traj_name if token_idx == 0 else None,
            )

            # Mark start point
            ax.scatter(
                trajectory[0, 0],
                trajectory[0, 1],
                trajectory[0, 2],
                color=color,
                marker="o",
                s=100,
                alpha=0.8,
                edgecolors="black",
                linewidths=1,
            )

            # Mark end point
            ax.scatter(
                trajectory[-1, 0],
                trajectory[-1, 1],
                trajectory[-1, 2],
                color=color,
                marker="*",
                s=200,
                alpha=0.8,
                edgecolors="black",
                linewidths=1,
            )

            # Add token label if provided
            if token_labels and token_idx < len(token_labels):
                ax.text(
                    trajectory[-1, 0],
                    trajectory[-1, 1],
                    trajectory[-1, 2],
                    token_labels[token_idx],
                    fontsize=8,
                    alpha=0.7,
                )

    ax.set_xlabel("PC1", fontsize=12)
    ax.set_ylabel("PC2", fontsize=12)
    ax.set_zlabel("PC3", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved figure to {save_path}")

    return fig


def visualize_token_drift(
    baseline_checkpoint: Path,
    reasoning_checkpoints: list[Path],
    token_ids: list[int],
    token_labels: list[str] | None = None,
    embedding_type: EmbeddingType = EmbeddingType.INPUT,
    n_components: int = 2,
    output_dir: Path = Path("fig"),
    experiment_name: str = "token_drift",
) -> dict[str, Figure]:
    """
    Visualize token embedding drift across training.

    This function compares three types of embeddings:
    1. Baseline: pretrained model embeddings (no training)
    2. Standard: standard vocab embeddings during reasoning vocab training
    3. Reasoning: reasoning vocab embeddings during reasoning vocab training

    Args:
        baseline_checkpoint: Path to baseline (pretrained) model checkpoint
        reasoning_checkpoints: List of checkpoint paths from reasoning vocab training
        token_ids: List of token IDs to track (these should be the standard token IDs
                  that were used to initialize reasoning tokens)
        token_labels: Optional string labels for each token
        embedding_type: Whether to visualize input embeddings or output unembeddings
        n_components: Number of PCA components (2 or 3)
        output_dir: Directory to save figures
        experiment_name: Name prefix for saved figures

    Returns:
        Dictionary mapping figure names to Figure objects
    """
    logger.info(f"Visualizing token drift for {len(token_ids)} tokens")
    logger.info(f"Baseline: {baseline_checkpoint}")
    logger.info(f"Reasoning checkpoints: {len(reasoning_checkpoints)}")

    # Collect trajectories
    logger.info("Collecting baseline embeddings...")
    baseline_traj = collect_embedding_trajectories([baseline_checkpoint], token_ids, embedding_type)

    logger.info("Collecting standard vocab trajectories...")
    standard_traj = collect_embedding_trajectories(reasoning_checkpoints, token_ids, embedding_type)

    logger.info("Collecting reasoning vocab trajectories...")
    reasoning_traj = collect_embedding_trajectories(
        reasoning_checkpoints, token_ids, embedding_type
    )

    # Combine all trajectories for unified PCA
    # Shape: (total_checkpoints, num_tokens, hidden_size)
    all_trajectories = np.concatenate([baseline_traj, standard_traj, reasoning_traj], axis=0)

    # Apply PCA
    logger.info(f"Applying PCA with {n_components} components...")
    all_pca_trajectories, pca_model = compute_pca_trajectories(
        all_trajectories, n_components=n_components
    )

    # Split back into separate trajectories
    num_baseline = baseline_traj.shape[0]
    num_standard = standard_traj.shape[0]

    baseline_pca = all_pca_trajectories[:num_baseline]
    standard_pca = all_pca_trajectories[num_baseline : num_baseline + num_standard]
    reasoning_pca = all_pca_trajectories[num_baseline + num_standard :]

    # Create visualizations
    trajectories_dict = {
        "baseline": baseline_pca,
        "standard": standard_pca,
        "reasoning": reasoning_pca,
    }

    figures = {}

    if n_components == 2:
        logger.info("Creating 2D visualization...")
        fig = plot_2d_drift(
            trajectories_dict,
            token_labels=token_labels,
            title=f"Token Embedding Drift - {embedding_type.value.capitalize()}",
            save_path=output_dir / f"{experiment_name}_{embedding_type.value}_2d.png",
        )
        figures["2d"] = fig
    elif n_components == 3:
        logger.info("Creating 3D visualization...")
        fig = plot_3d_drift(
            trajectories_dict,
            token_labels=token_labels,
            title=f"Token Embedding Drift - {embedding_type.value.capitalize()} (3D)",
            save_path=output_dir / f"{experiment_name}_{embedding_type.value}_3d.png",
        )
        figures["3d"] = fig
    else:
        raise ValueError(f"n_components must be 2 or 3, got {n_components}")

    logger.info("Visualization complete!")
    return figures


def main():
    """Command-line interface for token drift visualization."""
    parser = argparse.ArgumentParser(description="Visualize token embedding drift during training")
    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="Path to baseline model checkpoint",
    )
    parser.add_argument(
        "--checkpoints",
        type=Path,
        nargs="+",
        required=True,
        help="Paths to reasoning vocab training checkpoints (in chronological order)",
    )
    parser.add_argument(
        "--token-ids",
        type=int,
        nargs="+",
        required=True,
        help="Token IDs to visualize",
    )
    parser.add_argument(
        "--token-labels",
        type=str,
        nargs="+",
        help="Optional labels for tokens",
    )
    parser.add_argument(
        "--embedding-type",
        type=str,
        choices=["input", "output"],
        default="input",
        help="Type of embeddings to visualize",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        choices=[2, 3],
        default=2,
        help="Number of PCA components",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("fig"),
        help="Output directory for figures",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="token_drift",
        help="Experiment name for output files",
    )

    args = parser.parse_args()

    # Validate inputs
    if args.token_labels and len(args.token_labels) != len(args.token_ids):
        parser.error("Number of token labels must match number of token IDs")

    # Convert string to enum
    embedding_type = EmbeddingType(args.embedding_type)

    # Run visualization
    visualize_token_drift(
        baseline_checkpoint=args.baseline,
        reasoning_checkpoints=args.checkpoints,
        token_ids=args.token_ids,
        token_labels=args.token_labels,
        embedding_type=embedding_type,
        n_components=args.n_components,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
    )


if __name__ == "__main__":
    main()
