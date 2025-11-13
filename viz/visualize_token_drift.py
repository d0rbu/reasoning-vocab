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
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
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


@dataclass
class TokenTrajectories:
    """Represents a single token variant with its multiplicity and trajectory."""

    token_id: int
    trajectories: th.Tensor  # Shape: (num_variants, num_checkpoints, hidden_size)


def load_reasoning_token_map(checkpoint_path: Path) -> tuple[th.Tensor, th.Tensor]:
    """
    Load reasoning token map from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint directory

    Returns:
        Tuple of (standard_token_ids, multiplicities) where index i corresponds
        to reasoning token vocab_size + i. Returns empty tensors for baseline models.
    """
    map_path = checkpoint_path / "reasoning_token_map.json"

    if not map_path.exists():
        raise FileNotFoundError(
            f"reasoning_token_map.json not found at {map_path}. "
            "All checkpoints must have a reasoning_token_map.json file. "
        )

    with open(map_path) as f:
        data = json.load(f)

    raw_standard_token_ids = data["standard_token_ids"]
    raw_multiplicities = data["multiplicities"]

    standard_token_ids = th.tensor(raw_standard_token_ids)
    multiplicities = th.tensor(raw_multiplicities)

    assert standard_token_ids.dtype == th.long
    assert multiplicities.dtype == th.long
    assert standard_token_ids.shape == multiplicities.shape
    assert standard_token_ids.dim() == 1

    logger.debug(f"Loaded reasoning token map with {standard_token_ids.shape[0]} reasoning tokens")
    return standard_token_ids, multiplicities


def expand_token_ids_with_reasoning(
    standard_token_ids: th.Tensor,
    reasoning_std_ids: th.Tensor,
    vocab_size: int,
) -> th.Tensor:
    """
    Expand standard token IDs to include all their reasoning variants.

    Args:
        standard_token_ids: List of standard token IDs to track
        reasoning_std_ids: Standard token IDs for each reasoning token (from map)
        reasoning_multiplicities: Multiplicities for each reasoning token (from map)
        vocab_size: Size of standard vocabulary

    Returns:
        List of all token IDs (standard + reasoning variants) sorted by
        (standard_token_id, multiplicity)
    """
    # Validate that all standard token IDs are within vocab_size
    invalid_ids = standard_token_ids[standard_token_ids >= vocab_size]
    if invalid_ids.numel() > 0:
        raise ValueError(
            f"Invalid standard token IDs in reasoning map: {invalid_ids}. "
            f"All standard token IDs must be < vocab_size ({vocab_size})."
        )

    assert standard_token_ids.dim() == 1

    present_standard_token_ids = th.zeros(vocab_size, dtype=th.bool)
    present_standard_token_ids[standard_token_ids] = True

    present_reasoning_token_ids = present_standard_token_ids[reasoning_std_ids]

    present_token_ids = th.cat([present_standard_token_ids, present_reasoning_token_ids])

    return th.nonzero(present_token_ids).squeeze(-1)


def load_checkpoint_embeddings(
    checkpoint_path: Path,
    token_ids: th.Tensor,
    vocab_size: int,
    embedding_type: EmbeddingType = EmbeddingType.INPUT,
) -> th.Tensor:
    """
    Load embeddings for specific tokens from a model checkpoint.

    This function supports loading both standard and reasoning tokens from
    checkpoints with extended vocabulary (via resize_token_embeddings).

    Args:
        checkpoint_path: Path to model checkpoint directory
        token_ids: Tensor of token IDs to extract embeddings for (may include reasoning tokens)
        vocab_size: Size of standard vocabulary (reasoning tokens have IDs >= vocab_size)
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

    all_embeddings = embed_layer.weight.data

    # Validate the embedding layer has at least vocab_size embeddings
    # (it may have more if reasoning vocabulary is present)
    assert all_embeddings.shape[0] >= vocab_size, (
        f"Embedding layer has {all_embeddings.shape[0]} embeddings, but vocab_size is {vocab_size}"
    )

    assert token_ids.dim() == 1
    assert token_ids.dtype == th.long
    assert token_ids.min() >= 0
    assert token_ids.max() < all_embeddings.shape[0], (
        f"Token ID {token_ids.max()} exceeds embedding layer size {all_embeddings.shape[0]}"
    )

    embeddings = all_embeddings[token_ids]

    logger.debug(f"Loaded embeddings with shape: {embeddings.shape}")
    return embeddings


def collect_embedding_trajectories(
    checkpoint_dirs: list[Path],
    token_ids: th.Tensor,
    vocab_size: int,
    embedding_type: EmbeddingType = EmbeddingType.INPUT,
) -> th.Tensor:
    """
    Collect embedding trajectories across multiple checkpoints.

    Args:
        checkpoint_dirs: List of checkpoint directories in chronological order
        token_ids: Tensor of token IDs to track
        vocab_size: Size of standard vocabulary
        reasoning_standard_token_ids: Tensor of standard token IDs used to initialize reasoning tokens
        embedding_type: Whether to load input embeddings or output unembeddings

    Returns:
        Tensor of shape (num_tokens, num_checkpoints, hidden_size)
    """
    return th.stack(
        [
            load_checkpoint_embeddings(ckpt_dir, token_ids, vocab_size, embedding_type)
            for ckpt_dir in checkpoint_dirs
        ],
        dim=1,
    )


def compute_pca_trajectories(
    trajectories: th.Tensor,
    n_components: int = 2,
) -> tuple[th.Tensor, PCA]:
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
    trajectories_dict: dict[str, th.Tensor],
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
    trajectories_dict: dict[str, th.Tensor],
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


def group_trajectories_by_token_family(
    trajectories: th.Tensor,
    standard_token_ids: th.Tensor,
    reasoning_standard_token_ids: th.Tensor,
) -> list[TokenTrajectories]:
    """
    Group trajectories by standard token ID (token family).

    Args:
        trajectories: Shape (num_tokens, num_checkpoints, hidden_size)
        standard_token_ids: Original standard token IDs requested
        reasoning_std_ids: Standard token IDs for each reasoning token (from map)

    Returns:
        List[TokenTrajectories]: Each item contains a standard token family and its variants' trajectories.
    """
    token_trajectories: list[TokenTrajectories] = [
        TokenTrajectories(
            token_id=token_id.item(),
            trajectories=th.empty(
                0,
            ),
        )
        for token_id in standard_token_ids
    ]

    for standard_token_id in standard_token_ids:
        reasoning_variant_indices = th.nonzero(
            reasoning_standard_token_ids == standard_token_id
        ).squeeze(-1)

        token_trajectories[standard_token_id].trajectories = trajectories[reasoning_variant_indices]

    return token_trajectories


def visualize_token_drift(
    checkpoints: list[Path],
    token_ids_raw: list[int],
    token_labels: list[str] | None = None,
    embedding_type: EmbeddingType = EmbeddingType.INPUT,
    n_components: int = 2,
    output_dir: Path = Path("fig"),
    experiment_name: str = "token_drift",
) -> dict[str, Figure]:
    """
    Visualize token embedding drift across training.

    This function tracks token embeddings and their reasoning variants across training,
    showing how they evolve from initialization (checkpoint-0) through training.

    For each standard token, the visualization includes:
    - The standard token itself (multiplicity 0)
    - All reasoning variants initialized from that token (multiplicity 1, 2, 3, ...)

    The first checkpoint should be the initialized model (checkpoint-0) which has the
    reasoning vocabulary but hasn't been trained yet.

    Args:
        checkpoints: List of checkpoint paths in chronological order, starting with
                    checkpoint-0 (initialized model before training)
        token_ids_raw: List of standard token IDs to track (these should be the standard
                      token IDs that were used to initialize reasoning tokens)
        token_labels: Optional string labels for each standard token
        embedding_type: Whether to visualize input embeddings or output unembeddings
        n_components: Number of PCA components (2 or 3)
        output_dir: Directory to save figures
        experiment_name: Name prefix for saved figures

    Returns:
        Dictionary mapping figure names to Figure objects
    """
    logger.info(f"Visualizing token drift for {len(token_ids_raw)} standard tokens")
    logger.trace(f"Token IDs: {token_ids_raw}")
    logger.debug(f"Number of checkpoints: {len(checkpoints)}")
    logger.trace(f"Checkpoints: {checkpoints}")

    # Load reasoning token map from first checkpoint
    logger.debug("Loading reasoning token map...")
    reasoning_standard_token_ids, reasoning_multiplicities = load_reasoning_token_map(
        checkpoints[0]
    )
    token_ids = th.tensor(token_ids_raw, dtype=th.long)

    # Get vocab_size from config (required)
    config_path = checkpoints[0] / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"config.json not found at {config_path}. All checkpoints must have a config.json file."
        )

    with open(config_path) as f:
        config = json.load(f)
    vocab_size = config["vocab_size"]
    logger.debug(f"Standard vocab_size: {vocab_size}")

    # Expand token IDs to include all reasoning variants
    all_token_ids = expand_token_ids_with_reasoning(
        token_ids, reasoning_standard_token_ids, vocab_size
    )
    logger.debug(f"Tracking {len(all_token_ids)} total tokens (including reasoning variants)")

    # Collect all trajectories in one pass
    logger.debug(f"Collecting embeddings from {len(checkpoints)} checkpoints...")
    all_trajectories = collect_embedding_trajectories(
        checkpoints, all_token_ids, vocab_size, embedding_type
    )
    # Shape: (num_checkpoints, num_tokens, hidden_size)

    # Group trajectories by token family
    logger.info("Grouping trajectories by token family...")
    token_trajectories = group_trajectories_by_token_family(
        all_trajectories,
        token_ids,
        reasoning_standard_token_ids,
    )

    for token_trajectory in token_trajectories:
        logger.debug(
            f"Token {token_trajectory.token_id} trajectories shape: {token_trajectory.trajectories.shape}"
        )

    figures = {}

    # TODO: Implement global PCA visualization
    # TODO: Implement within-group PCA visualization

    logger.info(f"Created {len(figures)} visualizations")
    logger.info("Visualization complete!")
    return figures


def main():
    """Command-line interface for token drift visualization."""
    parser = argparse.ArgumentParser(description="Visualize token embedding drift during training")
    parser.add_argument(
        "--checkpoints",
        type=Path,
        nargs="+",
        required=True,
        help="Paths to checkpoints in chronological order (starting with checkpoint-0)",
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
        checkpoints=args.checkpoints,
        token_ids_raw=args.token_ids,
        token_labels=args.token_labels,
        embedding_type=embedding_type,
        n_components=args.n_components,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
    )


if __name__ == "__main__":
    main()
