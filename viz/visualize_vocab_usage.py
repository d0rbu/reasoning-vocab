"""
Visualize reasoning vs. normal token usage during training.

This script analyzes model outputs from training checkpoints and creates
visualizations showing how reasoning vocabulary tokens vs. standard vocabulary
tokens are used over the course of training.

Usage:
    python viz/visualize_vocab_usage.py --checkpoint_dir ./out/exp_name --output_dir ./fig
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch as th
from loguru import logger
from matplotlib.figure import Figure


def load_checkpoint_outputs(checkpoint_dir: Path) -> dict[int, list[th.Tensor]]:
    """
    Load generated outputs from training checkpoints.

    Expected structure:
        checkpoint_dir/
            checkpoint-100/
                generations.pt  # Contains generated token ids
            checkpoint-200/
                generations.pt
            ...

    Args:
        checkpoint_dir: Directory containing checkpoint subdirectories

    Returns:
        Dictionary mapping step number to list of integer tensors of generated token IDs.
        Each value is a list of tensors (ragged - different rollouts have different lengths).
    """
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    checkpoints = {}

    # Find all checkpoint directories
    checkpoint_subdirs = sorted(checkpoint_dir.glob("checkpoint-*"))

    if not checkpoint_subdirs:
        logger.warning(f"No checkpoint directories found in {checkpoint_dir}")
        return checkpoints

    for ckpt_dir in checkpoint_subdirs:
        # Extract step number from directory name (e.g., "checkpoint-100" -> 100)
        try:
            step = int(ckpt_dir.name.split("-")[-1])
        except ValueError:
            logger.warning(f"Could not parse step number from {ckpt_dir.name}, skipping")
            continue

        # Load generations file if it exists
        generations_file = ckpt_dir / "generations.pt"
        if not generations_file.exists():
            logger.debug(f"No generations.pt found in {ckpt_dir}, skipping")
            continue

        try:
            data = th.load(generations_file, map_location="cpu", weights_only=False)
            checkpoints[step] = data
            logger.debug(f"Loaded checkpoint at step {step}")
        except Exception as e:
            logger.warning(f"Failed to load {generations_file}: {e}")
            continue

    logger.info(f"Loaded {len(checkpoints)} checkpoints from {checkpoint_dir}")
    return checkpoints


def analyze_token_usage(
    token_ids: list[th.Tensor] | th.Tensor, vocab_size: int
) -> tuple[int, int, float, float]:
    """
    Analyze token usage to determine reasoning vs. standard token distribution.

    Tokens with ID < vocab_size are standard tokens.
    Tokens with ID >= vocab_size are reasoning tokens.

    Args:
        token_ids: List of integer tensors (ragged) or single integer tensor of token IDs
        vocab_size: Size of the standard vocabulary (not including reasoning tokens)

    Returns:
        Tuple of (num_standard, num_reasoning, pct_standard, pct_reasoning)
    """
    # Handle list of tensors (ragged case)
    if isinstance(token_ids, list):
        if len(token_ids) == 0:
            return 0, 0, 0.0, 0.0
        # Concatenate all tensors
        flat_tokens = th.cat([t.flatten() for t in token_ids])
    else:
        # Handle single tensor
        if token_ids.numel() == 0:
            return 0, 0, 0.0, 0.0
        flat_tokens = token_ids.flatten()

    # Count standard vs reasoning tokens
    num_standard = (flat_tokens < vocab_size).sum().item()
    num_reasoning = (flat_tokens >= vocab_size).sum().item()

    total = num_standard + num_reasoning
    pct_standard = 100.0 * num_standard / total if total > 0 else 0.0
    pct_reasoning = 100.0 * num_reasoning / total if total > 0 else 0.0

    assert isinstance(num_standard, int), "Number of standard tokens must be an integer"
    assert isinstance(num_reasoning, int), "Number of reasoning tokens must be an integer"

    return num_standard, num_reasoning, pct_standard, pct_reasoning


def create_usage_plot(
    steps: np.ndarray | list[int],
    standard_pcts: np.ndarray | list[float],
    reasoning_pcts: np.ndarray | list[float],
    title: str = "Token Vocabulary Usage Over Training",
) -> Figure:
    """
    Create a line plot showing token usage over training steps.

    Args:
        steps: List of training step numbers
        standard_pcts: Percentage of standard tokens at each step
        reasoning_pcts: Percentage of reasoning tokens at each step
        title: Plot title

    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot both lines
    ax.plot(steps, standard_pcts, label="Standard Tokens", marker="o", linewidth=2)
    ax.plot(steps, reasoning_pcts, label="Reasoning Tokens", marker="s", linewidth=2)

    # Styling
    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Token Usage (%)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    return fig


def create_stacked_area_plot(
    steps: np.ndarray | list[int],
    standard_counts: np.ndarray | list[int],
    reasoning_counts: np.ndarray | list[int],
    title: str = "Token Count Distribution Over Training",
) -> Figure:
    """
    Create a stacked area plot showing absolute token counts.

    Args:
        steps: List of training step numbers
        standard_counts: Count of standard tokens at each step
        reasoning_counts: Count of reasoning tokens at each step
        title: Plot title

    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Convert to numpy arrays
    steps_arr = np.array(steps)
    standard_arr = np.array(standard_counts)
    reasoning_arr = np.array(reasoning_counts)

    # Create stacked area plot
    ax.fill_between(
        steps_arr, 0, standard_arr, label="Standard Tokens", alpha=0.7, color="steelblue"
    )
    ax.fill_between(
        steps_arr,
        standard_arr,
        standard_arr + reasoning_arr,
        label="Reasoning Tokens",
        alpha=0.7,
        color="coral",
    )

    # Styling
    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Token Count", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="best")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    return fig


def visualize_vocab_usage(
    checkpoint_dir: Path, vocab_size: int, output_dir: Path = Path("./fig")
) -> None:
    """
    Main function to visualize vocabulary usage from checkpoints.

    Args:
        checkpoint_dir: Path to directory containing checkpoint subdirectories
        vocab_size: Size of the standard vocabulary (excluding reasoning tokens)
        output_dir: Directory to save output figures (default: ./fig)
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint data
    logger.info(f"Loading checkpoints from {checkpoint_dir}")
    checkpoints = load_checkpoint_outputs(checkpoint_dir)

    if not checkpoints:
        logger.error("No checkpoints found. Cannot create visualizations.")
        return

    # Analyze each checkpoint
    steps = []
    standard_counts = []
    reasoning_counts = []
    standard_pcts = []
    reasoning_pcts = []

    for step in sorted(checkpoints.keys()):
        token_ids = checkpoints[step]

        num_std, num_reas, pct_std, pct_reas = analyze_token_usage(token_ids, vocab_size)

        steps.append(step)
        standard_counts.append(num_std)
        reasoning_counts.append(num_reas)
        standard_pcts.append(pct_std)
        reasoning_pcts.append(pct_reas)

        logger.debug(
            f"Step {step}: {pct_std:.1f}% standard, {pct_reas:.1f}% reasoning "
            f"({num_std} std, {num_reas} reas)"
        )

    # Create visualizations
    logger.info("Creating usage percentage plot...")
    fig_pct = create_usage_plot(steps, standard_pcts, reasoning_pcts)
    pct_path = output_dir / "vocab_usage_percentage.png"
    fig_pct.savefig(pct_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved percentage plot to {pct_path}")
    plt.close(fig_pct)

    logger.info("Creating stacked area plot...")
    fig_stacked = create_stacked_area_plot(steps, standard_counts, reasoning_counts)
    stacked_path = output_dir / "vocab_usage_stacked.png"
    fig_stacked.savefig(stacked_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved stacked plot to {stacked_path}")
    plt.close(fig_stacked)

    logger.info("Visualization complete!")


def main():
    """Command-line interface for the visualization script."""
    parser = argparse.ArgumentParser(
        description="Visualize reasoning vs. standard token usage during training"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to directory containing checkpoint subdirectories",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=151936,
        help="Size of the standard vocabulary (excluding reasoning tokens, default: 151936 for Qwen3)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./fig", help="Directory to save output figures"
    )

    args = parser.parse_args()

    visualize_vocab_usage(
        checkpoint_dir=Path(args.checkpoint_dir),
        vocab_size=args.vocab_size,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
