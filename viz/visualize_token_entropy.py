"""
Visualize token embedding entropy over training.

This script computes and visualizes the entropy of token embeddings as they evolve
during training. It supports comparing:
- Baseline model (no reasoning vocab)
- Standard token in reasoning vocab model
- Reasoning token in reasoning vocab model

The entropy is computed over the probability distribution obtained by applying
softmax to the embedding/unembedding vectors.
"""

import argparse
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import torch as th
import torch.nn as nn
from loguru import logger
from transformers import AutoModelForCausalLM


def compute_embedding_entropy(embedding_vector: th.Tensor) -> float:
    """
    Compute entropy of an embedding vector.

    Applies softmax to convert the embedding to a probability distribution,
    then computes Shannon entropy.

    Args:
        embedding_vector: 1D tensor of embedding values

    Returns:
        Entropy value in nats (natural logarithm base)
    """
    # Convert embedding to probability distribution via softmax
    probs = th.softmax(embedding_vector, dim=0)

    # Compute Shannon entropy: -sum(p * log(p))
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    entropy = -th.sum(probs * th.log(probs + epsilon))

    return entropy.item()


def load_checkpoint_embeddings(
    checkpoint_path: Path,
    token_id: int,
    is_reasoning_token: bool = False,
) -> tuple[th.Tensor, th.Tensor]:
    """
    Load embedding and unembedding vectors for a specific token from a checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint directory
        token_id: Token ID to extract (for reasoning tokens, this is the offset from vocab_size)
        is_reasoning_token: Whether to load from reasoning vocab (True) or standard vocab (False)

    Returns:
        Tuple of (embedding_vector, unembedding_vector)
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")

    # Load model from checkpoint
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=th.float32,
        trust_remote_code=True,
    )

    # Extract embedding
    if is_reasoning_token:
        # For reasoning tokens, use reasoning_embed layer
        if not hasattr(model, "reasoning_embed"):
            raise ValueError(
                f"Model at {checkpoint_path} does not have reasoning_embed layer. "
                "Is this a reasoning vocab model?"
            )
        reasoning_embed = cast(nn.Embedding, model.reasoning_embed)
        reasoning_unembed = cast(nn.Linear, model.reasoning_unembed)
        embedding_vector = reasoning_embed.weight[token_id].detach().clone()
        unembedding_vector = reasoning_unembed.weight[:, token_id].detach().clone()
    else:
        # For standard tokens, use standard embedding layer
        # Get the base model's embeddings
        if hasattr(model, "model"):
            # For models with .model attribute (e.g., LlamaForCausalLM)
            embed_tokens = cast(nn.Embedding, model.model.embed_tokens)
            embedding_vector = embed_tokens.weight[token_id].detach().clone()
        elif hasattr(model, "transformer"):
            # For models with .transformer attribute (e.g., GPT-2)
            wte = cast(nn.Embedding, model.transformer.wte)
            embedding_vector = wte.weight[token_id].detach().clone()
        else:
            raise ValueError(f"Unknown model structure for {checkpoint_path}")

        # Get unembedding (lm_head)
        lm_head = cast(nn.Linear, model.lm_head)
        unembedding_vector = lm_head.weight[token_id].detach().clone()

    return embedding_vector, unembedding_vector


def compute_entropy_trajectory(
    checkpoint_dirs: list[Path],
    token_id: int,
    is_reasoning_token: bool = False,
    use_unembedding: bool = False,
) -> tuple[list[int], list[float]]:
    """
    Compute entropy trajectory across multiple checkpoints.

    Args:
        checkpoint_dirs: List of checkpoint directories in training order
        token_id: Token ID to track
        is_reasoning_token: Whether to track reasoning vocab token
        use_unembedding: If True, compute entropy from unembedding; else from embedding

    Returns:
        Tuple of (steps, entropies) where steps are checkpoint step numbers
    """
    steps = []
    entropies = []

    for checkpoint_dir in sorted(checkpoint_dirs):
        # Extract step number from checkpoint directory name
        # Expected format: checkpoint-{step}
        step_str = checkpoint_dir.name.split("-")[-1]
        try:
            step = int(step_str)
        except ValueError:
            logger.warning(f"Could not parse step number from {checkpoint_dir.name}, skipping")
            continue

        try:
            embedding_vec, unembedding_vec = load_checkpoint_embeddings(
                checkpoint_dir, token_id, is_reasoning_token
            )

            vector = unembedding_vec if use_unembedding else embedding_vec
            entropy = compute_embedding_entropy(vector)

            steps.append(step)
            entropies.append(entropy)

            logger.info(
                f"Step {step}: entropy = {entropy:.4f} "
                f"({'unembedding' if use_unembedding else 'embedding'})"
            )

        except Exception as e:
            logger.error(f"Error processing checkpoint {checkpoint_dir}: {e}")
            continue

    return steps, entropies


def plot_entropy_comparison(
    baseline_data: tuple[list[int], list[float]],
    standard_data: tuple[list[int], list[float]],
    reasoning_data: tuple[list[int], list[float]],
    token_str: str,
    output_path: Path,
    use_unembedding: bool = False,
) -> None:
    """
    Create comparison plot of entropy trajectories.

    Args:
        baseline_data: (steps, entropies) for baseline model
        standard_data: (steps, entropies) for standard token in reasoning model
        reasoning_data: (steps, entropies) for reasoning token in reasoning model
        token_str: String representation of the token for plot title
        output_path: Path to save the figure
        use_unembedding: Whether the data is from unembedding (vs embedding)
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    layer_type = "Unembedding" if use_unembedding else "Embedding"

    # Plot each trajectory
    if baseline_data[0]:  # Check if we have data
        ax.plot(
            baseline_data[0],
            baseline_data[1],
            marker="o",
            label="Baseline",
            linewidth=2,
            markersize=6,
        )

    if standard_data[0]:
        ax.plot(
            standard_data[0],
            standard_data[1],
            marker="s",
            label="Standard (Reasoning Model)",
            linewidth=2,
            markersize=6,
        )

    if reasoning_data[0]:
        ax.plot(
            reasoning_data[0],
            reasoning_data[1],
            marker="^",
            label="Reasoning Token",
            linewidth=2,
            markersize=6,
        )

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel(f"{layer_type} Entropy (nats)", fontsize=12)
    ax.set_title(f"{layer_type} Entropy Trajectory for Token: {token_str}", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved plot to {output_path}")
    plt.close()


def visualize_token_entropy(
    baseline_dir: Path | None,
    reasoning_dir: Path,
    token_id: int,
    reasoning_token_id: int | None = None,
    token_str: str | None = None,
    output_dir: Path | None = None,
    plot_both_layers: bool = True,
) -> dict[str, Any]:
    """
    Main function to visualize token entropy trajectories.

    Args:
        baseline_dir: Directory containing baseline model checkpoints
        reasoning_dir: Directory containing reasoning vocab model checkpoints
        token_id: Standard token ID to track
        reasoning_token_id: Reasoning token ID to track (offset from vocab_size).
                           If None, uses same as token_id
        token_str: String representation of token for plots. If None, uses token_id
        output_dir: Directory to save plots. If None, uses ./fig
        plot_both_layers: If True, plot both embedding and unembedding

    Returns:
        Dictionary containing computed entropy trajectories
    """
    if reasoning_token_id is None:
        reasoning_token_id = token_id

    if token_str is None:
        token_str = f"Token_{token_id}"

    if output_dir is None:
        output_dir = Path("fig")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find checkpoint directories
    baseline_checkpoints = []
    if baseline_dir is not None:
        baseline_checkpoints = list(baseline_dir.glob("checkpoint-*"))
        logger.info(f"Found {len(baseline_checkpoints)} baseline checkpoints")

    reasoning_checkpoints = list(reasoning_dir.glob("checkpoint-*"))
    logger.info(f"Found {len(reasoning_checkpoints)} reasoning checkpoints")

    results = {}

    # Process each layer type
    for use_unembedding in [False, True] if plot_both_layers else [False]:
        layer_name = "unembedding" if use_unembedding else "embedding"
        logger.info(f"\nProcessing {layer_name} layer")

        # Compute trajectories
        baseline_data = ([], [])
        if baseline_checkpoints:
            baseline_data = compute_entropy_trajectory(
                baseline_checkpoints,
                token_id,
                is_reasoning_token=False,
                use_unembedding=use_unembedding,
            )

        standard_data = compute_entropy_trajectory(
            reasoning_checkpoints,
            token_id,
            is_reasoning_token=False,
            use_unembedding=use_unembedding,
        )

        reasoning_data = compute_entropy_trajectory(
            reasoning_checkpoints,
            reasoning_token_id,
            is_reasoning_token=True,
            use_unembedding=use_unembedding,
        )

        # Store results
        results[layer_name] = {
            "baseline": {"steps": baseline_data[0], "entropies": baseline_data[1]},
            "standard": {"steps": standard_data[0], "entropies": standard_data[1]},
            "reasoning": {"steps": reasoning_data[0], "entropies": reasoning_data[1]},
        }

        # Create plot
        output_path = output_dir / f"token_entropy_{layer_name}_{token_str}.png"
        plot_entropy_comparison(
            baseline_data,
            standard_data,
            reasoning_data,
            token_str,
            output_path,
            use_unembedding=use_unembedding,
        )

    return results


def main():
    """Command-line interface for token entropy visualization."""
    parser = argparse.ArgumentParser(description="Visualize token embedding entropy over training")
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=None,
        help="Directory containing baseline model checkpoints",
    )
    parser.add_argument(
        "--reasoning-dir",
        type=Path,
        required=True,
        help="Directory containing reasoning vocab model checkpoints",
    )
    parser.add_argument(
        "--token-id",
        type=int,
        required=True,
        help="Standard token ID to track",
    )
    parser.add_argument(
        "--reasoning-token-id",
        type=int,
        default=None,
        help="Reasoning token ID (offset). If not provided, uses token-id",
    )
    parser.add_argument(
        "--token-str",
        type=str,
        default=None,
        help="String representation of token for plots",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("fig"),
        help="Directory to save output plots",
    )
    parser.add_argument(
        "--embedding-only",
        action="store_true",
        help="Only plot embedding layer (not unembedding)",
    )

    args = parser.parse_args()

    # Run visualization
    results = visualize_token_entropy(
        baseline_dir=args.baseline_dir,
        reasoning_dir=args.reasoning_dir,
        token_id=args.token_id,
        reasoning_token_id=args.reasoning_token_id,
        token_str=args.token_str,
        output_dir=args.output_dir,
        plot_both_layers=not args.embedding_only,
    )

    logger.info("\nVisualization complete!")
    logger.info(f"Results saved to {args.output_dir}")

    # Print summary statistics
    for layer_name, layer_data in results.items():
        logger.info(f"\n{layer_name.upper()} Summary:")
        for model_type, data in layer_data.items():
            if data["steps"]:
                entropies = data["entropies"]
                logger.info(
                    f"  {model_type}: "
                    f"mean={np.mean(entropies):.4f}, "
                    f"std={np.std(entropies):.4f}, "
                    f"min={np.min(entropies):.4f}, "
                    f"max={np.max(entropies):.4f}"
                )


if __name__ == "__main__":
    main()
