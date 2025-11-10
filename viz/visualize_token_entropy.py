"""
Visualize token entropy over training.

This script computes the entropy of output distributions when the model processes
validation data. For each token in the input, we measure the entropy of the model's
output distribution at that position, then aggregate these measurements by token ID.

This reveals which tokens the model is most/least confident about and how this
changes during training.
"""

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch as th
import torch.nn as nn
from datasets import load_dataset
from loguru import logger
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def compute_output_entropy(logits: th.Tensor, dim: int = -1) -> th.Tensor:
    """
    Compute entropy of output distributions from logits.

    Args:
        logits: Tensor of shape [..., vocab_size] containing unnormalized logits
        dim: Dimension along which to compute softmax and entropy

    Returns:
        Tensor of shape [...] containing entropy values in nats
    """
    # Convert to probabilities
    probs = th.softmax(logits, dim=dim)

    # Compute entropy: -sum(p * log(p))
    # Add epsilon for numerical stability
    epsilon = 1e-10
    entropy = -th.sum(probs * th.log(probs + epsilon), dim=dim)

    return entropy


def compute_token_entropies_for_checkpoint(
    model: nn.Module,
    tokenizer: Any,
    dataset: Any,
    max_samples: int | None = None,
    batch_size: int = 8,
    max_length: int = 512,
    device: str = "cuda" if th.cuda.is_available() else "cpu",
) -> dict[int, list[float]]:
    """
    Compute entropy statistics for each token by running model on validation data.

    Args:
        model: The language model to evaluate
        tokenizer: Tokenizer for the model
        dataset: Validation dataset (should have 'text' or 'problem' field)
        max_samples: Maximum number of samples to process (None = all)
        batch_size: Batch size for inference
        max_length: Maximum sequence length
        device: Device to run inference on

    Returns:
        Dictionary mapping token_id -> list of entropy values
    """
    model.eval()
    model.to(device)

    token_entropies: dict[int, list[float]] = defaultdict(list)

    # Prepare dataset
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    # Extract text field (handle different dataset formats)
    def get_text(example):
        if "text" in example:
            return example["text"]
        elif "problem" in example:
            return example["problem"]
        elif "question" in example:
            return example["question"]
        else:
            # Try to find any string field
            for _key, value in example.items():
                if isinstance(value, str) and len(value) > 0:
                    return value
            return ""

    texts = [get_text(example) for example in dataset]

    logger.info(f"Processing {len(texts)} validation samples")

    with th.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Computing entropies"):
            batch_texts = texts[i : i + batch_size]

            # Tokenize batch
            batch_encoding = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )

            input_ids = batch_encoding["input_ids"].to(device)
            attention_mask = batch_encoding["attention_mask"].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # Shape: [batch, seq_len, vocab_size]

            # Compute entropy at each position
            entropies = compute_output_entropy(logits)  # Shape: [batch, seq_len]

            # Process each sequence in batch
            for seq_idx in range(input_ids.size(0)):
                seq_input_ids = input_ids[seq_idx]
                seq_entropies = entropies[seq_idx]
                seq_attention_mask = attention_mask[seq_idx]

                # Only consider non-padding positions
                valid_positions = seq_attention_mask.bool()

                # For each valid position, store (token_id, entropy) pair
                for pos_idx in range(len(seq_input_ids)):
                    if not valid_positions[pos_idx]:
                        continue

                    token_id = seq_input_ids[pos_idx].item()
                    entropy_value = seq_entropies[pos_idx].item()

                    # Skip special tokens (optional - could make this configurable)
                    if token_id in [tokenizer.pad_token_id, tokenizer.bos_token_id]:
                        continue

                    token_entropies[token_id].append(entropy_value)

    # Convert to regular dict
    return dict(token_entropies)


def aggregate_token_statistics(
    token_entropies: dict[int, list[float]],
) -> dict[int, dict[str, float]]:
    """
    Compute aggregate statistics for each token.

    Args:
        token_entropies: Dictionary mapping token_id -> list of entropy values

    Returns:
        Dictionary mapping token_id -> {mean, std, count, min, max}
    """
    stats = {}

    for token_id, entropies in token_entropies.items():
        if len(entropies) == 0:
            continue

        stats[token_id] = {
            "mean": float(np.mean(entropies)),
            "std": float(np.std(entropies)),
            "count": len(entropies),
            "min": float(np.min(entropies)),
            "max": float(np.max(entropies)),
        }

    return stats


def compute_trajectory_across_checkpoints(
    checkpoint_dirs: list[Path],
    tokenizer: Any,
    dataset: Any,
    token_ids_to_track: list[int] | None = None,
    max_samples: int = 1000,
    batch_size: int = 8,
    device: str = "cuda" if th.cuda.is_available() else "cpu",
) -> dict[int, dict[int, dict[str, float]]]:
    """
    Compute token entropy trajectory across multiple training checkpoints.

    Args:
        checkpoint_dirs: List of checkpoint directories in training order
        tokenizer: Tokenizer for the model
        dataset: Validation dataset
        token_ids_to_track: Specific token IDs to track (None = track all)
        max_samples: Maximum validation samples to process per checkpoint
        batch_size: Batch size for inference
        device: Device to run inference on

    Returns:
        Dictionary mapping step -> token_id -> statistics
    """
    trajectory = {}

    for checkpoint_dir in sorted(checkpoint_dirs):
        # Extract step number
        step_str = checkpoint_dir.name.split("-")[-1]
        try:
            step = int(step_str)
        except ValueError:
            logger.warning(f"Could not parse step from {checkpoint_dir.name}, skipping")
            continue

        logger.info(f"\nProcessing checkpoint at step {step}")

        try:
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint_dir,
                torch_dtype=th.float32,
                trust_remote_code=True,
            )

            # Compute entropies
            token_entropies = compute_token_entropies_for_checkpoint(
                model, tokenizer, dataset, max_samples, batch_size, device=device
            )

            # Filter to tracked tokens if specified
            if token_ids_to_track is not None:
                token_entropies = {
                    tid: entropies
                    for tid, entropies in token_entropies.items()
                    if tid in token_ids_to_track
                }

            # Aggregate statistics
            stats = aggregate_token_statistics(token_entropies)

            trajectory[step] = stats

            logger.info(f"Step {step}: Computed entropy for {len(stats)} tokens")

            # Clean up
            del model
            th.cuda.empty_cache() if th.cuda.is_available() else None

        except Exception as e:
            logger.error(f"Error processing checkpoint {checkpoint_dir}: {e}")
            continue

    return trajectory


def plot_token_entropy_trajectories(
    baseline_trajectory: dict[int, dict[int, dict[str, float]]],
    reasoning_trajectory: dict[int, dict[int, dict[str, float]]],
    token_ids: list[int],
    tokenizer: Any,
    output_dir: Path,
    top_k: int = 20,
) -> None:
    """
    Plot entropy trajectories for specific tokens.

    Args:
        baseline_trajectory: Trajectory for baseline model (step -> token_id -> stats)
        reasoning_trajectory: Trajectory for reasoning model (step -> token_id -> stats)
        token_ids: List of token IDs to plot
        tokenizer: Tokenizer to decode token IDs
        output_dir: Directory to save plots
        top_k: Number of tokens to plot if token_ids is large
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select top_k most common tokens if we have too many
    if len(token_ids) > top_k:
        # Count total occurrences across all steps
        token_counts = defaultdict(int)
        for step_stats in reasoning_trajectory.values():
            for token_id, stats in step_stats.items():
                if token_id in token_ids:
                    token_counts[token_id] += stats["count"]

        # Select top_k by count
        top_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)[:top_k]
        token_ids = [tid for tid, _ in top_tokens]
        logger.info(f"Plotting top {top_k} tokens by frequency")

    # Create individual plots for each token
    for token_id in token_ids:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Get token string
        try:
            token_str = tokenizer.decode([token_id])
        except Exception:
            token_str = f"Token_{token_id}"

        # Extract baseline trajectory
        baseline_steps = []
        baseline_means = []
        baseline_stds = []

        for step in sorted(baseline_trajectory.keys()):
            if token_id in baseline_trajectory[step]:
                stats = baseline_trajectory[step][token_id]
                baseline_steps.append(step)
                baseline_means.append(stats["mean"])
                baseline_stds.append(stats["std"])

        # Extract reasoning trajectory
        reasoning_steps = []
        reasoning_means = []
        reasoning_stds = []

        for step in sorted(reasoning_trajectory.keys()):
            if token_id in reasoning_trajectory[step]:
                stats = reasoning_trajectory[step][token_id]
                reasoning_steps.append(step)
                reasoning_means.append(stats["mean"])
                reasoning_stds.append(stats["std"])

        # Plot baseline
        if baseline_steps:
            ax.plot(
                baseline_steps,
                baseline_means,
                marker="o",
                label="Baseline",
                linewidth=2,
                markersize=6,
            )
            ax.fill_between(
                baseline_steps,
                np.array(baseline_means) - np.array(baseline_stds),
                np.array(baseline_means) + np.array(baseline_stds),
                alpha=0.2,
            )

        # Plot reasoning
        if reasoning_steps:
            ax.plot(
                reasoning_steps,
                reasoning_means,
                marker="s",
                label="Reasoning Model",
                linewidth=2,
                markersize=6,
            )
            ax.fill_between(
                reasoning_steps,
                np.array(reasoning_means) - np.array(reasoning_stds),
                np.array(reasoning_means) + np.array(reasoning_stds),
                alpha=0.2,
            )

        ax.set_xlabel("Training Step", fontsize=12)
        ax.set_ylabel("Output Entropy (nats)", fontsize=12)
        ax.set_title(f"Output Entropy Trajectory for Token: {repr(token_str)}", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        safe_token_str = token_str.replace("/", "_").replace(" ", "_")[:50]
        output_path = output_dir / f"entropy_trajectory_token_{token_id}_{safe_token_str}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved plot to {output_path}")
        plt.close()


def visualize_token_entropy(
    baseline_dir: Path | None,
    reasoning_dir: Path,
    dataset_name: str = "gsm8k",
    dataset_split: str = "test",
    token_ids_to_track: list[int] | None = None,
    max_samples: int = 1000,
    batch_size: int = 8,
    output_dir: Path | None = None,
    device: str = "cuda" if th.cuda.is_available() else "cpu",
) -> dict[str, Any]:
    """
    Main function to visualize token entropy across training.

    Args:
        baseline_dir: Directory containing baseline model checkpoints
        reasoning_dir: Directory containing reasoning model checkpoints
        dataset_name: Name of validation dataset to use
        dataset_split: Split of dataset to use
        token_ids_to_track: Specific token IDs to track (None = track all)
        max_samples: Maximum samples to process per checkpoint
        batch_size: Batch size for inference
        output_dir: Output directory for plots
        device: Device to run inference on

    Returns:
        Dictionary containing computed trajectories
    """
    if output_dir is None:
        output_dir = Path("fig")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    logger.info(f"Loading dataset: {dataset_name} ({dataset_split})")
    if dataset_name == "gsm8k":
        dataset = load_dataset("gsm8k", "main", split=dataset_split)
    elif dataset_name == "math":
        dataset = load_dataset("hendrycks/math", split=dataset_split)
    else:
        # Try loading as generic dataset
        dataset = load_dataset(dataset_name, split=dataset_split)

    # Load tokenizer from reasoning checkpoint (should be same for baseline)
    logger.info("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(reasoning_dir, trust_remote_code=True)

    # Find checkpoints
    baseline_checkpoints = []
    if baseline_dir is not None:
        baseline_checkpoints = sorted(list(baseline_dir.glob("checkpoint-*")))
        logger.info(f"Found {len(baseline_checkpoints)} baseline checkpoints")

    reasoning_checkpoints = sorted(list(reasoning_dir.glob("checkpoint-*")))
    logger.info(f"Found {len(reasoning_checkpoints)} reasoning checkpoints")

    # Compute trajectories
    results = {}

    if baseline_checkpoints:
        logger.info("\nComputing baseline trajectory...")
        baseline_trajectory = compute_trajectory_across_checkpoints(
            baseline_checkpoints,
            tokenizer,
            dataset,
            token_ids_to_track,
            max_samples,
            batch_size,
            device,
        )
        results["baseline"] = baseline_trajectory
    else:
        baseline_trajectory = {}
        results["baseline"] = {}

    logger.info("\nComputing reasoning model trajectory...")
    reasoning_trajectory = compute_trajectory_across_checkpoints(
        reasoning_checkpoints,
        tokenizer,
        dataset,
        token_ids_to_track,
        max_samples,
        batch_size,
        device,
    )
    results["reasoning"] = reasoning_trajectory

    # Determine which tokens to plot
    all_token_ids = set()
    for step_stats in reasoning_trajectory.values():
        all_token_ids.update(step_stats.keys())

    if token_ids_to_track is not None:
        plot_token_ids = [tid for tid in token_ids_to_track if tid in all_token_ids]
    else:
        plot_token_ids = list(all_token_ids)

    logger.info(f"\nPlotting trajectories for {len(plot_token_ids)} tokens")

    # Generate plots
    plot_token_entropy_trajectories(
        baseline_trajectory,
        reasoning_trajectory,
        plot_token_ids,
        tokenizer,
        output_dir,
    )

    logger.info("\nVisualization complete!")
    return results


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description="Visualize token output entropy over training")
    parser.add_argument(
        "--baseline-dir", type=Path, default=None, help="Baseline model checkpoint directory"
    )
    parser.add_argument(
        "--reasoning-dir",
        type=Path,
        required=True,
        help="Reasoning model checkpoint directory",
    )
    parser.add_argument("--dataset", type=str, default="gsm8k", help="Validation dataset name")
    parser.add_argument("--dataset-split", type=str, default="test", help="Dataset split")
    parser.add_argument(
        "--token-ids",
        type=int,
        nargs="+",
        default=None,
        help="Specific token IDs to track",
    )
    parser.add_argument(
        "--max-samples", type=int, default=1000, help="Max validation samples per checkpoint"
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Inference batch size")
    parser.add_argument("--output-dir", type=Path, default=Path("fig"), help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")

    args = parser.parse_args()

    visualize_token_entropy(
        baseline_dir=args.baseline_dir,
        reasoning_dir=args.reasoning_dir,
        dataset_name=args.dataset,
        dataset_split=args.dataset_split,
        token_ids_to_track=args.token_ids,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        device=args.device,
    )

    logger.info(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
