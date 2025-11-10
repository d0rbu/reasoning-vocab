"""
Visualize token entropy over training.

This script computes the entropy of output distributions when processing tokens,
tracking how this evolves during training. It:
1. Runs validation data through model checkpoints
2. Computes entropy of output logits for each token position
3. Groups entropies by input token
4. Tracks average entropy per token across training steps
5. Compares baseline, standard, and reasoning token behavior
"""

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch as th
from datasets import load_dataset
from loguru import logger
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def compute_output_entropy(logits: th.Tensor) -> float:
    """
    Compute entropy of output distribution from logits.

    Args:
        logits: 1D tensor of logits for vocabulary

    Returns:
        Entropy value in nats (natural logarithm base)
    """
    # Convert logits to probability distribution via softmax
    probs = th.softmax(logits, dim=-1)

    # Compute Shannon entropy: -sum(p * log(p))
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    entropy = -th.sum(probs * th.log(probs + epsilon))

    return entropy.item()


def process_checkpoint_for_token_entropies(
    checkpoint_path: Path,
    tokenizer: AutoTokenizer,
    dataset: list[str],
    max_samples: int | None = None,
    max_length: int = 512,
) -> dict[int, list[float]]:
    """
    Process a checkpoint to compute token entropies on validation data.

    For each sample in the dataset:
    - Tokenize and run through model
    - For each token position, compute entropy of output distribution
    - Store entropy grouped by the input token at that position

    Args:
        checkpoint_path: Path to model checkpoint
        tokenizer: Tokenizer for the model
        dataset: List of text samples to process
        max_samples: Maximum number of samples to process (None = all)
        max_length: Maximum sequence length

    Returns:
        Dictionary mapping token_id -> list of entropy values observed for that token
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=th.float32,
        trust_remote_code=True,
        device_map="cpu",  # Use CPU for consistency
    )
    model.eval()

    # Track entropies by token
    token_entropies: dict[int, list[float]] = defaultdict(list)

    # Process samples
    samples_to_process = dataset[:max_samples] if max_samples else dataset

    with th.no_grad():
        for text in tqdm(samples_to_process, desc="Processing samples"):
            # Tokenize
            inputs = tokenizer(  # type: ignore[misc]
                text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
            )

            input_ids = inputs["input_ids"][0]  # Shape: [seq_len]

            # Run forward pass
            outputs = model(**inputs)
            logits = outputs.logits[0]  # Shape: [seq_len, vocab_size]

            # For each token position (except last, which has no next token to predict)
            for pos in range(len(input_ids) - 1):
                token_id = input_ids[pos].item()
                # Entropy of distribution when THIS token is the input
                output_logits = logits[pos]  # Shape: [vocab_size]
                entropy = compute_output_entropy(output_logits)

                token_entropies[token_id].append(entropy)

    logger.info(
        f"Processed {len(samples_to_process)} samples, "
        f"tracked entropies for {len(token_entropies)} unique tokens"
    )

    return dict(token_entropies)


def compute_token_entropy_trajectory(
    checkpoint_dirs: list[Path],
    tokenizer: AutoTokenizer,
    dataset: list[str],
    token_ids: list[int],
    max_samples: int | None = None,
) -> dict[int, dict[str, list[float]]]:
    """
    Compute entropy trajectory for specific tokens across checkpoints.

    Args:
        checkpoint_dirs: List of checkpoint directories in training order
        tokenizer: Tokenizer for the model
        dataset: Validation dataset
        token_ids: List of token IDs to track
        max_samples: Maximum samples per checkpoint

    Returns:
        Dictionary mapping token_id -> {"steps": [...], "entropies": [...]}
    """
    results: dict[int, dict[str, list[float]]] = {
        token_id: {"steps": [], "entropies": []} for token_id in token_ids
    }

    for checkpoint_dir in sorted(checkpoint_dirs):
        # Extract step number
        step_str = checkpoint_dir.name.split("-")[-1]
        try:
            step = int(step_str)
        except ValueError:
            logger.warning(f"Could not parse step from {checkpoint_dir.name}, skipping")
            continue

        try:
            # Process checkpoint
            token_entropies = process_checkpoint_for_token_entropies(
                checkpoint_dir, tokenizer, dataset, max_samples
            )

            # Record average entropy for each tracked token
            for token_id in token_ids:
                if token_id in token_entropies and token_entropies[token_id]:
                    avg_entropy = float(np.mean(token_entropies[token_id]))
                    results[token_id]["steps"].append(step)
                    results[token_id]["entropies"].append(avg_entropy)

                    logger.info(
                        f"Token {token_id} at step {step}: "
                        f"avg entropy = {avg_entropy:.4f} "
                        f"(from {len(token_entropies[token_id])} observations)"
                    )
                else:
                    logger.warning(f"Token {token_id} not found in checkpoint {step}")

        except Exception as e:
            logger.error(f"Error processing checkpoint {checkpoint_dir}: {e}")
            continue

    return results


def plot_token_entropy_comparison(
    baseline_data: dict[int, dict[str, list[float]]],
    reasoning_data: dict[int, dict[str, list[float]]],
    tokenizer: AutoTokenizer,
    output_path: Path,
) -> None:
    """
    Create comparison plot of token entropy trajectories.

    Args:
        baseline_data: Token entropy data from baseline model
        reasoning_data: Token entropy data from reasoning vocab model
        tokenizer: Tokenizer to decode token strings
        output_path: Path to save the figure
    """
    # Get all token IDs from both datasets
    all_token_ids = sorted(set(baseline_data.keys()) | set(reasoning_data.keys()))

    if not all_token_ids:
        logger.warning("No token data to plot")
        return

    fig, axes = plt.subplots(
        len(all_token_ids), 1, figsize=(10, 4 * len(all_token_ids)), squeeze=False
    )

    for idx, token_id in enumerate(all_token_ids):
        ax = axes[idx, 0]

        # Get token string
        token_str = tokenizer.decode([token_id])  # type: ignore[misc]

        # Plot baseline
        if token_id in baseline_data and baseline_data[token_id]["steps"]:
            ax.plot(  # type: ignore[misc]
                baseline_data[token_id]["steps"],
                baseline_data[token_id]["entropies"],
                marker="o",
                label="Baseline Model",
                linewidth=2,
                markersize=6,
            )

        # Plot reasoning vocab model
        if token_id in reasoning_data and reasoning_data[token_id]["steps"]:
            ax.plot(  # type: ignore[misc]
                reasoning_data[token_id]["steps"],
                reasoning_data[token_id]["entropies"],
                marker="s",
                label="Reasoning Vocab Model",
                linewidth=2,
                markersize=6,
            )

        ax.set_xlabel("Training Step", fontsize=11)  # type: ignore[misc]
        ax.set_ylabel("Average Output Entropy (nats)", fontsize=11)  # type: ignore[misc]
        ax.set_title(f'Token: "{token_str}" (ID: {token_id})', fontsize=12)  # type: ignore[misc]
        ax.legend(fontsize=10)  # type: ignore[misc]
        ax.grid(True, alpha=0.3)  # type: ignore[misc]

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved plot to {output_path}")
    plt.close()


def visualize_token_entropy(
    baseline_dir: Path | None,
    reasoning_dir: Path,
    token_ids: list[int],
    dataset_name: str = "openai/gsm8k",
    dataset_split: str = "test",
    max_samples: int = 100,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Main function to visualize token entropy trajectories.

    Args:
        baseline_dir: Directory containing baseline model checkpoints
        reasoning_dir: Directory containing reasoning vocab model checkpoints
        token_ids: List of token IDs to track
        dataset_name: HuggingFace dataset name
        dataset_split: Dataset split to use
        max_samples: Maximum number of samples to process per checkpoint
        output_dir: Directory to save plots (default: ./fig)

    Returns:
        Dictionary containing computed entropy trajectories
    """
    if output_dir is None:
        output_dir = Path("fig")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer (use from reasoning dir, should be same as baseline)
    logger.info("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(reasoning_dir, trust_remote_code=True)

    # Load validation dataset
    logger.info(f"Loading dataset {dataset_name} ({dataset_split})")
    dataset = load_dataset(dataset_name, split=dataset_split)

    # Extract text field (adapt based on dataset structure)
    if "question" in dataset.column_names:  # type: ignore[misc]
        texts = [sample["question"] for sample in dataset]  # type: ignore[misc]
    elif "text" in dataset.column_names:  # type: ignore[misc]
        texts = [sample["text"] for sample in dataset]  # type: ignore[misc]
    else:
        raise ValueError(f"Unknown dataset structure: {dataset.column_names}")

    logger.info(f"Loaded {len(texts)} samples from dataset")

    # Process baseline model
    baseline_data = {}
    if baseline_dir is not None:
        logger.info("\n=== Processing Baseline Model ===")
        baseline_checkpoints = sorted(list(baseline_dir.glob("checkpoint-*")))
        logger.info(f"Found {len(baseline_checkpoints)} baseline checkpoints")

        baseline_data = compute_token_entropy_trajectory(
            baseline_checkpoints, tokenizer, texts, token_ids, max_samples
        )

    # Process reasoning vocab model
    logger.info("\n=== Processing Reasoning Vocab Model ===")
    reasoning_checkpoints = sorted(list(reasoning_dir.glob("checkpoint-*")))
    logger.info(f"Found {len(reasoning_checkpoints)} reasoning checkpoints")

    reasoning_data = compute_token_entropy_trajectory(
        reasoning_checkpoints, tokenizer, texts, token_ids, max_samples
    )

    # Create plots
    if baseline_dir is not None:
        output_path = output_dir / "token_entropy_comparison.png"
        plot_token_entropy_comparison(baseline_data, reasoning_data, tokenizer, output_path)
    else:
        # Plot just reasoning model
        output_path = output_dir / "token_entropy_reasoning.png"
        plot_token_entropy_comparison({}, reasoning_data, tokenizer, output_path)

    return {
        "baseline": baseline_data,
        "reasoning": reasoning_data,
        "token_ids": token_ids,
    }


def main():
    """Command-line interface for token entropy visualization."""
    parser = argparse.ArgumentParser(description="Visualize token entropy over training")
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
        "--token-ids",
        type=int,
        nargs="+",
        required=True,
        help="Token IDs to track (space-separated)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="openai/gsm8k",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="test",
        help="Dataset split to use",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
        help="Maximum samples to process per checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("fig"),
        help="Directory to save output plots",
    )

    args = parser.parse_args()

    # Run visualization
    results = visualize_token_entropy(
        baseline_dir=args.baseline_dir,
        reasoning_dir=args.reasoning_dir,
        token_ids=args.token_ids,
        dataset_name=args.dataset,
        dataset_split=args.dataset_split,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
    )

    logger.info("\nVisualization complete!")
    logger.info(f"Tracked {len(results['token_ids'])} tokens")
    logger.info(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
