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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch as th
from datasets import load_dataset
from loguru import logger
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase


@dataclass
class TokenEntropyTrajectory:
    """Stores entropy trajectory data for a token across training steps."""

    steps: list[int]
    entropies: list[float]


def compute_output_entropy(logits: th.Tensor, vocab_size: int, dim: int = -1) -> th.Tensor:
    """
    Compute entropy of output distributions from logits (vectorized).

    Args:
        logits: 2D tensor of shape [seq_len, vocab_size] containing logits
        vocab_size: Size of vocabulary
        dim: Dimension along which to compute softmax (default: -1)

    Returns:
        1D tensor of shape [seq_len] containing entropy values in nats
    """
    # Convert logits to probability distribution via softmax
    probs = th.softmax(logits, dim=dim)

    # Compute Shannon entropy: -sum(p * log(p))
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    entropy = -th.sum(probs * th.log(probs + epsilon), dim=dim)

    return entropy


def compute_token_entropies_from_model(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    dataset: list[str],
    vocab_size: int,
    max_samples: int | None = None,
    max_length: int = 512,
    batch_size: int = 8,
    device: str | None = None,
) -> th.Tensor:
    """
    Compute average token entropies on validation data (vectorized, batched).

    For each sample in the dataset:
    - Tokenize and run through model in batches
    - Compute entropy of output distributions for all positions at once
    - Use scatter operations to aggregate entropies by token ID

    Args:
        model: The model to evaluate (must be in eval mode)
        tokenizer: Tokenizer for the model
        dataset: List of text samples to process
        vocab_size: Size of the vocabulary
        max_samples: Maximum number of samples to process (None = all)
        max_length: Maximum sequence length
        batch_size: Batch size for processing
        device: Device to use (None = auto-detect)

    Returns:
        1D tensor of shape [vocab_size] containing average entropy per token
    """
    if device is None:
        device = "cuda" if th.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()

    # Initialize accumulators for scatter operations
    entropy_sum = th.zeros(vocab_size, dtype=th.float32, device=device)
    token_count = th.zeros(vocab_size, dtype=th.int64, device=device)

    # Process samples
    samples_to_process = dataset[:max_samples] if max_samples else dataset
    num_batches = (len(samples_to_process) + batch_size - 1) // batch_size

    with th.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
            # Get batch
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(samples_to_process))
            batch_texts = samples_to_process[start_idx:end_idx]

            # Tokenize batch
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Run forward pass
            outputs = model(**inputs)
            logits = outputs.logits  # Shape: [batch_size, seq_len, vocab_size]

            # Compute entropies for all positions at once (vectorized)
            # logits shape: [batch_size, seq_len, vocab_size]
            batch_size_actual, seq_len, _ = logits.shape
            logits_2d = logits.view(-1, vocab_size)  # [batch_size * seq_len, vocab_size]
            entropies = compute_output_entropy(
                logits_2d, vocab_size, dim=-1
            )  # [batch_size * seq_len]
            entropies = entropies.view(batch_size_actual, seq_len)  # [batch_size, seq_len]

            # Get input token IDs (exclude last position as it has no next token)
            input_ids = inputs["input_ids"]  # [batch_size, seq_len]

            # Flatten for scatter operations (exclude last position)
            # We want entropies[i, j] to correspond to input_ids[i, j]
            for i in range(batch_size_actual):
                # Get valid sequence length (excluding padding and last token)
                attention_mask = inputs["attention_mask"][i]
                valid_length = attention_mask.sum().item()

                if valid_length > 1:  # Need at least 2 tokens
                    # Exclude last position (no next token to predict)
                    valid_tokens = input_ids[i, : valid_length - 1]  # [valid_length - 1]
                    valid_entropies = entropies[i, : valid_length - 1]  # [valid_length - 1]

                    # Scatter add entropies to accumulator
                    entropy_sum.scatter_add_(0, valid_tokens, valid_entropies)

                    # Scatter add counts (ones)
                    ones = th.ones_like(valid_tokens, dtype=th.int64)
                    token_count.scatter_add_(0, valid_tokens, ones)

    # Compute average entropy per token (avoid division by zero)
    avg_entropy = th.where(
        token_count > 0, entropy_sum / token_count.float(), th.zeros_like(entropy_sum)
    )

    logger.info(
        f"Processed {len(samples_to_process)} samples, "
        f"tracked entropies for {(token_count > 0).sum().item()} unique tokens"
    )

    return avg_entropy


def compute_token_entropy_trajectory(
    checkpoint_dirs: list[Path],
    tokenizer: PreTrainedTokenizerBase,
    dataset: list[str],
    vocab_size: int,
    token_ids: list[int],
    max_samples: int | None = None,
    batch_size: int = 8,
    device: str | None = None,
) -> dict[int, TokenEntropyTrajectory]:
    """
    Compute entropy trajectory for specific tokens across checkpoints.

    Args:
        checkpoint_dirs: List of checkpoint directories in training order
        tokenizer: Tokenizer for the model
        dataset: Validation dataset
        vocab_size: Size of vocabulary
        token_ids: List of token IDs to track
        max_samples: Maximum samples per checkpoint
        batch_size: Batch size for processing
        device: Device to use (None = auto-detect)

    Returns:
        Dictionary mapping token_id -> TokenEntropyTrajectory
    """
    results: dict[int, TokenEntropyTrajectory] = {
        token_id: TokenEntropyTrajectory(steps=[], entropies=[]) for token_id in token_ids
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
            logger.info(f"Loading checkpoint from {checkpoint_dir}")

            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint_dir,
                torch_dtype=th.float32,
                trust_remote_code=True,
            )

            # Compute average entropies for all tokens
            avg_entropies = compute_token_entropies_from_model(
                model,
                tokenizer,
                dataset,
                vocab_size,
                max_samples,
                batch_size=batch_size,
                device=device,
            )

            # Record entropy for each tracked token
            for token_id in token_ids:
                entropy_value = avg_entropies[token_id].item()

                if entropy_value > 0:  # Token was observed
                    results[token_id].steps.append(step)
                    results[token_id].entropies.append(entropy_value)

                    logger.info(
                        f"Token {token_id} at step {step}: avg entropy = {entropy_value:.4f}"
                    )
                else:
                    logger.warning(f"Token {token_id} not found in checkpoint {step}")

            # Clean up model to free memory
            del model
            th.cuda.empty_cache() if th.cuda.is_available() else None

        except Exception as e:
            logger.error(f"Error processing checkpoint {checkpoint_dir}: {e}")
            continue

    return results


def plot_token_entropy_comparison(
    baseline_data: dict[int, TokenEntropyTrajectory],
    reasoning_data: dict[int, TokenEntropyTrajectory],
    tokenizer: PreTrainedTokenizerBase,
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
        ax = axes[idx, 0]  # pyright: ignore[reportIndexIssue]

        # Get token string
        token_str = tokenizer.decode([token_id])

        # Plot baseline
        if token_id in baseline_data and baseline_data[token_id].steps:
            trajectory = baseline_data[token_id]
            ax.plot(  # pyright: ignore[reportAttributeAccessIssue]
                trajectory.steps,
                trajectory.entropies,
                marker="o",
                label="Baseline Model",
                linewidth=2,
                markersize=6,
            )

        # Plot reasoning vocab model
        if token_id in reasoning_data and reasoning_data[token_id].steps:
            trajectory = reasoning_data[token_id]
            ax.plot(  # pyright: ignore[reportAttributeAccessIssue]
                trajectory.steps,
                trajectory.entropies,
                marker="s",
                label="Reasoning Vocab Model",
                linewidth=2,
                markersize=6,
            )

        ax.set_xlabel("Training Step", fontsize=11)  # pyright: ignore[reportAttributeAccessIssue]
        ax.set_ylabel("Average Output Entropy (nats)", fontsize=11)  # pyright: ignore[reportAttributeAccessIssue]
        ax.set_title(f'Token: "{token_str}" (ID: {token_id})', fontsize=12)  # pyright: ignore[reportAttributeAccessIssue]
        ax.legend(fontsize=10)  # pyright: ignore[reportAttributeAccessIssue]
        ax.grid(True, alpha=0.3)  # pyright: ignore[reportAttributeAccessIssue]

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
    vocab_size = len(tokenizer)

    # Load validation dataset
    logger.info(f"Loading dataset {dataset_name} ({dataset_split})")
    dataset = load_dataset(dataset_name, split=dataset_split)

    # Extract text field (adapt based on dataset structure)
    column_names = dataset.column_names  # pyright: ignore[reportAttributeAccessIssue]
    if "question" in column_names:
        texts = [sample["question"] for sample in dataset]  # pyright: ignore[reportIndexIssue]
    elif "text" in column_names:
        texts = [sample["text"] for sample in dataset]  # pyright: ignore[reportIndexIssue]
    else:
        raise ValueError(f"Unknown dataset structure: {column_names}")

    logger.info(f"Loaded {len(texts)} samples from dataset")

    # Process baseline model
    baseline_data: dict[int, TokenEntropyTrajectory] = {}
    if baseline_dir is not None:
        logger.info("\n=== Processing Baseline Model ===")
        baseline_checkpoints = sorted(list(baseline_dir.glob("checkpoint-*")))
        logger.info(f"Found {len(baseline_checkpoints)} baseline checkpoints")

        baseline_data = compute_token_entropy_trajectory(
            baseline_checkpoints, tokenizer, texts, vocab_size, token_ids, max_samples
        )

    # Process reasoning vocab model
    logger.info("\n=== Processing Reasoning Vocab Model ===")
    reasoning_checkpoints = sorted(list(reasoning_dir.glob("checkpoint-*")))
    logger.info(f"Found {len(reasoning_checkpoints)} reasoning checkpoints")

    reasoning_data = compute_token_entropy_trajectory(
        reasoning_checkpoints, tokenizer, texts, vocab_size, token_ids, max_samples
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
