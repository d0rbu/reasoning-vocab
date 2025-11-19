"""
Sample reasoning sequences from a trained model.

This script loads a trained reasoning-vocab model and generates responses to queries,
tracking token multiplicities for visualization with visualize_reasoning.py.

Usage:
    python script/sample_reasoning.py \
        --checkpoint path/to/checkpoint \
        --dataset gsm8k \
        --num_samples 10 \
        --output_dir samples/

Example:
    python script/sample_reasoning.py \
        --checkpoint model/checkpoint-1000 \
        --dataset openai/gsm8k \
        --dataset_split test \
        --num_samples 50 \
        --output_dir samples/checkpoint-1000
"""

import argparse
import json
from pathlib import Path

import torch as th
from datasets import load_dataset
from loguru import logger
from transformers import AutoConfig, AutoTokenizer
from trl.rewards import accuracy_reward

from core.reasoning_vocab_model import ReasoningVocabLogitsProcessor, get_reasoning_class
from core.tokenizer_utils import ReasoningTokenizer


def load_model_and_tokenizer(
    checkpoint_path: str, device: str = "cuda"
) -> tuple[th.nn.Module, ReasoningTokenizer]:
    """
    Load a trained reasoning-vocab model and its tokenizer.

    Args:
        checkpoint_path: Path to model checkpoint directory
        device: Device to load model on

    Returns:
        Tuple of (model, reasoning_tokenizer)
    """
    logger.info(f"Loading model from {checkpoint_path}")

    # Load configuration and determine model class
    config = AutoConfig.from_pretrained(checkpoint_path)
    model_class_name = config.model_type

    # Get the appropriate reasoning model class
    from transformers import AutoModelForCausalLM

    base_model_class = AutoModelForCausalLM.from_config(config).__class__
    reasoning_model_class = get_reasoning_class(base_model_class)

    # Load the model
    model = reasoning_model_class.from_pretrained(checkpoint_path)
    model = model.to(device)
    model.eval()

    # Load tokenizer
    base_tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    # Get reasoning token IDs from model
    reasoning_token_ids = model.get_reasoning_token_ids()
    reasoning_tokenizer = ReasoningTokenizer(
        tokenizer=base_tokenizer, reasoning_token_ids=reasoning_token_ids
    )

    logger.info(f"Loaded {model_class_name} with {model.num_reasoning_tokens()} reasoning tokens")

    return model, reasoning_tokenizer


def prepare_dataset(dataset_name: str, split: str, num_samples: int):
    """
    Load and prepare a dataset for sampling.

    Args:
        dataset_name: HuggingFace dataset name (e.g., 'openai/gsm8k')
        split: Dataset split to use (e.g., 'test', 'train')
        num_samples: Number of samples to take

    Returns:
        Dataset subset
    """
    logger.info(f"Loading dataset: {dataset_name} (split: {split})")

    dataset = load_dataset(dataset_name, split=split)

    # Take only num_samples
    if num_samples < len(dataset):
        dataset = dataset.select(range(num_samples))

    return dataset


def format_prompt(example: dict, dataset_name: str) -> str:
    """
    Format a dataset example into a prompt.

    Args:
        example: Dataset example
        dataset_name: Name of the dataset for format selection

    Returns:
        Formatted prompt string
    """
    # GSM8K format
    if "gsm8k" in dataset_name.lower():
        return f"<|im_start|>user\n{example['question']}<|im_end|>\n<|im_start|>assistant\n"

    # MATH format
    if "math" in dataset_name.lower():
        return f"<|im_start|>user\n{example['problem']}<|im_end|>\n<|im_start|>assistant\n"

    # Default format
    if "question" in example:
        return f"<|im_start|>user\n{example['question']}<|im_end|>\n<|im_start|>assistant\n"

    raise ValueError(f"Unknown dataset format for: {dataset_name}")


def extract_answer(example: dict, dataset_name: str) -> str:
    """
    Extract the ground truth answer from a dataset example.

    Args:
        example: Dataset example
        dataset_name: Name of the dataset

    Returns:
        Ground truth answer string
    """
    # GSM8K format
    if "gsm8k" in dataset_name.lower():
        # GSM8K answers are formatted as "#### <answer>"
        answer = example["answer"]
        if "####" in answer:
            return answer.split("####")[-1].strip()
        return answer.strip()

    # MATH format
    if "math" in dataset_name.lower():
        return example["solution"].strip()

    # Default
    if "answer" in example:
        return example["answer"].strip()

    raise ValueError(f"Unknown answer format for: {dataset_name}")


def generate_sample(
    model: th.nn.Module,
    tokenizer: ReasoningTokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    do_sample: bool = True,
) -> tuple[list[int], str, list[int]]:
    """
    Generate a response from the model with reasoning token tracking.

    Args:
        model: Reasoning-vocab model
        tokenizer: ReasoningTokenizer
        prompt: Input prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        do_sample: Whether to sample (vs greedy decoding)

    Returns:
        Tuple of (token_ids, decoded_text, multiplicities)
    """
    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

    # Create logits processor for reasoning vocabulary
    logits_processor = ReasoningVocabLogitsProcessor(
        standard_vocab_size=model.standard_vocab_size,
        tokenizer=tokenizer,
        think_tag="<think>",
        end_think_tag="</think>",
    )

    # Generate
    with th.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=tokenizer.tokenizer.eos_token_id,
            logits_processor=[logits_processor],
        )

    # Extract generated tokens (remove prompt)
    generated_ids = output_ids[0, input_ids.shape[1] :].tolist()

    # Decode with multiplicity
    decoded_text, multiplicity_infos = tokenizer.decode_with_multiplicity(generated_ids)

    # Extract just the multiplicity values
    multiplicities = [info.multiplicity for info in multiplicity_infos]

    return generated_ids, decoded_text, multiplicities


def check_correctness(generated_text: str, ground_truth: str, dataset_name: str) -> bool:
    """
    Check if generated answer is correct.

    Args:
        generated_text: Model's generated response
        ground_truth: Ground truth answer
        dataset_name: Dataset name for format-specific checking

    Returns:
        True if correct, False otherwise
    """
    # Use TRL's accuracy reward function
    try:
        # Format as completion for accuracy_reward
        completions = [[{"role": "assistant", "content": generated_text}]]
        prompts = [[{"role": "user", "content": ""}]]  # Dummy prompt
        references = [[{"role": "assistant", "content": ground_truth}]]

        rewards = accuracy_reward(completions, prompts, references)
        return float(rewards[0]) > 0.5
    except Exception as e:
        logger.warning(f"Error checking correctness: {e}")
        # Fallback: simple string matching
        return ground_truth.lower() in generated_text.lower()


def save_sample(
    output_dir: Path,
    sample_idx: int,
    prompt: str,
    token_ids: list[int],
    decoded_text: str,
    multiplicities: list[int],
    ground_truth: str,
    is_correct: bool,
    was_truncated: bool,
) -> None:
    """
    Save a generated sample to disk.

    Args:
        output_dir: Directory to save samples
        sample_idx: Sample index
        prompt: Input prompt
        token_ids: Generated token IDs
        decoded_text: Decoded text
        multiplicities: Token multiplicities
        ground_truth: Ground truth answer
        is_correct: Whether answer is correct
        was_truncated: Whether generation was truncated
    """
    sample_data = {
        "sample_idx": sample_idx,
        "prompt": prompt,
        "token_ids": token_ids,
        "decoded_text": decoded_text,
        "multiplicities": multiplicities,
        "ground_truth": ground_truth,
        "is_correct": is_correct,
        "was_truncated": was_truncated,
    }

    output_file = output_dir / f"sample_{sample_idx:04d}.json"
    with open(output_file, "w") as f:
        json.dump(sample_data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Sample reasoning sequences from a trained model")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint directory"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="openai/gsm8k",
        help="HuggingFace dataset name (default: openai/gsm8k)",
    )
    parser.add_argument(
        "--dataset_split", type=str, default="test", help="Dataset split (default: test)"
    )
    parser.add_argument(
        "--num_samples", type=int, default=10, help="Number of samples to generate (default: 10)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="samples",
        help="Output directory for samples (default: samples)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate (default: 512)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature (default: 0.7)"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (default: cuda)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args()

    # Set seed
    th.manual_seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output directory: {output_dir}")

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.checkpoint, device=args.device)

    # Load dataset
    dataset = prepare_dataset(args.dataset, args.dataset_split, args.num_samples)

    # Generate samples
    logger.info(f"Generating {args.num_samples} samples...")

    correct_count = 0
    truncated_count = 0

    for idx, example in enumerate(dataset):
        logger.info(f"Sample {idx + 1}/{args.num_samples}")

        # Format prompt and get ground truth
        prompt = format_prompt(example, args.dataset)
        ground_truth = extract_answer(example, args.dataset)

        # Generate
        token_ids, decoded_text, multiplicities = generate_sample(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

        # Check correctness
        is_correct = check_correctness(decoded_text, ground_truth, args.dataset)
        if is_correct:
            correct_count += 1

        # Check if truncated (ends with EOS token)
        was_truncated = len(token_ids) >= args.max_new_tokens
        if was_truncated:
            truncated_count += 1

        # Save sample
        save_sample(
            output_dir=output_dir,
            sample_idx=idx,
            prompt=prompt,
            token_ids=token_ids,
            decoded_text=decoded_text,
            multiplicities=multiplicities,
            ground_truth=ground_truth,
            is_correct=is_correct,
            was_truncated=was_truncated,
        )

        logger.info(f"  Correct: {is_correct}, Truncated: {was_truncated}")

    # Summary
    accuracy = correct_count / args.num_samples
    truncation_rate = truncated_count / args.num_samples

    summary = {
        "num_samples": args.num_samples,
        "correct_count": correct_count,
        "accuracy": accuracy,
        "truncated_count": truncated_count,
        "truncation_rate": truncation_rate,
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
    }

    summary_file = output_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n{'=' * 50}")
    logger.info("SUMMARY")
    logger.info(f"{'=' * 50}")
    logger.info(f"Accuracy: {accuracy:.2%} ({correct_count}/{args.num_samples})")
    logger.info(f"Truncation rate: {truncation_rate:.2%} ({truncated_count}/{args.num_samples})")
    logger.info(f"Samples saved to: {output_dir}")
    logger.info(f"{'=' * 50}")


if __name__ == "__main__":
    main()
