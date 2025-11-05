"""
Reward functions for verifying math problem correctness.
"""

from typing import Any

import sympy
from sympy.parsing.latex import parse_latex

from .dataset import extract_answer_from_completion, normalize_answer


def compare_answers(generated: str, ground_truth: str) -> bool:
    """
    Compare two answers for mathematical equivalence.

    Tries multiple comparison methods:
    1. Exact string match (after normalization)
    2. Numeric comparison (with tolerance)
    3. Symbolic comparison using sympy

    Args:
        generated: Generated answer from model
        ground_truth: Ground truth answer

    Returns:
        True if answers match, False otherwise
    """
    # Normalize both answers
    gen_norm = normalize_answer(generated)
    gt_norm = normalize_answer(ground_truth)

    # Try exact match first
    if gen_norm == gt_norm:
        return True

    # Try numeric comparison
    try:
        gen_num = float(gen_norm.replace(",", ""))
        gt_num = float(gt_norm.replace(",", ""))
        if abs(gen_num - gt_num) < 1e-6:
            return True
    except (ValueError, AttributeError):
        pass

    # Try symbolic comparison with sympy
    try:
        gen_expr = sympy.sympify(gen_norm)
        gt_expr = sympy.sympify(gt_norm)
        if sympy.simplify(gen_expr - gt_expr) == 0:
            return True
    except (sympy.SympifyError, TypeError, AttributeError, ValueError):
        pass

    # Try parsing as LaTeX
    try:
        gen_expr = parse_latex(generated)
        gt_expr = parse_latex(ground_truth)
        if sympy.simplify(gen_expr - gt_expr) == 0:
            return True
    except Exception:
        pass

    return False


def math_correctness_reward(
    completions: list[list[dict[str, Any]]], ground_truths: list[str], **kwargs
) -> list[float]:
    """
    Reward function for math problem correctness.

    This function is designed to work with TRL's GRPOTrainer.

    Args:
        completions: List of completion batches, where each batch is a list of
                    message dicts with 'role' and 'content' keys
        ground_truths: List of ground truth answers
        **kwargs: Additional keyword arguments (unused)

    Returns:
        List of reward scores (1.0 for correct, 0.0 for incorrect)
    """
    rewards = []

    for completion_batch, ground_truth in zip(completions, ground_truths, strict=False):
        # Extract the assistant's response
        # completion_batch is a list of messages, we want the last assistant message
        assistant_message = None
        for msg in reversed(completion_batch):
            if msg.get("role") == "assistant":
                assistant_message = msg.get("content", "")
                break

        if assistant_message is None:
            # No assistant response found, give 0 reward
            rewards.append(0.0)
            continue

        # Extract answer from completion
        extracted_answer = extract_answer_from_completion(assistant_message)

        # Compare with ground truth
        is_correct = compare_answers(extracted_answer, ground_truth)

        # Binary reward: 1.0 for correct, 0.0 for incorrect
        reward = 1.0 if is_correct else 0.0
        rewards.append(reward)

    return rewards


def create_reward_function(dataset):
    """
    Create a reward function closure with access to ground truth answers.

    Args:
        dataset: Preprocessed dataset with 'ground_truth' field

    Returns:
        Reward function compatible with TRL's GRPOTrainer
    """
    # Extract ground truths into a list
    ground_truths = [example["ground_truth"] for example in dataset]

    # Create closure that captures ground_truths
    def reward_fn(completions: list[list[dict[str, Any]]], **kwargs) -> list[float]:
        """
        Reward function with access to ground truth answers.

        Args:
            completions: List of completion batches from GRPOTrainer
            **kwargs: Additional arguments (e.g., prompt_index)

        Returns:
            List of reward scores
        """
        # If prompt indices are provided, use them to look up ground truths
        # Otherwise, assume completions are in the same order as dataset
        if "prompt_indices" in kwargs:
            indices = kwargs["prompt_indices"]
            relevant_gts = [ground_truths[i] for i in indices]
        else:
            # Assume sequential processing
            relevant_gts = ground_truths[: len(completions)]

        return math_correctness_reward(completions, relevant_gts)

    return reward_fn
