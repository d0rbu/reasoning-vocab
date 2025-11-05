"""
Dataset loading and preprocessing for math problem datasets.
"""

import re

from datasets import load_dataset


class MathDataset:
    """Wrapper for math problem datasets with standardized preprocessing."""

    def __init__(
        self,
        dataset_name: str,
        train_split: str = "train",
        val_split: str | None = None,
        prompt_template: str = "Solve: {problem}\n\nAnswer:",
        max_train_samples: int | None = None,
        max_val_samples: int | None = None,
    ):
        """
        Initialize math dataset.

        Args:
            dataset_name: HuggingFace dataset identifier
            train_split: Name of training split
            val_split: Name of validation split (can be None)
            prompt_template: Template for formatting prompts
            max_train_samples: Maximum number of training samples to use
            max_val_samples: Maximum number of validation samples to use
        """
        self.dataset_name = dataset_name
        self.prompt_template = prompt_template

        # Load training dataset
        print(f"Loading dataset: {dataset_name}")
        self.train_dataset = load_dataset(dataset_name, split=train_split)

        if max_train_samples is not None:
            self.train_dataset = self.train_dataset.select(range(max_train_samples))

        # Load validation dataset if specified
        self.val_dataset = None
        if val_split is not None:
            self.val_dataset = load_dataset(dataset_name, split=val_split)
            if max_val_samples is not None:
                self.val_dataset = self.val_dataset.select(range(max_val_samples))

        print(f"Loaded {len(self.train_dataset)} training examples")
        if self.val_dataset:
            print(f"Loaded {len(self.val_dataset)} validation examples")

    def preprocess_example(self, example: dict) -> dict:
        """
        Preprocess a single example into GRPO format.

        Args:
            example: Raw example from dataset with keys: problem, answer, solution

        Returns:
            Processed example with keys: query, ground_truth
        """
        # Format the prompt
        problem = example["problem"]
        query = self.prompt_template.format(problem=problem)

        # Extract ground truth answer
        ground_truth = example["answer"].strip()

        return {
            "query": query,
            "ground_truth": ground_truth,
        }

    def get_train_dataset(self):
        """Get preprocessed training dataset."""
        return self.train_dataset.map(
            self.preprocess_example,
            remove_columns=self.train_dataset.column_names,
            desc="Preprocessing training data",
        )

    def get_val_dataset(self):
        """Get preprocessed validation dataset."""
        if self.val_dataset is None:
            return None
        return self.val_dataset.map(
            self.preprocess_example,
            remove_columns=self.val_dataset.column_names,
            desc="Preprocessing validation data",
        )


def normalize_answer(answer: str) -> str:
    """
    Normalize an answer string for comparison.

    Args:
        answer: Raw answer string

    Returns:
        Normalized answer string
    """
    # Remove LaTeX formatting
    answer = answer.replace("\\", "")
    answer = answer.replace("{", "").replace("}", "")
    answer = answer.replace("$", "")

    # Remove extra whitespace
    answer = " ".join(answer.split())

    # Convert to lowercase for comparison
    answer = answer.lower().strip()

    return answer


def extract_answer_from_completion(completion: str) -> str:
    """
    Extract the final answer from a model completion.

    This function looks for common answer patterns:
    - "Answer: X"
    - "The answer is X"
    - "= X" (for math expressions)
    - Boxed answers: \\boxed{X}

    Args:
        completion: Model's generated text

    Returns:
        Extracted answer string
    """
    # Try to find boxed answer first (common in math problems)
    boxed_match = re.search(r"\\boxed\{([^}]+)\}", completion)
    if boxed_match:
        return boxed_match.group(1).strip()

    # Try "Answer:" pattern
    answer_match = re.search(r"(?:answer|final answer):\s*(.+?)(?:\n|$)", completion, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).strip()

    # Try "The answer is" pattern
    is_match = re.search(
        r"(?:the answer is|answer is)\s+(.+?)(?:\n|\.|\,|$)", completion, re.IGNORECASE
    )
    if is_match:
        return is_match.group(1).strip()

    # Try to find last equation with equals sign
    equation_matches = re.findall(r"=\s*([^\n=]+?)(?:\n|$)", completion)
    if equation_matches:
        return equation_matches[-1].strip()

    # If no pattern found, return last non-empty line
    lines = [line.strip() for line in completion.split("\n") if line.strip()]
    if lines:
        return lines[-1]

    return completion.strip()
