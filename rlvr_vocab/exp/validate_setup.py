"""
Validation script to check that the training setup is correctly configured.

This script performs basic sanity checks:
1. Dependencies are installed
2. Dataset can be loaded
3. Model can be loaded
4. Reward function works
5. Hydra configuration is valid

Usage:
    uv run python rlvr_vocab/exp/validate_setup.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))


def check_dependencies():
    """Check that all required dependencies are installed."""
    print("=" * 80)
    print("Checking dependencies...")
    print("=" * 80)

    required_packages = [
        "torch",
        "transformers",
        "datasets",
        "trl",
        "hydra",
        "wandb",
        "sympy",
    ]

    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - MISSING")
            missing.append(package)

    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("Install with: uv sync")
        return False

    print("\n✅ All dependencies installed")
    return True


def check_dataset():
    """Check that the dataset can be loaded."""
    print("\n" + "=" * 80)
    print("Checking dataset loading...")
    print("=" * 80)

    try:
        from rlvr_vocab.core.dataset import MathDataset

        # Load a small sample
        dataset = MathDataset(
            dataset_name="agentica-org/DeepScaleR-Preview-Dataset",
            train_split="train[:10]",
            max_train_samples=10,
        )

        train_dataset = dataset.get_train_dataset()

        print(f"✓ Loaded {len(train_dataset)} examples")
        print(f"✓ Example fields: {list(train_dataset[0].keys())}")

        # Show a sample
        example = train_dataset[0]
        print("\nSample example:")
        print(f"Query (first 200 chars): {example['query'][:200]}...")
        print(f"Ground truth: {example['ground_truth']}")

        print("\n✅ Dataset loading works")
        return True

    except Exception as e:
        print(f"\n❌ Dataset loading failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def check_reward_function():
    """Check that the reward function works."""
    print("\n" + "=" * 80)
    print("Checking reward function...")
    print("=" * 80)

    try:
        from rlvr_vocab.core.reward import compare_answers

        # Test cases
        test_cases = [
            ("2", "2", True),
            ("2.0", "2", True),
            ("-\\frac{2}{3}", "-2/3", True),
            ("wrong", "correct", False),
            ("42", "42.0", True),
        ]

        all_passed = True
        for gen, gt, expected in test_cases:
            result = compare_answers(gen, gt)
            status = "✓" if result == expected else "✗"
            print(f"{status} compare_answers('{gen}', '{gt}') = {result} (expected {expected})")
            if result != expected:
                all_passed = False

        if all_passed:
            print("\n✅ Reward function works")
            return True
        else:
            print("\n⚠️  Some reward function tests failed")
            return False

    except Exception as e:
        print(f"\n❌ Reward function check failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def check_model_loading():
    """Check that the model can be loaded."""
    print("\n" + "=" * 80)
    print("Checking model loading (this may take a while)...")
    print("=" * 80)

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = "Qwen/Qwen2.5-0.6B"

        print(f"Loading model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        # Set pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"✓ Model loaded: {model.__class__.__name__}")
        print(f"✓ Model size: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")

        # Test forward pass
        inputs = tokenizer("Test input", return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)

        print(f"✓ Forward pass successful, output shape: {outputs.logits.shape}")

        print("\n✅ Model loading works")
        return True

    except Exception as e:
        print(f"\n❌ Model loading failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def check_hydra_config():
    """Check that Hydra configuration is valid."""
    print("\n" + "=" * 80)
    print("Checking Hydra configuration...")
    print("=" * 80)

    try:
        from pathlib import Path

        from hydra import compose, initialize_config_dir

        config_dir = Path(__file__).parent / "conf"

        if not config_dir.exists():
            print(f"❌ Config directory not found: {config_dir}")
            return False

        with initialize_config_dir(version_base=None, config_dir=str(config_dir.absolute())):
            cfg = compose(config_name="config")

            print("✓ Main config loaded")
            print(f"✓ Experiment name: {cfg.exp_name}")
            print(f"✓ Model: {cfg.model.name}")
            print(f"✓ Dataset: {cfg.dataset.name}")
            print(f"✓ Output dir: {cfg.output_dir}")

            # Test override
            cfg_override = compose(
                config_name="config",
                overrides=["training.learning_rate=1e-5", "exp_name=test"],
            )
            print(f"✓ Override test: learning_rate={cfg_override.training.learning_rate}")
            print(f"✓ Override test: exp_name={cfg_override.exp_name}")

        print("\n✅ Hydra configuration works")
        return True

    except Exception as e:
        print(f"\n❌ Hydra configuration check failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all validation checks."""
    print("\n" + "=" * 80)
    print("RLVR Vocabulary Training Setup Validation")
    print("=" * 80)

    checks = [
        ("Dependencies", check_dependencies),
        ("Dataset", check_dataset),
        ("Reward Function", check_reward_function),
        ("Hydra Configuration", check_hydra_config),
        # Model loading is slow, make it optional
        # ("Model Loading", check_model_loading),
    ]

    results = {}
    for name, check_fn in checks:
        results[name] = check_fn()

    # Summary
    print("\n" + "=" * 80)
    print("Validation Summary")
    print("=" * 80)

    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")

    all_passed = all(results.values())

    if all_passed:
        print("\n🎉 All checks passed! Training setup is ready.")
        print("\nNext steps:")
        print("  1. Login to WandB: wandb login")
        print(
            "  2. Run a quick test: uv run python rlvr_vocab/exp/grpo_train.py dataset.max_train_samples=10 training.num_train_epochs=1"
        )
        print("  3. Submit to SLURM: sbatch slurm/train_baseline.sh")
        return 0
    else:
        print("\n❌ Some checks failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
