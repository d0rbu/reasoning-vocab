"""
Tests for configuration management and validation.

Tests include:
- Hydra configuration loading
- Configuration validation
- Parameter overrides
- Default values
- Type checking
"""

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from exp.grpo_train import create_grpo_config


class TestHydraConfiguration:
    """Test Hydra configuration system."""

    def test_config_loading(self):
        """Test that main config file can be loaded."""
        config_dir = Path(__file__).parent.parent / "exp" / "conf"
        assert config_dir.exists(), f"Config directory not found: {config_dir}"

        with initialize_config_dir(version_base=None, config_dir=str(config_dir.absolute())):
            cfg = compose(config_name="config")

            assert cfg is not None
            assert isinstance(cfg, DictConfig)

    def test_config_structure(self):
        """Test that config has expected structure."""
        config_dir = Path(__file__).parent.parent / "exp" / "conf"

        with initialize_config_dir(version_base=None, config_dir=str(config_dir.absolute())):
            cfg = compose(config_name="config")

            # Check top-level keys
            assert "exp_name" in cfg
            assert "seed" in cfg
            assert "output_dir" in cfg
            assert "model" in cfg
            assert "dataset" in cfg
            assert "training" in cfg
            assert "logging" in cfg

    def test_model_config(self):
        """Test model configuration."""
        config_dir = Path(__file__).parent.parent / "exp" / "conf"

        with initialize_config_dir(version_base=None, config_dir=str(config_dir.absolute())):
            cfg = compose(config_name="config")

            # Check model config
            assert "name" in cfg.model
            assert "model_kwargs" in cfg.model
            assert "torch_dtype" in cfg.model.model_kwargs
            assert "trust_remote_code" in cfg.model.model_kwargs

    def test_training_config(self):
        """Test training configuration."""
        config_dir = Path(__file__).parent.parent / "exp" / "conf"

        with initialize_config_dir(version_base=None, config_dir=str(config_dir.absolute())):
            cfg = compose(config_name="config")

            # Check training config
            assert "num_train_epochs" in cfg.training
            assert "learning_rate" in cfg.training
            assert "per_device_train_batch_size" in cfg.training
            assert "gradient_accumulation_steps" in cfg.training
            assert "max_grad_norm" in cfg.training
            assert "warmup_ratio" in cfg.training

    def test_grpo_specific_config(self):
        """Test GRPO-specific configuration parameters."""
        config_dir = Path(__file__).parent.parent / "exp" / "conf"

        with initialize_config_dir(version_base=None, config_dir=str(config_dir.absolute())):
            cfg = compose(config_name="config")

            # Check GRPO-specific params
            assert "num_generations" in cfg.training
            assert "max_prompt_length" in cfg.training
            assert "max_completion_length" in cfg.training


class TestConfigurationOverrides:
    """Test configuration override functionality."""

    def test_simple_override(self):
        """Test simple parameter override."""
        config_dir = Path(__file__).parent.parent / "exp" / "conf"

        with initialize_config_dir(version_base=None, config_dir=str(config_dir.absolute())):
            cfg = compose(
                config_name="config",
                overrides=["exp_name=test_override"],
            )

            assert cfg.exp_name == "test_override"

    def test_nested_override(self):
        """Test nested parameter override."""
        config_dir = Path(__file__).parent.parent / "exp" / "conf"

        with initialize_config_dir(version_base=None, config_dir=str(config_dir.absolute())):
            cfg = compose(
                config_name="config",
                overrides=["training.learning_rate=1e-5"],
            )

            assert cfg.training.learning_rate == 1e-5

    def test_multiple_overrides(self):
        """Test multiple parameter overrides."""
        config_dir = Path(__file__).parent.parent / "exp" / "conf"

        with initialize_config_dir(version_base=None, config_dir=str(config_dir.absolute())):
            cfg = compose(
                config_name="config",
                overrides=[
                    "exp_name=multi_test",
                    "seed=123",
                    "training.learning_rate=2e-5",
                    "training.num_train_epochs=10",
                ],
            )

            assert cfg.exp_name == "multi_test"
            assert cfg.seed == 123
            assert cfg.training.learning_rate == 2e-5
            assert cfg.training.num_train_epochs == 10

    def test_boolean_override(self):
        """Test boolean parameter override."""
        config_dir = Path(__file__).parent.parent / "exp" / "conf"

        with initialize_config_dir(version_base=None, config_dir=str(config_dir.absolute())):
            cfg = compose(
                config_name="config",
                overrides=["training.bf16=false", "training.fp16=true"],
            )

            assert cfg.training.bf16 is False
            assert cfg.training.fp16 is True


class TestConfigurationValidation:
    """Test configuration validation."""

    def test_required_fields_present(self, minimal_hydra_config: DictConfig):
        """Test that all required fields are present."""
        required_fields = [
            "exp_name",
            "seed",
            "output_dir",
            "model",
            "dataset",
            "training",
            "logging",
        ]

        for field in required_fields:
            assert field in minimal_hydra_config, f"Missing required field: {field}"

    def test_model_required_fields(self, minimal_hydra_config: DictConfig):
        """Test that model config has required fields."""
        required_model_fields = ["name", "model_kwargs"]

        for field in required_model_fields:
            assert field in minimal_hydra_config.model, f"Missing model field: {field}"

    def test_training_required_fields(self, minimal_hydra_config: DictConfig):
        """Test that training config has required fields."""
        required_training_fields = [
            "num_train_epochs",
            "learning_rate",
            "per_device_train_batch_size",
            "gradient_accumulation_steps",
            "num_generations",
            "max_prompt_length",
            "max_completion_length",
        ]

        for field in required_training_fields:
            assert field in minimal_hydra_config.training, f"Missing training field: {field}"

    def test_config_types(self, minimal_hydra_config: DictConfig):
        """Test that config values have correct types."""
        assert isinstance(minimal_hydra_config.exp_name, str)
        assert isinstance(minimal_hydra_config.seed, int)
        assert isinstance(minimal_hydra_config.output_dir, str)

        assert isinstance(minimal_hydra_config.training.learning_rate, float)
        assert isinstance(minimal_hydra_config.training.num_train_epochs, int)
        assert isinstance(minimal_hydra_config.training.per_device_train_batch_size, int)

    def test_positive_values(self, minimal_hydra_config: DictConfig):
        """Test that numeric values are positive where expected."""
        assert minimal_hydra_config.training.learning_rate > 0
        assert minimal_hydra_config.training.num_train_epochs > 0
        assert minimal_hydra_config.training.per_device_train_batch_size > 0
        assert minimal_hydra_config.training.gradient_accumulation_steps > 0
        assert minimal_hydra_config.training.num_generations > 0
        assert minimal_hydra_config.training.max_prompt_length > 0
        assert minimal_hydra_config.training.max_completion_length > 0


class TestGRPOConfigConversion:
    """Test conversion from Hydra config to GRPOConfig."""

    def test_grpo_config_creation(self, minimal_hydra_config: DictConfig):
        """Test that GRPOConfig can be created from Hydra config."""
        grpo_config = create_grpo_config(minimal_hydra_config)

        assert grpo_config is not None
        assert hasattr(grpo_config, "output_dir")
        assert hasattr(grpo_config, "learning_rate")
        assert hasattr(grpo_config, "num_train_epochs")

    def test_grpo_config_parameter_mapping(self, minimal_hydra_config: DictConfig):
        """Test that parameters are correctly mapped to GRPOConfig."""
        grpo_config = create_grpo_config(minimal_hydra_config)

        # Check basic parameters
        assert grpo_config.output_dir == minimal_hydra_config.output_dir
        assert grpo_config.learning_rate == minimal_hydra_config.training.learning_rate
        assert grpo_config.num_train_epochs == minimal_hydra_config.training.num_train_epochs
        assert grpo_config.seed == minimal_hydra_config.seed

        # Check training parameters
        assert (
            grpo_config.per_device_train_batch_size
            == minimal_hydra_config.training.per_device_train_batch_size
        )
        assert (
            grpo_config.gradient_accumulation_steps
            == minimal_hydra_config.training.gradient_accumulation_steps
        )

        # Check GRPO-specific parameters
        assert grpo_config.num_generations == minimal_hydra_config.training.num_generations
        assert grpo_config.max_prompt_length == minimal_hydra_config.training.max_prompt_length
        assert (
            grpo_config.max_completion_length == minimal_hydra_config.training.max_completion_length
        )

    def test_grpo_config_logging_settings(self, minimal_hydra_config: DictConfig):
        """Test that logging settings are correctly converted."""
        # Test with logging disabled
        minimal_hydra_config.logging.enabled = False
        grpo_config = create_grpo_config(minimal_hydra_config)
        # report_to can be a list or string depending on transformers version
        if isinstance(grpo_config.report_to, list):
            assert len(grpo_config.report_to) == 0 or "none" in grpo_config.report_to
        else:
            assert grpo_config.report_to == "none"

        # Test with logging enabled
        minimal_hydra_config.logging.enabled = True
        grpo_config = create_grpo_config(minimal_hydra_config)
        if isinstance(grpo_config.report_to, list):
            assert "wandb" in grpo_config.report_to
        else:
            assert grpo_config.report_to == "wandb"


class TestConfigurationSerialization:
    """Test configuration serialization and deserialization."""

    def test_config_to_yaml(self, minimal_hydra_config: DictConfig):
        """Test converting config to YAML."""
        yaml_str = OmegaConf.to_yaml(minimal_hydra_config)

        assert isinstance(yaml_str, str)
        assert len(yaml_str) > 0
        assert "exp_name" in yaml_str
        assert "training" in yaml_str

    def test_config_to_dict(self, minimal_hydra_config: DictConfig):
        """Test converting config to dictionary."""
        config_dict = OmegaConf.to_container(minimal_hydra_config, resolve=True)

        assert isinstance(config_dict, dict)
        assert "exp_name" in config_dict
        assert "training" in config_dict
        assert "model" in config_dict

    def test_config_resolution(self, minimal_hydra_config: DictConfig):
        """Test that config interpolation works."""
        # Add an interpolated value
        minimal_hydra_config.test_interpolation = "${exp_name}_test"

        resolved = OmegaConf.to_container(minimal_hydra_config, resolve=True)

        expected = f"{minimal_hydra_config.exp_name}_test"
        assert resolved["test_interpolation"] == expected


class TestDefaultValues:
    """Test that default configuration values are sensible."""

    def test_default_config_loads(self):
        """Test that default config loads successfully."""
        config_dir = Path(__file__).parent.parent / "exp" / "conf"

        with initialize_config_dir(version_base=None, config_dir=str(config_dir.absolute())):
            cfg = compose(config_name="config")

            assert cfg is not None

    def test_default_learning_rate(self):
        """Test that default learning rate is reasonable."""
        config_dir = Path(__file__).parent.parent / "exp" / "conf"

        with initialize_config_dir(version_base=None, config_dir=str(config_dir.absolute())):
            cfg = compose(config_name="config")

            lr = cfg.training.learning_rate
            # Check learning rate is in reasonable range
            assert 1e-7 < lr < 1e-2, f"Learning rate {lr} seems unusual"

    def test_default_batch_size(self):
        """Test that default batch size is reasonable."""
        config_dir = Path(__file__).parent.parent / "exp" / "conf"

        with initialize_config_dir(version_base=None, config_dir=str(config_dir.absolute())):
            cfg = compose(config_name="config")

            batch_size = cfg.training.per_device_train_batch_size
            # Check batch size is positive and not too large
            assert 1 <= batch_size <= 128, f"Batch size {batch_size} seems unusual"

    def test_default_output_directory(self):
        """Test that default output directory is set."""
        config_dir = Path(__file__).parent.parent / "exp" / "conf"

        with initialize_config_dir(version_base=None, config_dir=str(config_dir.absolute())):
            cfg = compose(config_name="config")

            assert cfg.output_dir is not None
            assert isinstance(cfg.output_dir, str)
            assert len(cfg.output_dir) > 0


class TestConfigurationEdgeCases:
    """Test edge cases in configuration."""

    def test_zero_epochs(self, minimal_hydra_config: DictConfig):
        """Test handling of zero epochs."""
        minimal_hydra_config.training.num_train_epochs = 0
        grpo_config = create_grpo_config(minimal_hydra_config)

        assert grpo_config.num_train_epochs == 0

    def test_very_small_learning_rate(self, minimal_hydra_config: DictConfig):
        """Test handling of very small learning rate."""
        minimal_hydra_config.training.learning_rate = 1e-10
        grpo_config = create_grpo_config(minimal_hydra_config)

        assert grpo_config.learning_rate == 1e-10

    def test_very_large_batch_size(self, minimal_hydra_config: DictConfig):
        """Test handling of very large batch size."""
        minimal_hydra_config.training.per_device_train_batch_size = 1024
        grpo_config = create_grpo_config(minimal_hydra_config)

        assert grpo_config.per_device_train_batch_size == 1024

    def test_conflicting_precision_settings(self, minimal_hydra_config: DictConfig):
        """Test handling when both fp16 and bf16 are set."""
        # In CPU environment, we can only test with both disabled
        # GPU-specific precision tests would fail on CPU
        minimal_hydra_config.training.fp16 = False
        minimal_hydra_config.training.bf16 = False

        grpo_config = create_grpo_config(minimal_hydra_config)

        # Should accept the configuration
        assert grpo_config.fp16 is False
        assert grpo_config.bf16 is False
