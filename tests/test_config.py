"""
Tests for configuration loading and validation.
"""

import pytest
import os
import sys
from pathlib import Path

# Add project root to Python path
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

from src.config import (
    load_config,
    ProjectConfig,
    ModelConfig,
    DataConfig,
    TrainingConfig,
    EvaluationConfig,
    ExperimentConfig,
)


class TestConfigLoading:
    """Test configuration loading from YAML files"""

    def test_load_default_config(self):
        """Test loading default configuration"""
        config_path = "configs/default.yaml"
        assert os.path.exists(config_path), f"Default config not found: {config_path}"

        config = load_config(config_path)

        # Verify config structure
        assert isinstance(config, ProjectConfig)
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.data, DataConfig)
        assert isinstance(config.training, TrainingConfig)

        # Verify default model settings
        assert config.model.model_name == "openai/whisper-small"
        assert config.model.model_type == "whisper"
        assert config.model.freeze_feature_encoder is True
        assert config.model.freeze_encoder is True
        assert config.model.freeze_decoder is False
        assert config.model.unfreeze_last_n_encoder_layers == 0
        assert config.model.unfreeze_last_n_decoder_layers == 0


    def test_config_without_path(self):
        """Test loading default config when no path is provided"""
        config = load_config(None)

        assert isinstance(config, ProjectConfig)
        assert config.model.model_name == "openai/whisper-small"
        assert config.data.language == "ru"

    def test_freezing_settings_consistency(self):
        """Test that freezing settings are consistent in default config"""
        config = load_config("configs/default.yaml")

        # Verify default freezing settings
        assert config.model.freeze_feature_encoder is True
        assert config.model.freeze_encoder is True
        assert config.model.freeze_decoder is False

    def test_new_fields_present(self):
        """Test that new freezing fields are present in default config"""
        config = load_config("configs/default.yaml")

        # Check new fields exist
        assert hasattr(config.model, "freeze_decoder")
        assert hasattr(config.model, "unfreeze_last_n_encoder_layers")
        assert hasattr(config.model, "unfreeze_last_n_decoder_layers")

        # Check types
        assert isinstance(config.model.freeze_decoder, bool)
        assert isinstance(config.model.unfreeze_last_n_encoder_layers, int)
        assert isinstance(config.model.unfreeze_last_n_decoder_layers, int)

    def test_dropout_settings(self):
        """Test that dropout settings are properly loaded (standard HuggingFace names)"""
        config = load_config("configs/default.yaml")

        # Whisper defaults to 0.0 for all dropout parameters
        assert config.model.activation_dropout == 0.0
        assert config.model.attention_dropout == 0.0
        assert config.model.dropout == 0.0

    def test_augmentation_settings(self):
        """Test that augmentation settings are properly loaded"""
        config = load_config("configs/default.yaml")

        assert hasattr(config.data, "augmentation")
        assert config.data.augmentation.enabled is True
        assert config.data.augmentation.add_noise is True
        assert config.data.augmentation.spec_augment is True

    def test_scheduler_settings(self):
        """Test that scheduler settings are properly loaded"""
        # Test linear scheduler (default in main configs)
        config = load_config("configs/default.yaml")
        assert hasattr(config.training, "scheduler_name")
        assert config.training.scheduler_name == "linear"
        assert hasattr(config.training, "linear")
        assert config.training.linear is not None
        assert config.training.linear.warmup_ratio == 0.1

        # Test that all scheduler configs are present
        assert hasattr(config.training, "reduce_on_plateau")
        assert config.training.reduce_on_plateau is not None
        assert config.training.reduce_on_plateau.factor == 0.5

        assert hasattr(config.training, "cosine")
        assert config.training.cosine is not None
        assert config.training.cosine.T_max == 10

        assert hasattr(config.training, "onecycle")
        assert config.training.onecycle is not None


class TestUnifiedConfigs:
    """Test unified configuration files (for both training and evaluation)"""

    def test_default_config_has_required_blocks(self):
        """Test that default config has all required configuration blocks"""
        config = load_config("configs/default.yaml")

        # Every config must have all blocks
        assert hasattr(config, "experiment"), "missing 'experiment' block"
        assert hasattr(config, "data"), "missing 'data' block"
        assert hasattr(config, "model"), "missing 'model' block"
        assert hasattr(config, "training"), "missing 'training' block"
        assert hasattr(config, "evaluation"), "missing 'evaluation' block"
        assert hasattr(config, "logging"), "missing 'logging' block"

        # Verify types
        assert isinstance(config.experiment, ExperimentConfig), "experiment not ExperimentConfig"
        assert isinstance(config.data, DataConfig), "data not DataConfig"
        assert isinstance(config.model, ModelConfig), "model not ModelConfig"
        assert isinstance(config.training, TrainingConfig), "training not TrainingConfig"
        assert isinstance(config.evaluation, EvaluationConfig), "evaluation not EvaluationConfig"

    def test_evaluation_params_not_in_training(self):
        """Test that evaluation-specific parameters are in evaluation block, not training"""
        config = load_config("configs/default.yaml")

        # Evaluation params should be in evaluation block
        assert hasattr(config.evaluation, "batch_size"), "evaluation.batch_size missing"
        assert hasattr(config.evaluation, "num_workers"), "evaluation.num_workers missing"

        # Training should have its own separate batch_size and num_workers
        assert hasattr(config.training, "train_batch_size"), "training.train_batch_size missing"
        assert hasattr(config.training, "eval_batch_size"), "training.eval_batch_size missing"
        assert hasattr(config.training, "num_workers"), "training.num_workers missing"

        # Experiment params should NOT be in training
        assert hasattr(config.experiment, "experiment_name"), "experiment.experiment_name missing"
        assert hasattr(config.experiment, "output_dir"), "experiment.output_dir missing"
        assert hasattr(config.experiment, "seed"), "experiment.seed missing"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
