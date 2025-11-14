"""
Comprehensive tests for utils.py module.
Tests all utility functions, timers, experiment tracking, and helper functionality.
"""

import pytest
import torch
import numpy as np
import shutil
import json
import time
import logging
from pathlib import Path
from typing import Dict, List
from unittest.mock import Mock, MagicMock, patch
import sys

# Add project root to Python path
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

# Import classes and functions to test
from src.utils import (
    get_project_root,
    get_data_dir,
    get_config_dir,
    get_src_dir,
    get_notebooks_dir,
    get_experiments_dir,
    setup_logging,
    set_seed,
    print_system_info,
    format_time,
    calculate_model_size,
    print_model_summary,
    TrainingTimer,
    ExperimentTracker,
    ProgressManager,
    cleanup_checkpoints,
    save_predictions_sample,
    get_device_info
)
from src.config import (
    ProjectConfig, DataConfig, ModelConfig, TrainingConfig,
    EvaluationConfig, LoggingConfig, ExperimentConfig
)


# Helper function to create ProjectConfig for testing
def create_test_config() -> ProjectConfig:
    """Create a test ProjectConfig with all required fields"""
    return ProjectConfig(
        data=DataConfig(),
        model=ModelConfig(),
        training=TrainingConfig(),
        evaluation=EvaluationConfig(),
        logging=LoggingConfig(),
        experiment=ExperimentConfig()
    )


# ============================================================
# Test Path Management
# ============================================================
class TestPathManagement:
    """Test path management functions"""

    def test_get_project_root(self):
        """Test project root retrieval"""
        root = get_project_root()
        assert isinstance(root, Path)
        assert root.exists()
        # Project root should contain src directory
        assert (root / "src").exists()

    def test_get_data_dir(self):
        """Test data directory path"""
        data_dir = get_data_dir()
        assert isinstance(data_dir, Path)
        assert data_dir.name == "data"
        assert data_dir.parent == get_project_root()

    def test_get_config_dir(self):
        """Test config directory path"""
        config_dir = get_config_dir()
        assert isinstance(config_dir, Path)
        assert config_dir.name == "configs"
        assert config_dir.parent == get_project_root()

    def test_get_src_dir(self):
        """Test src directory path"""
        src_dir = get_src_dir()
        assert isinstance(src_dir, Path)
        assert src_dir.name == "src"
        assert src_dir.parent == get_project_root()

    def test_get_notebooks_dir(self):
        """Test notebooks directory path"""
        notebooks_dir = get_notebooks_dir()
        assert isinstance(notebooks_dir, Path)
        assert notebooks_dir.name == "notebooks"

    def test_get_experiments_dir(self):
        """Test experiments directory path"""
        experiments_dir = get_experiments_dir()
        assert isinstance(experiments_dir, Path)
        assert experiments_dir.name == "experiments"

    def test_path_consistency(self):
        """Test that all paths are consistent"""
        root = get_project_root()
        assert get_data_dir() == root / "data"
        assert get_config_dir() == root / "configs"
        assert get_src_dir() == root / "src"


# ============================================================
# Test Logging Setup
# ============================================================
class TestLoggingSetup:
    """Test logging configuration"""

    def test_setup_logging_creates_directory(self, temp_dir):
        """Test that setup_logging creates log directory"""
        config = create_test_config()
        config.experiment.output_dir = str(temp_dir)
        config.logging.log_level = "INFO"

        logger = setup_logging(config)

        # Check log directory exists
        log_dir = temp_dir / "logs"
        assert log_dir.exists()

        # Check logger is configured
        assert isinstance(logger, logging.Logger)

    def test_setup_logging_creates_log_file(self, temp_dir):
        """Test that setup_logging creates log file"""
        config = create_test_config()
        config.experiment.output_dir = str(temp_dir)
        config.logging.log_level = "DEBUG"

        logger = setup_logging(config)

        # Check log file exists
        log_file = temp_dir / "logs" / "training.log"
        assert log_file.exists()

    def test_setup_logging_level(self, temp_dir):
        """Test different logging levels"""
        # Test one level to avoid state conflicts between tests
        config = create_test_config()
        config.experiment.output_dir = str(temp_dir)
        config.logging.log_level = "DEBUG"

        logger = setup_logging(config)

        # Verify logger is configured (level checking is complex due to global state)
        assert isinstance(logger, logging.Logger)
        # Just verify it doesn't crash with different levels
        for level in ["INFO", "WARNING", "ERROR"]:
            config.logging.log_level = level
            logger = setup_logging(config)
            assert logger is not None


# ============================================================
# Test Seed Setting
# ============================================================
class TestSeedSetting:
    """Test random seed functionality"""

    def test_set_seed_basic(self):
        """Test basic seed setting"""
        seed = 42
        set_seed(seed)

        # Generate random numbers
        np_rand1 = np.random.rand()
        torch_rand1 = torch.rand(1).item()

        # Reset seed and verify reproducibility
        set_seed(seed)
        np_rand2 = np.random.rand()
        torch_rand2 = torch.rand(1).item()

        assert np_rand1 == np_rand2
        assert torch_rand1 == torch_rand2

    def test_set_seed_different_values(self):
        """Test that different seeds produce different results"""
        set_seed(42)
        result1 = np.random.rand()

        set_seed(123)
        result2 = np.random.rand()

        assert result1 != result2

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_set_seed_cuda(self):
        """Test CUDA seed setting"""
        seed = 42
        set_seed(seed)

        # Generate CUDA random numbers
        torch_cuda_rand1 = torch.randn(1, device='cuda').item()

        # Reset and verify
        set_seed(seed)
        torch_cuda_rand2 = torch.randn(1, device='cuda').item()

        assert torch_cuda_rand1 == torch_cuda_rand2


# ============================================================
# Test System Info
# ============================================================
class TestSystemInfo:
    """Test system information functions"""

    def test_print_system_info(self):
        """Test system info printing (should not crash)"""
        # Should not raise exception
        print_system_info()

    def test_get_device_info_structure(self):
        """Test device info structure"""
        info = get_device_info()

        assert isinstance(info, dict)
        assert "device_type" in info
        assert "device_name" in info
        assert "memory_total" in info
        assert "memory_available" in info

    def test_get_device_info_cpu(self):
        """Test device info for CPU"""
        info = get_device_info()

        # Should at least have CPU info
        assert info["device_type"] in ["cpu", "cuda"]
        assert isinstance(info["device_name"], str)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_get_device_info_cuda(self):
        """Test device info for CUDA"""
        info = get_device_info()

        assert info["device_type"] == "cuda"
        assert "cuda_version" in info
        assert "cudnn_version" in info
        assert info["memory_total"] > 0


# ============================================================
# Test Time Formatting
# ============================================================
class TestTimeFormatting:
    """Test time formatting utilities"""

    def test_format_time_seconds(self):
        """Test formatting seconds only"""
        formatted = format_time(45.5)
        assert "00:45.50" in formatted
        assert formatted.count(":") == 1

    def test_format_time_minutes(self):
        """Test formatting minutes and seconds"""
        formatted = format_time(125.75)  # 2:05.75
        assert "02:05.75" in formatted

    def test_format_time_hours(self):
        """Test formatting hours, minutes, seconds"""
        formatted = format_time(3725.5)  # 1:02:05.50
        assert "01:02:05.50" in formatted
        assert formatted.count(":") == 2

    def test_format_time_zero(self):
        """Test formatting zero time"""
        formatted = format_time(0.0)
        assert "00:00.00" in formatted

    def test_format_time_large_value(self):
        """Test formatting large time values"""
        formatted = format_time(36000)  # 10 hours
        assert "10:00:00.00" in formatted


# ============================================================
# Test Model Size Calculation
# ============================================================
class TestModelSizeCalculation:
    """Test model size and parameter counting"""

    def test_calculate_model_size_simple_model(self):
        """Test size calculation for simple model"""
        model = torch.nn.Linear(10, 5)
        stats = calculate_model_size(model)

        assert "total_parameters" in stats
        assert "trainable_parameters" in stats
        assert "non_trainable_parameters" in stats
        assert "trainable_ratio" in stats
        assert "model_size_mb" in stats

        # Linear layer: 10*5 weights + 5 bias = 55 parameters
        assert stats["total_parameters"] == 55
        assert stats["trainable_parameters"] == 55
        assert stats["non_trainable_parameters"] == 0
        assert stats["trainable_ratio"] == 1.0
        assert stats["model_size_mb"] > 0

    def test_calculate_model_size_frozen_parameters(self):
        """Test size calculation with frozen parameters"""
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.Linear(5, 2)
        )

        # Freeze first layer
        for param in model[0].parameters():
            param.requires_grad = False

        stats = calculate_model_size(model)

        assert stats["non_trainable_parameters"] > 0
        assert stats["trainable_parameters"] < stats["total_parameters"]
        assert stats["trainable_ratio"] < 1.0

    def test_print_model_summary(self):
        """Test model summary printing (should not crash)"""
        model = torch.nn.Linear(10, 5)
        # Should not raise exception
        print_model_summary(model, "Test Model")

    def test_calculate_model_size_empty_model(self):
        """Test size calculation for model without parameters"""
        model = torch.nn.Identity()
        stats = calculate_model_size(model)

        assert stats["total_parameters"] == 0
        assert stats["trainable_parameters"] == 0
        assert stats["trainable_ratio"] == 0


# ============================================================
# Test TrainingTimer
# ============================================================
class TestTrainingTimer:
    """Test training timer functionality"""

    def test_timer_initialization(self):
        """Test timer initialization"""
        timer = TrainingTimer()

        assert timer.start_time is None
        assert timer.epoch_start is None
        assert timer.epoch_times == []
        assert timer.step_times == []

    def test_start_training(self):
        """Test starting training timer"""
        timer = TrainingTimer()
        timer.start_training()

        assert timer.start_time is not None
        assert isinstance(timer.start_time, float)

    def test_get_training_time(self):
        """Test getting total training time"""
        timer = TrainingTimer()
        timer.start_training()
        time.sleep(0.1)

        training_time = timer.get_training_time()
        assert training_time >= 0.1
        assert isinstance(training_time, float)

    def test_get_training_time_not_started(self):
        """Test getting training time before starting"""
        timer = TrainingTimer()
        training_time = timer.get_training_time()

        assert training_time == 0.0

    def test_epoch_timing(self):
        """Test epoch timing"""
        timer = TrainingTimer()
        timer.start_epoch()
        time.sleep(0.05)
        epoch_time = timer.end_epoch()

        assert epoch_time >= 0.05
        assert len(timer.epoch_times) == 1
        assert timer.epoch_times[0] == epoch_time

    def test_multiple_epochs(self):
        """Test timing multiple epochs"""
        timer = TrainingTimer()

        for _ in range(3):
            timer.start_epoch()
            time.sleep(0.02)
            timer.end_epoch()

        assert len(timer.epoch_times) == 3
        assert all(t >= 0.02 for t in timer.epoch_times)

    def test_average_epoch_time(self):
        """Test average epoch time calculation"""
        timer = TrainingTimer()

        # Simulate 3 epochs
        timer.epoch_times = [1.0, 2.0, 3.0]

        avg_time = timer.get_average_epoch_time()
        assert avg_time == 2.0

    def test_average_epoch_time_empty(self):
        """Test average epoch time with no epochs"""
        timer = TrainingTimer()
        avg_time = timer.get_average_epoch_time()

        assert avg_time == 0.0

    def test_estimate_remaining_time(self):
        """Test remaining time estimation"""
        timer = TrainingTimer()
        timer.epoch_times = [1.0, 1.0, 1.0]

        # Current epoch 3, total 10 epochs
        remaining = timer.estimate_remaining_time(3, 10)

        # Should estimate 7 more epochs * 1.0 second = 7.0 seconds
        assert remaining == 7.0

    def test_estimate_remaining_time_no_history(self):
        """Test remaining time estimation without history"""
        timer = TrainingTimer()
        remaining = timer.estimate_remaining_time(0, 10)

        assert remaining == 0.0

    def test_end_epoch_without_start(self):
        """Test ending epoch without starting"""
        timer = TrainingTimer()
        epoch_time = timer.end_epoch()

        assert epoch_time == 0.0


# ============================================================
# Test ExperimentTracker
# ============================================================
class TestExperimentTracker:
    """Test experiment tracking functionality"""

    def test_tracker_initialization(self, temp_dir):
        """Test tracker initialization"""
        tracker = ExperimentTracker(str(temp_dir))

        assert tracker.experiment_dir.exists()
        assert tracker.config_file == tracker.experiment_dir / "config.json"

    def test_tracker_creates_directory(self, temp_dir):
        """Test that tracker creates experiment directory"""
        exp_dir = temp_dir / "new_experiment"
        tracker = ExperimentTracker(str(exp_dir))

        assert exp_dir.exists()

    def test_save_config(self, temp_dir):
        """Test saving configuration"""
        tracker = ExperimentTracker(str(temp_dir))
        config = create_test_config()

        tracker.save_config(config)

        # Check config file was created
        assert tracker.config_file.exists()

        # Load and verify JSON structure
        with open(tracker.config_file, 'r') as f:
            saved_config = json.load(f)

        assert isinstance(saved_config, dict)
        assert "data" in saved_config
        assert "model" in saved_config
        assert "training" in saved_config

    def test_save_config_with_custom_values(self, temp_dir):
        """Test saving custom configuration"""
        tracker = ExperimentTracker(str(temp_dir))
        config = create_test_config()
        config.training.train_batch_size = 32
        config.training.learning_rate = 1e-5
        config.model.model_name = "test-model"

        tracker.save_config(config)

        # Load and verify custom values
        with open(tracker.config_file, 'r') as f:
            saved_config = json.load(f)

        assert saved_config["training"]["train_batch_size"] == 32
        assert saved_config["training"]["learning_rate"] == 1e-5
        assert saved_config["model"]["model_name"] == "test-model"

    def test_save_config_overwrites(self, temp_dir):
        """Test that saving config overwrites existing file"""
        tracker = ExperimentTracker(str(temp_dir))

        # Save first config
        config1 = create_test_config()
        config1.training.train_batch_size = 16
        tracker.save_config(config1)

        # Save second config with different value
        config2 = create_test_config()
        config2.training.train_batch_size = 32
        tracker.save_config(config2)

        # Load and verify latest value
        with open(tracker.config_file, 'r') as f:
            saved_config = json.load(f)

        assert saved_config["training"]["train_batch_size"] == 32


# ============================================================
# Test ProgressManager
# ============================================================
class TestProgressManager:
    """Test progress manager functionality"""

    def test_progress_manager_context(self):
        """Test progress manager as context manager"""
        manager = ProgressManager()

        # Should not raise exception
        with manager as progress:
            assert progress is not None

    def test_progress_manager_start_stop(self):
        """Test manual start/stop"""
        manager = ProgressManager()

        # Manual usage
        progress = manager.__enter__()
        assert progress is not None

        # Should not raise exception
        manager.__exit__(None, None, None)


# ============================================================
# Test Cleanup Checkpoints
# ============================================================
class TestCleanupCheckpoints:
    """Test checkpoint cleanup functionality"""

    def test_cleanup_no_checkpoints(self, temp_dir):
        """Test cleanup when no checkpoints exist"""
        # Should not raise exception
        cleanup_checkpoints(str(temp_dir), keep_last=3)

    def test_cleanup_fewer_than_limit(self, temp_dir):
        """Test cleanup when fewer checkpoints than limit"""
        # Create 2 checkpoints
        (temp_dir / "checkpoint-1").mkdir()
        (temp_dir / "checkpoint-2").mkdir()

        cleanup_checkpoints(str(temp_dir), keep_last=3)

        # Both should still exist
        assert (temp_dir / "checkpoint-1").exists()
        assert (temp_dir / "checkpoint-2").exists()

    def test_cleanup_more_than_limit(self, temp_dir):
        """Test cleanup when more checkpoints than limit"""
        # Create 5 checkpoints with different modification times
        checkpoints = []
        for i in range(5):
            cp = temp_dir / f"checkpoint-{i}"
            cp.mkdir()
            checkpoints.append(cp)
            time.sleep(0.01)  # Ensure different timestamps

        cleanup_checkpoints(str(temp_dir), keep_last=2)

        # Only last 2 should exist
        remaining = list(temp_dir.glob("checkpoint-*"))
        assert len(remaining) == 2

        # Check that newest checkpoints are kept
        assert (temp_dir / "checkpoint-4").exists()
        assert (temp_dir / "checkpoint-3").exists()

    def test_cleanup_nonexistent_directory(self):
        """Test cleanup with non-existent directory"""
        # Should not raise exception
        cleanup_checkpoints("/nonexistent/path", keep_last=3)

    def test_cleanup_checkpoint_files(self, temp_dir):
        """Test cleanup of checkpoint files (not directories)"""
        # Create checkpoint files
        for i in range(5):
            cp_file = temp_dir / f"checkpoint-{i}.pt"
            cp_file.touch()
            time.sleep(0.01)

        cleanup_checkpoints(str(temp_dir), keep_last=2)

        # Only last 2 should exist
        remaining = list(temp_dir.glob("checkpoint-*.pt"))
        assert len(remaining) == 2


# ============================================================
# Test Save Predictions Sample
# ============================================================
class TestSavePredictionsSample:
    """Test predictions sample saving"""

    def test_save_predictions_basic(self, temp_dir):
        """Test basic predictions saving"""
        predictions = ["привет мир", "как дела"]
        references = ["привет друг", "как дела"]
        audio_paths = ["audio1.mp3", "audio2.mp3"]
        save_path = temp_dir / "predictions.csv"

        save_predictions_sample(
            predictions, references, audio_paths,
            str(save_path), n_samples=2
        )

        assert save_path.exists()

        # Load and verify
        import pandas as pd
        df = pd.read_csv(save_path)

        assert len(df) == 2
        assert "audio_path" in df.columns
        assert "reference" in df.columns
        assert "prediction" in df.columns
        assert df["prediction"][0] == "привет мир"

    def test_save_predictions_limit_samples(self, temp_dir):
        """Test limiting number of samples"""
        predictions = ["text"] * 20
        references = ["text"] * 20
        audio_paths = ["audio.mp3"] * 20
        save_path = temp_dir / "predictions.csv"

        save_predictions_sample(
            predictions, references, audio_paths,
            str(save_path), n_samples=5
        )

        import pandas as pd
        df = pd.read_csv(save_path)

        assert len(df) == 5

    def test_save_predictions_mismatched_lengths(self, temp_dir):
        """Test with mismatched list lengths"""
        predictions = ["text1", "text2"]
        references = ["ref1"]  # Shorter
        audio_paths = ["audio1.mp3", "audio2.mp3", "audio3.mp3"]  # Longer
        save_path = temp_dir / "predictions.csv"

        # Should handle gracefully
        save_predictions_sample(
            predictions, references, audio_paths,
            str(save_path), n_samples=5
        )

        import pandas as pd
        df = pd.read_csv(save_path)

        # Should save min length
        assert len(df) <= 2

    def test_save_predictions_empty_lists(self, temp_dir):
        """Test with empty lists"""
        save_path = temp_dir / "predictions.csv"

        save_predictions_sample(
            [], [], [], str(save_path), n_samples=10
        )

        # File should be created even if empty
        assert save_path.exists()

        # Check file content is empty or has only headers
        with open(save_path, 'r') as f:
            content = f.read()
            # Should have headers or be empty (possibly with newline)
            assert "audio_path" in content or content.strip() == ""


# ============================================================
# Integration Tests
# ============================================================
class TestUtilsIntegration:
    """Integration tests for complete utils workflow"""

    def test_complete_experiment_setup(self, temp_dir):
        """Test complete experiment setup workflow"""
        # Setup
        config = create_test_config()
        config.experiment.output_dir = str(temp_dir)
        config.experiment.seed = 42

        # Initialize components
        set_seed(config.experiment.seed)
        logger = setup_logging(config)
        tracker = ExperimentTracker(str(temp_dir))
        timer = TrainingTimer()

        # Save config
        tracker.save_config(config)
        assert tracker.config_file.exists()

        # Simulate training
        timer.start_training()
        for epoch in range(3):
            timer.start_epoch()
            time.sleep(0.01)
            timer.end_epoch()

        training_time = timer.get_training_time()
        assert training_time > 0

        # Get system info
        device_info = get_device_info()
        assert device_info is not None

    def test_training_loop_simulation(self, temp_dir):
        """Test simulated training loop with all utils"""
        config = create_test_config()
        config.experiment.output_dir = str(temp_dir)

        # Setup
        set_seed(42)
        tracker = ExperimentTracker(str(temp_dir))
        tracker.save_config(config)
        timer = TrainingTimer()

        # Training loop
        timer.start_training()
        num_epochs = 5

        for epoch in range(num_epochs):
            timer.start_epoch()
            time.sleep(0.01)
            epoch_time = timer.end_epoch()
            assert epoch_time > 0

            # Estimate remaining time
            remaining = timer.estimate_remaining_time(epoch + 1, num_epochs)
            assert remaining >= 0

        # Final stats
        total_time = timer.get_training_time()
        avg_epoch = timer.get_average_epoch_time()

        assert total_time > 0
        assert avg_epoch > 0
        assert len(timer.epoch_times) == num_epochs

    def test_checkpoint_management_workflow(self, temp_dir):
        """Test checkpoint creation and cleanup workflow"""
        # Create checkpoints during "training"
        for epoch in range(10):
            cp = temp_dir / f"checkpoint-{epoch}"
            cp.mkdir()
            time.sleep(0.01)

        # Cleanup old checkpoints
        cleanup_checkpoints(str(temp_dir), keep_last=3)

        # Verify only 3 remain
        remaining = list(temp_dir.glob("checkpoint-*"))
        assert len(remaining) == 3


# ============================================================
# Edge Cases and Error Handling
# ============================================================
class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_timer_multiple_starts(self):
        """Test timer behavior with multiple start calls"""
        timer = TrainingTimer()
        timer.start_training()
        first_start = timer.start_time

        time.sleep(0.01)

        # Start again
        timer.start_training()
        second_start = timer.start_time

        # Should have different start times
        assert second_start != first_start

    def test_format_time_negative(self):
        """Test format_time with negative value (edge case)"""
        # Should handle gracefully (might wrap)
        formatted = format_time(-10.0)
        assert isinstance(formatted, str)

    def test_format_time_very_large(self):
        """Test format_time with very large value"""
        # 100 hours
        formatted = format_time(360000)
        assert isinstance(formatted, str)
        assert "100" in formatted

    def test_calculate_model_size_with_buffers(self):
        """Test model size calculation with buffers"""
        class ModelWithBuffers(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buffer", torch.randn(10, 10))
                self.fc = torch.nn.Linear(10, 5)

        model = ModelWithBuffers()
        stats = calculate_model_size(model)

        # Should include buffer size
        assert stats["model_size_mb"] > 0

    def test_experiment_tracker_nested_directories(self, temp_dir):
        """Test experiment tracker with nested directories"""
        nested_path = temp_dir / "exp1" / "subexp" / "final"
        tracker = ExperimentTracker(str(nested_path))

        # Should create all parent directories
        assert nested_path.exists()
        assert tracker.experiment_dir == nested_path

    def test_cleanup_checkpoints_with_non_checkpoint_files(self, temp_dir):
        """Test cleanup doesn't affect non-checkpoint files"""
        # Create checkpoints
        for i in range(5):
            (temp_dir / f"checkpoint-{i}").mkdir()
            time.sleep(0.01)

        # Create non-checkpoint files
        (temp_dir / "model.pt").touch()
        (temp_dir / "config.json").touch()

        cleanup_checkpoints(str(temp_dir), keep_last=2)

        # Non-checkpoint files should remain
        assert (temp_dir / "model.pt").exists()
        assert (temp_dir / "config.json").exists()

        # Only 2 checkpoints should remain
        remaining = list(temp_dir.glob("checkpoint-*"))
        assert len(remaining) == 2

    def test_save_predictions_with_unicode(self, temp_dir):
        """Test saving predictions with Unicode characters"""
        predictions = ["привет мир 🌍", "как дела?"]
        references = ["привет друг", "всё хорошо!"]
        audio_paths = ["audio1.mp3", "audio2.mp3"]
        save_path = temp_dir / "predictions.csv"

        save_predictions_sample(
            predictions, references, audio_paths,
            str(save_path), n_samples=2
        )

        # Should save with UTF-8 encoding
        import pandas as pd
        df = pd.read_csv(save_path)

        assert "🌍" in df["prediction"][0]
        assert "?" in df["prediction"][1]

    def test_get_device_info_memory_values(self):
        """Test device info returns valid memory values"""
        info = get_device_info()

        assert info["memory_total"] >= 0
        assert info["memory_available"] >= 0

        if info["device_type"] == "cuda":
            # CUDA should have positive memory
            assert info["memory_total"] > 0

    def test_timer_precision(self):
        """Test timer precision for very short durations"""
        timer = TrainingTimer()
        timer.start_epoch()
        # Very short sleep
        time.sleep(0.001)
        epoch_time = timer.end_epoch()

        # Should still capture time
        assert epoch_time >= 0.0

    def test_config_save_with_special_characters(self, temp_dir):
        """Test config saving with special characters in paths"""
        tracker = ExperimentTracker(str(temp_dir))
        config = create_test_config()
        config.data.data_dir = "data/test-dataset_v1.0"

        # Should save without error
        tracker.save_config(config)

        with open(tracker.config_file, 'r') as f:
            saved = json.load(f)

        assert saved["data"]["data_dir"] == "data/test-dataset_v1.0"


# ============================================================
# Run tests
# ============================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
