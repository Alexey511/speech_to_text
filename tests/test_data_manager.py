#!/usr/bin/env python3
"""
Unit tests for DataManager using full Common Voice dataset
"""

import sys
import pytest
import logging
import pandas as pd
from pathlib import Path
import warnings
import numpy as np

# Add project root to Python path
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

from src.config import load_config
from src.data import DataManager
from transformers import WhisperProcessor, Speech2TextProcessor

# Setup logging for tests
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
    force=True
)

# Suppress torchaudio MP3 warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="The MPEG_LAYER_III subtype is unknown to TorchAudio"
)

logger = logging.getLogger(__name__)

@pytest.fixture
def config():
    """Load default configuration"""
    return load_config("configs/default.yaml")

@pytest.fixture
def data_manager(config):
    """Initialize DataManager"""
    return DataManager(config)

class TestDataManager:
    """Test suite for DataManager with full Common Voice dataset"""

    def test_setup_processor_whisper(self, data_manager, config):
        """Test setup_processor for Whisper model"""
        processor = data_manager.setup_processor(
            model_name=config.model.model_name,
            model_type="whisper",
            language=config.data.language,
            task=config.data.task
        )
        assert isinstance(processor, WhisperProcessor)
        assert data_manager.processor is processor
        # Safely access feature_extractor and tokenizer
        assert hasattr(processor, 'feature_extractor'), "Processor missing feature_extractor"
        assert hasattr(processor, 'tokenizer'), "Processor missing tokenizer"

        # Safe attribute access with getattr
        feature_extractor = getattr(processor, 'feature_extractor', None)
        tokenizer = getattr(processor, 'tokenizer', None)

        assert feature_extractor is not None
        assert tokenizer is not None
        assert feature_extractor.sampling_rate == config.data.sample_rate

        # CRITICAL: Verify language and task are set correctly
        assert tokenizer.language == config.data.language, (
            f"Expected language='{config.data.language}', got '{tokenizer.language}'"
        )
        assert tokenizer.task == config.data.task, (
            f"Expected task='{config.data.task}', got '{tokenizer.task}'"
        )

    def test_setup_processor_speech2text(self, data_manager, config):
        """Test setup_processor for Speech2Text Medium Multilingual model"""
        processor = data_manager.setup_processor(
            model_name="facebook/s2t-medium-mustc-multilingual-st",
            model_type="speech2text",
            language=config.data.language
        )
        assert isinstance(processor, Speech2TextProcessor)
        assert data_manager.processor is processor
        # Safely access feature_extractor and tokenizer
        assert hasattr(processor, 'feature_extractor'), "Processor missing feature_extractor"
        assert hasattr(processor, 'tokenizer'), "Processor missing tokenizer"

        # Safe attribute access with getattr
        feature_extractor = getattr(processor, 'feature_extractor', None)

        assert feature_extractor is not None
        assert feature_extractor.sampling_rate == config.data.sample_rate

    def test_setup_processor_invalid(self, data_manager):
        """Test setup_processor with invalid model type"""
        with pytest.raises(ValueError, match="Unsupported model type: invalid"):
            data_manager.setup_processor(
                model_name="some_model",
                model_type="invalid",
                language="ru"
            )

    def test_is_dataset_available(self, data_manager, monkeypatch):
        """Test is_dataset_available method"""
        # Mock Path.exists to simulate dataset presence
        def mock_exists(self):
            return True
        monkeypatch.setattr(Path, "exists", mock_exists)
        assert data_manager.is_dataset_available()

        # Mock to simulate missing files
        def mock_exists_missing(self):
            return False
        monkeypatch.setattr(Path, "exists", mock_exists_missing)
        assert not data_manager.is_dataset_available()

    def test_load_dataset_train(self, data_manager, caplog):
        """Test load_dataset for train split"""
        caplog.set_level(logging.INFO)
        if not data_manager.is_dataset_available():
            pytest.skip("Common Voice dataset not available")
        
        df = data_manager.load_dataset("train")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "path" in df.columns
        assert "sentence" in df.columns
        assert "duration" in df.columns
        assert "client_id" in df.columns
        durations = df["duration"].to_numpy()
        assert np.all(durations >= data_manager.config.data.min_duration)
        assert np.all(durations <= data_manager.config.data.max_duration)
        assert "Loading train split from TSV file" in caplog.text
        assert "Cached train split" in caplog.text

    def test_load_dataset_val(self, data_manager, caplog):
        """Test load_dataset for validation split"""
        caplog.set_level(logging.INFO)
        if not data_manager.is_dataset_available():
            pytest.skip("Common Voice dataset not available")
        
        df = data_manager.load_dataset("val")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "path" in df.columns
        assert "sentence" in df.columns
        assert "duration" in df.columns
        assert "client_id" in df.columns
        durations = df["duration"].to_numpy()
        assert np.all(durations >= data_manager.config.data.min_duration)
        assert np.all(durations <= data_manager.config.data.max_duration)
        assert "Loading val split from TSV file" in caplog.text
        assert "Cached val split" in caplog.text

    def test_load_dataset_test(self, data_manager, caplog):
        """Test load_dataset for test split"""
        caplog.set_level(logging.INFO)
        if not data_manager.is_dataset_available():
            pytest.skip("Common Voice dataset not available")
        
        df = data_manager.load_dataset("test")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "path" in df.columns
        assert "sentence" in df.columns
        assert "duration" in df.columns
        assert "client_id" in df.columns
        durations = df["duration"].to_numpy()
        assert np.all(durations >= data_manager.config.data.min_duration)
        assert np.all(durations <= data_manager.config.data.max_duration)
        assert "Loading test split from TSV file" in caplog.text
        assert "Cached test split" in caplog.text

    def test_load_dataset_cache(self, data_manager, caplog, monkeypatch):
        """Test load_dataset caching for all splits"""
        if not data_manager.is_dataset_available():
            pytest.skip("Common Voice dataset not available")
        
        caplog.set_level(logging.INFO)
        
        # Clear cache to ensure fresh loading
        data_manager.dataset_cache.clear()
        
        # Load all splits to populate cache
        for split in ["train", "val", "test"]:
            df = data_manager.load_dataset(split)
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
            assert f"Loading {split} split from TSV file" in caplog.text
            assert f"Cached {split} split" in caplog.text
        
        # Reset caplog
        caplog.clear()
        
        # Load again and check cache usage
        for split, cache_key in [
            ("train", "cv22_train"),
            ("val", "cv22_dev"),
            ("test", "cv22_test")
        ]:
            df = data_manager.load_dataset(split)
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
            assert f"Loading {split} split from cache (key: {cache_key})" in caplog.text
            assert f"Cached {split} split" not in caplog.text  # Should not re-cache
            assert cache_key in data_manager.dataset_cache

    def test_load_dataset_invalid_split(self, data_manager):
        """Test load_dataset with invalid split"""
        with pytest.raises(ValueError, match="Unknown split 'invalid'"):
            data_manager.load_dataset("invalid")

    def test_load_dataset_missing_tsv(self, data_manager, monkeypatch):
        """Test load_dataset with missing dataset directory"""
        def mock_exists(self):
            return False
        monkeypatch.setattr(Path, "exists", mock_exists)
        with pytest.raises(ValueError, match="Common Voice dataset not found at"):
            data_manager.load_dataset("train")

    def test_load_dataset_error_handling(self, data_manager, monkeypatch, caplog):
        """Test load_dataset error handling"""
        caplog.set_level(logging.ERROR)
        def mock_read_csv(*args, **kwargs):
            raise Exception("CSV read error")
        monkeypatch.setattr(pd, "read_csv", mock_read_csv)
        if not data_manager.is_dataset_available():
            pytest.skip("Common Voice dataset not available")
        with pytest.raises(Exception, match="CSV read error"):
            data_manager.load_dataset("train")
        assert "Failed to load dataset" in caplog.text

    def test_get_dataset_info_available(self, data_manager, caplog):
        """Test get_dataset_info when dataset is available"""
        caplog.set_level(logging.INFO)
        if not data_manager.is_dataset_available():
            pytest.skip("Common Voice dataset not available")
        
        info = data_manager.get_dataset_info()
        assert info["status"] == "available"
        assert info["dataset_version"] == "Common Voice 22.0"
        assert info["language"] == "ru"
        assert info["local_path"] == str(data_manager.dataset_dir)
        assert info["clips_path"] == str(data_manager.clips_dir)
        assert "splits" in info
        for split in ["train", "dev", "test"]:
            assert split in info["splits"]
            if "num_samples" in info["splits"][split]:
                assert info["splits"][split]["num_samples"] > 0
                assert "tsv_file" in info["splits"][split]
                assert "total_duration_hours" in info["splits"][split]
        assert info["total_samples"] > 0

    def test_get_dataset_info_not_available(self, data_manager, monkeypatch):
        """Test get_dataset_info when dataset is not available"""
        def mock_exists(self):
            return False
        monkeypatch.setattr(Path, "exists", mock_exists)
        info = data_manager.get_dataset_info()
        assert info["status"] == "not_available"
        assert "message" in info
        assert "Common Voice 22.0 dataset not found" in info["message"]

    def test_get_dataset_stats(self, data_manager, caplog):
        """Test get_dataset_stats"""
        caplog.set_level(logging.INFO)
        if not data_manager.is_dataset_available():
            pytest.skip("Common Voice dataset not available")
        
        stats = data_manager.get_dataset_stats("train")
        assert isinstance(stats, dict)
        assert stats["total_samples"] > 0
        assert stats["audio_samples_analyzed"] > 0
        assert stats["estimated_total_duration_hours"] >= 0
        assert stats["avg_duration"] >= 0
        assert stats["median_duration"] >= 0
        assert stats["min_duration"] >= 0
        assert stats["max_duration"] >= 0
        assert stats["avg_text_length"] >= 0
        assert stats["median_text_length"] >= 0
        assert stats["min_text_length"] >= 0
        assert stats["max_text_length"] >= 0
        assert stats["avg_up_votes"] >= 0
        assert stats["avg_down_votes"] >= 0

    def test_apply_filters(self, data_manager, monkeypatch):
        """Test _apply_filters method"""
        # Create a sample DataFrame
        # Note: config has min_duration=0.2s, max_duration=30.0s
        df = pd.DataFrame({
            "path": ["file1.mp3", "file2.mp3", "file3.mp3"],
            "sentence": ["text1", "text2", "text3"],
            "duration": [0.1, 10.0, 40.0],  # 0.1s is too short, 40.0s is too long, only 10.0s is valid
            "client_id": ["id1", "id2", "id3"],
            "up_votes": [2, 3, 4],
            "down_votes": [1, 0, 2]
        })

        # Test with filter_by_duration=True
        monkeypatch.setattr(data_manager.config.data, "filter_by_duration", True)
        filtered_df = data_manager._apply_filters(df)
        assert len(filtered_df) == 1
        assert filtered_df["duration"].iloc[0] == 10.0
        durations = filtered_df["duration"].to_numpy()
        assert np.all(durations >= data_manager.config.data.min_duration)
        assert np.all(durations <= data_manager.config.data.max_duration)

        # Test with filter_by_duration=False
        monkeypatch.setattr(data_manager.config.data, "filter_by_duration", False)
        filtered_df = data_manager._apply_filters(df)
        assert len(filtered_df) == 3

    def test_apply_filters_with_long_audio(self, data_manager, config):
        """Test that _apply_filters correctly filters out samples exceeding max_duration"""
        if not data_manager.is_dataset_available():
            pytest.skip("Common Voice dataset not available")

        # Load real dataset
        df_real = data_manager.load_dataset("train")

        # Create fake samples with durations exceeding max_duration
        max_duration = config.data.max_duration
        fake_samples = pd.DataFrame({
            'path': ['fake_long1.mp3', 'fake_long2.mp3'],
            'sentence': ['fake sentence 1', 'fake sentence 2'],
            'duration': [max_duration + 1.0, max_duration + 5.0],  # Both exceed max
            'client_id': ['fake_client1', 'fake_client2'],
            'up_votes': [0, 0],
            'down_votes': [0, 0]
        })

        # Combine real and fake data
        df_combined = pd.concat([df_real, fake_samples], ignore_index=True)
        initial_real_count = len(df_real)
        initial_combined_count = len(df_combined)

        # Verify that fake samples were added
        assert initial_combined_count == initial_real_count + 2, "Fake samples not added correctly"

        # Enable duration filtering
        original_filter_setting = config.data.filter_by_duration
        config.data.filter_by_duration = True

        try:
            # Apply filters
            filtered_df = data_manager._apply_filters(df_combined)

            # Verify that fake long samples are filtered out
            assert len(filtered_df) == initial_real_count, \
                f"Expected {initial_real_count} samples after filtering, got {len(filtered_df)}"

            # Verify no fake samples remain
            assert not any(filtered_df['path'].str.contains('fake_long')), \
                "Fake long audio samples were not filtered out"

            # Verify all remaining samples are within bounds
            assert all(filtered_df['duration'] <= max_duration), "Some samples exceed max_duration"
            assert all(filtered_df['duration'] >= config.data.min_duration), "Some samples below min_duration"

        finally:
            # Restore original setting
            config.data.filter_by_duration = original_filter_setting

    def test_create_dataloader_transpose_features(self, data_manager, config):
        """Test create_dataloader with transpose_features parameter"""
        if not data_manager.is_dataset_available():
            pytest.skip("Common Voice dataset not available")

        # Setup processor
        data_manager.setup_processor(
            model_name=config.model.model_name,
            model_type="whisper",
            language=config.data.language,
            task=config.data.task
        )

        # Create dataset
        dataset = data_manager.create_dataset("train")

        # Test with transpose_features=False (default for Whisper)
        dataloader_whisper = data_manager.create_dataloader(
            dataset=dataset,
            batch_size=8,
            shuffle=False,
            transpose_features=False
        )
        assert dataloader_whisper is not None
        assert dataloader_whisper.collate_fn.transpose_features is False

        # Test with transpose_features=True (for Speech2Text)
        dataloader_speech2text = data_manager.create_dataloader(
            dataset=dataset,
            batch_size=8,
            shuffle=False,
            transpose_features=True
        )
        assert dataloader_speech2text is not None
        assert dataloader_speech2text.collate_fn.transpose_features is True

    def test_dataset_extracts_max_positions_for_speech2text(self, data_manager):
        """Test that CommonVoiceDataset extracts max_source_positions and max_target_positions for Speech2Text"""
        if not data_manager.is_dataset_available():
            pytest.skip("Common Voice dataset not available")

        # Setup Speech2Text processor
        data_manager.setup_processor(
            model_name="facebook/s2t-medium-mustc-multilingual-st",
            model_type="speech2text",
            language=data_manager.config.data.language
        )

        # Create dataset - should extract max positions in __init__
        dataset = data_manager.create_dataset("train")

        # Check that max positions were extracted
        assert dataset.max_source_positions is not None, "max_source_positions should be extracted for Speech2Text"
        assert dataset.max_target_positions is not None, "max_target_positions should be extracted for Speech2Text"
        assert isinstance(dataset.max_source_positions, int)
        assert isinstance(dataset.max_target_positions, int)
        assert dataset.max_source_positions > 0
        assert dataset.max_target_positions > 0

    def test_dataset_no_max_positions_for_whisper(self, data_manager):
        """Test that CommonVoiceDataset does not extract max positions for Whisper (stays None)"""
        if not data_manager.is_dataset_available():
            pytest.skip("Common Voice dataset not available")

        # Setup Whisper processor
        data_manager.setup_processor(
            model_name=data_manager.config.model.model_name,
            model_type="whisper",
            language=data_manager.config.data.language,
            task=data_manager.config.data.task
        )

        # Create dataset - should NOT extract max positions for Whisper
        dataset = data_manager.create_dataset("train")

        # Check that max positions remain None for Whisper
        assert dataset.max_source_positions is None, "max_source_positions should be None for Whisper"
        assert dataset.max_target_positions is None, "max_target_positions should be None for Whisper"

    def test_dataset_no_max_positions_for_speech2text_gpu_mode(self, data_manager, monkeypatch):
        """Test that CommonVoiceDataset does not extract max positions for Speech2Text in GPU mode"""
        if not data_manager.is_dataset_available():
            pytest.skip("Common Voice dataset not available")

        # Setup Speech2Text processor
        data_manager.setup_processor(
            model_name="facebook/s2t-medium-mustc-multilingual-st",
            model_type="speech2text",
            language=data_manager.config.data.language
        )

        # Switch to GPU mode
        monkeypatch.setattr(data_manager.config.data, "processing_and_augmentation_device", "cuda")

        # Create dataset in GPU mode - should NOT extract max positions
        dataset = data_manager.create_dataset("train")

        # Check that max positions remain None in GPU mode (not needed for GPU processing)
        assert dataset.max_source_positions is None, "max_source_positions should be None for Speech2Text in GPU mode"
        assert dataset.max_target_positions is None, "max_target_positions should be None for Speech2Text in GPU mode"

    def test_set_already_loaded_processor_whisper(self, data_manager, config, caplog):
        """Test set_already_loaded_processor with WhisperProcessor"""
        caplog.set_level(logging.INFO)

        # Create processor via setup_processor first
        processor = data_manager.setup_processor(
            model_name=config.model.model_name,
            model_type="whisper",
            language=config.data.language,
            task=config.data.task
        )

        # Clear processor and token IDs to test set_already_loaded_processor
        data_manager.processor = None
        data_manager.pad_token_id = None
        data_manager.forced_bos_token_id = None

        # Use set_already_loaded_processor to set the processor
        caplog.clear()
        data_manager.set_already_loaded_processor(processor)

        # Verify processor is set
        assert data_manager.processor is processor
        assert isinstance(data_manager.processor, WhisperProcessor)

        # Verify pad_token_id was extracted
        assert data_manager.pad_token_id is not None
        assert "Set pad_token_id:" in caplog.text

        # Whisper doesn't use forced_bos_token_id
        assert data_manager.forced_bos_token_id is None

    def test_set_already_loaded_processor_speech2text_multilingual(self, data_manager, caplog):
        """Test set_already_loaded_processor with Speech2Text multilingual processor"""
        caplog.set_level(logging.INFO)

        # Create multilingual Speech2Text processor with target language
        processor = data_manager.setup_processor(
            model_name="facebook/s2t-medium-mustc-multilingual-st",
            model_type="speech2text",
            language="ru"
        )

        # Clear processor and token IDs to test set_already_loaded_processor
        data_manager.processor = None
        data_manager.pad_token_id = None
        data_manager.forced_bos_token_id = None

        # Use set_already_loaded_processor to set the processor
        caplog.clear()
        data_manager.set_already_loaded_processor(processor)

        # Verify processor is set
        assert data_manager.processor is processor
        assert isinstance(data_manager.processor, Speech2TextProcessor)

        # Verify pad_token_id was extracted
        assert data_manager.pad_token_id is not None
        assert "Set pad_token_id:" in caplog.text

        # Verify forced_bos_token_id was extracted for multilingual model
        assert data_manager.forced_bos_token_id is not None
        assert "Set forced_bos_token_id for target language 'ru':" in caplog.text

    def test_set_already_loaded_processor_speech2text_monolingual(self, data_manager, caplog):
        """Test set_already_loaded_processor with Speech2Text monolingual processor (no forced_bos_token_id)"""
        caplog.set_level(logging.INFO)

        # Create monolingual Speech2Text processor (English LibriSpeech model)
        processor = data_manager.setup_processor(
            model_name="facebook/s2t-small-librispeech-asr",
            model_type="speech2text",
            language="en"
        )

        # Clear processor and token IDs to test set_already_loaded_processor
        data_manager.processor = None
        data_manager.pad_token_id = None
        data_manager.forced_bos_token_id = None

        # Use set_already_loaded_processor to set the processor
        caplog.clear()
        data_manager.set_already_loaded_processor(processor)

        # Verify processor is set
        assert data_manager.processor is processor
        assert isinstance(data_manager.processor, Speech2TextProcessor)

        # Verify pad_token_id was extracted
        assert data_manager.pad_token_id is not None
        assert "Set pad_token_id:" in caplog.text

        # Monolingual model should NOT have forced_bos_token_id
        assert data_manager.forced_bos_token_id is None

if __name__ == "__main__":
    """Run tests directly"""
    pytest.main([__file__, "-v"])