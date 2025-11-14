#!/usr/bin/env python3
"""
Unit tests for CommonVoiceDataset and DataLoader using a subset of real Common Voice data
"""

import sys
import pytest
import logging
import pickle
import pandas as pd
import torch
from pathlib import Path
import torchaudio
import warnings
from transformers import WhisperProcessor, Speech2TextProcessor

# Add project root to Python path
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

from src.config import load_config
from src.data import DataManager, CommonVoiceDataset, AudioPreprocessor, AudioAugmentationCPU, AudioCollator

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

# Define mock_getitem at module level to ensure it is picklable
def mock_getitem(self, idx):
    """Mock __getitem__ that returns None for testing empty batches"""
    return None

@pytest.fixture
def max_rows():
    """Maximum number of rows to load from TSV"""
    return 100

@pytest.fixture
def config():
    """Load default configuration"""
    return load_config("configs/default.yaml")

@pytest.fixture
def data_manager(config):
    """Initialize DataManager"""
    return DataManager(config)

@pytest.fixture
def processor(config):
    """Initialize WhisperProcessor"""
    return WhisperProcessor.from_pretrained(config.model.model_name)

@pytest.fixture(params=[8])
def batch_size(request):
    """Parameterized batch size"""
    return request.param

@pytest.fixture(params=[4])
def num_workers(request):
    """Parameterized number of workers"""
    return request.param

@pytest.fixture
def dataset_subset(data_manager, config, processor, max_rows):
    """Load a subset of real dataset (up to max_rows)"""
    if not data_manager.is_dataset_available():
        pytest.skip("Common Voice dataset not available")
    
    # Override load_dataset to read only up to max_rows
    original_load_dataset = data_manager.load_dataset
    
    def load_subset(split: str = "train") -> pd.DataFrame:
        split_mapping = {
            "train": ("cv22_train", data_manager.train_tsv),
            "validation": ("cv22_dev", data_manager.dev_tsv),
            "val": ("cv22_dev", data_manager.dev_tsv),
            "dev": ("cv22_dev", data_manager.dev_tsv),
            "test": ("cv22_test", data_manager.test_tsv)
        }
        cache_key, tsv_file = split_mapping[split]
        
        if cache_key in data_manager.dataset_cache:
            logger.info(f"Loading {split} subset from cache (key: {cache_key})...")
            return data_manager.dataset_cache[cache_key].iloc[:max_rows]
        
        if not tsv_file.exists():
            raise ValueError(f"TSV file not found: {tsv_file}")
        
        logger.info(f"Loading first {max_rows} rows from {tsv_file}...")
        df = pd.read_csv(tsv_file, sep='\t', encoding='utf-8', nrows=max_rows)
        
        # Calculate durations
        durations = []
        for idx, row in df.iterrows():
            audio_path = data_manager.clips_dir / row['path']
            try:
                if audio_path.exists():
                    info = torchaudio.info(str(audio_path))
                    duration = info.num_frames / info.sample_rate
                else:
                    logger.warning(f"Audio file not found: {audio_path}")
                    duration = 0.0
                durations.append(duration)
            except Exception as e:
                logger.warning(f"Could not get duration for {audio_path}: {e}")
                durations.append(0.0)
        
        df['duration'] = durations
        df = data_manager._apply_filters(df)
        data_manager.dataset_cache[cache_key] = df
        return df.iloc[:max_rows]

    data_manager.load_dataset = load_subset
    data_manager.setup_processor(
        model_name=config.model.model_name,
        model_type=config.model.model_type,
        language=config.data.language,
        task=config.data.task
    )
    dataset = data_manager.create_dataset("train")
    data_manager.load_dataset = original_load_dataset  # Restore original method
    return dataset

class TestCommonVoiceDataset:
    """Test suite for CommonVoiceDataset with real data subset"""

    def test_initialization(self, dataset_subset, data_manager, config):
        """Test dataset initialization"""
        assert isinstance(dataset_subset.dataframe, pd.DataFrame)
        assert dataset_subset.audio_dir == data_manager.clips_dir
        assert dataset_subset.config == config
        assert dataset_subset.split == "train"
        assert dataset_subset.sample_rate == config.data.sample_rate
        assert isinstance(dataset_subset.preprocessor, AudioPreprocessor)
        assert dataset_subset.preprocessor.sample_rate == config.data.sample_rate
        assert dataset_subset.preprocessor.normalize == config.data.normalize
        assert dataset_subset.preprocessor.trim_silence == config.data.trim_silence
        if config.data.processing_and_augmentation_device == "cpu" and config.data.augmentation.enabled:
            assert isinstance(dataset_subset.augmentation, AudioAugmentationCPU)
        else:
            assert dataset_subset.augmentation is None
        if config.data.processing_and_augmentation_device == "cpu":
            assert isinstance(dataset_subset.processor, (WhisperProcessor, Speech2TextProcessor))

    def test_len(self, dataset_subset, max_rows):
        """Test __len__ method"""
        assert len(dataset_subset) <= max_rows
        assert len(dataset_subset) == len(dataset_subset.dataframe)

    def test_getitem_missing_file(self, dataset_subset, caplog):
        """Test __getitem__ with missing audio file"""
        caplog.set_level(logging.WARNING)
        
        found_missing = False
        for idx in range(len(dataset_subset)):
            row = dataset_subset.dataframe.iloc[idx]
            audio_path = dataset_subset.audio_dir / row["path"]
            if not audio_path.exists():
                result = dataset_subset[idx]
                assert result is None
                assert f"Audio file not found: {audio_path}" in caplog.text
                found_missing = True
                break
        
        if not found_missing:
            pytest.skip("No missing files found in dataset subset")

    def test_getitem_valid_cpu(self, dataset_subset, monkeypatch):
        """Test __getitem__ with valid audio file in CPU mode"""
        monkeypatch.setattr(dataset_subset.config.data, "processing_and_augmentation_device", "cpu")
        monkeypatch.setattr(dataset_subset.config.data.augmentation, "enabled", False)
        for idx in range(len(dataset_subset)):
            result = dataset_subset[idx]
            if result is not None:
                assert torch.is_tensor(result["input_features"])
                assert len(result["input_features"].shape) == 2
                assert result["input_features"].shape[0] == 80
                assert torch.is_tensor(result["labels"])
                assert len(result["labels"].shape) == 1
                assert result["input_features"].device == torch.device("cpu")
                assert result["labels"].device == torch.device("cpu")
                assert isinstance(result["path"], str)
                assert isinstance(result["duration"], float)
                assert result["client_id"] == dataset_subset.dataframe.iloc[idx].get("client_id", "unknown")
                assert result["up_votes"] == dataset_subset.dataframe.iloc[idx].get("up_votes", 0)
                assert result["down_votes"] == dataset_subset.dataframe.iloc[idx].get("down_votes", 0)
                assert dataset_subset.preprocessor.min_duration <= result["duration"] <= dataset_subset.preprocessor.max_duration
                break
        else:
            pytest.skip("No valid audio files found in dataset subset")

    def test_getitem_valid_cuda(self, dataset_subset, monkeypatch):
        """Test __getitem__ with valid audio file in CUDA mode"""
        monkeypatch.setattr(dataset_subset.config.data, "processing_and_augmentation_device", "cuda")
        for idx in range(len(dataset_subset)):
            result = dataset_subset[idx]
            if result is not None:
                assert torch.is_tensor(result["audio"])
                assert len(result["audio"].shape) == 2
                assert result["audio"].shape[0] == 1
                assert result["audio"].device == torch.device("cpu")
                assert isinstance(result["text"], str)
                assert isinstance(result["path"], str)
                assert isinstance(result["duration"], float)
                assert result["client_id"] == dataset_subset.dataframe.iloc[idx].get("client_id", "unknown")
                assert result["up_votes"] == dataset_subset.dataframe.iloc[idx].get("up_votes", 0)
                assert result["down_votes"] == dataset_subset.dataframe.iloc[idx].get("down_votes", 0)
                assert dataset_subset.preprocessor.min_duration <= result["duration"] <= dataset_subset.preprocessor.max_duration
                break
        else:
            pytest.skip("No valid audio files found in dataset subset")

    def test_getitem_sample_rate_mismatch_cpu(self, dataset_subset, monkeypatch, caplog):
        """Test __getitem__ with sample rate mismatch in CPU mode"""
        monkeypatch.setattr(dataset_subset.config.data, "processing_and_augmentation_device", "cpu")
        monkeypatch.setattr(dataset_subset.config.data.augmentation, "enabled", False)
        caplog.set_level(logging.WARNING)
        
        found_mismatch = False
        for idx in range(len(dataset_subset)):
            row = dataset_subset.dataframe.iloc[idx]
            audio_path = dataset_subset.audio_dir / row["path"]
            if audio_path.exists():
                info = torchaudio.info(audio_path)
                if info.sample_rate != dataset_subset.sample_rate:
                    result = dataset_subset[idx]
                    if result is not None:
                        assert torch.is_tensor(result["input_features"])
                        assert len(result["input_features"].shape) == 2
                        assert result["input_features"].shape[0] == 80
                        assert result["input_features"].device == torch.device("cpu")
                        found_mismatch = True
                        break
        
        if not found_mismatch:
            pytest.skip("No sample rate mismatches found in dataset subset")

    def test_getitem_sample_rate_mismatch_cuda(self, dataset_subset, monkeypatch, caplog):
        """Test __getitem__ with sample rate mismatch in CUDA mode"""
        monkeypatch.setattr(dataset_subset.config.data, "processing_and_augmentation_device", "cuda")
        caplog.set_level(logging.WARNING)
        
        found_mismatch = False
        for idx in range(len(dataset_subset)):
            row = dataset_subset.dataframe.iloc[idx]
            audio_path = dataset_subset.audio_dir / row["path"]
            if audio_path.exists():
                info = torchaudio.info(audio_path)
                if info.sample_rate != dataset_subset.sample_rate:
                    result = dataset_subset[idx]
                    if result is not None:
                        assert torch.is_tensor(result["audio"])
                        assert len(result["audio"].shape) == 2
                        assert result["audio"].shape[0] == 1
                        assert result["audio"].device == torch.device("cpu")
                        found_mismatch = True
                        break
        
        if not found_mismatch:
            pytest.skip("No sample rate mismatches found in dataset subset")

    def test_getitem_multi_channel_cpu(self, dataset_subset, monkeypatch):
        """Test __getitem__ with multi-channel audio in CPU mode"""
        monkeypatch.setattr(dataset_subset.config.data, "processing_and_augmentation_device", "cpu")
        monkeypatch.setattr(dataset_subset.config.data.augmentation, "enabled", False)
        found_multi_channel = False
        for idx in range(len(dataset_subset)):
            row = dataset_subset.dataframe.iloc[idx]
            audio_path = dataset_subset.audio_dir / row["path"]
            if audio_path.exists():
                audio, _ = torchaudio.load(audio_path)
                if audio.shape[0] > 1:
                    result = dataset_subset[idx]
                    if result is not None:
                        assert torch.is_tensor(result["input_features"])
                        assert len(result["input_features"].shape) == 2
                        assert result["input_features"].shape[0] == 80
                        assert result["input_features"].device == torch.device("cpu")
                        found_multi_channel = True
                        break
        
        if not found_multi_channel:
            pytest.skip("No multi-channel audio found in dataset subset")

    def test_getitem_multi_channel_cuda(self, dataset_subset, monkeypatch):
        """Test __getitem__ with multi-channel audio in CUDA mode"""
        monkeypatch.setattr(dataset_subset.config.data, "processing_and_augmentation_device", "cuda")
        found_multi_channel = False
        for idx in range(len(dataset_subset)):
            row = dataset_subset.dataframe.iloc[idx]
            audio_path = dataset_subset.audio_dir / row["path"]
            if audio_path.exists():
                audio, _ = torchaudio.load(audio_path)
                if audio.shape[0] > 1:
                    result = dataset_subset[idx]
                    if result is not None:
                        assert torch.is_tensor(result["audio"])
                        assert len(result["audio"].shape) == 2
                        assert result["audio"].shape[0] == 1
                        assert result["audio"].device == torch.device("cpu")
                        found_multi_channel = True
                        break
        
        if not found_multi_channel:
            pytest.skip("No multi-channel audio found in dataset subset")

    def test_getitem_error_handling(self, dataset_subset, caplog, monkeypatch):
        """Test __getitem__ error handling"""
        caplog.set_level(logging.ERROR)
        
        def mock_load(*args, **kwargs):
            raise Exception("Load error")
        
        monkeypatch.setattr("torchaudio.load", mock_load)
        
        for idx in range(len(dataset_subset)):
            audio_path = dataset_subset.audio_dir / dataset_subset.dataframe.iloc[idx]["path"]
            if audio_path.exists():
                result = dataset_subset[idx]
                assert result is None
                assert f"Error loading audio {audio_path}: Load error" in caplog.text
                break
        else:
            pytest.skip("No valid audio files found in dataset subset")

    def test_getitem_with_augmentation(self, dataset_subset, monkeypatch):
        """Test __getitem__ with augmentations enabled in CPU mode"""
        monkeypatch.setattr(dataset_subset.config.data, "processing_and_augmentation_device", "cpu")
        monkeypatch.setattr(dataset_subset.config.data.augmentation, "enabled", True)
        monkeypatch.setattr(dataset_subset.config.data.augmentation, "add_noise", True)
        monkeypatch.setattr(dataset_subset.config.data.augmentation, "noise_probability", 1.0)
        monkeypatch.setattr(dataset_subset.config.data.augmentation, "noise_factor", 0.1)

        for idx in range(len(dataset_subset)):
            row = dataset_subset.dataframe.iloc[idx]
            audio_path = dataset_subset.audio_dir / row["path"]
            if audio_path.exists():
                audio_orig, sr = torchaudio.load(audio_path)
                logger.info(f"Original audio sample rate: {sr}")
                processed = dataset_subset.preprocessor.preprocess(audio_orig, sr)
                if processed is None:
                    logger.warning(f"Preprocessing failed for {audio_path}")
                    continue
                audio_orig = processed["audio"]
                logger.info(f"Processed audio sample rate: {dataset_subset.sample_rate}")
                result = dataset_subset[idx]
                if result is not None:
                    input_features = result["input_features"]
                    assert torch.is_tensor(input_features)
                    assert len(input_features.shape) == 2
                    assert input_features.shape[0] == 80
                    orig_features = dataset_subset.processor.feature_extractor(
                        audio_orig.squeeze(0).numpy(),
                        sampling_rate=dataset_subset.sample_rate
                    ).input_features[0]
                    assert not torch.allclose(input_features, torch.tensor(orig_features), atol=1e-2)
                    logger.info("Augmentation test passed: features differ due to noise")
                    break
        else:
            pytest.skip("No valid audio files found in dataset subset")

    def test_serialization(self, dataset_subset):
        """Test dataset serialization"""
        try:
            pickle.dumps(dataset_subset)
            assert True, "Serialization successful"
        except Exception as e:
            pytest.fail(f"Serialization failed: {e}")

    def test_repr(self, dataset_subset):
        """Test __repr__ method"""
        expected = (
            f"CommonVoiceDataset(split=train, "
            f"samples={len(dataset_subset)}, "
            f"sample_rate={dataset_subset.sample_rate}, "
            f"audio_dir={dataset_subset.audio_dir})"
        )
        assert str(dataset_subset) == expected

class TestAudioCollator:
    """Test suite for AudioCollator in both CPU and CUDA modes"""

    @pytest.fixture
    def mock_samples_cpu_whisper(self):
        """Mock samples for CPU mode (Whisper format)"""
        # Whisper: input_features shape (freq=80, time=3000) - fixed time dimension
        # Whisper processor always returns fixed-length features
        return [
            {
                "input_features": torch.randn(80, 3000),  # (freq, time) - fixed
                "labels": torch.tensor([1, 2, 3, 4, 5]),
                "path": "sample1.mp3",
                "duration": 1.0,
                "client_id": "client1",
                "up_votes": 2,
                "down_votes": 0,
            },
            {
                "input_features": torch.randn(80, 3000),  # (freq, time) - fixed
                "labels": torch.tensor([6, 7, 8]),
                "path": "sample2.mp3",
                "duration": 1.5,
                "client_id": "client2",
                "up_votes": 3,
                "down_votes": 1,
            },
            {
                "input_features": torch.randn(80, 3000),  # (freq, time) - fixed
                "labels": torch.tensor([9, 10]),
                "path": "sample3.mp3",
                "duration": 1.2,
                "client_id": "client3",
                "up_votes": 1,
                "down_votes": 0,
            },
        ]

    @pytest.fixture
    def mock_samples_cpu_speech2text(self):
        """Mock samples for CPU mode (Speech2Text format)"""
        # Speech2Text: input_features shape (freq, time) before transpose
        # Using fixed time dimension for simplicity (same as Whisper)
        return [
            {
                "input_features": torch.randn(80, 3000),  # (freq, time) - fixed
                "labels": torch.tensor([1, 2, 3, 4, 5]),
                "path": "sample1.mp3",
                "duration": 1.0,
                "client_id": "client1",
                "up_votes": 2,
                "down_votes": 0,
            },
            {
                "input_features": torch.randn(80, 3000),  # (freq, time) - fixed
                "labels": torch.tensor([6, 7, 8]),
                "path": "sample2.mp3",
                "duration": 1.5,
                "client_id": "client2",
                "up_votes": 3,
                "down_votes": 1,
            },
        ]

    @pytest.fixture
    def mock_samples_cpu_speech2text_variable(self):
        """Mock samples for CPU mode (Speech2Text format) with VARIABLE time lengths"""
        # Speech2Text with different time dimensions to test padding
        return [
            {
                "input_features": torch.randn(80, 500),  # (freq, time) - SHORT
                "labels": torch.tensor([1, 2, 3]),
                "path": "sample1.mp3",
                "duration": 0.5,
                "client_id": "client1",
                "up_votes": 2,
                "down_votes": 0,
            },
            {
                "input_features": torch.randn(80, 1500),  # (freq, time) - MEDIUM
                "labels": torch.tensor([4, 5, 6, 7]),
                "path": "sample2.mp3",
                "duration": 1.5,
                "client_id": "client2",
                "up_votes": 3,
                "down_votes": 1,
            },
            {
                "input_features": torch.randn(80, 2000),  # (freq, time) - LONG
                "labels": torch.tensor([8, 9]),
                "path": "sample3.mp3",
                "duration": 2.0,
                "client_id": "client3",
                "up_votes": 1,
                "down_votes": 0,
            },
        ]

    @pytest.fixture
    def mock_samples_cuda(self):
        """Mock samples for CUDA mode (raw audio)"""
        return [
            {
                "audio": torch.randn(1, 16000),  # (1, time)
                "text": "первый текст",
                "path": "sample1.mp3",
                "duration": 1.0,
                "client_id": "client1",
                "up_votes": 2,
                "down_votes": 0,
            },
            {
                "audio": torch.randn(1, 24000),  # (1, time) - different length
                "text": "второй текст",
                "path": "sample2.mp3",
                "duration": 1.5,
                "client_id": "client2",
                "up_votes": 3,
                "down_votes": 1,
            },
            {
                "audio": torch.randn(1, 19200),  # (1, time) - different length
                "text": "третий текст",
                "path": "sample3.mp3",
                "duration": 1.2,
                "client_id": "client3",
                "up_votes": 1,
                "down_votes": 0,
            },
        ]

    def test_cpu_whisper_batching(self, config, mock_samples_cpu_whisper):
        """Test CPU mode batching for Whisper (no transpose)"""
        collator = AudioCollator(config.data, model_type="whisper", transpose_features=False)

        batch = collator(mock_samples_cpu_whisper)

        assert batch is not None
        assert "input_features" in batch
        assert "labels" in batch
        # Metadata fields removed (handled by remove_unused_columns in Trainer)
        assert "paths" not in batch
        assert "durations" not in batch

        # Check shapes
        assert batch["input_features"].shape[0] == 3  # batch size
        assert batch["input_features"].shape[1] == 80  # freq dimension
        assert batch["input_features"].shape[2] == 3000  # time dimension (fixed for Whisper)
        assert batch["labels"].shape[0] == 3  # batch size
        assert batch["labels"].shape[1] == 5  # max label length

    def test_cpu_whisper_padding(self, config, mock_samples_cpu_whisper):
        """Test CPU mode padding for Whisper (labels should be padded with -100)"""
        collator = AudioCollator(config.data, model_type="whisper", transpose_features=False)

        batch = collator(mock_samples_cpu_whisper)
        assert batch is not None  # Type guard

        # For Whisper, input_features have fixed time dimension (3000), so no padding needed
        # Just check that all samples have the same shape
        assert batch["input_features"].shape == (3, 80, 3000)

        # Check labels padding (should be padded with -100)
        # First sample has 5 labels, second has 3 labels (should be padded to 5)
        assert batch["labels"][0, :5].tolist() != [-100, -100, -100, -100, -100]  # Real labels
        assert batch["labels"][1, 3:].tolist() == [-100, -100]  # Padded labels
        assert batch["labels"][2, 2:].tolist() == [-100, -100, -100]  # Padded labels

    def test_cpu_speech2text_transpose(self, config, mock_samples_cpu_speech2text):
        """Test CPU mode with transpose for Speech2Text"""
        collator = AudioCollator(config.data, model_type="speech2text", transpose_features=True)

        batch = collator(mock_samples_cpu_speech2text)

        assert batch is not None
        # After transpose: (batch, time, freq)
        assert batch["input_features"].shape[0] == 2  # batch size
        assert batch["input_features"].shape[1] == 3000  # time dimension (fixed)
        assert batch["input_features"].shape[2] == 80  # freq dimension

        # Check attention_mask is created for Speech2Text
        assert "attention_mask" in batch
        assert batch["attention_mask"].shape == (2, 3000)  # (batch, time)

        # All samples have same length, so all attention_mask should be 1
        assert batch["attention_mask"][0].sum() == 3000
        assert batch["attention_mask"][1].sum() == 3000

    def test_cpu_speech2text_attention_mask(self, config, mock_samples_cpu_speech2text):
        """Test attention_mask creation for Speech2Text"""
        collator = AudioCollator(config.data, model_type="speech2text", transpose_features=True)

        batch = collator(mock_samples_cpu_speech2text)
        assert batch is not None  # Type guard

        # Check attention_mask shape and values
        assert "attention_mask" in batch
        assert batch["attention_mask"].dtype == torch.long
        assert batch["attention_mask"].shape[0] == 2  # batch size
        assert batch["attention_mask"].shape[1] == 3000  # time length (fixed)

        # Verify attention_mask values (1 for real data, 0 for padding)
        assert torch.all((batch["attention_mask"] == 0) | (batch["attention_mask"] == 1))

        # All samples have same length, all attention_mask should be 1
        assert batch["attention_mask"][0].sum() == 3000  # All real data
        assert batch["attention_mask"][1].sum() == 3000  # All real data

    def test_cpu_whisper_no_attention_mask(self, config, mock_samples_cpu_whisper):
        """Test that Whisper does not create attention_mask"""
        collator = AudioCollator(config.data, model_type="whisper", transpose_features=False)

        batch = collator(mock_samples_cpu_whisper)
        assert batch is not None  # Type guard

        # Whisper should NOT have attention_mask
        assert "attention_mask" not in batch

    def test_cpu_speech2text_variable_lengths_padding(self, config, mock_samples_cpu_speech2text_variable):
        """Test Speech2Text with variable time lengths - padding correctness"""
        collator = AudioCollator(config.data, model_type="speech2text", transpose_features=True)

        batch = collator(mock_samples_cpu_speech2text_variable)
        assert batch is not None  # Type guard

        # After transpose: (batch, time, freq)
        assert batch["input_features"].shape[0] == 3  # batch size
        assert batch["input_features"].shape[1] == 2000  # max time dimension (padded to longest)
        assert batch["input_features"].shape[2] == 80  # freq dimension

        # Check that padding happened correctly
        # Sample 0: original time=500, should be padded to 2000
        # Sample 1: original time=1500, should be padded to 2000
        # Sample 2: original time=2000, should NOT be padded

        # Verify attention_mask reflects the original lengths
        assert "attention_mask" in batch
        assert batch["attention_mask"].shape == (3, 2000)  # (batch, max_time)

        # Check attention_mask values
        assert batch["attention_mask"][0].sum() == 500  # Sample 0: 500 real, 1500 padding
        assert batch["attention_mask"][1].sum() == 1500  # Sample 1: 1500 real, 500 padding
        assert batch["attention_mask"][2].sum() == 2000  # Sample 2: all real

        # Verify padding positions (attention_mask should be 0 for padding)
        assert torch.all(batch["attention_mask"][0, :500] == 1)  # Real data
        assert torch.all(batch["attention_mask"][0, 500:] == 0)  # Padding
        assert torch.all(batch["attention_mask"][1, :1500] == 1)  # Real data
        assert torch.all(batch["attention_mask"][1, 1500:] == 0)  # Padding
        assert torch.all(batch["attention_mask"][2, :] == 1)  # All real (no padding)

    def test_cpu_speech2text_variable_lengths_no_transpose(self, config, mock_samples_cpu_speech2text_variable):
        """Test Speech2Text with variable time lengths - no transpose (Whisper format)"""
        collator = AudioCollator(config.data, model_type="speech2text", transpose_features=False)

        batch = collator(mock_samples_cpu_speech2text_variable)
        assert batch is not None  # Type guard

        # Without transpose: (batch, freq, time)
        assert batch["input_features"].shape[0] == 3  # batch size
        assert batch["input_features"].shape[1] == 80  # freq dimension
        assert batch["input_features"].shape[2] == 2000  # max time dimension (padded to longest)

        # Check attention_mask still created correctly
        assert "attention_mask" in batch
        assert batch["attention_mask"].shape == (3, 2000)  # (batch, max_time)

        # Verify same attention_mask as transposed version
        assert batch["attention_mask"][0].sum() == 500
        assert batch["attention_mask"][1].sum() == 1500
        assert batch["attention_mask"][2].sum() == 2000

    def test_cpu_whisper_variable_lengths_padding(self, config):
        """Test Whisper format with variable time lengths (edge case)"""
        # Create mock samples with variable lengths (simulating non-standard Whisper)
        mock_samples = [
            {
                "input_features": torch.randn(80, 1000),  # SHORT
                "labels": torch.tensor([1, 2]),
                "path": "sample1.mp3",
                "duration": 1.0,
                "client_id": "client1",
                "up_votes": 2,
                "down_votes": 0,
            },
            {
                "input_features": torch.randn(80, 2500),  # LONG
                "labels": torch.tensor([3, 4, 5]),
                "path": "sample2.mp3",
                "duration": 2.5,
                "client_id": "client2",
                "up_votes": 1,
                "down_votes": 0,
            },
        ]

        collator = AudioCollator(config.data, model_type="whisper", transpose_features=False)
        batch = collator(mock_samples)
        assert batch is not None  # Type guard

        # Check that padding worked correctly
        assert batch["input_features"].shape == (2, 80, 2500)  # (batch, freq, max_time)

        # Whisper doesn't create attention_mask
        assert "attention_mask" not in batch

    def test_cuda_batching(self, config, mock_samples_cuda):
        """Test CUDA mode batching (raw audio)"""
        # Mock CUDA mode in config
        config.data.processing_and_augmentation_device = "cuda"
        collator = AudioCollator(config.data, model_type="whisper", transpose_features=False)

        batch = collator(mock_samples_cuda)

        assert batch is not None
        assert "audio" in batch
        assert "texts" in batch
        # Metadata fields removed (handled by remove_unused_columns in Trainer)
        assert "paths" not in batch
        assert "durations" not in batch

        # Check shapes
        assert batch["audio"].shape[0] == 3  # batch size
        assert batch["audio"].shape[1] == 1  # mono channel
        assert batch["audio"].shape[2] == 24000  # time dimension (padded to max)

        # Check texts
        assert len(batch["texts"]) == 3

    def test_cuda_padding(self, config, mock_samples_cuda):
        """Test CUDA mode audio padding"""
        # Mock CUDA mode in config
        config.data.processing_and_augmentation_device = "cuda"
        collator = AudioCollator(config.data, model_type="whisper", transpose_features=False)

        batch = collator(mock_samples_cuda)
        assert batch is not None  # Type guard

        # First sample has time=16000, should be padded to 24000
        assert batch["audio"][0, 0, 16000:].abs().sum() == 0  # Check padding is zero

        # Third sample has time=19200, should be padded to 24000
        assert batch["audio"][2, 0, 19200:].abs().sum() == 0  # Check padding is zero

    def test_empty_batch_returns_none(self, config):
        """Test that collator returns None for empty batch"""
        collator = AudioCollator(config.data, model_type="whisper", transpose_features=False)

        # Empty batch
        batch = collator([])
        assert batch is None

        # Batch with all None samples (using type: ignore for test purposes)
        batch = collator([None, None, None])  # type: ignore
        assert batch is None

    def test_batch_with_some_none_samples(self, config, mock_samples_cpu_whisper):
        """Test that collator filters None samples correctly"""
        collator = AudioCollator(config.data, model_type="whisper", transpose_features=False)

        # Mix valid and None samples
        mixed_batch = [mock_samples_cpu_whisper[0], None, mock_samples_cpu_whisper[1], None]
        batch = collator(mixed_batch)

        assert batch is not None
        assert batch["input_features"].shape[0] == 2  # Only 2 valid samples
        assert batch["labels"].shape[0] == 2

    def test_single_sample_batch(self, config, mock_samples_cpu_whisper):
        """Test batching with single sample"""
        collator = AudioCollator(config.data, model_type="whisper", transpose_features=False)

        batch = collator([mock_samples_cpu_whisper[0]])

        assert batch is not None
        assert batch["input_features"].shape[0] == 1  # batch size = 1
        assert batch["input_features"].shape[1] == 80
        assert batch["input_features"].shape[2] == 3000  # Fixed time dimension
        assert batch["labels"].shape[0] == 1

    def test_variable_length_padding(self, config):
        """Test padding labels with highly variable sequence lengths"""
        # Create samples with very different label lengths (but same input_features shape for Whisper)
        samples = [
            {
                "input_features": torch.randn(80, 3000),
                "labels": torch.tensor([1]),
                "path": "short.mp3",
                "duration": 0.5,
                "client_id": "client1",
                "up_votes": 0,
                "down_votes": 0,
            },
            {
                "input_features": torch.randn(80, 3000),
                "labels": torch.tensor([2, 3, 4, 5, 6, 7, 8, 9, 10]),
                "path": "long.mp3",
                "duration": 3.0,
                "client_id": "client2",
                "up_votes": 0,
                "down_votes": 0,
            },
        ]

        collator = AudioCollator(config.data, model_type="whisper", transpose_features=False)
        batch = collator(samples)
        assert batch is not None  # Type guard

        # Check shapes
        assert batch["input_features"].shape == (2, 80, 3000)  # Fixed time dimension
        assert batch["labels"].shape == (2, 9)  # Padded to max labels=9

        # Verify labels padding (short labels should be padded with -100)
        assert batch["labels"][0, 1:].tolist() == [-100] * 8  # Short labels padded

    def test_transpose_does_not_affect_attention_mask(self, config, mock_samples_cpu_speech2text):
        """Test that attention_mask is not transposed (stays as (batch, time))"""
        collator = AudioCollator(config.data, model_type="speech2text", transpose_features=True)

        batch = collator(mock_samples_cpu_speech2text)
        assert batch is not None  # Type guard

        # input_features should be (batch, time, freq) after transpose
        assert batch["input_features"].shape == (2, 3000, 80)

        # attention_mask should be (batch, time) - NOT transposed
        assert batch["attention_mask"].shape == (2, 3000)

    def test_invalid_device_raises_error(self, config):
        """Test that invalid device raises ValueError"""
        config.data.processing_and_augmentation_device = "invalid"
        collator = AudioCollator(config.data, model_type="whisper", transpose_features=False)

        with pytest.raises(ValueError, match="Unsupported processing device"):
            collator([{"input_features": torch.randn(80, 100), "labels": torch.tensor([1, 2, 3])}])

    def test_only_model_inputs_returned(self, config, mock_samples_cpu_whisper):
        """Test that collator returns only model inputs (no metadata)"""
        collator = AudioCollator(config.data, model_type="whisper", transpose_features=False)

        batch = collator(mock_samples_cpu_whisper)
        assert batch is not None  # Type guard

        # Only model inputs should be in batch (metadata removed by remove_unused_columns)
        assert "input_features" in batch
        assert "labels" in batch
        # Metadata should NOT be in batch
        assert "paths" not in batch
        assert "client_ids" not in batch
        assert "durations" not in batch
        assert "up_votes" not in batch
        assert "down_votes" not in batch

class TestDataLoader:
    """Test suite for DataLoader with CommonVoiceDataset"""

    def test_dataloader_serialization_windows(self, data_manager, config, max_rows, batch_size, num_workers):
        """Test DataLoader serialization with workers on Windows"""
        if not data_manager.is_dataset_available():
            pytest.skip("Common Voice dataset not available")
        
        # Override load_dataset to read only up to max_rows
        original_load_dataset = data_manager.load_dataset
        
        def load_subset(split: str = "train") -> pd.DataFrame:
            split_mapping = {
                "train": ("cv22_train", data_manager.train_tsv),
                "validation": ("cv22_dev", data_manager.dev_tsv),
                "val": ("cv22_dev", data_manager.dev_tsv),
                "dev": ("cv22_dev", data_manager.dev_tsv),
                "test": ("cv22_test", data_manager.test_tsv)
            }
            cache_key, tsv_file = split_mapping[split]
            
            if cache_key in data_manager.dataset_cache:
                logger.info(f"Loading {split} subset from cache (key: {cache_key})...")
                return data_manager.dataset_cache[cache_key].iloc[:max_rows]
            
            if not tsv_file.exists():
                raise ValueError(f"TSV file not found: {tsv_file}")
            
            logger.info(f"Loading first {max_rows} rows from {tsv_file}...")
            df = pd.read_csv(tsv_file, sep='\t', encoding='utf-8', nrows=max_rows)
            
            # Calculate durations
            durations = []
            for idx, row in df.iterrows():
                audio_path = data_manager.clips_dir / row['path']
                try:
                    if audio_path.exists():
                        info = torchaudio.info(str(audio_path))
                        duration = info.num_frames / info.sample_rate
                    else:
                        logger.warning(f"Audio file not found: {audio_path}")
                        duration = 0.0
                    durations.append(duration)
                except Exception as e:
                    logger.warning(f"Could not get duration for {audio_path}: {e}")
                    durations.append(0.0)
            
            df['duration'] = durations
            df = data_manager._apply_filters(df)
            data_manager.dataset_cache[cache_key] = df
            return df.iloc[:max_rows]

        data_manager.load_dataset = load_subset

        # Setup processor
        data_manager.setup_processor(
            model_name=config.model.model_name,
            model_type="whisper",
            language=config.data.language,
            task=config.data.task
        )

        # Create dataset and dataloader
        dataset = data_manager.create_dataset("train")
        dataloader = data_manager.create_dataloader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        try:
            pickle.dumps(dataloader)
            assert True, "DataLoader serialization successful"
        except Exception as e:
            pytest.fail(f"DataLoader serialization failed: {e}")
        finally:
            data_manager.load_dataset = original_load_dataset  # Restore original method

    def test_dataloader_cpu_batch(self, data_manager, config, monkeypatch, max_rows, batch_size, num_workers):
        """Test DataLoader batch correctness in CPU mode"""
        if not data_manager.is_dataset_available():
            pytest.skip("Common Voice dataset not available")
        
        monkeypatch.setattr(config.data, "processing_and_augmentation_device", "cpu")
        
        # Override load_dataset to read only up to max_rows
        original_load_dataset = data_manager.load_dataset
        
        def load_subset(split: str = "train") -> pd.DataFrame:
            split_mapping = {
                "train": ("cv22_train", data_manager.train_tsv),
                "validation": ("cv22_dev", data_manager.dev_tsv),
                "val": ("cv22_dev", data_manager.dev_tsv),
                "dev": ("cv22_dev", data_manager.dev_tsv),
                "test": ("cv22_test", data_manager.test_tsv)
            }
            cache_key, tsv_file = split_mapping[split]
            
            if cache_key in data_manager.dataset_cache:
                logger.info(f"Loading {split} subset from cache (key: {cache_key})...")
                return data_manager.dataset_cache[cache_key].iloc[:max_rows]
            
            if not tsv_file.exists():
                raise ValueError(f"TSV file not found: {tsv_file}")
            
            logger.info(f"Loading first {max_rows} rows from {tsv_file}...")
            df = pd.read_csv(tsv_file, sep='\t', encoding='utf-8', nrows=max_rows)
            
            # Calculate durations
            durations = []
            for idx, row in df.iterrows():
                audio_path = data_manager.clips_dir / row['path']
                try:
                    if audio_path.exists():
                        info = torchaudio.info(str(audio_path))
                        duration = info.num_frames / info.sample_rate
                    else:
                        logger.warning(f"Audio file not found: {audio_path}")
                        duration = 0.0
                    durations.append(duration)
                except Exception as e:
                    logger.warning(f"Could not get duration for {audio_path}: {e}")
                    durations.append(0.0)
            
            df['duration'] = durations
            df = data_manager._apply_filters(df)
            data_manager.dataset_cache[cache_key] = df
            return df.iloc[:max_rows]

        data_manager.load_dataset = load_subset

        # Setup processor
        data_manager.setup_processor(
            model_name=config.model.model_name,
            model_type="whisper",
            language=config.data.language,
            task=config.data.task
        )

        # Create dataset and dataloader
        dataset = data_manager.create_dataset("train")
        dataloader = data_manager.create_dataloader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        try:
            for batch in dataloader:
                assert isinstance(batch, dict)
                assert "input_features" in batch
                assert "labels" in batch
                # Metadata removed by collator
                assert "durations" not in batch
                assert "paths" not in batch
                assert torch.is_tensor(batch["input_features"])
                assert torch.is_tensor(batch["labels"])
                assert batch["input_features"].shape[0] <= batch_size
                assert batch["input_features"].shape[1] == 80
                assert batch["labels"].shape[0] <= batch_size
                assert batch["input_features"].device == torch.device("cpu")
                assert batch["labels"].device == torch.device("cpu")
                break
            else:
                pytest.skip("No valid batches produced by DataLoader")
        finally:
            data_manager.load_dataset = original_load_dataset  # Restore original method

    def test_dataloader_cuda_batch(self, data_manager, config, monkeypatch, max_rows, batch_size, num_workers):
        """Test DataLoader batch correctness in CUDA mode"""
        if not data_manager.is_dataset_available():
            pytest.skip("Common Voice dataset not available")

        monkeypatch.setattr(config.data, "processing_and_augmentation_device", "cuda")

        # Override load_dataset to read only up to max_rows
        original_load_dataset = data_manager.load_dataset

        def load_subset(split: str = "train") -> pd.DataFrame:
            split_mapping = {
                "train": ("cv22_train", data_manager.train_tsv),
                "validation": ("cv22_dev", data_manager.dev_tsv),
                "val": ("cv22_dev", data_manager.dev_tsv),
                "dev": ("cv22_dev", data_manager.dev_tsv),
                "test": ("cv22_test", data_manager.test_tsv)
            }
            cache_key, tsv_file = split_mapping[split]

            if cache_key in data_manager.dataset_cache:
                logger.info(f"Loading {split} subset from cache (key: {cache_key})...")
                return data_manager.dataset_cache[cache_key].iloc[:max_rows]

            if not tsv_file.exists():
                raise ValueError(f"TSV file not found: {tsv_file}")

            logger.info(f"Loading first {max_rows} rows from {tsv_file}...")
            df = pd.read_csv(tsv_file, sep='\t', encoding='utf-8', nrows=max_rows)

            # Calculate durations
            durations = []
            for idx, row in df.iterrows():
                audio_path = data_manager.clips_dir / str(row['path'])
                try:
                    if audio_path.exists():
                        info = torchaudio.info(str(audio_path))
                        duration = info.num_frames / info.sample_rate
                    else:
                        logger.warning(f"Audio file not found: {audio_path}")
                        duration = 0.0
                    durations.append(duration)
                except Exception as e:
                    logger.warning(f"Could not get duration for {audio_path}: {e}")
                    durations.append(0.0)

            df['duration'] = durations
            df = data_manager._apply_filters(df)
            data_manager.dataset_cache[cache_key] = df
            return df.iloc[:max_rows]

        data_manager.load_dataset = load_subset

        # Setup processor
        data_manager.setup_processor(
            model_name=config.model.model_name,
            model_type="whisper",
            language=config.data.language,
            task=config.data.task
        )

        # Create dataset and dataloader
        dataset = data_manager.create_dataset("train")
        dataloader = data_manager.create_dataloader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        try:
            for batch in dataloader:
                assert isinstance(batch, dict)
                assert "audio" in batch
                assert "texts" in batch
                # Metadata removed by collator
                assert "durations" not in batch
                assert "paths" not in batch
                assert torch.is_tensor(batch["audio"])
                assert isinstance(batch["texts"], list)
                assert batch["audio"].shape[0] <= batch_size
                assert batch["audio"].shape[1] == 1
                assert batch["audio"].device == torch.device("cpu")
                assert len(batch["texts"]) <= batch_size
                break
            else:
                pytest.skip("No valid batches produced by DataLoader")
        finally:
            data_manager.load_dataset = original_load_dataset  # Restore original method

    def test_empty_batch_returns_none(self, data_manager, config, monkeypatch, caplog):
        """Test that collator returns None when all samples in batch are None"""
        if not data_manager.is_dataset_available():
            pytest.skip("Common Voice dataset not available")

        caplog.set_level(logging.INFO)

        # Create empty dataframe cache to force dataset with data
        data_manager.dataset_cache["cv22_train"] = pd.DataFrame({
            'path': ['dummy1.mp3', 'dummy2.mp3', 'dummy3.mp3'],
            'sentence': ['test1', 'test2', 'test3'],
            'duration': [1.0, 1.5, 2.0],
            'client_id': ['client1', 'client2', 'client3'],
            'up_votes': [0, 0, 0],
            'down_votes': [0, 0, 0]
        })

        # Setup processor
        data_manager.setup_processor(
            model_name=config.model.model_name,
            model_type="whisper",
            language=config.data.language,
            task=config.data.task
        )

        # Create dataset
        dataset = data_manager.create_dataset("train")

        # Mock __getitem__ to always return None
        monkeypatch.setattr(CommonVoiceDataset, "__getitem__", mock_getitem)

        # Create dataloader
        dataloader = data_manager.create_dataloader(
            dataset=dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0
        )

        # Iterate through dataloader - collator should return None for empty batches
        none_batch_count = 0
        total_batches = 0
        for batch in dataloader:
            total_batches += 1
            if batch is None:
                none_batch_count += 1

        # Verify that collator returned None (check logs)
        assert "Empty batch after filtering None samples, skipping..." in caplog.text

        # All batches should be None since all samples return None
        assert total_batches > 0, "DataLoader should produce batches"
        assert none_batch_count == total_batches, f"All batches should be None, got {none_batch_count}/{total_batches}"

        # Clear cache
        data_manager.dataset_cache.clear()

if __name__ == "__main__":
    """Run tests directly"""
    pytest.main([__file__, "-v"])