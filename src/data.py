"""
Data loading and preprocessing for speech-to-text models.
Handles official Mozilla Common Voice dataset (TSV format) and audio preprocessing.
"""

import os
import logging
import random
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor
from transformers import Speech2TextFeatureExtractor, Speech2TextTokenizer, Speech2TextProcessor
from transformers import AutoConfig

from .utils import get_project_root, get_data_dir
from .config import DataConfig, ProjectConfig, AugmentationConfig

logger = logging.getLogger(__name__)


@dataclass
class AudioSample:
    """Container for audio sample with metadata"""
    audio: np.ndarray
    sampling_rate: int
    text: str
    duration: float
    file_path: Optional[str] = None
    speaker_id: Optional[str] = None


class AudioPreprocessor:
    """Single audio sample preprocessing utilities using torchaudio"""
    
    def __init__(self, config: Any):
        self.config = config
        self.sample_rate = config.data.sample_rate
        self.min_duration = config.data.min_duration
        self.max_duration = config.data.max_duration
        self.normalize = config.data.normalize
        self.trim_silence = config.data.trim_silence
        self.silence_threshold = config.data.silence_threshold

    def preprocess(self, audio: torch.Tensor, original_sample_rate: int) -> Optional[Dict[str, Any]]:
        """Apply preprocessing to a single audio sample
        
        Args:
            audio: torch.Tensor (1, time) on CPU
            original_sample_rate: Original sample rate of the audio from torchaudio.info
        
        Returns:
            Dict with preprocessed audio and duration, or None if sample invalid
        """
        try:
            # Ensure mono audio
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)

            # Resample to target sample_rate if needed
            if original_sample_rate != self.sample_rate:
                resampler = torchaudio.transforms.Resample(original_sample_rate, self.sample_rate)
                audio = resampler(audio)

            # Calculate duration
            duration = audio.shape[1] / self.sample_rate

            # Normalize
            if self.normalize:
                max_abs = torch.max(torch.abs(audio))
                if max_abs > 0:
                    audio = audio / (max_abs + 1e-8)

            # Trim silence
            if self.trim_silence:
                energy = torch.abs(audio).mean(dim=0)
                mask = energy > (energy.max() * self.silence_threshold)
                if mask.any():
                    start = torch.nonzero(mask)[0].item()
                    end = torch.nonzero(mask)[-1].item() + 1
                    audio = audio[:, start:end]
                    duration = (end - start) / self.sample_rate

            # Final duration check AFTER all preprocessing (including trim_silence)
            # This ensures we filter based on the actual audio length that will be fed to the model
            if not (self.min_duration <= duration <= self.max_duration):
                logger.debug(f"Sample duration {duration:.2f}s out of bounds "
                             f"[{self.min_duration}, {self.max_duration}] after preprocessing")
                return None
            
            return {
                "audio": audio,  # Remains on CPU
                "sample_rate": self.sample_rate,
                "duration": duration
            }

        except Exception as e:
            logger.error(f"Error in preprocessing: {e}")
            return None

    def __repr__(self) -> str:
        """Return string representation of AudioPreprocessor"""
        return (
            f"AudioPreprocessor(sample_rate={self.sample_rate}, "
            f"normalize={self.normalize}, "
            f"trim_silence={self.trim_silence})"
        )

class AudioAugmentationCPU:
    """Audio augmentation using torchaudio on CPU"""
    
    def __init__(self, config: AugmentationConfig, sample_rate: int = 16000):
        self.config = config
        self.sample_rate = sample_rate
        self._init_transforms()
    
    def _init_transforms(self):
        """Initialize torchaudio transforms"""
        # Time masking for SpecAugment
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=self.config.time_mask_max_size)
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=self.config.freq_mask_max_size)
        
        # Pitch shift
        self.pitch_shift = torchaudio.transforms.PitchShift(
            sample_rate=self.sample_rate, 
            n_steps=0  # Will be set dynamically
        )
        
        # Volume
        self.vol_aug = torchaudio.transforms.Vol(gain=1.0, gain_type="amplitude")
        
        # Fade in/out
        self.fade = torchaudio.transforms.Fade(fade_in_len=0, fade_out_len=0)

        # Speed perturbation
        speed_factors = np.linspace(
            self.config.speed_factor_range[0],
            self.config.speed_factor_range[1],
            self.config.speed_num_steps
        ).tolist()
        self.speed_perturb = T.SpeedPerturbation(self.sample_rate, speed_factors)
    
    def apply_time_domain_augmentations(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply time-domain augmentations"""
        # Add noise
        if self.config.add_noise and random.random() < self.config.noise_probability:
            noise = torch.randn_like(audio) * self.config.noise_factor
            audio = audio + noise
        
        # Speed perturbation
        if self.config.speed_perturbation and random.random() < self.config.speed_probability:
            audio, _ = self.speed_perturb(audio)
        
        # Pitch shift
        if self.config.pitch_shift and random.random() < self.config.pitch_probability:
            n_steps = random.randint(*self.config.pitch_shift_range)
            self.pitch_shift.n_steps = n_steps
            audio = self.pitch_shift(audio)
        
        # Volume perturbation
        if self.config.volume_perturbation and random.random() < self.config.volume_probability:
            gain_db = random.uniform(*self.config.volume_range_db)
            gain = 10 ** (gain_db / 20.0)
            self.vol_aug.gain = gain
            audio = self.vol_aug(audio)
        
        # Fade in/out
        if self.config.fade_inout and random.random() < self.config.fade_probability:
            fade_samples = int(self.config.fade_duration * self.sample_rate)
            self.fade.fade_in_len = fade_samples
            self.fade.fade_out_len = fade_samples
            audio = self.fade(audio)
        
        return audio

    def apply_spectrogram_augmentations(self, log_mel: torch.Tensor) -> torch.Tensor:
        """Apply spectrogram-domain augmentations (spec_augment)"""
        if self.config.spec_augment and random.random() < self.config.spec_augment_probability:
            # Apply time and frequency masking
            for _ in range(self.config.num_time_masks):
                log_mel = self.time_mask(log_mel)
            for _ in range(self.config.num_freq_masks):
                log_mel = self.freq_mask(log_mel)
        
        return log_mel

class CommonVoiceDataset(Dataset):
    """PyTorch Dataset for Mozilla Common Voice (Official TSV format)"""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        audio_dir: Path,
        config: ProjectConfig,
        processor: Optional[Union[WhisperProcessor, Speech2TextProcessor]] = None,
        split: str = "train"
    ):
        """Initialize dataset with pre-filtered DataFrame from DataManager.

        Args:
            dataframe: DataFrame with columns ['path', 'sentence', 'duration', 'client_id', 'up_votes', 'down_votes']
            audio_dir: Path to directory containing audio files
            config: Configuration object with data.sample_rate, data.min_duration, data.max_duration
            processor: Model-specific processor (WhisperProcessor or Speech2TextProcessor) or None if on GPU
            split: Dataset split ('train', 'dev', 'test', 'validation')
        """
        self.dataframe = dataframe
        self.audio_dir = Path(audio_dir)
        self.config = config
        self.split = split
        self.sample_rate = config.data.sample_rate
        self.preprocessor = AudioPreprocessor(config)
        self.augmentation = AudioAugmentationCPU(config.data.augmentation, self.sample_rate) if config.data.processing_and_augmentation_device == "cpu" and config.data.augmentation.enabled else None
        self.processor = processor if config.data.processing_and_augmentation_device == "cpu" else None

        # Validate processor for CPU mode (fail fast - check at dataset creation, not during iteration)
        if config.data.processing_and_augmentation_device == "cpu":
            assert self.processor is not None, (
                "Processor required for CPU mode. "
                "Processor must be passed to DataManager BEFORE creating datasets. "
                "Use data_manager.setup_processor() or data_manager.set_already_loaded_processor()."
            )

            # Extract and validate processor components
            self.feature_extractor = getattr(self.processor, "feature_extractor", None)
            self.tokenizer = getattr(self.processor, "tokenizer", None)

            assert self.feature_extractor is not None and self.tokenizer is not None, (
                f"Processor {type(self.processor).__name__} missing feature_extractor or tokenizer. "
                f"Got: feature_extractor={self.feature_extractor}, tokenizer={self.tokenizer}. "
                f"Processor must be passed to DataManager BEFORE creating datasets using "
                f"data_manager.setup_processor() or data_manager.set_already_loaded_processor()."
            )
        else:
            # GPU mode: processor components not needed during dataset iteration
            self.feature_extractor = None
            self.tokenizer = None

        # Extract max_source_positions and max_target_positions from model config for Speech2Text in CPU mode
        # These parameters are only needed in CPU mode for truncation during feature extraction and tokenization
        # In GPU mode, processing happens later on GPU, so these parameters are not needed
        self.max_source_positions: Optional[int] = None
        self.max_target_positions: Optional[int] = None

        if isinstance(self.processor, Speech2TextProcessor) and config.data.processing_and_augmentation_device == "cpu":
            # Load model config to extract max positions for Speech2Text (CPU mode only)
            model_config = AutoConfig.from_pretrained(config.model.model_name)
            self.max_source_positions = getattr(model_config, 'max_source_positions', None)
            self.max_target_positions = getattr(model_config, 'max_target_positions', None)

            # Validate that we got the required parameters for Speech2Text in CPU mode
            if self.max_source_positions is None or self.max_target_positions is None:
                missing_params = []
                if self.max_source_positions is None:
                    missing_params.append('max_source_positions')
                if self.max_target_positions is None:
                    missing_params.append('max_target_positions')
                raise ValueError(
                    f"Failed to extract required parameters {missing_params} from model config for "
                    f"Speech2Text processor '{config.model.model_name}'. These parameters are required "
                    f"for truncation during feature extraction and tokenization in CPU mode."
                )

            logger.info(f"Extracted from Speech2Text config (CPU mode): max_source_positions={self.max_source_positions}, "
                       f"max_target_positions={self.max_target_positions}")

    def __len__(self) -> int:
        """Return number of samples in the dataset"""
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        """Get a single sample: preprocessed audio features or raw audio, tokenized text, and metadata.

        CPU mode behavior:
            - Always returns input_features in (freq, time) format regardless of model type
            - Whisper features: (freq, time) directly from feature_extractor
            - Speech2Text features: (time, freq) from feature_extractor -> transposed to (freq, time)
            - Format conversion to model-specific format happens in AudioCollator (via transpose_features parameter)

        GPU mode behavior:
            - Returns raw preprocessed audio tensor in (1, time) format
            - Audio is resampled, normalized, and trimmed (if configured)
            - Processing and feature extraction will be done on GPU later
            - Text is returned as string (not tokenized)

        Returns:
            CPU mode: Dict with keys 'input_features' (torch.Tensor log-mel in (freq, time) format),
                     'labels' (torch.Tensor), 'text', 'path', 'duration', 'client_id', 'up_votes', 'down_votes'
            GPU mode: Dict with keys 'audio' (torch.Tensor in (1, time) format), 'text', 'path',
                     'duration', 'client_id', 'up_votes', 'down_votes'
            Returns None if loading/preprocessing fails.
        """
        row = self.dataframe.iloc[idx]
        audio_filename = row["path"]
        text = row["sentence"]
        audio_path = self.audio_dir / audio_filename

        if not audio_path.exists():
            logger.warning(f"Audio file not found: {audio_path}")
            return None

        # Load audio and get original sample rate
        try:
            audio, sr = torchaudio.load(audio_path)
        except Exception as e:
            logger.error(f"Error loading audio {audio_path}: {e}")
            return None

        # Preprocess audio
        preprocessed = self.preprocessor.preprocess(audio, sr)
        if preprocessed is None:
            return None

        audio = preprocessed["audio"]
        duration = preprocessed["duration"]

        if self.config.data.processing_and_augmentation_device == "cpu":
            # Apply time-domain augmentations (only for train split)
            if self.augmentation and self.split == "train":
                audio = self.augmentation.apply_time_domain_augmentations(audio)

            # Process audio to features and tokenize text
            # Processor components validated in __init__, guaranteed to be not None in CPU mode
            assert self.feature_extractor is not None and self.tokenizer is not None
            audio_np = audio.squeeze(0).numpy()

            # Extract features from audio
            if isinstance(self.processor, WhisperProcessor):
                audio_inputs = self.feature_extractor(
                    audio_np,
                    sampling_rate=self.sample_rate,
                    return_tensors="pt"
                )
                input_features = audio_inputs.input_features.squeeze(0)  # (freq, time)
                # Apply spectrogram augmentations (if enabled and training)
                # Return (freq, time) format - unified with Speech2Text
            elif isinstance(self.processor, Speech2TextProcessor):
                # Speech2Text: truncate audio to max_source_positions from model config
                truncation = self.max_source_positions is not None
                audio_inputs = self.feature_extractor(
                    audio_np,
                    sampling_rate=self.sample_rate,
                    return_tensors="pt",
                    truncation=truncation,
                    max_length=self.max_source_positions if truncation else None
                )
                input_features = audio_inputs.input_features.squeeze(0)
                # Speech2Text extractor returns (time, freq) - transpose to unified (freq, time) format
                input_features = input_features.T  # (time, freq) -> (freq, time)
                # Augmentations expect (freq, time) format
            if self.augmentation and self.split == "train":
                input_features = self.augmentation.apply_spectrogram_augmentations(input_features)
                # Keep (freq, time) format - AudioCollator will transpose back to (time, freq) if needed

            # Tokenize text
            if isinstance(self.processor, WhisperProcessor):
                # Whisper: no explicit truncation (handled internally)
                text_inputs = self.tokenizer(
                    text,
                    return_tensors="pt"
                )
            elif isinstance(self.processor, Speech2TextProcessor):
                # Speech2Text: truncate text to max_target_positions from model config
                truncation = self.max_target_positions is not None
                text_inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=truncation,
                    max_length=self.max_target_positions if truncation else None
                )
            labels = text_inputs.input_ids.squeeze(0)
            
            return {
                "input_features": input_features,  # torch.Tensor
                "labels": labels,  # torch.Tensor
                "text": text,
                "duration": duration,
                "path": row["path"],
                "client_id": row.get("client_id", "unknown"),
                "up_votes": row.get("up_votes", 0),
                "down_votes": row.get("down_votes", 0)
            }
        elif self.config.data.processing_and_augmentation_device == "cuda":
            # CUDA mode: Return preprocessed audio for GPU processing
            return {
                "audio": audio,  # torch.Tensor
                "text": text,
                "path": str(audio_path),
                "duration": duration,
                "client_id": row.get("client_id", "unknown"),
                "up_votes": row.get("up_votes", 0),
                "down_votes": row.get("down_votes", 0)
            }
        else:
            raise ValueError(f"Unsupported processing device: {self.config.data.processing_and_augmentation_device}. "
                           f"Must be 'cpu' or 'cuda'.")

    def __repr__(self) -> str:
        """Return string representation of CommonVoiceDataset"""
        return (
            f"CommonVoiceDataset(split={self.split}, "
            f"samples={len(self)}, "
            f"sample_rate={self.sample_rate}, "
            f"audio_dir={self.audio_dir})"
        )


class AudioCollator:
    """Batch collator for CommonVoiceDataset

    Args:
        config: DataConfig with processing settings
        model_type: Model type ("whisper" or "speech2text")
        transpose_features: If True, transpose input_features from (batch, freq, time) to (batch, time, freq).
                          Default False (for Whisper which expects (batch, freq, time)).
                          Must be explicitly set to True for Speech2Text (expects (batch, time, freq)).
    """

    def __init__(
        self,
        config: DataConfig,
        model_type: str = "whisper",
        transpose_features: bool = False
    ):
        self.config = config
        self.device = config.processing_and_augmentation_device  # 'cpu' or 'cuda'
        self.model_type = model_type.lower()  # Store model type to determine if attention_mask is needed
        self.transpose_features = transpose_features  # Whether to transpose features from (freq, time) to (time, freq)
        # Always use -100 for labels padding (standard for seq2seq models - ignored in loss)
        logger.info(f"Initialized AudioCollator with device: {self.device}, model_type: {self.model_type}, "
                   f"labels padding: -100, transpose_features: {self.transpose_features}")

    def __call__(self, batch: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Collate function for DataLoader

        Args:
            batch: List of samples from __getitem__

        Returns:
            Dict with batched tensors:
            - If transpose_features=False (Whisper): input_features shape (batch, freq, time)
            - If transpose_features=True (Speech2Text): input_features shape (batch, time, freq)
            - None if batch is empty after filtering None samples
        """
        # Filter None samples
        batch = [sample for sample in batch if sample is not None]

        # Return None if batch is empty after filtering
        if not batch:
            logger.info("Empty batch after filtering None samples, skipping...")
            return None

        if self.device == "cpu":
            # CPU mode: batch input_features and labels (already processed)
            # Dataset returns (freq, time) format for all models
            # IMPORTANT: Need to pad along time dimension, but pad_sequence pads along dim 0

            # Step 1: Transpose each tensor from (freq, time) to (time, freq) for padding
            features_transposed = [sample["input_features"].T for sample in batch]  # List of (time, freq)

            # Get original time lengths for attention_mask
            time_lengths = [f.shape[0] for f in features_transposed]  # time is now dim 0

            # Step 2: Pad along time dimension (dim 0): (time, freq) -> (batch, time_padded, freq)
            input_features = pad_sequence(features_transposed, batch_first=True)

            # Step 3: Transpose back to (batch, freq, time_padded)
            input_features = input_features.transpose(1, 2)

            # Create attention_mask for Speech2Text
            if self.model_type == "speech2text":
                max_len = input_features.shape[2]  # time dimension in (batch, freq, time_padded)

                # Create attention_mask (1 for real data, 0 for padding)
                attention_mask = torch.ones(len(batch), max_len, dtype=torch.long)
                for i, length in enumerate(time_lengths):
                    if length < max_len:
                        attention_mask[i, length:] = 0
            elif self.model_type == "whisper":
                # Whisper: fixed time dimension, no attention_mask needed
                attention_mask = None

            # Step 4: Final transpose if needed (Speech2Text expects (batch, time, freq))
            if self.transpose_features:
                # Before transpose: (batch, freq, time_padded)
                # After transpose: (batch, time_padded, freq)
                input_features = input_features.transpose(1, 2)
                # Attention mask doesn't need transposition - it's (batch, time)

            # Always use -100 for labels (ignored in cross-entropy loss for both Whisper and Speech2Text)
            labels = pad_sequence([sample["labels"] for sample in batch], batch_first=True, padding_value=-100)

            # Extract reference texts (useful for debugging and custom evaluation loops)
            # Note: 'text' field may not be present in test/mock samples
            texts = [sample.get("text", "") for sample in batch]

            # Return fields needed by the model + reference texts
            # (remove_unused_columns=True in Trainer already removed extra fields like path, duration, etc.)
            result = {
                "input_features": input_features,  # (batch, freq, time) or (batch, time, freq) depending on transpose_features
                "labels": labels,
                "text": texts,  # Reference texts for debugging/evaluation
            }

            # Add attention_mask for Speech2Text
            if attention_mask is not None:
                result["attention_mask"] = attention_mask

            return result
        elif self.device == "cuda":
            # CUDA mode: batch raw audio and texts (processing will be done on GPU later)
            # Return ONLY fields needed for GPU processing
            audio = pad_sequence([sample["audio"].squeeze(0) for sample in batch], batch_first=True).unsqueeze(1)
            texts = [sample["text"] for sample in batch]
            return {
                "audio": audio,
                "texts": texts,
            }
        else:
            raise ValueError(f"Unsupported processing device: {self.device}. Must be 'cpu' or 'cuda'.")


class DataManager:
    """Main data management class for official Mozilla Common Voice TSV dataset"""

    def __init__(self, config: ProjectConfig):
        self.config: ProjectConfig = config
        self.dataset_cache: Dict[str, pd.DataFrame] = {}  # Cache for loaded DataFrames
        self.processor: Optional[Union[WhisperProcessor, Speech2TextProcessor]] = None
        self.pad_token_id: Optional[int] = None  # Store pad_token_id for tokenizer
        self.forced_bos_token_id: Optional[int] = None  # Store forced_bos_token_id for Speech2Text generation

        # Setup directories using centralized path management
        self.project_root: Path = get_project_root()
        self.data_dir: Path = get_data_dir()
        os.makedirs(self.data_dir, exist_ok=True)

        # Common Voice dataset paths from config
        self.dataset_dir: Path = self.data_dir / config.data.dataset_path
        self.clips_dir: Path = self.dataset_dir / "clips"

        # TSV file paths
        self.train_tsv: Path = self.dataset_dir / "train.tsv"
        self.dev_tsv: Path = self.dataset_dir / "dev.tsv"
        self.test_tsv: Path = self.dataset_dir / "test.tsv"
    
    def setup_processor(
        self,
        model_name: str,
        model_type: str,
        language: Optional[str] = None,
        task: Optional[str] = None,
        tokenizer_name_or_path: Optional[str] = None
    ) -> Union[WhisperProcessor, Speech2TextProcessor]:
        """Setup model-specific processor and token_ids.

        IMPORTANT: language and task must be passed EXPLICITLY - no fallback to config!

        Args:
            model_name: Model checkpoint for feature extractor
            model_type: Model type ("whisper", "speech2text")
            language: Language code (e.g., "ru"). REQUIRED for all models - must be passed explicitly.
            task: Task (e.g., "transcribe", "translate"). REQUIRED for Whisper, not used for Speech2Text.
            tokenizer_name_or_path: Alternative tokenizer (for cross-lingual transfer). Optional.
        """
        logger.info(f"Setting up processor for {model_type} model: {model_name}")

        if model_type.lower() == "whisper":
            # CRITICAL: Whisper requires language and task to be set in tokenizer
            if language is None or language == "":
                raise ValueError(
                    f"language parameter is REQUIRED for Whisper models and must be passed explicitly to setup_processor(). "
                    f"Got language={language!r}. Example: setup_processor(..., language='ru', task='transcribe')"
                )
            if task is None or task == "":
                raise ValueError(
                    f"task parameter is REQUIRED for Whisper models and must be passed explicitly to setup_processor(). "
                    f"Got task={task!r}. Example: setup_processor(..., language='ru', task='transcribe')"
                )

            logger.info(f"Creating WhisperProcessor with language='{language}', task='{task}'")

            feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
            tokenizer = WhisperTokenizer.from_pretrained(
                model_name,
                language=language,
                task=task
            )
            self.processor = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
            self.pad_token_id = tokenizer.pad_token_id  # Extract pad_token_id
            self.forced_bos_token_id = None  # Whisper doesn't use forced_bos_token_id

        elif model_type.lower() == "speech2text":
            # Speech2Text requires language for multilingual models
            if language is None or language == "":
                raise ValueError(
                    f"language parameter is REQUIRED for Speech2Text models and must be passed explicitly to setup_processor(). "
                    f"Got language={language!r}. Example: setup_processor(..., language='ru')"
                )

            feature_extractor = Speech2TextFeatureExtractor.from_pretrained(model_name)

            # Use alternative tokenizer if specified (for cross-lingual transfer)
            tokenizer_source = tokenizer_name_or_path if tokenizer_name_or_path else model_name
            if tokenizer_name_or_path:
                logger.info(f"Using alternative tokenizer for cross-lingual transfer: {tokenizer_name_or_path}")

            tokenizer = Speech2TextTokenizer.from_pretrained(tokenizer_source)

            # Get lang_code_to_id (empty for monolingual, populated for multilingual)
            lang_code_to_id = getattr(tokenizer, 'lang_code_to_id', {})

            # Set tgt_lang attribute ONLY for multilingual tokenizers that support language
            if lang_code_to_id and language in lang_code_to_id:
                tokenizer.tgt_lang = language
                logger.info(f"Set tokenizer.tgt_lang = '{language}' (multilingual tokenizer)")
            elif lang_code_to_id and language not in lang_code_to_id:
                # Multilingual tokenizer but language not supported - FAIL FAST
                raise ValueError(
                    f"Target language '{language}' not found in tokenizer.lang_code_to_id. "
                    f"Available languages: {list(lang_code_to_id.keys())}"
                )
            # else: Monolingual tokenizer (lang_code_to_id empty), skip setting tgt_lang

            # Set forced_bos_token_id for generation (multilingual models only)
            if lang_code_to_id and language in lang_code_to_id:
                # Store forced_bos_token_id for generation (inference)
                self.forced_bos_token_id = lang_code_to_id[language]
                logger.info(f"Set forced_bos_token_id for target language '{language}': {self.forced_bos_token_id}")
            else:
                self.forced_bos_token_id = None

            self.processor = Speech2TextProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
            self.pad_token_id = tokenizer.pad_token_id  # Extract pad_token_id

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        logger.info(f"Set pad_token_id: {self.pad_token_id}")
        if self.forced_bos_token_id is not None:
            logger.info(f"Set forced_bos_token_id: {self.forced_bos_token_id}")
        return self.processor

    def set_already_loaded_processor(self, processor: Union[WhisperProcessor, Speech2TextProcessor]) -> None:
        """Set an already-loaded processor from external source and extract token IDs

        Use this method when you have a processor from an external source
        (e.g., loaded from a checkpoint via ModelManager.load_checkpoint()).
        This extracts pad_token_id and forced_bos_token_id automatically.

        Args:
            processor: Pre-created processor (WhisperProcessor or Speech2TextProcessor)
        """
        self.processor = processor

        # CRITICAL: Processor must have both feature_extractor and tokenizer
        feature_extractor = getattr(processor, 'feature_extractor', None)
        tokenizer = getattr(processor, 'tokenizer', None)

        assert feature_extractor is not None, (
            f"Processor must have 'feature_extractor' attribute! "
            f"processor type: {type(processor)}"
        )
        assert tokenizer is not None, (
            f"Processor must have 'tokenizer' attribute! "
            f"processor type: {type(processor)}"
        )

        # Extract pad_token_id from tokenizer
        self.pad_token_id = getattr(tokenizer, 'pad_token_id', None)
        logger.info(f"Set pad_token_id: {self.pad_token_id}")

        # For Whisper models: verify language and task are set
        if isinstance(processor, WhisperProcessor):
            language = getattr(tokenizer, 'language', None)
            task = getattr(tokenizer, 'task', None)

            assert language is not None and language != "", (
                f"WhisperTokenizer must have 'language' attribute set! "
                f"tokenizer.language = {language!r}. "
                f"Processor was created incorrectly - must use WhisperTokenizer.from_pretrained(..., language='ru', task='transcribe')"
            )
            assert task is not None and task != "", (
                f"WhisperTokenizer must have 'task' attribute set! "
                f"tokenizer.task = {task!r}. "
                f"Processor was created incorrectly - must use WhisperTokenizer.from_pretrained(..., language='ru', task='transcribe')"
            )

            logger.info(f"WhisperProcessor verified: language='{language}', task='{task}'")

            self.forced_bos_token_id = None  # Whisper doesn't use forced_bos_token_id

        # For Speech2Text models: extract forced_bos_token_id for multilingual models
        elif isinstance(processor, Speech2TextProcessor):
            # Check if tokenizer has lang_code_to_id and tgt_lang attributes
            lang_code_to_id = getattr(tokenizer, 'lang_code_to_id', {})
            tgt_lang = getattr(tokenizer, 'tgt_lang', None)

            if lang_code_to_id and tgt_lang and tgt_lang in lang_code_to_id:
                self.forced_bos_token_id = lang_code_to_id[tgt_lang]
                logger.info(f"Set forced_bos_token_id for target language '{tgt_lang}': {self.forced_bos_token_id}")
            else:
                self.forced_bos_token_id = None

    def is_dataset_available(self) -> bool:
        """Check if official Common Voice dataset is available locally"""
        required_files = [self.train_tsv, self.dev_tsv, self.test_tsv, self.clips_dir]
        return all(path.exists() for path in required_files)
    
    def load_dataset(self, split: str = "train") -> pd.DataFrame:
            """Load Common Voice dataset from TSV files and merge with clip_durations.tsv for fast duration lookup"""
            # Map split names to cache keys and TSV files
            split_mapping = {
                "train": ("cv22_train", self.train_tsv),
                "validation": ("cv22_dev", self.dev_tsv),
                "val": ("cv22_dev", self.dev_tsv),
                "dev": ("cv22_dev", self.dev_tsv),
                "test": ("cv22_test", self.test_tsv)
            }
            
            if split not in split_mapping:
                available_splits = list(split_mapping.keys())
                raise ValueError(f"Unknown split '{split}'. Available: {available_splits}")
            
            cache_key, tsv_file = split_mapping[split]
            
            if cache_key in self.dataset_cache:
                logger.info(f"Loading {split} split from cache (key: {cache_key})...")
                return self.dataset_cache[cache_key]
            
            # Check if dataset is available
            if not self.is_dataset_available():
                raise ValueError(
                    f"Common Voice dataset not found at {self.dataset_dir}!\n"
                    f"Please download the official dataset from:\n"
                    f"https://commonvoice.mozilla.org/en/datasets\n"
                    f"And extract it to: {self.config.data.data_dir}/{self.config.data.dataset_path}"
                )
            
            if not tsv_file.exists():
                raise ValueError(f"TSV file not found: {tsv_file}")
            
            try:
                logger.info(f"Loading {split} split from TSV file {tsv_file}...")
                # Load TSV file with pandas
                df = pd.read_csv(tsv_file, sep='\t', encoding='utf-8')
                logger.info(f"Loaded {len(df):,} samples from {split} split")
                
                # Load clip_durations.tsv (contains durations for ALL clips in milliseconds)
                durations_file = self.dataset_dir / "clip_durations.tsv"
                if durations_file.exists():
                    logger.info("Loading clip durations from clip_durations.tsv...")
                    durations_df = pd.read_csv(durations_file, sep='\t', encoding='utf-8')
                    # Convert duration from milliseconds to seconds
                    durations_df['duration'] = durations_df['duration[ms]'] / 1000.0
                    # Merge on clip name (df['path'] == durations_df['clip'])
                    df = df.merge(durations_df[['clip', 'duration']], left_on='path', right_on='clip', how='left')
                    # Drop the redundant 'clip' column
                    df = df.drop(columns=['clip'])
                    logger.info(f"Merged durations for {len(df)} samples")
                    
                    # Check if any samples are missing durations (shouldn't happen, but just in case)
                    missing_durations = df['duration'].isna().sum()
                    if missing_durations > 0:
                        logger.warning(f"⚠️ {missing_durations} samples have missing durations. Calculating manually...")
                        # Calculate durations only for missing samples
                        for idx in df[df['duration'].isna()].index:
                            audio_path = self.clips_dir / str(df.loc[idx, 'path'])
                            try:
                                if audio_path.exists():
                                    info = torchaudio.info(str(audio_path))
                                    df.loc[idx, 'duration'] = info.num_frames / info.sample_rate
                                else:
                                    logger.warning(f"Audio file not found: {audio_path}")
                                    df.loc[idx, 'duration'] = 0.0
                            except Exception as e:
                                logger.warning(f"Could not get duration for {audio_path}: {e}")
                                df.loc[idx, 'duration'] = 0.0
                else:
                    logger.warning(f"⚠️ clip_durations.tsv not found at {durations_file}. Falling back to manual calculation...")
                    # Fallback: calculate durations manually (old slow method)
                    durations = []
                    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Calculating durations for {split}", disable=False):
                        audio_path = self.clips_dir / str(row['path'])
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
                
                # Apply filtering
                df = self._apply_filters(df)
                logger.info(f"After filtering: {len(df):,} samples remaining")
                
                # Cache the DataFrame
                self.dataset_cache[cache_key] = df
                logger.info(f"Cached {split} split with {len(df)} samples")
                return df
                
            except Exception as e:
                logger.error(f"Failed to load dataset from {tsv_file}: {e}")
                raise
    
    def _apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply initial filtering to dataframe based on config and cached durations

        This is a fast pre-filtering step based on metadata durations before audio loading.
        Final accurate filtering happens in AudioPreprocessor.preprocess() after trim_silence.
        """
        if hasattr(self.config.data, 'filter_by_duration') and self.config.data.filter_by_duration:
            initial_size = len(df)
            # Use loc to ensure DataFrame output
            filtered_df = df.loc[
                (df['duration'] >= self.config.data.min_duration) &
                (df['duration'] <= self.config.data.max_duration)
            ].copy()  # Create a copy to avoid SettingWithCopyWarning
            logger.info(f"Pre-filtered {initial_size - len(filtered_df)} samples by metadata duration "
                    f"(min: {self.config.data.min_duration}s, max: {self.config.data.max_duration}s). "
                    f"Final filtering after trim_silence happens in AudioPreprocessor.")
            return filtered_df
        return df
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about Common Voice dataset"""
        if not self.is_dataset_available():
            return {
                "status": "not_available",
                "message": "Common Voice 22.0 dataset not found. Download from https://commonvoice.mozilla.org/en/datasets"
            }
        
        try:
            info = {
                "status": "available",
                "dataset_version": "Common Voice 22.0",
                "language": "ru",
                "local_path": str(self.dataset_dir),
                "clips_path": str(self.clips_dir),
                "splits": {}
            }
            
            # Get info for each split
            for split_name, tsv_file in [("train", self.train_tsv), ("dev", self.dev_tsv), ("test", self.test_tsv)]:
                if tsv_file.exists():
                    try:
                        df = self.load_dataset(split_name)  # Use load_dataset to get cached DataFrame with durations
                        info["splits"][split_name] = {
                            "num_samples": len(df),
                            "tsv_file": str(tsv_file),
                            "total_duration_hours": df['duration'].sum() / 3600 if 'duration' in df else 0
                        }
                    except Exception as e:
                        logger.warning(f"Could not read {split_name} TSV: {e}")
                        info["splits"][split_name] = {"error": str(e)}
            
            # Calculate total samples
            total_samples = sum(split_info.get("num_samples", 0) for split_info in info["splits"].values())
            info["total_samples"] = total_samples
            
            return info
        except Exception as e:
            logger.error(f"Error reading dataset info: {e}")
            return {"status": "error", "error": str(e)}
    
    def get_dataset_stats(self, split: str = "train", sample_size: int = 1000) -> Dict[str, Any]:
        """Get dataset statistics (text-based + audio sampling)"""
        dataframe = self.load_dataset(split)
        
        # Text statistics from TSV (fast)
        text_lengths = dataframe['sentence'].str.len().to_numpy()  # Convert to numpy.ndarray
        up_votes = (
            dataframe['up_votes'].to_numpy() if 'up_votes' in dataframe.columns 
            else np.zeros(len(dataframe), dtype=np.int64)
        )
        down_votes = (
            dataframe['down_votes'].to_numpy() if 'down_votes' in dataframe.columns 
            else np.zeros(len(dataframe), dtype=np.int64)
        )
        
        # Use cached durations (already numpy.ndarray)
        durations = dataframe['duration'].to_numpy()
        
        stats = {
            "total_samples": len(dataframe),
            "audio_samples_analyzed": len(durations),
            "estimated_total_duration_hours": durations.sum() / 3600 if len(durations) > 0 else 0,
            "avg_duration": durations.mean() if len(durations) > 0 else 0,
            "median_duration": np.median(durations) if len(durations) > 0 else 0,
            "min_duration": durations.min() if len(durations) > 0 else 0,
            "max_duration": durations.max() if len(durations) > 0 else 0,
            "avg_text_length": text_lengths.mean() if len(text_lengths) > 0 else 0,
            "median_text_length": np.median(text_lengths) if len(text_lengths) > 0 else 0,
            "min_text_length": text_lengths.min() if len(text_lengths) > 0 else 0,
            "max_text_length": text_lengths.max() if len(text_lengths) > 0 else 0,
            "avg_up_votes": up_votes.mean() if len(up_votes) > 0 else 0,
            "avg_down_votes": down_votes.mean() if len(down_votes) > 0 else 0,
        }
        
        return stats

    def create_dataset(self, split: str = "train") -> CommonVoiceDataset:
        """Create PyTorch Dataset for the specified split"""
        df = self.load_dataset(split)
        processor = self.processor if self.config.data.processing_and_augmentation_device == "cpu" else None
        return CommonVoiceDataset(
            dataframe=df,
            audio_dir=self.clips_dir,
            config=self.config,
            processor=processor,
            split=split
        )

    def create_dataloader(
        self,
        dataset: CommonVoiceDataset,
        batch_size: int,
        shuffle: bool = False,
        num_workers: Optional[int] = None,
        transpose_features: bool = False
    ) -> DataLoader:
        """Create DataLoader for the dataset

        Args:
            dataset: CommonVoiceDataset instance
            batch_size: Batch size
            shuffle: Whether to shuffle the dataset
            num_workers: Number of data loading workers (defaults to config.data.num_workers)
            transpose_features: If True, transpose input_features from (batch, freq, time) to (batch, time, freq).
                              Default False (for Whisper which expects (batch, freq, time)).
                              Set to True for Speech2Text which expects (batch, time, freq).
        """
        if self.processor is None:
            raise ValueError("Processor must be set up before creating DataLoader. Call setup_processor first.")
        if num_workers is None:
            num_workers = self.config.data.num_workers

        # Setup collator with model_type and transpose_features
        collator = AudioCollator(
            self.config.data,
            model_type=self.config.model.model_type,
            transpose_features=transpose_features
        )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collator,
            pin_memory=self.config.data.pin_memory
        )