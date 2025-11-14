"""
Speech-to-text models implementation.
Supports Whisper, Speech2Text, and custom architectures.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, Union, Any, List, TYPE_CHECKING, TypeAlias
import torch
import torch.nn as nn
from transformers import (
    WhisperForConditionalGeneration,
    Speech2TextForConditionalGeneration,
    WhisperProcessor,
    Speech2TextProcessor,
    Speech2TextFeatureExtractor,
    Speech2TextTokenizer,
)
from transformers.modeling_outputs import CausalLMOutput

from .config import ModelConfig

# Type aliases for processors
ProcessorType = Union[WhisperProcessor, Speech2TextProcessor]

# Type for compiled models (torch.compile returns an OptimizedModule wrapper)
# For type checking: Union[BaseSTTModel, OptimizedModule]
# For runtime: Any (torch.compile may not be available)
if TYPE_CHECKING:
    # Using string annotation to avoid import errors in older PyTorch versions
    # OptimizedModule is from torch._dynamo (PyTorch 2.0+)
    CompiledModelType: TypeAlias = Union["BaseSTTModel", Any]
else:
    # At runtime, we don't need strict typing for compiled models
    CompiledModelType: TypeAlias = Any

logger = logging.getLogger(__name__)


# ============================================================
# Base class
# ============================================================
class BaseSTTModel(nn.Module):
    """Base class for all speech-to-text models"""

    # Expected processor type for each model (override in subclasses)
    expected_processor_type: Optional[type] = None

    def __init__(self, config: ModelConfig, processor: Optional[ProcessorType] = None):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model_name = config.model_name
        self.processor = processor

        # Validate processor
        self.validate_processor(processor)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def generate(
        self,
        input_data,
        language: Optional[str] = None,
        task: Optional[str] = None,
        return_text: bool = True,
        **kwargs
    ) -> Union[torch.Tensor, List[str]]:
        """
        Generate transcription from input audio.

        Args:
            input_data: Input audio tensor (features or waveform, model-dependent)
            language: Language code (e.g., 'ru', 'en'). Required for Whisper, ignored for CTC models.
            task: Task type ('transcribe' or 'translate'). Required for Whisper, ignored for CTC models.
            return_text: If True and processor is available, decode tokens to text. Otherwise return token IDs.
            **kwargs: Additional model-specific generation arguments

        Returns:
            List of decoded transcriptions if return_text=True and processor available,
            otherwise torch.Tensor of token IDs
        """
        raise NotImplementedError

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def validate_processor(self, processor: Optional[ProcessorType]) -> None:
        """Validate that processor matches the expected type for this model"""
        if self.expected_processor_type is not None:
            if processor is None:
                raise ValueError(
                    f"{self.__class__.__name__} expects a processor of type "
                    f"{self.expected_processor_type.__name__}, but got None."
                )
            if not isinstance(processor, self.expected_processor_type):
                raise TypeError(
                    f"{self.__class__.__name__} expected processor of type "
                    f"{self.expected_processor_type.__name__}, "
                    f"but got {type(processor).__name__}."
                )

    def freeze_feature_encoder(self):
        pass

    def freeze_encoder(self):
        pass

    def freeze_decoder(self):
        pass

    def unfreeze_last_n_encoder_layers(self, n: int):
        pass

    def unfreeze_last_n_decoder_layers(self, n: int):
        pass

    def unfreeze_embed_tokens(self):
        """Unfreeze decoder token embeddings (critical for cross-lingual transfer)"""
        pass

    def unfreeze_embed_positions_decoder(self):
        """Unfreeze decoder positional embeddings"""
        pass

    def unfreeze_lm_head(self):
        """Unfreeze output projection layer (lm_head or proj_out)"""
        pass

    def unfreeze_layer_norm_decoder(self):
        """Unfreeze final layer norm in decoder"""
        pass

    def resize_embeddings_for_tokenizer(self, new_vocab_size: int) -> None:
        """
        Resize token embeddings for cross-lingual transfer.
        Called when using alternative tokenizer with different vocabulary size.

        Args:
            new_vocab_size: New vocabulary size from the alternative tokenizer
        """
        pass

    def apply_freezing_from_config(self, config: Optional[ModelConfig] = None):
        """
        Apply freezing/unfreezing according to ModelConfig.

        Args:
            config: ModelConfig to apply. If None, uses self.config.

        Note:
            First unfreezes all parameters, then applies freeze/unfreeze logic.
            This ensures correct behavior when loading baseline models (fully frozen)
            and applying different freeze settings for training.
        """
        c = config if config is not None else self.config

        # First, unfreeze ALL parameters (handles baseline models that are fully frozen)
        for param in self.parameters():
            param.requires_grad = True

        # Apply freezing (uses config values, no fallback defaults)
        if c.freeze_feature_encoder:
            self.freeze_feature_encoder()
        if c.freeze_encoder:
            self.freeze_encoder()
        if c.freeze_decoder:
            self.freeze_decoder()

        # Apply selective unfreezing
        if c.unfreeze_last_n_encoder_layers > 0:
            self.unfreeze_last_n_encoder_layers(c.unfreeze_last_n_encoder_layers)
        if c.unfreeze_last_n_decoder_layers > 0:
            self.unfreeze_last_n_decoder_layers(c.unfreeze_last_n_decoder_layers)

        # Apply fine-grained unfreezing for critical components (AFTER freeze/unfreeze operations)
        # This allows scenarios like: freeze_decoder=True + unfreeze_embed_tokens=True
        # which is critical for cross-lingual transfer (decoder frozen but new embeddings trainable)
        if c.unfreeze_embed_tokens:
            self.unfreeze_embed_tokens()
        if c.unfreeze_embed_positions_decoder:
            self.unfreeze_embed_positions_decoder()
        if c.unfreeze_lm_head:
            self.unfreeze_lm_head()
        if c.unfreeze_layer_norm_decoder:
            self.unfreeze_layer_norm_decoder()


# ============================================================
# WhisperSTT
# ============================================================
class WhisperSTT(BaseSTTModel):
    """Whisper model wrapper with additional functionality"""

    expected_processor_type = WhisperProcessor

    def __init__(self, config: ModelConfig, processor: Optional[WhisperProcessor] = None):
        super().__init__(config, processor)

        logger.info(f"Loading Whisper model: {config.model_name}")

        self.model = WhisperForConditionalGeneration.from_pretrained(
            config.model_name,
            dropout=0.0,
            attention_dropout=0.0,
            activation_dropout=0.0,
        )

        self.apply_freezing_from_config()

        logger.info(
            f"Model loaded. Total parameters: {self.get_num_parameters():,}, "
            f"Trainable: {self.get_trainable_parameters():,}"
        )

    def forward(self, input_features, labels=None, **kwargs):
        """
        Forward pass for Whisper model.

        Args:
            input_features: torch.Tensor
                Mel-spectrogram features of shape (batch_size, feature_dim, sequence_length)
                Typically feature_dim=80 for Whisper models
                Example shape: (8, 80, 3000) for batch_size=8
            labels: torch.Tensor, optional
                Token IDs for teacher forcing, shape (batch_size, target_sequence_length)
                Example shape: (8, 448) for batch_size=8, max_length=448
                If provided, loss will be computed
            **kwargs: Additional arguments passed to the model

        Returns:
            transformers.modeling_outputs.Seq2SeqLMOutput containing:
                - loss: torch.Tensor, scalar
                    Cross-entropy loss if labels are provided, else None
                - logits: torch.Tensor, shape (batch_size, target_sequence_length, vocab_size)
                    Predicted token logits
                - past_key_values: cached key/value pairs for generation
                - decoder_hidden_states: hidden states if output_hidden_states=True
                - encoder_hidden_states: encoder hidden states if output_hidden_states=True
        """

        return self.model(
            input_features=input_features,
            labels=labels,
             **kwargs,
        )

    def generate(
        self,
        input_data,
        language: Optional[str] = None,
        task: Optional[str] = None,
        return_text: bool = True,
        **generation_kwargs
    ) -> Union[torch.Tensor, List[str]]:
        """
        Generate transcription using language and task parameters.

        Args:
            input_data: Input audio features (mel-spectrogram)
            language: Language code (e.g., 'ru', 'en'). Required for Whisper.
            task: Task type ('transcribe' or 'translate'). Required for Whisper.
            return_text: If True and processor is available, decode tokens to text. Otherwise return token IDs.
            **generation_kwargs: Additional generation arguments

        Returns:
            List of decoded transcriptions if return_text=True and processor available,
            otherwise torch.Tensor of token IDs
        """
        # Validate required parameters for Whisper
        if language is None:
            raise ValueError(
                "language parameter is required for Whisper generation. "
                "Please provide a language code (e.g., 'ru', 'en', 'de'). "
                "Example: model.generate(input_data, language='ru', task='transcribe')"
            )
        if task is None:
            raise ValueError(
                "task parameter is required for Whisper generation. "
                "Please provide either 'transcribe' or 'translate'. "
                "Example: model.generate(input_data, language='ru', task='transcribe')"
            )

        # Validate processor
        if self.processor is None or not isinstance(self.processor, WhisperProcessor):
            raise ValueError("WhisperProcessor must be provided for generation")

        # Use modern API: pass language and task directly to generate()
        # This replaces the deprecated forced_decoder_ids approach
        token_ids: torch.Tensor = self.model.generate(
            input_data,
            language=language,
            task=task,
            **generation_kwargs,
        )  # type: ignore

        # Decode to text if requested
        if return_text:
            return self.processor.batch_decode(token_ids, skip_special_tokens=True)

        return token_ids

    def freeze_feature_encoder(self):
        encoder = self.model.model.encoder
        for param in encoder.conv1.parameters():
            param.requires_grad = False
        for param in encoder.conv2.parameters():
            param.requires_grad = False
        for param in encoder.embed_positions.parameters():
            param.requires_grad = False
        logger.info("Feature encoder frozen (conv1, conv2, embed_positions)")

    def freeze_encoder(self):
        for param in self.model.model.encoder.parameters():
            param.requires_grad = False
        logger.info("Encoder frozen")

    def freeze_decoder(self):
        for param in self.model.model.decoder.parameters():
            param.requires_grad = False
        logger.info("Decoder frozen")

    def unfreeze_last_n_encoder_layers(self, n: int):
        encoder_layers = self.model.model.encoder.layers
        for layer in encoder_layers[-n:]:
            for param in layer.parameters():
                param.requires_grad = True
        logger.info(f"Unfroze last {n} encoder layers")

    def unfreeze_last_n_decoder_layers(self, n: int):
        decoder_layers = self.model.model.decoder.layers
        for layer in decoder_layers[-n:]:
            for param in layer.parameters():
                param.requires_grad = True
        logger.info(f"Unfroze last {n} decoder layers")

    def unfreeze_embed_tokens(self):
        """Unfreeze decoder token embeddings (critical for cross-lingual transfer)"""
        for param in self.model.model.decoder.embed_tokens.parameters():
            param.requires_grad = True
        logger.info("Decoder embed_tokens unfrozen")

    def unfreeze_embed_positions_decoder(self):
        """Unfreeze decoder positional embeddings"""
        for param in self.model.model.decoder.embed_positions.parameters():
            param.requires_grad = True
        logger.info("Decoder embed_positions unfrozen")

    def unfreeze_lm_head(self):
        """Unfreeze output projection layer (proj_out for Whisper)"""
        for param in self.model.proj_out.parameters():
            param.requires_grad = True
        logger.info("proj_out (lm_head) unfrozen")

    def unfreeze_layer_norm_decoder(self):
        """Unfreeze final layer norm in decoder"""
        for param in self.model.model.decoder.layer_norm.parameters():
            param.requires_grad = True
        logger.info("Decoder layer_norm unfrozen")


# ============================================================
# Speech2TextSTT
# ============================================================
class Speech2TextSTT(BaseSTTModel):
    """Speech2Text model wrapper for seq2seq speech recognition"""

    expected_processor_type = Speech2TextProcessor

    def __init__(self, config: ModelConfig, processor: Optional[Speech2TextProcessor] = None):
        super().__init__(config, processor)

        logger.info(f"Loading Speech2Text model: {config.model_name}")

        # Используем safetensors для совместимости с PyTorch < 2.6 (CVE-2025-32434)
        self.model = Speech2TextForConditionalGeneration.from_pretrained(
            config.model_name,
            dropout=config.dropout,
            attention_dropout=config.attention_dropout,
            activation_dropout=config.activation_dropout,
            use_safetensors=True,
        )

        self.apply_freezing_from_config()

        logger.info(
            f"Model loaded. Total parameters: {self.get_num_parameters():,}, "
            f"Trainable: {self.get_trainable_parameters():,}"
        )

    def forward(self, input_features, attention_mask=None, labels=None, **kwargs):
        """
        Forward pass for Speech2Text model.

        Args:
            input_features: torch.Tensor
                Mel-spectrogram features of shape (batch_size, sequence_length, feature_dim)
                Typically feature_dim=80 for Speech2Text models
                Example shape: (8, 3000, 80) for batch_size=8
                Note: Speech2Text expects (batch, time, freq) format (transposed from Whisper)
            attention_mask: torch.Tensor, optional
                Attention mask for input_features, shape (batch_size, sequence_length)
                1 for real data, 0 for padding
                Example shape: (8, 3000)
            labels: torch.Tensor, optional
                Token IDs for teacher forcing, shape (batch_size, target_sequence_length)
                Example shape: (8, 448) for batch_size=8, max_length=448
                If provided, loss will be computed
            **kwargs: Additional arguments passed to the model

        Returns:
            transformers.modeling_outputs.Seq2SeqLMOutput containing:
                - loss: torch.Tensor, scalar
                    Cross-entropy loss if labels are provided, else None
                - logits: torch.Tensor, shape (batch_size, target_sequence_length, vocab_size)
                    Predicted token logits
                - past_key_values: cached key/value pairs for generation
                - decoder_hidden_states: hidden states if output_hidden_states=True
                - encoder_hidden_states: encoder hidden states if output_hidden_states=True
        """

        return self.model(
            input_features=input_features,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

    def generate(
        self,
        input_data,
        language: Optional[str] = None,
        task: Optional[str] = None,
        return_text: bool = True,
        **generation_kwargs
    ) -> Union[torch.Tensor, List[str]]:
        """
        Generate transcription with Speech2Text.

        Args:
            input_data: Input audio features (mel-spectrogram)
                Shape: (batch_size, sequence_length, feature_dim)
            language: Language code (e.g., 'ru', 'en').
                Optional for monolingual models (e.g., librispeech-asr).
                Required for multilingual models to set forced_bos_token_id.
            task: Task type (ignored for Speech2Text, included for API consistency)
            return_text: If True and processor is available, decode tokens to text. Otherwise return token IDs.
            **generation_kwargs: Additional generation arguments
                attention_mask: Attention mask for input_features (can be passed here)

        Returns:
            List of decoded transcriptions if return_text=True and processor available,
            otherwise torch.Tensor of token IDs
        """
        # Validate processor
        if self.processor is None or not isinstance(self.processor, Speech2TextProcessor):
            raise ValueError("Speech2TextProcessor must be provided for generation")

        # Get tokenizer from processor
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            raise ValueError("Tokenizer not found in Speech2TextProcessor")

        # Get token IDs from tokenizer
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        bos_token_id = getattr(tokenizer, "bos_token_id", None)

        if pad_token_id is None:
            raise ValueError("pad_token_id not found in processor tokenizer")
        if eos_token_id is None:
            raise ValueError("eos_token_id not found in processor tokenizer")

        # Handle language-specific BOS token for multilingual models
        # For monolingual models (e.g., librispeech-asr), lang_code_to_id is empty
        forced_bos_token_id = None
        lang_code_to_id = getattr(tokenizer, "lang_code_to_id", {})

        if lang_code_to_id and language:
            # Multilingual model with language specified
            forced_bos_token_id = lang_code_to_id.get(language, None)
            if forced_bos_token_id is None:
                available_langs = list(lang_code_to_id.keys())
                raise ValueError(
                    f"Language '{language}' not found in tokenizer.lang_code_to_id. "
                    f"Available languages: {available_langs}"
                )
        elif lang_code_to_id and not language:
            # Multilingual model but language not specified
            raise ValueError(
                f"This is a multilingual model. Please specify language parameter. "
                f"Available languages: {list(lang_code_to_id.keys())}"
            )
        # else: Monolingual model, no forced_bos_token_id needed

        # Prepare generation kwargs
        gen_kwargs = {
            "pad_token_id": pad_token_id,
            "eos_token_id": eos_token_id,
            **generation_kwargs
        }

        # Add forced_bos_token_id only if available (multilingual models)
        if forced_bos_token_id is not None:
            gen_kwargs["forced_bos_token_id"] = forced_bos_token_id
        elif bos_token_id is not None:
            # For monolingual models, use regular bos_token_id
            gen_kwargs["bos_token_id"] = bos_token_id

        # Generate token IDs with explicit token parameters
        token_ids: torch.Tensor = self.model.generate(
            input_data,
            **gen_kwargs,
        )  # type: ignore

        # Decode to text if requested
        if return_text:
            return self.processor.batch_decode(token_ids, skip_special_tokens=True)

        return token_ids

    def freeze_feature_encoder(self):
        """
        Freeze the feature projection layer.

        Freezes:
        - Conv layers (parameters with names starting with 'conv')
        - Positional embeddings (embed_positions)
        """
        encoder = self.model.model.encoder
        frozen_params = 0

        # Speech2Text encoder has conv layers for feature extraction
        for name, param in encoder.named_parameters():
            # Check if this is a conv layer (name starts with 'conv') or positional embeddings
            name_parts = name.split('.')
            if any(part.startswith('conv') for part in name_parts) or 'embed_positions' in name:
                param.requires_grad = False
                frozen_params += 1

        logger.info(f"Feature encoder frozen: {frozen_params} parameter groups (conv layers and positional embeddings)")

    def freeze_encoder(self):
        for param in self.model.model.encoder.parameters():
            param.requires_grad = False
        logger.info("Encoder frozen")

    def freeze_decoder(self):
        for param in self.model.model.decoder.parameters():
            param.requires_grad = False
        logger.info("Decoder frozen")

    def unfreeze_last_n_encoder_layers(self, n: int):
        encoder_layers = self.model.model.encoder.layers
        for layer in encoder_layers[-n:]:
            for param in layer.parameters():
                param.requires_grad = True
        logger.info(f"Unfroze last {n} encoder layers")

    def unfreeze_last_n_decoder_layers(self, n: int):
        decoder_layers = self.model.model.decoder.layers
        for layer in decoder_layers[-n:]:
            for param in layer.parameters():
                param.requires_grad = True
        logger.info(f"Unfroze last {n} decoder layers")

    def unfreeze_embed_tokens(self):
        """Unfreeze decoder token embeddings (critical for cross-lingual transfer)"""
        for param in self.model.model.decoder.embed_tokens.parameters():
            param.requires_grad = True
        logger.info("Decoder embed_tokens unfrozen")

    def unfreeze_embed_positions_decoder(self):
        """Unfreeze decoder positional embeddings"""
        for param in self.model.model.decoder.embed_positions.parameters():
            param.requires_grad = True
        logger.info("Decoder embed_positions unfrozen")

    def unfreeze_lm_head(self):
        """Unfreeze output projection layer (lm_head for Speech2Text)"""
        for param in self.model.lm_head.parameters():
            param.requires_grad = True
        logger.info("lm_head unfrozen")

    def unfreeze_layer_norm_decoder(self):
        """Unfreeze final layer norm in decoder"""
        for param in self.model.model.decoder.layer_norm.parameters():
            param.requires_grad = True
        logger.info("Decoder layer_norm unfrozen")

    def resize_embeddings_for_tokenizer(self, new_vocab_size: int) -> None:
        """
        Resize token embeddings for cross-lingual transfer.
        This method changes the model's output layer to match the new tokenizer vocabulary.

        Args:
            new_vocab_size: New vocabulary size from the alternative tokenizer

        Note:
            The new embeddings for added tokens are initialized randomly and will be
            trained from scratch. This is expected for cross-lingual transfer where
            the decoder needs to learn a new vocabulary.
        """
        original_vocab_size = self.model.config.vocab_size

        if new_vocab_size == original_vocab_size:
            logger.info(f"Vocabulary size unchanged ({new_vocab_size}), skipping resize")
            return

        logger.info(
            f"Resizing token embeddings for cross-lingual transfer: "
            f"{original_vocab_size} -> {new_vocab_size}"
        )

        # Resize embeddings (modifies model in-place)
        self.model.resize_token_embeddings(new_vocab_size)

        # Update config to reflect new vocab size
        self.model.config.vocab_size = new_vocab_size

        logger.info(
            f"Token embeddings resized successfully. "
            f"New tokens will be trained from scratch during fine-tuning."
        )

# ============================================================
# ModelFactory
# ============================================================
class ModelFactory:
    """Factory class for creating models"""

    @staticmethod
    def create_model(config: ModelConfig, processor: Optional[ProcessorType] = None) -> BaseSTTModel:
        """
        Create model based on configuration.

        Args:
            config: Model configuration
            processor: Processor for the model (WhisperProcessor, Speech2TextProcessor, etc.)
                      Required for Speech2Text and custom models, optional for Whisper.
        """
        model_type = config.model_type.lower()

        if model_type == "whisper":
            return WhisperSTT(config, processor=processor)  # type: ignore
        elif model_type == "speech2text":
            model = Speech2TextSTT(config, processor=processor)  # type: ignore

            # Cross-lingual transfer: resize embeddings if using alternative tokenizer
            if config.tokenizer_name_or_path and processor is not None:
                tokenizer = getattr(processor, 'tokenizer', None)
                if tokenizer is not None:
                    new_vocab_size = len(tokenizer)
                    model.resize_embeddings_for_tokenizer(new_vocab_size)

            return model
        elif model_type == "custom":
            raise NotImplementedError("Custom model type is not implemented yet") 
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    @staticmethod
    def get_model_info(model: BaseSTTModel) -> Dict[str, Any]:
        return {
            "model_type": model.model_type,
            "model_name": model.model_name,
            "total_parameters": model.get_num_parameters(),
            "trainable_parameters": model.get_trainable_parameters(),
            "trainable_ratio": model.get_trainable_parameters() / model.get_num_parameters(),
        }


# ============================================================
# ModelManager
# ============================================================


class ModelManager:
    """
    High-level model management class for creating, loading, and managing STT models.

    This class is stateless and can work with multiple models. All model-specific
    parameters are passed as method arguments rather than stored as instance attributes.

    Features:
        - Device management (CPU/CUDA)
        - Model compilation with torch.compile
        - Automatic Mixed Precision (AMP) support
        - Checkpoint save/load with compiled model handling

    """

    def __init__(self):
        """Initialize ModelManager."""
        # Enable CUDA optimizations if available
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            logger.info("CUDA optimizations enabled (cudnn.benchmark=True)")

        logger.info("ModelManager initialized")

    def create_model(
        self,
        config: ModelConfig,
        processor: Optional[ProcessorType] = None
    ) -> BaseSTTModel:
        """
        Create model without moving to device.

        Args:
            config: Model configuration
            processor: Processor for the model (required for Speech2Text and custom models)

        Returns:
            Created model instance
        """
        model = ModelFactory.create_model(config, processor=processor)
        logger.info(f"Model created: {config.model_type} - {config.model_name}")
        return model

    def compile_model(
        self,
        model: BaseSTTModel,
        device: torch.device,
        compile: bool = False
    ) -> CompiledModelType:
        """
        Move model to device and optionally compile with torch.compile.

        Args:
            model: Model to prepare
            device: Target device (e.g., torch.device("cuda") or torch.device("cpu"))
            compile: Whether to compile model with torch.compile

        Returns:
            Model instance (BaseSTTModel or compiled version)
        """
        # Move to device
        model.to(device)  # type: ignore
        logger.info(f"Model moved to device: {device}")

        # Compile model if requested
        compiled_model: CompiledModelType = model
        if compile and hasattr(torch, "compile"):
            try:
                logger.info("Compiling model with PyTorch 2.0")
                compiled_model = torch.compile(model)  # type: ignore
                logger.info("Model compilation successful")
            except Exception as e:
                logger.warning(f"Model compilation failed: {e}. Continuing without compilation.")
                compiled_model = model

        # Log model info
        info = ModelFactory.get_model_info(model)
        logger.info(f"Model Info: {info}")

        return compiled_model

    def save_checkpoint(
        self,
        model: CompiledModelType,
        config: ModelConfig,
        save_dir: str
    ):
        """
        Save model checkpoint safely (weights and metadata separately).

        Epoch is NOT saved here - it belongs in experiment_metadata.json (see train.py).

        Args:
            model: Model to save (can be compiled or regular)
            config: Model configuration
            save_dir: Directory to save checkpoint (will be created if doesn't exist)

        File structure:
            save_dir/
                model_weights.pt      - Model state_dict (safe to load with weights_only=True)
                model_metadata.json   - Model configuration and info (no epoch)
        """
        # Create save directory
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # If model is compiled, get the original model for saving
        model_to_save = model
        if hasattr(model, "_orig_mod"):
            model_to_save = model._orig_mod  # type: ignore

        # Save weights only (safe format)
        weights_path = save_path / "model_weights.pt"
        torch.save(model_to_save.state_dict(), weights_path)
        logger.info(f"Model weights saved to {weights_path}")

        # Check for weight tying (shared weights between embeddings and output projection)
        # This is an important architectural feature that can be lost when saving/loading state_dict
        has_weight_tying = False
        model_type = config.model_type.lower()

        if model_type == "whisper":
            # Whisper: check if decoder.embed_tokens and proj_out share weights
            if hasattr(model_to_save, 'model') and hasattr(model_to_save.model, 'model'):
                has_weight_tying = (
                    model_to_save.model.model.decoder.embed_tokens.weight
                    is model_to_save.model.proj_out.weight
                )
        elif model_type == "speech2text":
            # Speech2Text: check if decoder.embed_tokens and lm_head share weights
            if hasattr(model_to_save, 'model') and hasattr(model_to_save.model, 'model'):
                has_weight_tying = (
                    model_to_save.model.model.decoder.embed_tokens.weight
                    is model_to_save.model.lm_head.weight
                )

        if has_weight_tying:
            logger.info("Detected weight tying: decoder.embed_tokens shares weights with output projection")
        else:
            logger.info("No weight tying detected: decoder.embed_tokens and output projection have separate weights")

        # Prepare metadata (JSON-serializable, no epoch)
        model_info = ModelFactory.get_model_info(model_to_save)  # type: ignore
        metadata = {
            "model_name": config.model_name,
            "model_type": config.model_type,
            "model_info": model_info,
            "config": asdict(config),  # Convert dataclass to dict
            "weight_tying": has_weight_tying,  # Flag for weight tying restoration
        }

        # Save metadata as JSON
        metadata_path = save_path / "model_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"Model metadata saved to {metadata_path}")

        logger.info(f"Checkpoint saved to {save_dir}")

    def load_checkpoint(
        self,
        checkpoint_dir: str,
        device: torch.device,
        processor: Optional[ProcessorType] = None,
        compile: bool = False,
        language: Optional[str] = None,
        task: Optional[str] = None
    ) -> tuple[CompiledModelType, ProcessorType, Dict[str, Any]]:
        """
        Load model from checkpoint safely with automatic processor creation.

        Args:
            checkpoint_dir: Directory containing checkpoint (model_weights.pt and model_metadata.json)
            device: Target device for the model (e.g., torch.device("cuda") or torch.device("cpu"))
            processor: Optional processor (if None, will be created from checkpoint config)
            compile: Whether to compile loaded model
            language: Language code for Whisper models (e.g., "ru"). Required for Whisper if processor is None.
            task: Task for STT models (e.g., "transcribe"). Required for Whisper if processor is None.

        Returns:
            Tuple of (loaded model, processor, checkpoint metadata dict)
        """
        checkpoint_path = Path(checkpoint_dir)

        # Load metadata from JSON
        metadata_path = checkpoint_path / "model_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Model metadata file not found: {metadata_path}. "
                f"Expected checkpoint structure: {checkpoint_dir}/model_metadata.json and model_weights.pt"
            )

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata: Dict[str, Any] = json.load(f)

        # Reconstruct ModelConfig from dict
        config_dict = metadata["config"]
        config = ModelConfig(**config_dict)

        # Create processor if not provided
        if processor is None:
            logger.info(f"Creating processor from checkpoint config: {config.model_name}")
            model_type = config.model_type.lower()
            if model_type == "whisper":
                # CRITICAL: Whisper requires language and task to be set in tokenizer
                assert language is not None, (
                    "language parameter is required for Whisper models when processor is not provided. "
                    f"Pass language='ru' (or other language code) to load_checkpoint()"
                )
                assert task is not None, (
                    "task parameter is required for Whisper models when processor is not provided. "
                    f"Pass task='transcribe' (or 'translate') to load_checkpoint()"
                )

                logger.info(f"Creating WhisperProcessor with language='{language}', task='{task}'")

                # Create tokenizer with language and task
                from transformers import WhisperTokenizer, WhisperFeatureExtractor
                tokenizer = WhisperTokenizer.from_pretrained(
                    config.model_name,
                    language=language,
                    task=task
                )

                # Create feature extractor
                feature_extractor = WhisperFeatureExtractor.from_pretrained(config.model_name)

                # Assemble processor from components
                processor = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

            elif model_type == "speech2text":
                # Use alternative tokenizer if specified (cross-lingual transfer)
                if config.tokenizer_name_or_path:
                    logger.info(
                        f"Loading processor with alternative tokenizer for cross-lingual transfer: "
                        f"{config.tokenizer_name_or_path}"
                    )
                    feature_extractor = Speech2TextFeatureExtractor.from_pretrained(config.model_name)
                    tokenizer = Speech2TextTokenizer.from_pretrained(config.tokenizer_name_or_path)

                    # Check if tokenizer is multilingual and requires language parameter
                    # Multilingual tokenizers have 'supported_languages' or similar attribute
                    is_multilingual = hasattr(tokenizer, 'set_tgt_lang_special_tokens')

                    if is_multilingual:
                        if not language:
                            raise ValueError(
                                f"Multilingual Speech2Text tokenizer requires 'language' parameter. "
                                f"Tokenizer: {config.tokenizer_name_or_path}. "
                                f"Pass language='ru' (or other language code) to load_checkpoint()"
                            )
                        tokenizer.tgt_lang = language
                        logger.info(f"Set multilingual tokenizer.tgt_lang = '{language}'")

                    processor = Speech2TextProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
                else:
                    processor = Speech2TextProcessor.from_pretrained(config.model_name)
            else:
                raise ValueError(
                    f"Cannot auto-create processor for model_type '{model_type}'. "
                    f"Please provide processor explicitly."
                )
        else:
            # Processor was provided - validate for Whisper models
            model_type = config.model_type.lower()
            if model_type == "whisper":
                assert isinstance(processor, WhisperProcessor), (
                    f"Expected WhisperProcessor for Whisper models, got {type(processor)}"
                )

                tokenizer = processor.tokenizer  # type: ignore[attr-defined]
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

                logger.info(f"Using provided WhisperProcessor: language='{language}', task='{task}'")

        # Create model
        model = self.create_model(config, processor)

        # Load weights safely (weights_only=True prevents arbitrary code execution)
        weights_path = checkpoint_path / "model_weights.pt"
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Weights file not found: {weights_path}. "
                f"Expected checkpoint structure: {checkpoint_dir}/model_weights.pt and metadata.json"
            )

        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        logger.info(f"Loaded weights from {weights_path}")

        # Restore weight tying if it was present in the original model
        # Weight tying means embed_tokens and output projection share the same weights (single tensor)
        # This architectural feature is lost when using state_dict, so we restore it explicitly
        if metadata.get("weight_tying", False):
            logger.info("Restoring weight tying from metadata...")
            model_type = config.model_type.lower()

            if model_type == "whisper":
                # Whisper: tie decoder.embed_tokens and proj_out
                model.model.proj_out.weight = model.model.model.decoder.embed_tokens.weight
                logger.info("Weight tying restored: proj_out.weight now shares tensor with decoder.embed_tokens.weight")
            elif model_type == "speech2text":
                # Speech2Text: tie decoder.embed_tokens and lm_head
                model.model.lm_head.weight = model.model.model.decoder.embed_tokens.weight
                logger.info("Weight tying restored: lm_head.weight now shares tensor with decoder.embed_tokens.weight")

        # Move to device and compile
        compiled_model = self.compile_model(model, device=device, compile=compile)

        return compiled_model, processor, metadata

    def inference_with_amp(
        self,
        model: CompiledModelType,
        input_data: torch.Tensor,
        use_amp: bool = True,
        amp_dtype: torch.dtype = torch.float16,
        **generate_kwargs
    ) -> Union[torch.Tensor, List[str]]:
        """
        Run inference with optional Automatic Mixed Precision (AMP).
        Supports both single samples and batches.

        Args:
            model: Model to use for inference (can be compiled or regular)
            input_data: Input tensor (audio features or waveform).
                Single sample or batch:
                - Whisper: (batch_size, n_mels, time) or (n_mels, time)
                - Speech2Text: (batch_size, time, n_mels) or (time, n_mels)
                - Custom: (batch_size, sequence_length) or (sequence_length,)
            use_amp: Whether to use AMP for faster inference
            amp_dtype: Data type for AMP (torch.float16 or torch.bfloat16)
            **generate_kwargs: Additional arguments for model.generate()
                For Whisper: language, task, return_text, etc.
                For Speech2Text/Custom: return_text, etc.

        Returns:
            Generated output:
            - If return_text=True: List[str] (transcriptions)
            - If return_text=False: torch.Tensor (token IDs)
            For batches, returns list with length=batch_size

        Example:
            >>> manager = ModelManager()
            >>> model = manager.create_model(config, processor)
            >>> model = manager.compile_model(model, compile=False)

            >>> # Single sample (Whisper)
            >>> transcription = manager.inference_with_amp(
            ...     model, mel_features,  # shape: (80, 3000)
            ...     language="ru", task="transcribe", return_text=True
            ... )

            >>> # Batch (Whisper)
            >>> transcriptions = manager.inference_with_amp(
            ...     model, mel_features_batch,  # shape: (8, 80, 3000)
            ...     language="ru", task="transcribe", return_text=True
            ... )  # Returns list of 8 transcriptions

            >>> # Batch (Speech2Text)
            >>> transcriptions = manager.inference_with_amp(
            ...     model, mel_features_batch,  # shape: (8, 3000, 80)
            ...     return_text=True
            ... )  # Returns list of 8 transcriptions
        """
        model.eval()

        # Determine device from input tensor
        device = input_data.device if isinstance(input_data, torch.Tensor) else torch.device("cpu")

        # Use AMP if requested and on CUDA
        if use_amp and device.type == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):  # type: ignore
                with torch.no_grad():
                    output = model.generate(input_data, **generate_kwargs)  # type: ignore
        else:
            with torch.no_grad():
                output = model.generate(input_data, **generate_kwargs)  # type: ignore

        return output

    def get_memory_usage(self, device: Optional[torch.device] = None) -> Dict[str, Union[float, str]]:
        """
        Get current GPU memory usage.

        Args:
            device: Device to check (if None, checks CUDA if available)
        """
        check_device = device if device is not None else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        if check_device.type == "cuda":
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            return {
                "allocated_gb": allocated,
                "reserved_gb": reserved,
                "total_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3,
            }
        return {"message": "CUDA not available"}
