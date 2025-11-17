"""
Configuration management using OmegaConf and Hydra.
Handles all project configurations and hyperparameters.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from omegaconf import OmegaConf
import os

@dataclass
class AugmentationConfig:
    """Audio augmentation configuration"""
    enabled: bool = False  # Enable/disable augmentation
    
    # Time-domain augmentations
    add_noise: bool = True
    noise_probability: float = 0.3
    noise_factor: float = 0.02  # Noise amplitude factor
    
    speed_perturbation: bool = True
    speed_probability: float = 0.3
    speed_factor_range: tuple = (0.9, 1.1)  # Speed factor range
    speed_num_steps: int = 3  # Number of steps in speed range (e.g., 3 for [0.9, 1.0, 1.1])
    
    pitch_shift: bool = True
    pitch_probability: float = 0.3
    pitch_shift_range: tuple = (-2, 2)  # Semitones
    
    volume_perturbation: bool = True
    volume_probability: float = 0.3
    volume_range_db: tuple = (-10, 10)  # dB range
    
    fade_inout: bool = True
    fade_probability: float = 0.2
    fade_duration: float = 0.1  # seconds
    
    # Frequency-domain augmentations
    spec_augment: bool = True
    spec_augment_probability: float = 0.4
    time_mask_max_size: int = 20  # frames
    freq_mask_max_size: int = 30  # frequency bins
    num_time_masks: int = 2
    num_freq_masks: int = 2
    
    time_stretch: bool = True
    time_stretch_probability: float = 0.3
    time_stretch_range: tuple = (0.95, 1.05)
    
    # Environmental augmentations
    reverb: bool = True
    reverb_probability: float = 0.2
    reverb_time: float = 0.3  # seconds
    reverb_decay_rate: float = 0.3  # decay factor
    reverb_impulse_amplitude: float = 0.1  # impulse response amplitude
    reverb_mix_level: float = 0.3  # how much reverb to mix (0.0-1.0)


@dataclass
class DataConfig:
    """Data-related configuration for Common Voice 22.0 TSV dataset"""
    language: str = "ru"
    task: str = "transcribe"  # Task for STT models (transcribe, translate for Whisper)
    train_split: str = "train"
    validation_split: str = "dev"  # Common Voice uses 'dev' for validation
    test_split: str = "test"
    max_duration: float = 30.0  # seconds
    min_duration: float = 0.5   # seconds
    sample_rate: int = 16000    #target sample rate
    data_dir: str = "data"  # Directory containing cv-corpus-22.0-2025-06-20/
    dataset_path: str = "cv-corpus-22.0-2025-06-20/ru"  # Path to specific dataset within data_dir
    num_workers: int = 4
    preprocessing_num_workers: int = 8
    normalize: bool = True  # Normalize audio amplitude
    trim_silence: bool = True  # Trim silence from audio
    silence_threshold: float = 0.01  # Threshold for silence detection (fraction of max energy, 0.0-1.0)

    processing_and_augmentation_device: str = "cpu"  # Device for processing and augmentations ('cpu' or 'cuda')

    # Data processing settings
    filter_by_duration: bool = False  # Apply duration filtering during Dataset loading

    # DataLoader settings
    pin_memory: bool = True  # Pin memory for faster data transfer to GPU

    # Speech2Text specific settings (model-dependent, overridden in model-specific configs)
    max_source_positions: int = 6000  # Maximum audio length in frames (~60s for Speech2Text)
    max_target_positions: int = 1024  # Maximum text length in tokens

    # Augmentation settings
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)


@dataclass
class ConvLayerConfig:
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int
    padding: int


@dataclass
class CustomModelConfig:
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    vocab_size: int = 33000
    max_seq_len: int = 448
    dim_feedforward: int = 2048
    feature_extractor: Dict[str, ConvLayerConfig] = field(default_factory=lambda: {
        "conv1": ConvLayerConfig(1, 64, 10, 5, 2),
        "conv2": ConvLayerConfig(64, 128, 8, 4, 2),
        "conv3": ConvLayerConfig(128, 256, 4, 2, 1),
        "conv4": ConvLayerConfig(256, 512, 4, 2, 1)  # out_channels will be overridden by d_model ref in YAML
    })
    attention_dropout: float = 0.1  # Can inherit from model


@dataclass
class ModelConfig:
    model_name: str = "openai/whisper-small"
    model_type: str = "whisper"
    # Cross-lingual transfer: alternative tokenizer for models with mismatched vocabulary
    tokenizer_name_or_path: Optional[str] = None  # If set, overrides model_name tokenizer
    freeze_feature_encoder: bool = False
    freeze_encoder: bool = False
    freeze_decoder: bool = False
    unfreeze_last_n_encoder_layers: int = 0
    unfreeze_last_n_decoder_layers: int = 0

    # Fine-grained unfreezing for critical components (applied AFTER freeze/unfreeze operations above)
    # These flags are critical for cross-lingual transfer when decoder is frozen but embeddings need training
    unfreeze_embed_tokens: bool = False  # Unfreeze decoder token embeddings (critical for new vocabulary)
    unfreeze_embed_positions_decoder: bool = False  # Unfreeze decoder positional embeddings
    unfreeze_lm_head: bool = False  # Unfreeze output projection (lm_head/proj_out)
    unfreeze_layer_norm_decoder: bool = False  # Unfreeze final decoder layer norm

    # Dropout parameters (standard HuggingFace names)
    activation_dropout: float = 0.0  # Dropout for activation functions
    attention_dropout: float = 0.0  # Dropout for attention weights
    dropout: float = 0.0  # General dropout rate

    # Hardware and compilation settings
    use_gpu: bool = True  # Use GPU if available
    # gpu_memory_fraction parameter removed - let PyTorch manage memory automatically
    compile_model: bool = False  # PyTorch 2.0 compilation for faster inference

    # Custom model architecture parameters (used when model_type="custom")
    # For Whisper and Speech2Text, these are ignored (they use pre-trained configs)
    hidden_size: int = 256
    num_attention_heads: int = 4
    num_hidden_layers: int = 4
    intermediate_size: int = 1024

    custom: Optional[CustomModelConfig] = None


@dataclass
class ExperimentConfig:
    """Experiment-level configuration (common for both training and evaluation)"""
    output_dir: str = "experiments"  # Root directory for all experiments
    experiment_name: Optional[str] = None  # Name of this experiment (auto-generated if not specified)
    seed: int = 42  # Random seed for reproducibility


@dataclass
class ReduceLROnPlateauConfig:
    """Configuration for ReduceLROnPlateau scheduler"""
    factor: float = 0.5  # Factor to reduce LR by (new_lr = lr * factor)
    patience: int = 2  # Number of epochs with no improvement before reducing LR
    min_lr: float = 1e-7  # Minimum learning rate
    mode: str = "min"  # 'min' or 'max' - whether to minimize or maximize the monitored metric
    threshold: float = 1e-4  # Threshold for measuring the new optimum


@dataclass
class CosineAnnealingLRConfig:
    """Configuration for CosineAnnealingLR scheduler"""
    T_max: int = 10  # Maximum number of iterations (epochs)
    eta_min: float = 1e-7  # Minimum learning rate
    last_epoch: int = -1  # The index of last epoch


@dataclass
class OneCycleLRConfig:
    """Configuration for OneCycleLR scheduler"""
    max_lr: float = 1e-3  # Upper learning rate boundary
    pct_start: float = 0.3  # Percentage of cycle spent increasing LR
    anneal_strategy: str = "cos"  # 'cos' or 'linear'
    div_factor: float = 25.0  # Initial LR = max_lr / div_factor
    final_div_factor: float = 1e4  # Final LR = max_lr / final_div_factor


@dataclass
class LinearLRConfig:
    """Configuration for Linear scheduler with warmup (transformers get_linear_schedule_with_warmup)"""
    warmup_ratio: float = 0.1  # Ratio of total training steps for warmup (default 10%)
    # LR increases linearly from 0 to learning_rate during warmup, then decreases linearly to 0


@dataclass
class WarmupPlateauDecayConfig:
    """Configuration for custom Warmup + Plateau + Linear Decay scheduler"""
    warmup_ratio: float = 0.1  # Ratio of total steps for warmup phase (LR: 0 -> 1)
    plateau_ratio: float = 0.5  # Ratio of total steps when plateau ends (LR: 1 constant until this point)
    # After plateau_ratio, LR decays linearly from 1 to 0 until end of training


@dataclass
class TrainingConfig:
    """Training configuration"""
    num_train_epochs: int = 10
    train_batch_size: int = 8  # Batch size for training
    eval_batch_size: int = 8  # Batch size for evaluation during training
    gradient_accumulation_steps: int = 2
    eval_accumulation_steps: Optional[int] = None  # Move predictions to CPU every N batches during eval (None = all at once)
    gradient_checkpointing: bool = False
    max_grad_norm: float = 1.0  # Max gradient norm for gradient clipping (0 = no clipping)
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    label_smoothing_factor: float = 0.0  # Label smoothing factor (0.0 = no smoothing, 0.1 = 10% smoothing)

    # Learning rate scheduler configuration
    scheduler_name: str = "linear"  # Scheduler type: "reduce_on_plateau", "cosine", "onecycle", "linear", "warmup_plateau_decay"
    reduce_on_plateau: Optional[ReduceLROnPlateauConfig] = None
    cosine: Optional[CosineAnnealingLRConfig] = None
    onecycle: Optional[OneCycleLRConfig] = None
    linear: Optional[LinearLRConfig] = None
    warmup_plateau_decay: Optional[WarmupPlateauDecayConfig] = None

    max_steps: Optional[int] = None
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_wer"
    greater_is_better: bool = False
    save_total_limit: int = 3
    fp16: bool = True  # Use mixed precision for RTX 4070ti
    num_workers: int = 4  # DataLoader workers for training
    remove_unused_columns: bool = True  # Remove unused columns (safer, faster)
    use_cpu_offload: bool = False  # CPU offload for large models (experimental)

    # Early stopping configuration
    use_early_stopping: bool = True  # Enable/disable early stopping
    early_stopping_patience: int = 3  # Number of evaluations with no improvement before stopping
    early_stopping_threshold: float = 0.01  # Minimum change to qualify as improvement


@dataclass
class EvaluationConfig:
    """Evaluation configuration"""
    # DataLoader settings
    batch_size: int = 16  # Batch size for evaluation
    num_workers: int = 4  # DataLoader workers for evaluation

    # Metrics computation
    compute_metrics: bool = True
    prediction_loss_only: bool = False
    eval_on_start: bool = False
    eval_accumulation_steps: Optional[int] = None

    # Metrics to calculate
    calculate_wer: bool = True
    calculate_cer: bool = True
    calculate_bleu: bool = False

    # Decoding parameters
    num_beams: int = 1
    max_length: int = 448
    language: str = "ru"
    task: str = "transcribe"

    # Anti-repetition parameters (prevent model from looping/hallucinating)
    repetition_penalty: float = 1.2  # Penalize repetition (>1.0), set to 1.0 to disable
    no_repeat_ngram_size: int = 3  # Prevent repeating n-grams, set to 0 to disable


@dataclass
class LoggingConfig:
    """Logging and monitoring configuration"""
    use_wandb: bool = True
    wandb_project: str = "speech-to-text-ru"
    wandb_entity: Optional[str] = None
    log_level: str = "INFO"
    report_to: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.report_to is None:
            self.report_to = ["tensorboard"]
            if self.use_wandb:
                self.report_to.append("wandb")


@dataclass
class ProjectConfig:
    """Main project configuration combining all sub-configurations"""
    experiment: ExperimentConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    logging: LoggingConfig

    # Project metadata
    project_name: str = "speech-to-text-ru"
    description: str = "Russian speech-to-text fine-tuning project"
    version: str = "0.1.0"


def load_config(config_path: Optional[str] = None) -> ProjectConfig:
    """
    Load configuration from YAML file or use defaults.
    Supports 'defaults' directive for config composition (similar to Hydra).
    """
    if config_path and os.path.exists(config_path):
        cfg = OmegaConf.load(config_path)

        # Convert to dict to check for 'defaults' key
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)  # type: ignore

        # Handle 'defaults' directive for config composition
        if "defaults" in cfg_dict:  # type: ignore
            defaults_list = cfg_dict.pop("defaults")  # type: ignore

            # Load and merge default configs
            base_dir = os.path.dirname(config_path)

            # Create default ProjectConfig with all required sub-configs
            default_project_config = ProjectConfig(
                experiment=ExperimentConfig(),
                data=DataConfig(),
                model=ModelConfig(),
                training=TrainingConfig(),
                evaluation=EvaluationConfig(),
                logging=LoggingConfig()
            )
            merged_defaults = OmegaConf.structured(default_project_config)

            for default_name in defaults_list:
                default_path = os.path.join(base_dir, f"{default_name}.yaml")
                if os.path.exists(default_path):
                    # Recursively load default config (supports nested defaults)
                    default_cfg = load_config(default_path)
                    merged_defaults = OmegaConf.merge(
                        merged_defaults,
                        OmegaConf.structured(default_cfg)
                    )

            # Merge defaults with current config (overrides)
            final_cfg = OmegaConf.merge(merged_defaults, cfg_dict)
        else:
            # No defaults, merge with base structured config
            default_project_config = ProjectConfig(
                experiment=ExperimentConfig(),
                data=DataConfig(),
                model=ModelConfig(),
                training=TrainingConfig(),
                evaluation=EvaluationConfig(),
                logging=LoggingConfig()
            )
            structured_cfg = OmegaConf.structured(default_project_config)
            final_cfg = OmegaConf.merge(structured_cfg, cfg_dict)

        return OmegaConf.to_object(final_cfg)  # type: ignore
    else:
        # Return default configuration
        return ProjectConfig(
            experiment=ExperimentConfig(),
            data=DataConfig(),
            model=ModelConfig(),
            training=TrainingConfig(),
            evaluation=EvaluationConfig(),
            logging=LoggingConfig()
        )


def save_config(config: ProjectConfig, save_path: str) -> None:
    """Save configuration to YAML file"""
    cfg = OmegaConf.structured(config)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    OmegaConf.save(cfg, save_path)


def print_config(config: ProjectConfig) -> None:
    """Pretty print configuration"""
    import logging

    cfg = OmegaConf.structured(config)
    config_yaml = OmegaConf.to_yaml(cfg)

    # Log to file and console (logger has both handlers)
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("CONFIGURATION")
    logger.info("=" * 60)
    for line in config_yaml.splitlines():
        logger.info(line)
    logger.info("=" * 60)