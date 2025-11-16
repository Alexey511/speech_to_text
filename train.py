"""
Training script for speech-to-text models with custom training loop.
Based on the same architecture as evaluation.py for consistency.
"""

import os
import sys
import logging
import warnings
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import argparse
import json
from datetime import datetime

# Подавить pydantic warnings при работе с OmegaConf
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic._internal._generate_schema")

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR, LambdaLR
from tqdm import tqdm
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.config import ProjectConfig, load_config, save_config, print_config
from src.data import DataManager, AudioCollator
from src.models import ModelManager
from src.metrics import STTMetrics
from src.utils import (
    setup_logging, set_seed, print_system_info
)

# Import transformers scheduler
from transformers import get_linear_schedule_with_warmup

# Suppress warnings and configure CUDA memory management
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Optional WandB import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ==============================================================================
# CONFIGURATION - Edit these parameters to run via VSCode
# ==============================================================================
DEFAULT_CONFIG = {
    "model_path": "experiments/baselines/s2t-cross-lingual",
    "config": None,  # None = auto-search in model directory
    "experiment_name": "s2t-cross-lingual_train_5_decoder_3e-3",  # None = auto-generate from model name + "_trained"
    "epochs": None,  # None = use config value
}
# Example model paths:
# experiments/baselines/whisper-small
# experiments/baselines/whisper-base
# experiments/baselines/s2t-cross-lingual
# experiments/whisper-small-finetuned/checkpoints/epoch_5  (for resume)
# ==============================================================================


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train Russian Speech-to-Text Model")
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_CONFIG["model_path"],
        help=f"Path to model directory or checkpoint (default: {DEFAULT_CONFIG['model_path']})"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG["config"],
        help="Path to configuration file (optional, will search in model directory)"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=DEFAULT_CONFIG["experiment_name"],
        help="Override experiment name"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_CONFIG["epochs"],
        help="Override number of training epochs"
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (fast testing with reduced epochs/steps)"
    )

    return parser.parse_args()


def find_config(experiment_path: Path, config_path: Optional[str] = None) -> str:
    """Find configuration file in experiment/checkpoint directory"""
    # If config explicitly provided, use it
    if config_path:
        if Path(config_path).exists():
            return config_path
        else:
            raise ValueError(f"Config file not found: {config_path}")

    # Otherwise, look ONLY in experiment directory
    config_in_dir = experiment_path / "config.yaml"
    if config_in_dir.exists():
        return str(config_in_dir)

    raise ValueError(
        f"Could not find config.yaml in experiment directory: {experiment_path}\n"
        f"Please either:\n"
        f"  1. Ensure config.yaml exists in {experiment_path}/\n"
        f"  2. Specify --config explicitly"
    )


def setup_experiment(
    config: ProjectConfig,
    args,
    experiment_path: str,
    is_resume: bool = False,
    experiment_dir: Optional[Path] = None
) -> Tuple[str, SummaryWriter]:
    """
    Setup experiment directory and logging.

    Args:
        config: Project configuration
        args: Command line arguments
        experiment_path: Path to experiment/checkpoint directory (for auto-generating experiment name)
        is_resume: Whether this is resuming from checkpoint
        experiment_dir: Optional pre-created experiment directory (if None, will be created)

    Returns:
        Tuple of (experiment_dir, tensorboard_writer)
    """
    # Use pre-created experiment directory if provided
    if experiment_dir is None:
        # Determine experiment name (priority: CLI arg > config > experiment path name)
        if args.experiment_name:
            # CLI argument has highest priority
            config.experiment.experiment_name = args.experiment_name
        elif config.experiment.experiment_name:
            # Use experiment name from config if specified
            pass  # Already set, do nothing
        else:
            # Fallback: auto-generate from experiment directory name + "_trained"
            experiment_name = Path(experiment_path).name
            if is_resume:
                # If resuming, use parent directory name (e.g., whisper_base_trained)
                config.experiment.experiment_name = Path(experiment_path).parent.parent.name
            else:
                # New training: add _trained suffix
                config.experiment.experiment_name = f"{experiment_name}_trained"

        # Ensure experiment_name is set (for type checker)
        assert config.experiment.experiment_name is not None, "Experiment name must be set"

        # Create experiment directory structure
        experiment_dir = Path(config.experiment.output_dir) / config.experiment.experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    checkpoints_dir = experiment_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    # Get logger (already setup in main)
    logger = logging.getLogger(__name__)

    logger.info(f"Experiment directory: {experiment_dir}")
    logger.info(f"Checkpoints directory: {checkpoints_dir}")
    logger.info(f"Experiment path: {experiment_path}")
    logger.info(f"Resume training: {is_resume}")

    # Save configuration
    config_path = experiment_dir / "config.yaml"
    save_config(config, str(config_path))
    logger.info(f"Configuration saved to: {config_path}")

    # Setup TensorBoard writer (save directly in experiment_dir, no subdirectory)
    writer = SummaryWriter(log_dir=str(experiment_dir))

    # Setup WandB if enabled
    if config.logging.use_wandb and not args.no_wandb and WANDB_AVAILABLE:
        try:
            from omegaconf import OmegaConf
            config_dict = OmegaConf.to_container(OmegaConf.structured(config), resolve=True)
            wandb_config: Dict[str, Any] = dict(config_dict) if isinstance(config_dict, dict) else {}  # type: ignore
            wandb.init(
                project=config.logging.wandb_project,
                entity=config.logging.wandb_entity,
                name=config.experiment.experiment_name,
                config=wandb_config,
                dir=str(experiment_dir),
                resume="allow" if is_resume else None
            )
            logger.info("Weights & Biases initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")
            config.logging.use_wandb = False
    elif config.logging.use_wandb and not WANDB_AVAILABLE:
        logger.warning("WandB requested but not installed. Install with: pip install wandb")
        config.logging.use_wandb = False

    return str(experiment_dir), writer


def train_one_epoch(
    model,
    train_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    config: ProjectConfig,
    epoch: int,
    global_step: int,
    writer: SummaryWriter,
    logger: logging.Logger
) -> Tuple[float, int]:
    """
    Train for one epoch.

    Args:
        model: Model to train
        train_dataloader: Training data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        config: Project configuration
        epoch: Current epoch number (0-indexed)
        global_step: Current global step
        writer: TensorBoard writer
        logger: Logger instance

    Returns:
        Tuple of (average_loss, new_global_step)
    """
    model.train()
    device = next(model.parameters()).device

    total_loss = 0.0
    num_batches = 0

    # Progress bar
    pbar = tqdm(train_dataloader, desc=f"Epoch {epoch} Training")

    for batch_idx, batch in enumerate(pbar):
        # Move batch to device
        input_features = batch['input_features'].to(device)
        labels = batch['labels'].to(device)

        # Get attention_mask if present (Speech2Text needs it)
        attention_mask = batch.get('attention_mask')
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        # Forward pass
        if config.model.model_type.lower() == "whisper":
            outputs = model.forward(input_features, labels=labels)
        else:  # speech2text or custom
            outputs = model.forward(
                input_features,
                attention_mask=attention_mask,
                labels=labels
            )

        loss = outputs['loss']

        # Backward pass
        loss.backward()

        # Gradient clipping (optional but recommended)
        if config.training.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()

        # Linear, OneCycleLR, and WarmupPlateauDecay schedulers step per batch
        if config.training.scheduler_name.lower() in ["linear", "onecycle", "warmup_plateau_decay"]:
            scheduler.step()

        # Accumulate loss
        total_loss += loss.item()
        num_batches += 1

        # Update global step
        global_step += 1

        # Update progress bar
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss/num_batches:.4f}',
            'lr': f'{current_lr:.2e}'
        })

        # Log to TensorBoard
        if global_step % config.training.logging_steps == 0:
            writer.add_scalar('train/loss', loss.item(), global_step)
            writer.add_scalar('train/learning_rate', current_lr, global_step)

            # Log to WandB if enabled
            if config.logging.use_wandb and WANDB_AVAILABLE and wandb.run:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/learning_rate': current_lr,
                    'epoch': epoch,
                    'global_step': global_step
                })

        # Clear CUDA cache periodically
        if (batch_idx + 1) % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Calculate average loss
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    logger.info(f"Epoch {epoch} - Average training loss: {avg_loss:.4f}")

    return avg_loss, global_step


def validate_model(
    model,
    val_dataloader: DataLoader,
    data_manager: DataManager,
    config: ProjectConfig,
    epoch: int,
    global_step: int,
    writer: SummaryWriter,
    logger: logging.Logger
) -> Dict[str, float]:
    """
    Validate model (similar to evaluate_model in evaluation.py).

    Args:
        model: Model to validate
        val_dataloader: Validation data loader
        data_manager: Data manager instance
        config: Project configuration
        epoch: Current epoch number (0-indexed)
        global_step: Current global step
        writer: TensorBoard writer
        logger: Logger instance

    Returns:
        Dictionary of validation metrics
    """
    logger.info("Running validation...")

    model.eval()
    device = next(model.parameters()).device

    # Get tokenizer from processor
    tokenizer = getattr(data_manager.processor, 'tokenizer', None)
    if tokenizer is None:
        raise ValueError("Tokenizer not found in processor")

    # Lists to accumulate decoded predictions on CPU
    all_pred_str = []
    all_label_str = []

    # Manual validation loop
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_dataloader, desc="Validation")):
            # Move batch to device
            input_features = batch['input_features'].to(device)
            labels = batch['labels'].to(device)

            # Get attention_mask if present
            attention_mask = batch.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            # Generate predictions
            if config.model.model_type.lower() == "whisper":
                generated_ids = model.generate(
                    input_data=input_features,
                    language=config.evaluation.language,
                    task=config.evaluation.task,
                    return_text=False,
                    max_length=config.evaluation.max_length,
                    num_beams=config.evaluation.num_beams,
                    repetition_penalty=config.evaluation.repetition_penalty,
                    no_repeat_ngram_size=config.evaluation.no_repeat_ngram_size,
                )
            else:  # speech2text or custom
                generated_ids = model.generate(
                    input_data=input_features,
                    attention_mask=attention_mask,
                    language=config.evaluation.language,
                    return_text=False,
                    max_length=config.evaluation.max_length,
                    num_beams=config.evaluation.num_beams,
                    repetition_penalty=config.evaluation.repetition_penalty,
                    no_repeat_ngram_size=config.evaluation.no_repeat_ngram_size,
                )

            # Move to CPU immediately
            pred_ids_cpu = generated_ids.cpu().numpy()
            label_ids_cpu = labels.cpu().numpy()

            # Decode predictions
            batch_pred_str = tokenizer.batch_decode(pred_ids_cpu, skip_special_tokens=True)

            # Decode labels (replace -100 with pad_token_id)
            label_ids_cpu_copy = label_ids_cpu.copy()
            label_ids_cpu_copy[label_ids_cpu_copy == -100] = tokenizer.pad_token_id
            batch_label_str = tokenizer.batch_decode(label_ids_cpu_copy, skip_special_tokens=True)

            # Accumulate
            all_pred_str.extend(batch_pred_str)
            all_label_str.extend(batch_label_str)

            # Clear CUDA cache periodically
            if (batch_idx + 1) % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Compute metrics
    logger.info("Computing validation metrics...")
    metrics_computer = STTMetrics(
        language=config.evaluation.language,
        use_bleu=config.evaluation.calculate_bleu
    )

    # Compute detailed metrics
    metrics = metrics_computer.compute_detailed_measures(all_pred_str, all_label_str)

    # Compute BLEU if enabled
    if config.evaluation.calculate_bleu:
        try:
            bleu_score = metrics_computer.compute_bleu(all_pred_str, all_label_str)
            metrics["bleu"] = bleu_score
        except Exception as e:
            logger.warning(f"Failed to compute BLEU: {e}")
            metrics["bleu"] = 0.0

    # Log metrics
    logger.info(f"Validation - Epoch {epoch} - WER: {metrics['wer']:.4f}, CER: {metrics['cer']:.4f}")

    # Log to TensorBoard
    writer.add_scalar('val/wer', metrics['wer'], global_step)
    writer.add_scalar('val/cer', metrics['cer'], global_step)
    if 'bleu' in metrics and metrics['bleu'] > 0:
        writer.add_scalar('val/bleu', metrics['bleu'], global_step)
    if 'mer' in metrics:
        writer.add_scalar('val/mer', metrics['mer'], global_step)
    if 'wil' in metrics:
        writer.add_scalar('val/wil', metrics['wil'], global_step)

    # Log error breakdown
    writer.add_scalar('val/substitutions', metrics['substitutions'], global_step)
    writer.add_scalar('val/deletions', metrics['deletions'], global_step)
    writer.add_scalar('val/insertions', metrics['insertions'], global_step)
    writer.add_scalar('val/hits', metrics['hits'], global_step)

    # Log to WandB if enabled
    if config.logging.use_wandb and WANDB_AVAILABLE and wandb.run:
        wandb_metrics = {
            'val/wer': metrics['wer'],
            'val/cer': metrics['cer'],
            'val/substitutions': metrics['substitutions'],
            'val/deletions': metrics['deletions'],
            'val/insertions': metrics['insertions'],
            'val/hits': metrics['hits'],
            'epoch': epoch,
            'global_step': global_step
        }
        if 'bleu' in metrics and metrics['bleu'] > 0:
            wandb_metrics['val/bleu'] = metrics['bleu']
        if 'mer' in metrics:
            wandb_metrics['val/mer'] = metrics['mer']
        if 'wil' in metrics:
            wandb_metrics['val/wil'] = metrics['wil']
        wandb.log(wandb_metrics)

    return metrics


def save_metrics(
    metrics: Dict[str, float],
    save_path: Path,
    logger: logging.Logger
) -> None:
    """
    Save metrics to JSON file in human-readable format.

    Args:
        metrics: Training/validation metrics dictionary
        save_path: Path to save metrics JSON file
        logger: Logger instance
    """
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.info(f"Metrics saved to: {save_path}")


def save_optimizer_state(
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    save_path: Path,
    logger: logging.Logger
) -> None:
    """
    Save optimizer state with epoch and global_step to file.

    Args:
        optimizer: Optimizer with state to save
        epoch: Current epoch number
        global_step: Current global step
        save_path: Path to save optimizer_state.pt
        logger: Logger instance
    """
    optimizer_state = {
        'epoch': epoch,
        'global_step': global_step,
        'optimizer_state_dict': optimizer.state_dict(),
    }

    torch.save(optimizer_state, save_path)
    logger.info(f"Optimizer state saved: {save_path}")


def save_scheduler_state(
    scheduler: Any,
    save_path: Path,
    logger: logging.Logger
) -> None:
    """
    Save scheduler state to file.

    Args:
        scheduler: Scheduler with state to save
        save_path: Path to save scheduler_state.pt
        logger: Logger instance
    """
    scheduler_state = {
        'scheduler_state_dict': scheduler.state_dict(),
    }

    torch.save(scheduler_state, save_path)
    logger.info(f"Scheduler state saved: {save_path}")


def save_experiment_metadata(
    last_finished_epoch: int,
    global_step: int,
    save_path: Path,
    logger: logging.Logger
) -> None:
    """
    Save experiment metadata (last finished epoch, global step, etc.)

    Args:
        last_finished_epoch: Last completed epoch number (0-indexed)
        global_step: Current global step
        save_path: Path to save experiment_metadata.json
        logger: Logger instance
    """
    metadata = {
        'last_finished_epoch': last_finished_epoch,
        'global_step': global_step,
    }

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info(f"Experiment metadata saved: {save_path}")


def load_experiment_metadata(checkpoint_path: Path, logger: logging.Logger) -> Tuple[int, int]:
    """
    Load experiment metadata from checkpoint directory.

    Args:
        checkpoint_path: Path to checkpoint directory
        logger: Logger instance

    Returns:
        Tuple of (last_finished_epoch, global_step)
        If metadata doesn't exist, returns (-1, 0) indicating new training
    """
    metadata_path = checkpoint_path / "experiment_metadata.json"

    if metadata_path.exists():
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        last_finished_epoch = metadata.get('last_finished_epoch', -1)
        global_step = metadata.get('global_step', 0)

        logger.info(f"Loaded experiment metadata: last_finished_epoch={last_finished_epoch}, global_step={global_step}")
        return last_finished_epoch, global_step
    else:
        logger.info("No experiment_metadata.json found - starting new training")
        return -1, 0


def save_checkpoint(
    model,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    global_step: int,
    metrics: Dict[str, float],
    experiment_dir: str,
    config: ProjectConfig,
    logger: logging.Logger,
    is_best: bool = False
):
    """
    Save training checkpoint (model + optimizer + scheduler + experiment metadata).

    Model is saved via ModelManager (model_weights.pt + model_metadata.json - without epoch).
    Optimizer state is saved separately (optimizer_state.pt with global_step).
    Scheduler state is saved separately (scheduler_state.pt).
    Experiment metadata is saved separately (experiment_metadata.json with last_finished_epoch).

    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Scheduler state
        epoch: Current finished epoch number (0-indexed)
        global_step: Current global step
        metrics: Validation metrics
        experiment_dir: Experiment directory
        config: Project configuration
        logger: Logger instance
        is_best: Whether this is the best model so far
    """
    checkpoints_dir = Path(experiment_dir) / "checkpoints"
    model_manager = ModelManager()

    # ALWAYS save checkpoint in epoch directory (0-indexed)
    epoch_checkpoint_dir = checkpoints_dir / f"epoch_{epoch}"
    model_manager.save_checkpoint(model, config.model, str(epoch_checkpoint_dir))
    logger.info(f"Model checkpoint saved: {epoch_checkpoint_dir}")

    # Save optimizer state separately (includes global_step only, epoch is in experiment_metadata)
    optimizer_state_path = epoch_checkpoint_dir / "optimizer_state.pt"
    save_optimizer_state(optimizer, epoch, global_step, optimizer_state_path, logger)

    # Save scheduler state separately
    scheduler_state_path = epoch_checkpoint_dir / "scheduler_state.pt"
    save_scheduler_state(scheduler, scheduler_state_path, logger)

    # Save experiment metadata (last_finished_epoch, global_step)
    experiment_metadata_path = epoch_checkpoint_dir / "experiment_metadata.json"
    save_experiment_metadata(epoch, global_step, experiment_metadata_path, logger)

    # Save full config for compatibility with find_config and evaluation
    config_path = epoch_checkpoint_dir / "config.yaml"
    save_config(config, str(config_path))
    logger.info(f"Config saved to: {config_path}")

    # Save metrics separately in human-readable JSON format
    metrics_path = epoch_checkpoint_dir / "metrics.json"
    save_metrics(metrics, metrics_path, logger)

    # If this is the best checkpoint, also save to best_checkpoint directory (in experiment root)
    if is_best:
        import shutil
        best_checkpoint_dir = Path(experiment_dir) / "best_checkpoint"

        # Remove old best_checkpoint if exists
        if best_checkpoint_dir.exists():
            shutil.rmtree(best_checkpoint_dir)

        # Copy entire epoch checkpoint to best_checkpoint (includes all state files)
        shutil.copytree(epoch_checkpoint_dir, best_checkpoint_dir)
        logger.info(f"Best checkpoint also saved to: {best_checkpoint_dir}")

    # Cleanup old checkpoints if needed
    if config.training.save_total_limit > 0:
        cleanup_old_checkpoints(checkpoints_dir, config.training.save_total_limit, logger)


def cleanup_old_checkpoints(checkpoints_dir: Path, keep_last: int, logger: logging.Logger):
    """
    Remove old checkpoints, keeping only the last N.

    Args:
        checkpoints_dir: Directory containing checkpoints
        keep_last: Number of recent checkpoints to keep
        logger: Logger instance
    """
    # Find all epoch directories (best_checkpoint is in experiment root, not here)
    epoch_dirs = [d for d in checkpoints_dir.glob("epoch_*") if d.is_dir()]

    # Sort by epoch number (not lexicographically) to correctly identify oldest/newest
    def extract_epoch_number(path: Path) -> int:
        """Extract epoch number from directory name like 'epoch_7' -> 7"""
        return int(path.name.split('_')[1])

    epoch_dirs = sorted(epoch_dirs, key=extract_epoch_number)

    # Keep only last N epochs (best_checkpoint is preserved automatically as it's outside checkpoints/)
    if len(epoch_dirs) > keep_last:
        to_remove = epoch_dirs[:-keep_last]
        for dir_path in to_remove:
            try:
                import shutil
                shutil.rmtree(dir_path)
                logger.info(f"Removed old checkpoint: {dir_path}")
            except Exception as e:
                logger.warning(f"Failed to remove checkpoint {dir_path}: {e}")


def evaluate_test_set(
    model,
    data_manager: DataManager,
    config: ProjectConfig,
    experiment_dir: str,
    logger: logging.Logger
):
    """
    Evaluate model on test set (final evaluation after training).

    Args:
        model: Trained model
        data_manager: Data manager instance
        config: Project configuration
        experiment_dir: Experiment directory
        logger: Logger instance
    """
    logger.info("="*60)
    logger.info("Running final evaluation on test set...")
    logger.info("="*60)

    # Create test dataset
    test_dataset = data_manager.create_dataset("test")
    logger.info(f"Test dataset size: {len(test_dataset)}")

    # Setup data collator
    transpose_features = config.model.model_type.lower() == "speech2text"
    data_collator = AudioCollator(
        config=config.data,
        model_type=config.model.model_type,
        transpose_features=transpose_features
    )

    # Create DataLoader
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.evaluation.batch_size,
        collate_fn=data_collator,
        num_workers=config.evaluation.num_workers,
        pin_memory=config.data.pin_memory if torch.cuda.is_available() else False,
        shuffle=False
    )

    model.eval()
    device = next(model.parameters()).device

    # Get tokenizer
    tokenizer = getattr(data_manager.processor, 'tokenizer', None)
    if tokenizer is None:
        raise ValueError("Tokenizer not found in processor")

    # Accumulate predictions
    all_pred_str = []
    all_label_str = []
    all_file_paths = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_dataloader, desc="Test Evaluation")):
            # Move batch to device
            input_features = batch['input_features'].to(device)
            labels = batch['labels'].to(device)

            attention_mask = batch.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            # Generate predictions
            if config.model.model_type.lower() == "whisper":
                generated_ids = model.generate(
                    input_data=input_features,
                    language=config.evaluation.language,
                    task=config.evaluation.task,
                    return_text=False,
                    max_length=config.evaluation.max_length,
                    num_beams=config.evaluation.num_beams,
                    repetition_penalty=config.evaluation.repetition_penalty,
                    no_repeat_ngram_size=config.evaluation.no_repeat_ngram_size,
                )
            else:
                generated_ids = model.generate(
                    input_data=input_features,
                    attention_mask=attention_mask,
                    language=config.evaluation.language,
                    return_text=False,
                    max_length=config.evaluation.max_length,
                    num_beams=config.evaluation.num_beams,
                    repetition_penalty=config.evaluation.repetition_penalty,
                    no_repeat_ngram_size=config.evaluation.no_repeat_ngram_size,
                )

            # Decode on CPU
            pred_ids_cpu = generated_ids.cpu().numpy()
            label_ids_cpu = labels.cpu().numpy()

            batch_pred_str = tokenizer.batch_decode(pred_ids_cpu, skip_special_tokens=True)

            label_ids_cpu_copy = label_ids_cpu.copy()
            label_ids_cpu_copy[label_ids_cpu_copy == -100] = tokenizer.pad_token_id
            batch_label_str = tokenizer.batch_decode(label_ids_cpu_copy, skip_special_tokens=True)

            all_pred_str.extend(batch_pred_str)
            all_label_str.extend(batch_label_str)

            # Get file paths
            start_idx = batch_idx * config.evaluation.batch_size
            end_idx = start_idx + len(labels)
            batch_paths = test_dataset.dataframe['path'].iloc[start_idx:end_idx].tolist()
            all_file_paths.extend(batch_paths)

            if (batch_idx + 1) % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Compute metrics
    logger.info("Computing test metrics...")
    metrics_computer = STTMetrics(
        language=config.evaluation.language,
        use_bleu=config.evaluation.calculate_bleu
    )

    metrics = metrics_computer.compute_detailed_measures(all_pred_str, all_label_str)

    if config.evaluation.calculate_bleu:
        try:
            bleu_score = metrics_computer.compute_bleu(all_pred_str, all_label_str)
            metrics["bleu"] = bleu_score
        except Exception as e:
            logger.warning(f"Failed to compute BLEU: {e}")
            metrics["bleu"] = 0.0

    # Print results
    logger.info("="*60)
    logger.info("Test Set Results:")
    logger.info("="*60)
    logger.info(f"Word Error Rate (WER): {metrics['wer']:.4f} ({metrics['wer']:.2%})")
    logger.info(f"Character Error Rate (CER): {metrics['cer']:.4f} ({metrics['cer']:.2%})")
    if "bleu" in metrics and metrics["bleu"] > 0:
        logger.info(f"BLEU Score: {metrics['bleu']:.4f}")
    logger.info(f"Substitutions: {metrics['substitutions']:,}")
    logger.info(f"Deletions: {metrics['deletions']:,}")
    logger.info(f"Insertions: {metrics['insertions']:,}")
    logger.info(f"Correct Words: {metrics['hits']:,}")
    logger.info("="*60)

    # Save metrics
    metrics_file = Path(experiment_dir) / "test_results.json"
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.info(f"Test metrics saved to: {metrics_file}")

    # Save predictions
    predictions_file = Path(experiment_dir) / "test_predictions.csv"
    df = pd.DataFrame({
        "file_path": all_file_paths,
        "reference": all_label_str,
        "prediction": all_pred_str
    })
    df.to_csv(predictions_file, index=False, encoding='utf-8')
    logger.info(f"Test predictions saved to: {predictions_file}")

    # Log to WandB if enabled
    if config.logging.use_wandb and WANDB_AVAILABLE and wandb.run:
        wandb_metrics = {
            'test/wer': metrics['wer'],
            'test/cer': metrics['cer'],
            'test/substitutions': metrics['substitutions'],
            'test/deletions': metrics['deletions'],
            'test/insertions': metrics['insertions'],
            'test/hits': metrics['hits']
        }
        if 'bleu' in metrics and metrics['bleu'] > 0:
            wandb_metrics['test/bleu'] = metrics['bleu']
        wandb.log(wandb_metrics)


def create_scheduler(
    optimizer,
    config: ProjectConfig,
    total_steps: int
):
    """
    Create learning rate scheduler based on configuration.

    Args:
        optimizer: PyTorch optimizer
        config: Project configuration
        total_steps: Total training steps (required for all schedulers)

    Returns:
        Learning rate scheduler
    """
    scheduler_name = config.training.scheduler_name.lower()

    if scheduler_name == "reduce_on_plateau":
        if config.training.reduce_on_plateau is None:
            raise ValueError("reduce_on_plateau config is None but scheduler_name is 'reduce_on_plateau'")

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=config.training.reduce_on_plateau.mode,  # type: ignore
            factor=config.training.reduce_on_plateau.factor,
            patience=config.training.reduce_on_plateau.patience,
            min_lr=config.training.reduce_on_plateau.min_lr,
            threshold=config.training.reduce_on_plateau.threshold
        )
        return scheduler

    elif scheduler_name == "cosine":
        if config.training.cosine is None:
            raise ValueError("cosine config is None but scheduler_name is 'cosine'")

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config.training.cosine.T_max,
            eta_min=config.training.cosine.eta_min,
            last_epoch=config.training.cosine.last_epoch
        )
        return scheduler

    elif scheduler_name == "onecycle":
        if config.training.onecycle is None:
            raise ValueError("onecycle config is None but scheduler_name is 'onecycle'")
        if total_steps is None:
            raise ValueError("total_steps is required for OneCycleLR scheduler")

        scheduler = OneCycleLR(
            optimizer,
            max_lr=config.training.onecycle.max_lr,
            total_steps=total_steps,
            pct_start=config.training.onecycle.pct_start,
            anneal_strategy=config.training.onecycle.anneal_strategy,  # type: ignore
            div_factor=config.training.onecycle.div_factor,
            final_div_factor=config.training.onecycle.final_div_factor
        )
        return scheduler

    elif scheduler_name == "linear":
        if config.training.linear is None:
            raise ValueError("linear config is None but scheduler_name is 'linear'")
        if total_steps is None:
            raise ValueError("total_steps is required for linear scheduler with warmup")

        # Calculate warmup steps as a ratio of total training steps
        num_warmup_steps = int(config.training.linear.warmup_ratio * total_steps)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_steps
        )
        return scheduler

    elif scheduler_name == "warmup_plateau_decay":
        if config.training.warmup_plateau_decay is None:
            raise ValueError("warmup_plateau_decay config is None but scheduler_name is 'warmup_plateau_decay'")
        if total_steps is None:
            raise ValueError("total_steps is required for warmup_plateau_decay scheduler")

        # Calculate phase boundaries
        warmup_steps = int(config.training.warmup_plateau_decay.warmup_ratio * total_steps)
        plateau_end_step = int(config.training.warmup_plateau_decay.plateau_ratio * total_steps)

        def lr_lambda(current_step: int) -> float:
            """
            Learning rate schedule: warmup + plateau + linear decay

            Phase 1 (0 to warmup_steps): Linear warmup from 0 to 1
            Phase 2 (warmup_steps to plateau_end_step): Constant at 1 (plateau)
            Phase 3 (plateau_end_step to total_steps): Linear decay from 1 to 0
            """
            if current_step < warmup_steps:
                # Warmup phase: linear increase from 0 to 1
                return float(current_step) / float(max(1, warmup_steps))
            elif current_step < plateau_end_step:
                # Plateau phase: constant at 1
                return 1.0
            else:
                # Decay phase: linear decrease from 1 to 0
                decay_steps = total_steps - plateau_end_step
                progress = (current_step - plateau_end_step) / float(max(1, decay_steps))
                return max(0.0, 1.0 - progress)

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        return scheduler

    else:
        raise ValueError(
            f"Unknown scheduler_name: '{scheduler_name}'. "
            f"Supported schedulers: 'reduce_on_plateau', 'cosine', 'onecycle', 'linear', 'warmup_plateau_decay'"
        )


def load_scheduler(
    optimizer: torch.optim.Optimizer,
    config: ProjectConfig,
    total_steps: int,
    checkpoint_path: str
) -> Any:
    """
    Create scheduler and optionally load its state from checkpoint.

    Creates scheduler based on config, then loads state if scheduler_state.pt exists.
    When resuming mid-training, the loaded scheduler state includes current step/epoch,
    so the scheduler will continue from the correct position automatically.

    Args:
        optimizer: Optimizer instance
        config: Project configuration
        total_steps: Total training steps (from DataLoader length * num_epochs)
        checkpoint_path: Path to checkpoint directory (may contain scheduler_state.pt)

    Returns:
        Learning rate scheduler
    """
    logger = logging.getLogger(__name__)
    checkpoint_path_obj = Path(checkpoint_path)

    # Create scheduler (always create fresh first)
    logger.info(f"Creating {config.training.scheduler_name} scheduler with total_steps={total_steps}")
    scheduler = create_scheduler(optimizer, config, total_steps)

    # Try to load scheduler state
    scheduler_state_path = checkpoint_path_obj / "scheduler_state.pt"

    if scheduler_state_path.exists():
        logger.info(f"Found scheduler_state.pt - loading scheduler state...")
        scheduler_state = torch.load(scheduler_state_path, map_location='cpu', weights_only=False)

        if 'scheduler_state_dict' in scheduler_state:
            scheduler.load_state_dict(scheduler_state['scheduler_state_dict'])
            logger.info("Loaded scheduler state - scheduler will continue from saved step")
        else:
            logger.warning("scheduler_state.pt exists but missing 'scheduler_state_dict' key")
    else:
        logger.info("No scheduler_state.pt found - using fresh scheduler")

    return scheduler


def load_experiment_objects(
    checkpoint_path: str,
    config: ProjectConfig
):
    """
    Load experiment objects from checkpoint directory.

    Loads model (always from model_weights.pt), processor, and optimizer.
    If optimizer_state.pt exists, loads optimizer state (resume training).
    If experiment_metadata.json exists, loads last_finished_epoch and continues from next epoch.
    Otherwise creates fresh optimizer and starts from epoch 0 (new training from baseline).
    Scheduler is NOT loaded here - use load_scheduler() after DataLoader.

    Args:
        checkpoint_path: Path to checkpoint directory (baseline or training checkpoint)
        config: Project configuration

    Returns:
        Dict with keys: model, processor, optimizer, start_epoch, global_step
    """
    logger = logging.getLogger(__name__)
    checkpoint_path_obj = Path(checkpoint_path)

    # Initialize model manager
    model_manager = ModelManager()

    # Determine device
    device = torch.device("cuda" if config.model.use_gpu and torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load model and processor from checkpoint
    logger.info(f"Loading model from checkpoint: {checkpoint_path}")
    model, processor, checkpoint_info = model_manager.load_checkpoint(
        checkpoint_path,
        device=device,
        processor=None,  # Will be auto-created from checkpoint model_metadata
        compile=config.model.compile_model,
        language=config.data.language,  # Required for Whisper processor creation
        task=config.data.task  # Required for Whisper processor creation
    )

    # Apply freezing settings from config
    logger.info("Applying freeze settings from config...")
    model.apply_freezing_from_config(config.model)

    trainable_params = model.get_trainable_parameters()
    total_params = model.get_num_parameters()
    logger.info(f"Model parameters: {trainable_params:,} / {total_params:,} trainable")

    # Check if model has trainable parameters
    if trainable_params == 0:
        raise ValueError(
            "Model has 0 trainable parameters! All parameters are frozen.\n"
            "Please check your freezing configuration:\n"
            f"  - freeze_feature_encoder: {config.model.freeze_feature_encoder}\n"
            f"  - freeze_encoder: {config.model.freeze_encoder}\n"
            f"  - freeze_decoder: {config.model.freeze_decoder}\n"
            "At least one component must be trainable for fine-tuning."
        )

    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )

    # Load experiment metadata (contains last_finished_epoch and global_step)
    last_finished_epoch, global_step = load_experiment_metadata(checkpoint_path_obj, logger)

    # Calculate start_epoch: if resuming, continue from next epoch after last_finished_epoch
    # last_finished_epoch = -1 means new training, so start_epoch = 0
    start_epoch = last_finished_epoch + 1

    # Load optimizer state if exists
    optimizer_state_path = checkpoint_path_obj / "optimizer_state.pt"

    if optimizer_state_path.exists():
        logger.info("Found optimizer_state.pt - loading optimizer state (resuming training)...")
        optimizer_state = torch.load(optimizer_state_path, map_location='cpu', weights_only=False)

        # Load optimizer state
        if 'optimizer_state_dict' in optimizer_state:
            optimizer.load_state_dict(optimizer_state['optimizer_state_dict'])
            logger.info("Loaded optimizer state")
    else:
        logger.info("No optimizer_state.pt found - starting new training from baseline")

    logger.info(f"Experiment loaded: {checkpoint_info.get('model_name', 'unknown')}")
    logger.info(f"Will start training from epoch {start_epoch}, global_step {global_step}")

    return {
        'model': model,
        'processor': processor,
        'optimizer': optimizer,
        'start_epoch': start_epoch,
        'global_step': global_step
    }


def main():
    """Main training function"""
    # Parse arguments
    args = parse_args()

    # Find and load configuration
    checkpoint_path = Path(args.model_path)
    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")

    # Find config file
    config_file_path = find_config(checkpoint_path, args.config)
    config = load_config(config_file_path)

    # Override config with CLI args
    if args.epochs:
        config.training.num_train_epochs = args.epochs

    if args.no_wandb:
        config.logging.use_wandb = False

    # Debug mode
    if args.debug:
        config.training.num_train_epochs = 2
        config.training.logging_steps = 10
        config.training.save_steps = 50
        config.logging.use_wandb = False

    # Set random seed early
    set_seed(config.experiment.seed)

    # Determine experiment name early (needed for logging setup)
    if args.experiment_name:
        config.experiment.experiment_name = args.experiment_name
    elif not config.experiment.experiment_name:
        # Auto-generate from experiment directory name
        experiment_name = Path(checkpoint_path).name
        config.experiment.experiment_name = f"{experiment_name}_trained"

    # Ensure experiment_name is set (for type checker)
    assert config.experiment.experiment_name is not None, "Experiment name must be set"

    # Create experiment directory early (needed for logging setup)
    experiment_dir = Path(config.experiment.output_dir) / config.experiment.experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging EARLY - before any other operations
    log_file = experiment_dir / "training.log"
    logger = setup_logging(config, log_file=str(log_file))

    # Now log all the information
    logger.info(f"Checkpoint path: {checkpoint_path}")
    logger.info(f"Configuration file: {config_file_path}")
    if args.epochs:
        logger.info(f"Overriding epochs to: {args.epochs}")
    if args.debug:
        logger.info("Debug mode enabled")

    # Print system info (now it will be logged too)
    print_system_info()

    try:
        # Initialize data manager
        logger.info("Initializing data manager...")
        data_manager = DataManager(config)

        # Load experiment objects (model, processor, optimizer)
        # Automatically detects if resuming (has optimizer_state.pt) or new training
        logger.info("Loading experiment objects...")
        exp_objects = load_experiment_objects(
            str(checkpoint_path),
            config
        )

        # Extract experiment objects
        model = exp_objects['model']
        processor = exp_objects['processor']
        optimizer = exp_objects['optimizer']
        start_epoch = exp_objects['start_epoch']
        global_step = exp_objects['global_step']

        # Determine if this is resume based on start_epoch
        is_resume = start_epoch > 0

        # Update data manager with loaded processor (explicit control)
        data_manager.set_already_loaded_processor(processor)

        # Create datasets AFTER processor is loaded (needed for CPU mode)
        logger.info("Loading datasets...")
        train_dataset = data_manager.create_dataset("train")
        val_dataset = data_manager.create_dataset("validation")
        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Validation dataset size: {len(val_dataset)}")

        # Setup experiment (now only creates checkpoints dir and TensorBoard/WandB)
        experiment_dir, writer = setup_experiment(
            config,
            args,
            str(checkpoint_path),
            is_resume=is_resume,
            experiment_dir=experiment_dir  # Pass already created dir
        )

        # Print configuration
        print_config(config)

        # Print GPU memory info
        if torch.cuda.is_available():
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            logger.info("Using CPU")

        # Print model info
        trainable_params = model.get_trainable_parameters()
        total_params = model.get_num_parameters()
        logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

        # Setup data collator
        transpose_features = config.model.model_type.lower() == "speech2text"
        data_collator = AudioCollator(
            config=config.data,
            model_type=config.model.model_type,
            transpose_features=transpose_features
        )

        # Create data loaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.training.train_batch_size,
            collate_fn=data_collator,
            num_workers=config.training.num_workers,
            pin_memory=config.data.pin_memory if torch.cuda.is_available() else False,
            shuffle=True
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config.training.eval_batch_size,
            collate_fn=data_collator,
            num_workers=config.training.num_workers,
            pin_memory=config.data.pin_memory if torch.cuda.is_available() else False,
            shuffle=False
        )

        logger.info(f"Training batches per epoch: {len(train_dataloader)}")
        logger.info(f"Validation batches: {len(val_dataloader)}")

        # Calculate total training steps
        num_train_epochs = config.training.num_train_epochs
        steps_per_epoch = len(train_dataloader)
        total_steps = steps_per_epoch * num_train_epochs

        logger.info(f"Total training steps: {total_steps}")

        # Create/load scheduler (automatically loads state if scheduler_state.pt exists)
        scheduler = load_scheduler(optimizer, config, total_steps, str(checkpoint_path))

        # Optimizer and scheduler ready
        logger.info(f"Using optimizer: AdamW (lr={config.training.learning_rate})")
        logger.info(f"Using scheduler: {config.training.scheduler_name}")
        logger.info(f"Starting from epoch {start_epoch}, global_step {global_step}")

        # Training loop
        logger.info("="*60)
        logger.info("Starting training...")
        logger.info("="*60)

        # Initialize best_wer from previous best_checkpoint if resuming
        best_wer = float('inf')
        best_epoch = -1
        patience_counter = 0

        if is_resume:
            # Try to load best WER from previous best_checkpoint
            best_checkpoint_metrics = Path(experiment_dir) / "best_checkpoint" / "metrics.json"
            if best_checkpoint_metrics.exists():
                try:
                    with open(best_checkpoint_metrics, 'r', encoding='utf-8') as f:
                        prev_best_metrics = json.load(f)
                        best_wer = prev_best_metrics.get('wer', float('inf'))
                        logger.info(f"Loaded previous best WER from best_checkpoint: {best_wer:.4f}")
                except Exception as e:
                    logger.warning(f"Failed to load previous best_checkpoint metrics: {e}")

        start_time = datetime.now()

        for epoch in range(start_epoch, num_train_epochs):
            logger.info(f"\nEpoch {epoch}/{num_train_epochs-1}")

            # Train for one epoch
            avg_train_loss, global_step = train_one_epoch(
                model=model,
                train_dataloader=train_dataloader,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
                epoch=epoch,
                global_step=global_step,
                writer=writer,
                logger=logger
            )

            # Validate
            val_metrics = validate_model(
                model=model,
                val_dataloader=val_dataloader,
                data_manager=data_manager,
                config=config,
                epoch=epoch,
                global_step=global_step,
                writer=writer,
                logger=logger
            )

            current_wer = val_metrics['wer']

            # Update learning rate based on scheduler type
            if config.training.scheduler_name.lower() == "reduce_on_plateau":
                # ReduceLROnPlateau requires metric
                scheduler.step(current_wer)  # type: ignore
            elif config.training.scheduler_name.lower() == "cosine":
                # CosineAnnealingLR steps without metric
                scheduler.step()  # type: ignore
            # Linear and OneCycleLR step per batch, not per epoch - handled in train_one_epoch

            logger.info(f"Current LR: {optimizer.param_groups[0]['lr']:.2e}")

            # Check if this is the best model
            is_best = current_wer < best_wer
            if is_best:
                best_wer = current_wer
                best_epoch = epoch
                patience_counter = 0
                logger.info(f"New best WER: {best_wer:.4f}")
            else:
                patience_counter += 1

            # Save checkpoint
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
                metrics=val_metrics,
                experiment_dir=experiment_dir,
                config=config,
                logger=logger,
                is_best=is_best
            )

            # Early stopping
            if config.training.use_early_stopping:
                if patience_counter >= config.training.early_stopping_patience:
                    logger.info(f"Early stopping triggered after {patience_counter} epochs without improvement")
                    break

        # Training completed
        training_time = datetime.now() - start_time
        logger.info("="*60)
        logger.info("Training completed!")
        logger.info(f"Total training time: {training_time}")
        logger.info(f"Best WER: {best_wer:.4f} (Epoch {best_epoch})")
        logger.info("="*60)

        # Evaluate on test set
        evaluate_test_set(
            model=model,
            data_manager=data_manager,
            config=config,
            experiment_dir=experiment_dir,
            logger=logger
        )

        # Close TensorBoard writer
        writer.close()

        # Finish WandB run
        if config.logging.use_wandb and WANDB_AVAILABLE and wandb.run:
            wandb.finish()

        logger.info("="*60)
        logger.info("All done!")
        logger.info(f"Results saved to: {experiment_dir}")
        logger.info(f"TensorBoard: tensorboard --logdir {experiment_dir}/tensorboard")
        logger.info("="*60)

    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
