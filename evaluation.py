"""
Evaluation script for trained speech-to-text models.
Supports comprehensive evaluation with detailed error analysis.
"""

import os
import sys
import logging
import warnings
from pathlib import Path
from typing import Optional
import argparse
import json

# Подавить pydantic warnings при работе с OmegaConf
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic._internal._generate_schema")

import torch
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.config import ProjectConfig, load_config, save_config, print_config
from src.data import DataManager, AudioCollator
from src.models import ModelManager
from src.metrics import HuggingFaceMetricsComputer, PerformanceAnalyzer
from src.utils import (
    setup_logging, set_seed, print_system_info
)

# Suppress warnings and configure CUDA memory management
os.environ["TOKENIZERS_PARALLELISM"] = "false"
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # Reduce memory fragmentation


# ==============================================================================
# CONFIGURATION - Edit these parameters to run via VSCode
# ==============================================================================
# При запуске через VSCode "Run" будут использованы эти значения
DEFAULT_CONFIG = {
    "model_path": "experiments/whisper-base_train_full_decoder/checkpoints/epoch_4",
    "config": None,  # None = auto-search in model directory
    "dataset_split": "validation",  # "train", "validation", "test"
    "experiment_name": "whisper-base_full_dec_eval_val",  # None = auto-generate from model name
}
# Example model paths:
# experiments/baselines/whisper-small
# experiments/baselines/whisper-base
# experiments/baselines/s2t-cross-lingual
# ==============================================================================


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate Russian Speech-to-Text Model")
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_CONFIG["model_path"],
        help=f"Path to trained model checkpoint directory (default: {DEFAULT_CONFIG['model_path']})"
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
        help="Override experiment name for evaluation output"
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default=DEFAULT_CONFIG["dataset_split"],
        choices=["train", "validation", "test"],
        help=f"Dataset split to evaluate (default: {DEFAULT_CONFIG['dataset_split']})"
    )

    return parser.parse_args()


def find_config(model_path: Path, config_path: Optional[str] = None) -> str:
    """Find configuration file in model directory"""
    # If config explicitly provided, use it
    if config_path:
        if Path(config_path).exists():
            return config_path
        else:
            raise ValueError(f"Config file not found: {config_path}")

    # Otherwise, look ONLY in model directory
    config_in_model_dir = model_path / "config.yaml"
    if config_in_model_dir.exists():
        return str(config_in_model_dir)

    raise ValueError(
        f"Could not find config.yaml in model directory: {model_path}\n"
        f"Please either:\n"
        f"  1. Ensure config.yaml exists in {model_path}/\n"
        f"  2. Specify --config explicitly"
    )


def setup_evaluation(config: ProjectConfig, args, model_path: str) -> str:
    """Setup evaluation directory and logging

    Returns:
        str: Path to evaluation output directory
    """
    # Determine experiment name (priority: CLI arg > config > model name)
    if args.experiment_name:
        # CLI argument has highest priority
        eval_name = args.experiment_name
    elif config.experiment.experiment_name:
        # Use experiment name from config (e.g., "whisper_base_baseline_eval")
        eval_name = config.experiment.experiment_name
    else:
        # Fallback: use model directory name + "_evaluation"
        model_name = Path(model_path).name
        eval_name = f"{model_name}_evaluation"

    # Create evaluation directory
    eval_dir = Path(config.experiment.output_dir) / eval_name
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging to file and console
    log_file = eval_dir / "evaluation.log"
    logger = setup_logging(config, log_file=str(log_file))

    logger.info(f"Evaluation directory: {eval_dir}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Model path: {model_path}")

    # Save configuration
    config_path = eval_dir / "config.yaml"
    save_config(config, str(config_path))
    logger.info(f"Configuration saved to: {config_path}")

    return str(eval_dir)


def load_model_and_processor(
    model_path: str,
    config: ProjectConfig,
    data_manager: DataManager
):
    """
    Load trained model and processor from checkpoint.

    Args:
        model_path: Path to checkpoint directory (containing model_weights.pt and model_metadata.json)
        config: Project configuration
        data_manager: Data manager instance

    Returns:
        Tuple of (model, processor)
    """
    logger = logging.getLogger(__name__)

    # Initialize model manager
    logger.info("Initializing model manager...")
    model_manager = ModelManager()

    # Determine device
    device = torch.device("cuda" if config.model.use_gpu and torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load checkpoint (automatically creates processor from model_metadata)
    logger.info(f"Loading checkpoint from: {model_path}")
    model, processor, checkpoint_info = model_manager.load_checkpoint(
        model_path,
        device=device,
        processor=None,  # Will be auto-created from checkpoint model_metadata
        compile=False,
        language=config.evaluation.language,  # Required for Whisper processor creation
        task=config.evaluation.task  # Required for Whisper processor creation
    )

    # Update data manager with loaded processor (extracts pad_token_id and forced_bos_token_id automatically)
    data_manager.set_already_loaded_processor(processor)

    # Log checkpoint info
    model_name = checkpoint_info.get('model_name', 'unknown')
    logger.info(f"Loaded checkpoint: model={model_name}")

    return model, processor


def evaluate_model(
    model,
    data_manager: DataManager,
    config: ProjectConfig,
    split: str,
    eval_dir: str,
    save_predictions: bool = True,
    detailed_analysis: bool = True
):
    """
    Run evaluation with manual loop for full GPU memory control.

    All predictions accumulate on CPU only, preventing GPU OOM.
    """

    logger = logging.getLogger(__name__)

    # Create dataset
    logger.info(f"Loading {split} dataset...")
    dataset = data_manager.create_dataset(split)
    logger.info(f"Evaluating on {len(dataset)} samples from {split} split")

    # Setup data collator
    transpose_features = config.model.model_type.lower() == "speech2text"
    data_collator = AudioCollator(
        config=config.data,
        model_type=config.model.model_type,
        transpose_features=transpose_features
    )

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=config.evaluation.batch_size,
        collate_fn=data_collator,
        num_workers=config.evaluation.num_workers,
        pin_memory=config.data.pin_memory if torch.cuda.is_available() else False,
        shuffle=False  # Important: don't shuffle for evaluation
    )

    logger.info(f"Batch size: {config.evaluation.batch_size}")
    logger.info(f"Total batches: {len(dataloader)}")
    logger.info(f"Device: {next(model.parameters()).device}")

    # Prepare model for evaluation
    model.eval()
    device = next(model.parameters()).device

    # Ensure processor is available
    if data_manager.processor is None:
        raise ValueError("Processor not initialized")

    # Get tokenizer from processor
    tokenizer = getattr(data_manager.processor, 'tokenizer', None)
    if tokenizer is None:
        raise ValueError("Tokenizer not found in processor")

    # Lists to accumulate decoded predictions on CPU
    all_pred_str = []
    all_label_str = []
    all_file_paths = []

    # Manual evaluation loop
    logger.info("Starting evaluation loop...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluating {split}")):
            # Move batch to device
            input_features = batch['input_features'].to(device)
            labels = batch['labels'].to(device)

            # Get attention_mask if present (Speech2Text needs it)
            attention_mask = batch.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            # Generate predictions
            # Use model.generate() for actual transcription
            if config.model.model_type.lower() == "whisper":
                generated_ids = model.generate(
                    input_data=input_features,
                    language=config.evaluation.language,
                    task=config.evaluation.task,
                    return_text=False,  # Return token IDs, not text
                    max_length=config.evaluation.max_length,
                    num_beams=config.evaluation.num_beams,
                    repetition_penalty=config.evaluation.repetition_penalty,
                    no_repeat_ngram_size=config.evaluation.no_repeat_ngram_size,
                )
            else:  # speech2text or custom
                generated_ids = model.generate(
                    input_data=input_features,
                    attention_mask=attention_mask,
                    language=config.evaluation.language,  # Required for multilingual Speech2Text models
                    return_text=False,
                    max_length=config.evaluation.max_length,
                    num_beams=config.evaluation.num_beams,
                    repetition_penalty=config.evaluation.repetition_penalty,
                    no_repeat_ngram_size=config.evaluation.no_repeat_ngram_size,
                )

            # Move to CPU immediately to free GPU memory
            pred_ids_cpu = generated_ids.cpu().numpy()
            label_ids_cpu = labels.cpu().numpy()

            # Decode predictions for this batch
            batch_pred_str = tokenizer.batch_decode(pred_ids_cpu, skip_special_tokens=True)

            # Decode labels for this batch (replace -100 with pad_token_id)
            label_ids_cpu_copy = label_ids_cpu.copy()
            label_ids_cpu_copy[label_ids_cpu_copy == -100] = tokenizer.pad_token_id
            batch_label_str = tokenizer.batch_decode(label_ids_cpu_copy, skip_special_tokens=True)

            # Accumulate decoded strings
            all_pred_str.extend(batch_pred_str)
            all_label_str.extend(batch_label_str)

            # Get file paths for this batch from dataset
            # Calculate indices for current batch
            start_idx = batch_idx * config.evaluation.batch_size
            end_idx = start_idx + len(labels)
            batch_paths = dataset.dataframe['path'].iloc[start_idx:end_idx].tolist()
            all_file_paths.extend(batch_paths)

            # Clear CUDA cache periodically
            if (batch_idx + 1) % 10 == 0:
                torch.cuda.empty_cache()

    logger.info("Evaluation loop completed. Processing results...")

    # Use accumulated decoded strings
    pred_str = all_pred_str
    label_str = all_label_str
    file_paths = all_file_paths

    # Compute metrics
    logger.info("Computing metrics...")
    from src.metrics import STTMetrics
    metrics_computer = STTMetrics(
        language=config.evaluation.language,
        use_bleu=config.evaluation.calculate_bleu
    )

    # Compute detailed metrics (includes WER, CER, substitutions, etc.)
    metrics = metrics_computer.compute_detailed_measures(pred_str, label_str)

    # Compute BLEU if enabled
    if config.evaluation.calculate_bleu:
        try:
            bleu_score = metrics_computer.compute_bleu(pred_str, label_str)
            metrics["bleu"] = bleu_score
        except Exception as e:
            logger.warning(f"Failed to compute BLEU: {e}")
            metrics["bleu"] = 0.0

    logger.info(f"WER: {metrics['wer']:.4f}, CER: {metrics['cer']:.4f}")

    # Log metrics to TensorBoard (save directly in eval_dir)
    writer = SummaryWriter(log_dir=str(eval_dir))

    # Log main metrics
    writer.add_scalar(f"eval/{split}/wer", metrics["wer"], 0)
    writer.add_scalar(f"eval/{split}/cer", metrics["cer"], 0)
    if "bleu" in metrics and metrics["bleu"] > 0:
        writer.add_scalar(f"eval/{split}/bleu", metrics["bleu"], 0)
    if "mer" in metrics:
        writer.add_scalar(f"eval/{split}/mer", metrics["mer"], 0)
    if "wil" in metrics:
        writer.add_scalar(f"eval/{split}/wil", metrics["wil"], 0)

    # Log error breakdown
    writer.add_scalar(f"eval/{split}/substitutions", metrics["substitutions"], 0)
    writer.add_scalar(f"eval/{split}/deletions", metrics["deletions"], 0)
    writer.add_scalar(f"eval/{split}/insertions", metrics["insertions"], 0)
    writer.add_scalar(f"eval/{split}/hits", metrics["hits"], 0)

    # Log empty prediction rate if available
    if "empty_prediction_rate" in metrics:
        writer.add_scalar(f"eval/{split}/empty_prediction_rate", metrics["empty_prediction_rate"], 0)

    writer.close()
    logger.info(f"TensorBoard logs saved to: {eval_dir}")

    # Save metrics
    metrics_file = Path(eval_dir) / f"{split}_results.json"
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.info(f"Metrics saved to: {metrics_file}")

    # Print metrics summary
    logger.info("="*60)
    logger.info(f"Evaluation Results on {split} split:")
    logger.info("="*60)

    # Print metrics
    if metrics:
        logger.info(f"Word Error Rate (WER): {metrics['wer']:.4f} ({metrics['wer']:.2%})")
        logger.info(f"Character Error Rate (CER): {metrics['cer']:.4f} ({metrics['cer']:.2%})")

        # BLEU if available
        if "bleu" in metrics and metrics["bleu"] > 0:
            logger.info(f"BLEU Score: {metrics['bleu']:.4f}")

        # Detailed metrics
        logger.info(f"Substitutions: {metrics['substitutions']:,}")
        logger.info(f"Deletions: {metrics['deletions']:,}")
        logger.info(f"Insertions: {metrics['insertions']:,}")
        logger.info(f"Correct Words: {metrics['hits']:,}")

    logger.info("="*60)

    # Save predictions if requested
    if save_predictions:
        predictions_file = Path(eval_dir) / f"{split}_predictions.csv"
        df = pd.DataFrame({
            "file_path": file_paths,
            "reference": label_str,
            "prediction": pred_str
        })
        df.to_csv(predictions_file, index=False, encoding='utf-8')
        logger.info(f"Predictions saved to: {predictions_file}")

        # Log sample predictions
        logger.info("Sample predictions:")
        num_samples = min(3, len(df))
        for i in range(num_samples):
            logger.info(f"  Reference:  {df.iloc[i]['reference']}")
            logger.info(f"  Prediction: {df.iloc[i]['prediction']}")
            logger.info("  ---")

    # Detailed analysis if requested
    if detailed_analysis:
        logger.info("Performing detailed error analysis...")

        analyzer = PerformanceAnalyzer(language=config.evaluation.language)

        # Get durations if available (optional)
        durations = None  # We don't have durations in current pipeline

        analysis = analyzer.analyze_errors(pred_str, label_str, durations)

        analysis_file = Path(eval_dir) / f"{split}_detailed_analysis.json"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)

        logger.info(f"Detailed analysis saved to: {analysis_file}")

        # Print analysis summary
        logger.info("="*60)
        logger.info("Detailed Error Analysis:")
        logger.info("="*60)
        logger.info(f"Total Samples: {analysis['total_samples']:,}")
        logger.info(f"Perfect Matches: {analysis['perfect_matches']:,} ({analysis['perfect_match_rate']:.2%})")
        logger.info(f"Empty Predictions: {analysis['empty_predictions']:,} ({analysis['empty_prediction_rate']:.2%})")
        logger.info(f"Overall WER: {analysis['overall_wer']:.4f}")

        # WER by text length
        if analysis.get("wer_by_length"):
            logger.info("\nWER by Text Length (words in reference):")
            for length_range, stats in analysis["wer_by_length"].items():
                logger.info(f"  {length_range}: WER={stats['wer']:.4f}, Count={stats['count']:,}")

        logger.info("="*60)

    # Print TensorBoard command
    logger.info("\n" + "="*60)
    logger.info("View evaluation metrics in TensorBoard:")
    logger.info(f"  tensorboard --logdir {eval_dir}")
    logger.info("="*60)


def main():
    """Main evaluation function"""
    # Parse arguments
    args = parse_args()

    # Find and load configuration
    model_path = Path(args.model_path)
    config_file = find_config(model_path, args.config)

    print(f"Loading configuration from: {config_file}")
    config = load_config(config_file)

    # Setup evaluation
    eval_dir = setup_evaluation(config, args, args.model_path)
    logger = logging.getLogger(__name__)

    # Print configuration and system info
    print_config(config)
    print_system_info()

    # Set random seed
    set_seed(config.experiment.seed)

    try:
        logger.info("Initializing data manager...")
        data_manager = DataManager(config)

        # Load model and processor
        logger.info("Loading model and processor...")
        model, processor = load_model_and_processor(
            args.model_path,
            config,
            data_manager
        )

        # Ensure data manager has processor
        if data_manager.processor is None:
            data_manager.processor = processor
            tokenizer = getattr(processor, 'tokenizer', None)
            if tokenizer is not None:
                data_manager.pad_token_id = getattr(tokenizer, 'pad_token_id', None)

        # Print GPU memory info
        if torch.cuda.is_available():
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            logger.info("Using CPU")

        # Run evaluation with manual loop (no Trainer, full GPU memory control)
        logger.info(f"Evaluating on {args.dataset_split} dataset...")
        evaluate_model(
            model=model,
            data_manager=data_manager,
            config=config,
            split=args.dataset_split,
            eval_dir=eval_dir,
            save_predictions=True,
            detailed_analysis=True
        )

        logger.info("="*60)
        logger.info("Evaluation completed successfully!")
        logger.info(f"Results saved to: {eval_dir}")
        logger.info("="*60)

        # Final memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        logger.error(f"Evaluation failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
