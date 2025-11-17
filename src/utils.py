"""
Utility functions for the speech-to-text project.
Includes logging, visualization, and helper functions.
"""

import os
import logging
import random
import json
import time
from typing import Dict, List, Optional, Any
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from .config import ProjectConfig

# Set matplotlib style for scientific plots
matplotlib.use('Agg')  # Non-interactive backend for server environments
plt.style.use('seaborn-v0_8-paper')  # Scientific paper style

console = Console()

# ===========================
# PATH MANAGEMENT
# ===========================

# Project root directory (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def get_project_root() -> Path:
    """Get the project root directory"""
    return PROJECT_ROOT

def get_data_dir() -> Path:
    """Get the data directory path"""
    return PROJECT_ROOT / "data"

def get_config_dir() -> Path:
    """Get the configs directory path"""
    return PROJECT_ROOT / "configs"

def get_src_dir() -> Path:
    """Get the src directory path"""
    return PROJECT_ROOT / "src"

def get_notebooks_dir() -> Path:
    """Get the notebooks directory path"""
    return PROJECT_ROOT / "notebooks"

def get_experiments_dir() -> Path:
    """Get the experiments directory path"""
    return PROJECT_ROOT / "experiments"


def setup_logging(config: ProjectConfig, log_file: Optional[str] = None) -> logging.Logger:
    """Setup logging configuration with proper Unicode handling for Windows

    Args:
        config: Project configuration
        log_file: Optional path to log file. If not provided, uses default location.
    """
    import sys

    log_level = getattr(logging, config.logging.log_level.upper())

    # Determine log file path
    if log_file is None:
        log_dir = Path(config.experiment.output_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file_path = log_dir / "training.log"
    else:
        log_file_path = Path(log_file)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)

    # Create handlers with proper encoding
    # File handler: UTF-8 encoding for log files
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setLevel(log_level)

    # Console handler: handle Unicode errors gracefully (important for Windows)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    # Fix for Windows console: replace unencodable characters instead of crashing
    # This prevents UnicodeEncodeError when logging multilingual text in Windows (cp1251)
    try:
        # Python 3.7+: reconfigure stream to replace unencodable chars
        console_handler.stream.reconfigure(errors='replace')  # type: ignore
    except AttributeError:
        # Fallback for older Python versions
        import codecs
        console_handler.stream = codecs.getwriter('utf-8')(sys.stdout.buffer, errors='replace')  # type: ignore

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    # Clear existing handlers to avoid duplicates
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    logger = logging.getLogger(__name__)
    logger.info(f"Logging setup complete. Level: {config.logging.log_level}")
    logger.info(f"Log file: {log_file_path}")

    return logger


def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    console.print(f"[green]Random seed set to {seed}[/green]")


def print_system_info():
    """Print system information"""
    import sys

    # Collect system info
    python_version = sys.version.split()[0]
    pytorch_version = torch.__version__

    # GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        cuda_version = getattr(torch.version, 'cuda', None) if hasattr(torch, 'version') else None  # type: ignore
        gpu_info = f"{gpu_name}"
        gpu_memory_info = f"{gpu_memory:.1f} GB"
        cuda_info = cuda_version or "Unknown"
    else:
        gpu_info = "Not available"
        gpu_memory_info = "N/A"
        cuda_info = "N/A"

    cpu_cores = str(os.cpu_count())

    # Log to file
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("SYSTEM INFORMATION")
    logger.info("=" * 60)
    logger.info(f"Python Version: {python_version}")
    logger.info(f"PyTorch Version: {pytorch_version}")
    logger.info(f"GPU: {gpu_info}")
    if torch.cuda.is_available():
        logger.info(f"GPU Memory: {gpu_memory_info}")
        logger.info(f"CUDA Version: {cuda_info}")
    logger.info(f"CPU Cores: {cpu_cores}")
    logger.info("=" * 60)

    # Print to console (Rich table)
    table = Table(title="System Information")
    table.add_column("Component", style="cyan")
    table.add_column("Information", style="magenta")

    table.add_row("Python Version", python_version)
    table.add_row("PyTorch Version", pytorch_version)

    if torch.cuda.is_available():
        table.add_row("GPU", gpu_info)
        table.add_row("GPU Memory", gpu_memory_info)
        table.add_row("CUDA Version", cuda_info)
    else:
        table.add_row("GPU", gpu_info)

    table.add_row("CPU Cores", cpu_cores)

    console.print(table)


def format_time(seconds: float) -> str:
    """Format time in human readable format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:05.2f}"
    else:
        return f"{minutes:02d}:{seconds:05.2f}"


def calculate_model_size(model: torch.nn.Module) -> Dict[str, Any]:
    """Calculate model size and parameter count"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate model size in MB
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    model_size_mb = (param_size + buffer_size) / 1024**2
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": total_params - trainable_params,
        "trainable_ratio": trainable_params / total_params if total_params > 0 else 0,
        "model_size_mb": model_size_mb
    }


def print_model_summary(model: torch.nn.Module, model_name: str = "Model"):
    """Print detailed model summary"""
    stats = calculate_model_size(model)
    
    table = Table(title=f"{model_name} Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    
    table.add_row("Total Parameters", f"{stats['total_parameters']:,}")
    table.add_row("Trainable Parameters", f"{stats['trainable_parameters']:,}")
    table.add_row("Non-trainable Parameters", f"{stats['non_trainable_parameters']:,}")
    table.add_row("Trainable Ratio", f"{stats['trainable_ratio']:.2%}")
    table.add_row("Model Size", f"{stats['model_size_mb']:.2f} MB")
    
    console.print(table)


class TrainingTimer:
    """Timer for tracking training progress"""

    def __init__(self):
        self.start_time: Optional[float] = None
        self.epoch_start: Optional[float] = None
        self.epoch_times: List[float] = []
        self.step_times: List[float] = []

    def start_training(self) -> None:
        """Start training timer"""
        self.start_time = time.time()
        console.print("[green]Training started[/green]")

    def start_epoch(self) -> None:
        """Start epoch timer"""
        self.epoch_start = time.time()

    def end_epoch(self) -> float:
        """End epoch timer"""
        if self.epoch_start is not None:
            epoch_time = time.time() - self.epoch_start
            self.epoch_times.append(epoch_time)
            return epoch_time
        return 0.0

    def get_training_time(self) -> float:
        """Get total training time"""
        if self.start_time is not None:
            return time.time() - self.start_time
        return 0.0
    
    def get_average_epoch_time(self) -> float:
        """Get average epoch time"""
        if not self.epoch_times:
            return 0.0
        return float(np.mean(self.epoch_times))
    
    def estimate_remaining_time(self, current_epoch: int, total_epochs: int) -> float:
        """Estimate remaining training time"""
        if not self.epoch_times:
            return 0
        
        avg_epoch_time = self.get_average_epoch_time()
        remaining_epochs = total_epochs - current_epoch
        return avg_epoch_time * remaining_epochs


class ExperimentTracker:
    """
    Lightweight experiment tracker for managing experiment directories and configs.
    For metrics logging and visualization, use TensorBoard or WandB instead.
    """

    def __init__(self, experiment_dir: str):
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.config_file = self.experiment_dir / "config.json"

    def save_config(self, config: ProjectConfig) -> None:
        """Save experiment configuration to JSON"""
        from omegaconf import OmegaConf
        from dataclasses import asdict

        try:
            # Try OmegaConf first for better handling of complex configs
            config_dict = OmegaConf.to_container(OmegaConf.structured(config), resolve=True)
        except Exception:
            # Fallback to dataclasses.asdict for nested dataclasses
            config_dict = asdict(config)  # type: ignore

        with open(self.config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)

        console.print(f"[green]Config saved to {self.config_file}[/green]")


def save_predictions_sample(
    predictions: List[str], 
    references: List[str], 
    audio_paths: List[str],
    save_path: str,
    n_samples: int = 10
):
    """Save sample predictions for manual inspection"""
    samples = []
    for i in range(min(n_samples, len(predictions))):
        samples.append({
            "audio_path": audio_paths[i] if i < len(audio_paths) else "N/A",
            "reference": references[i] if i < len(references) else "N/A",
            "prediction": predictions[i] if i < len(predictions) else "N/A"
        })
    
    df = pd.DataFrame(samples)
    df.to_csv(save_path, index=False, encoding='utf-8')
    console.print(f"[green]Saved {len(samples)} prediction samples to {save_path}[/green]")


class ProgressManager:
    """Manage progress display for long-running operations"""
    
    def __init__(self):
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        )
    
    def __enter__(self):
        self.progress.start()
        return self.progress
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.progress.stop()


def cleanup_checkpoints(checkpoint_dir: str, keep_last: int = 3):
    """Clean up old checkpoints, keeping only the latest ones"""
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        return
    
    # Find all checkpoint files
    checkpoints = list(checkpoint_path.glob("checkpoint-*"))
    if len(checkpoints) <= keep_last:
        return
    
    # Sort by modification time and remove old ones
    checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    for checkpoint in checkpoints[keep_last:]:
        if checkpoint.is_dir():
            import shutil
            shutil.rmtree(checkpoint)
        else:
            checkpoint.unlink()
        console.print(f"[yellow]Removed old checkpoint: {checkpoint.name}[/yellow]")


def get_device_info() -> Dict[str, Any]:
    """Get detailed device information"""
    info = {
        "device_type": "cpu",
        "device_name": "CPU",
        "memory_total": 0,
        "memory_available": 0
    }
    
    if torch.cuda.is_available():
        info.update({
            "device_type": "cuda",
            "device_name": torch.cuda.get_device_name(0),
            "memory_total": torch.cuda.get_device_properties(0).total_memory / 1024**3,
            "memory_available": (torch.cuda.get_device_properties(0).total_memory -
                               torch.cuda.memory_allocated()) / 1024**3,
            "cuda_version": torch.version.cuda if hasattr(torch.version, 'cuda') else "Unknown",  # type: ignore
            "cudnn_version": torch.backends.cudnn.version()
        })

    return info


def plot_training_metrics(metrics_file: str, output_dir: Optional[str] = None) -> None:
    """
    Plot training metrics from metrics_on_all_epochs.json file.

    Creates several plots:
    1. Combined WER, CER, MER, WIL plot
    2. BLEU score plot
    3. Error breakdown (substitutions, deletions, insertions, hits) as subplots
    4. Hits/total_predictions ratio plot
    5. Combined WER, CER, and train loss plot

    Args:
        metrics_file: Path to metrics_on_all_epochs.json file
        output_dir: Directory to save plots (default: same directory as metrics file + '/graphs')
    """
    logger = logging.getLogger(__name__)

    # Load metrics
    metrics_path = Path(metrics_file)
    if not metrics_path.exists():
        logger.error(f"Metrics file not found: {metrics_file}")
        return

    with open(metrics_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    epochs_data = data.get('epochs', [])
    if not epochs_data:
        logger.error("No epochs data found in metrics file")
        return

    # Create output directory
    if output_dir is None:
        output_dir = str(metrics_path.parent / "graphs")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving plots to: {output_path}")

    # Extract data
    epochs = [e['epoch'] for e in epochs_data]
    wer = [e['wer'] for e in epochs_data]
    cer = [e['cer'] for e in epochs_data]
    mer = [e['mer'] for e in epochs_data]
    wil = [e['wil'] for e in epochs_data]
    bleu = [e['bleu'] for e in epochs_data]
    train_loss = [e['train_loss'] for e in epochs_data]
    substitutions = [e['substitutions'] for e in epochs_data]
    deletions = [e['deletions'] for e in epochs_data]
    insertions = [e['insertions'] for e in epochs_data]
    hits = [e['hits'] for e in epochs_data]
    total_predictions = [e['total_predictions'] for e in epochs_data]

    # Calculate hits ratio
    hits_ratio = [h / t for h, t in zip(hits, total_predictions)]

    # Find best epoch (min WER)
    best_epoch_idx = wer.index(min(wer))
    best_epoch = epochs[best_epoch_idx]

    # ========== Plot 1: Combined WER, CER, MER, WIL ==========
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(epochs, wer, label='WER', linewidth=2.5, marker='o', markersize=4)
    ax.plot(epochs, cer, label='CER', linewidth=2.5, marker='s', markersize=4)
    ax.plot(epochs, mer, label='MER', linewidth=1.5, marker='^', markersize=4, alpha=0.8)
    ax.plot(epochs, wil, label='WIL', linewidth=1.5, marker='v', markersize=4, alpha=0.8)

    # Mark best epoch
    ax.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.5, label=f'Best Epoch ({best_epoch})')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Error Rate', fontsize=12)
    ax.set_title('Word and Character Error Rates', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / '1_error_rates.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 1_error_rates.png")

    # ========== Plot 2: BLEU Score ==========
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(epochs, bleu, label='BLEU', linewidth=2, marker='o', markersize=4, color='green')
    ax.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.5, label=f'Best Epoch ({best_epoch})')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('BLEU Score', fontsize=12)
    ax.set_title('BLEU Score over Epochs', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / '2_bleu_score.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 2_bleu_score.png")

    # ========== Plot 3: Error Breakdown (4 subplots) ==========
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Substitutions
    axes[0, 0].plot(epochs, substitutions, linewidth=2, marker='o', markersize=4, color='red')  # type: ignore[index]
    axes[0, 0].axvline(x=best_epoch, color='red', linestyle='--', alpha=0.3)  # type: ignore[index]
    axes[0, 0].set_xlabel('Epoch', fontsize=11)  # type: ignore[index]
    axes[0, 0].set_ylabel('Count', fontsize=11)  # type: ignore[index]
    axes[0, 0].set_title('Substitutions', fontsize=12, fontweight='bold')  # type: ignore[index]
    axes[0, 0].grid(True, alpha=0.3)  # type: ignore[index]

    # Deletions
    axes[0, 1].plot(epochs, deletions, linewidth=2, marker='o', markersize=4, color='orange')  # type: ignore[index]
    axes[0, 1].axvline(x=best_epoch, color='red', linestyle='--', alpha=0.3)  # type: ignore[index]
    axes[0, 1].set_xlabel('Epoch', fontsize=11)  # type: ignore[index]
    axes[0, 1].set_ylabel('Count', fontsize=11)  # type: ignore[index]
    axes[0, 1].set_title('Deletions', fontsize=12, fontweight='bold')  # type: ignore[index]
    axes[0, 1].grid(True, alpha=0.3)  # type: ignore[index]

    # Insertions
    axes[1, 0].plot(epochs, insertions, linewidth=2, marker='o', markersize=4, color='purple')  # type: ignore[index]
    axes[1, 0].axvline(x=best_epoch, color='red', linestyle='--', alpha=0.3)  # type: ignore[index]
    axes[1, 0].set_xlabel('Epoch', fontsize=11)  # type: ignore[index]
    axes[1, 0].set_ylabel('Count', fontsize=11)  # type: ignore[index]
    axes[1, 0].set_title('Insertions', fontsize=12, fontweight='bold')  # type: ignore[index]
    axes[1, 0].grid(True, alpha=0.3)  # type: ignore[index]

    # Hits
    axes[1, 1].plot(epochs, hits, linewidth=2, marker='o', markersize=4, color='green')  # type: ignore[index]
    axes[1, 1].axvline(x=best_epoch, color='red', linestyle='--', alpha=0.3)  # type: ignore[index]
    axes[1, 1].set_xlabel('Epoch', fontsize=11)  # type: ignore[index]
    axes[1, 1].set_ylabel('Count', fontsize=11)  # type: ignore[index]
    axes[1, 1].set_title('Hits (Correct Words)', fontsize=12, fontweight='bold')  # type: ignore[index]
    axes[1, 1].grid(True, alpha=0.3)  # type: ignore[index]

    plt.suptitle('Error Breakdown over Epochs', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path / '3_error_breakdown.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 3_error_breakdown.png")

    # ========== Plot 4: Hits / Total Predictions Ratio ==========
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(epochs, hits_ratio, linewidth=2, marker='o', markersize=4, color='blue')
    ax.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.5, label=f'Best Epoch ({best_epoch})')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Hits / Total Predictions', fontsize=12)
    ax.set_title('Word Accuracy (Hits per Sample)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / '4_hits_ratio.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 4_hits_ratio.png")

    # ========== Plot 5: Combined WER, CER, Train Loss ==========
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot error rates on primary y-axis
    ax1.plot(epochs, wer, label='WER', linewidth=2.5, marker='o', markersize=4, color='C0')
    ax1.plot(epochs, cer, label='CER', linewidth=2.5, marker='s', markersize=4, color='C1')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Error Rate', fontsize=12, color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, alpha=0.3)

    # Create secondary y-axis for train loss
    ax2 = ax1.twinx()
    ax2.plot(epochs, train_loss, label='Train Loss', linewidth=2, marker='^', markersize=4, color='C2')  # type: ignore[attr-defined]
    ax2.set_ylabel('Train Loss', fontsize=12, color='C2')  # type: ignore[attr-defined]
    ax2.tick_params(axis='y', labelcolor='C2')  # type: ignore[attr-defined]

    # Mark best epoch
    ax1.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.5, label=f'Best Epoch ({best_epoch})')

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()  # type: ignore[attr-defined]
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=10)

    ax1.set_title('Error Rates and Training Loss', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path / '5_combined_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 5_combined_metrics.png")

    logger.info(f"All plots saved successfully to: {output_path}")
    logger.info(f"Best epoch: {best_epoch} (WER: {min(wer):.4f})")
