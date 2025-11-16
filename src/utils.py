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
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from .config import ProjectConfig

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
