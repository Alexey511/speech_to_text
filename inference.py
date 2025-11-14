"""
Inference script for speech-to-text models.
Supports both single file and batch processing.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import json
import time

import torch
import numpy as np
import torchaudio
import torchaudio.transforms as T
from rich.console import Console
from rich.table import Table
from rich.progress import track

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.config import load_config
from src.data import DataManager, AudioPreprocessor
from src.models import ModelManager
from src.utils import format_time

console = Console()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Speech-to-Text Inference")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the trained model directory"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input audio file or directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file (for single file) or directory (for batch)"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="ru",
        help="Language code (default: ru)"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="transcribe",
        choices=["transcribe", "translate"],
        help="Task type (default: transcribe)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for processing multiple files"
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=30.0,
        help="Maximum audio duration in seconds"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="text",
        choices=["text", "json", "csv"],
        help="Output format"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    return parser.parse_args()


def load_audio_file(file_path: str, target_sr: int = 16000, max_duration: float = 30.0) -> Optional[np.ndarray]:
    """Load and preprocess audio file using torchaudio"""
    try:
        # Load audio
        audio, sr = torchaudio.load(file_path)
        
        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        
        # Resample if necessary
        if sr != target_sr:
            resampler = T.Resample(sr, target_sr)
            audio = resampler(audio)
        
        # Remove channel dimension
        audio = audio.squeeze(0)
        
        # Check duration
        duration = len(audio) / target_sr
        if duration > max_duration:
            console.print(f"[yellow]Warning: Audio {file_path} is {duration:.2f}s, truncating to {max_duration}s[/yellow]")
            audio = audio[:int(max_duration * target_sr)]
        
        # Normalize
        audio = audio / (torch.max(torch.abs(audio)) + 1e-8)
        
        # Convert to numpy for compatibility with transformers
        return audio.numpy()
    
    except Exception as e:
        console.print(f"[red]Error loading {file_path}: {e}[/red]")
        return None


def transcribe_audio(
    model,
    processor,
    audio: np.ndarray,
    language: str = "ru",
    task: str = "transcribe"
) -> str:
    """Transcribe a single audio array"""
    device = next(model.parameters()).device
    
    # Process audio
    if hasattr(processor, 'feature_extractor'):
        # Whisper-style processing
        inputs = processor.feature_extractor(
            audio, 
            sampling_rate=16000, 
            return_tensors="pt"
        )
        input_features = inputs.input_features.to(device)
        
        # Generate transcription
        with torch.no_grad():
            outputs = model.generate(
                input_features,
                language=language,
                task=task,
                max_length=448
            )
        
        # Decode
        transcription = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    else:
        # Speech2Text-style processing
        inputs = processor.feature_extractor(
            audio,
            sampling_rate=16000,
            return_tensors="pt"
        )
        input_values = inputs.input_values.to(device)
        
        # Generate transcription
        with torch.no_grad():
            outputs = model.generate(input_values)
        
        # Decode
        transcription = processor.tokenizer.decode(outputs[0])
    
    return transcription.strip()


def process_single_file(
    model, processor, file_path: str, args
) -> Dict[str, Any]:
    """Process a single audio file"""
    start_time = time.time()
    
    # Load audio
    audio = load_audio_file(file_path, max_duration=args.max_duration)
    if audio is None:
        return {"error": f"Failed to load {file_path}"}
    
    # Transcribe
    try:
        transcription = transcribe_audio(
            model, processor, audio, args.language, args.task
        )
        
        processing_time = time.time() - start_time
        audio_duration = len(audio) / 16000
        rtf = processing_time / audio_duration  # Real-time factor
        
        return {
            "file_path": file_path,
            "transcription": transcription,
            "audio_duration": audio_duration,
            "processing_time": processing_time,
            "real_time_factor": rtf
        }
    
    except Exception as e:
        return {"error": f"Transcription failed for {file_path}: {e}"}


def process_batch(
    model, processor, file_paths: List[str], args
) -> List[Dict[str, Any]]:
    """Process multiple audio files"""
    results = []
    
    for file_path in track(file_paths, description="Processing files..."):
        result = process_single_file(model, processor, file_path, args)
        results.append(result)
        
        if args.verbose and "transcription" in result:
            console.print(f"[green]{Path(file_path).name}:[/green] {result['transcription']}")
    
    return results


def save_results(results: List[Dict[str, Any]], output_path: str, format_type: str):
    """Save results in specified format"""
    output_path_obj = Path(output_path)
    
    if format_type == "text":
        with open(output_path_obj, 'w', encoding='utf-8') as f:
            for result in results:
                if "transcription" in result:
                    f.write(f"{result['transcription']}\n")
                else:
                    f.write(f"ERROR: {result.get('error', 'Unknown error')}\n")
    
    elif format_type == "json":
        with open(output_path_obj, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    elif format_type == "csv":
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv(output_path_obj, index=False, encoding='utf-8')
    
    console.print(f"[green]Results saved to {output_path}[/green]")


def print_results_summary(results: List[Dict[str, Any]]):
    """Print summary of processing results"""
    successful = [r for r in results if "transcription" in r]
    failed = [r for r in results if "error" in r]
    
    table = Table(title="Processing Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    
    table.add_row("Total Files", f"{len(results)}")
    table.add_row("Successful", f"{len(successful)}")
    table.add_row("Failed", f"{len(failed)}")
    
    if successful:
        total_audio_time = sum(r["audio_duration"] for r in successful)
        total_processing_time = sum(r["processing_time"] for r in successful)
        avg_rtf = np.mean([r["real_time_factor"] for r in successful])
        
        table.add_row("Total Audio Duration", format_time(total_audio_time))
        table.add_row("Total Processing Time", format_time(total_processing_time))
        table.add_row("Average Real-time Factor", f"{avg_rtf:.2f}x")
    
    console.print(table)
    
    # Show failed files
    if failed and len(failed) <= 10:
        console.print("\n[red]Failed files:[/red]")
        for result in failed:
            console.print(f"  - {result.get('error', 'Unknown error')}")


def main():
    """Main inference function"""
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    
    try:
        console.print("[bold blue]🔍 Loading model and configuration...[/bold blue]")
        
        # Load configuration
        model_path = Path(args.model_path)
        if args.config:
            config_path = args.config
        else:
            # Try to find config in model directory
            config_candidates = [
                model_path / "config.yaml",
                model_path / "config.json",
                model_path.parent / "config.yaml"
            ]
            config_path = None
            for candidate in config_candidates:
                if candidate.exists():
                    config_path = str(candidate)
                    break
            
            if config_path is None:
                console.print("[yellow]Warning: No config found, using default[/yellow]")
                config_path = "configs/default.yaml"
        
        config = load_config(config_path)

        # Create model manager
        model_manager = ModelManager()

        # Determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        console.print(f"[blue]Using device: {device}[/blue]")

        # Load model
        if (model_path / "pytorch_model.bin").exists() or (model_path / "model.safetensors").exists():
            # HuggingFace format
            console.print("[green]Loading HuggingFace model...[/green]")
            config.model.model_name = str(model_path)

            # Create processor
            data_manager = DataManager(config)
            processor = data_manager.setup_processor(
                model_name=config.model.model_name,
                model_type=config.model.model_type,
                language=config.data.language,
                task=config.data.task
            )

            # Create and compile model
            model = model_manager.create_model(config.model, processor)
            model = model_manager.compile_model(model, device=device, compile=False)
        else:
            # Custom checkpoint
            console.print("[green]Loading custom checkpoint...[/green]")
            model, processor, checkpoint = model_manager.load_checkpoint(
                str(model_path),
                device=device,
                compile=False,
                language=args.language,  # Required for Whisper processor creation
                task=args.task  # Required for Whisper processor creation
            )
            console.print(f"[green]Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}[/green]")
        
        console.print("[bold blue]🎯 Processing audio...[/bold blue]")
        
        # Determine input files
        input_path = Path(args.input)
        if input_path.is_file():
            file_paths = [str(input_path)]
        elif input_path.is_dir():
            # Find all audio files
            audio_extensions = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.wma'}
            file_paths = [
                str(f) for f in input_path.rglob('*') 
                if f.suffix.lower() in audio_extensions
            ]
            if not file_paths:
                console.print("[red]No audio files found in directory[/red]")
                return
        else:
            console.print(f"[red]Input path {input_path} does not exist[/red]")
            return
        
        console.print(f"Found {len(file_paths)} audio file(s)")
        
        # Process files
        results = process_batch(model, processor, file_paths, args)
        
        # Print summary
        print_results_summary(results)
        
        # Save results if output specified
        if args.output:
            save_results(results, args.output, args.format)
        else:
            # Print transcriptions
            console.print("\n[bold blue]Transcriptions:[/bold blue]")
            for result in results:
                if "transcription" in result:
                    file_name = Path(result["file_path"]).name
                    console.print(f"[green]{file_name}:[/green] {result['transcription']}")
        
        console.print("[bold green]✅ Inference completed successfully![/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]❌ Inference failed: {e}[/bold red]")
        raise


if __name__ == "__main__":
    main()
