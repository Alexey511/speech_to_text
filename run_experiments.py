"""
Script to run multiple experiments comparing different models and approaches.
Useful for systematic comparison of SOTA fine-tuning vs training from scratch.
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Any
import json

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run Speech-to-Text Experiments")
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        default=["whisper_small", "whisper_base", "wav2vec2", "custom"],
        choices=["whisper_tiny", "whisper_base", "whisper_small", "wav2vec2", "custom"],
        help="List of experiments to run"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode (faster, limited training)"
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments_comparison",
        help="Output directory for comparison results"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run experiments in parallel (experimental)"
    )
    
    return parser.parse_args()


def get_experiment_config(experiment_name: str, args) -> Dict[str, Any]:
    """Get configuration for specific experiment"""
    
    configs = {
        "whisper_tiny": {
            "config": "configs/debug.yaml",  # Use tiny model from debug config
            "name": "whisper_tiny_ru",
            "description": "Whisper Tiny (39M params) - fastest baseline"
        },
        "whisper_base": {
            "config": "configs/whisper_base.yaml",
            "name": "whisper_base_ru",
            "description": "Whisper Base (74M params) - good balance"
        },
        "whisper_small": {
            "config": "configs/default.yaml",
            "name": "whisper_small_ru",
            "description": "Whisper Small (244M params) - high quality"
        },
        "wav2vec2": {
            "config": "configs/wav2vec2.yaml",
            "name": "wav2vec2_xlsr_ru",
            "description": "Wav2Vec2 XLSR (300M params) - CTC approach"
        },
        "custom": {
            "config": "configs/custom_model.yaml",
            "name": "custom_model_ru",
            "description": "Custom Transformer (50M params) - from scratch"
        }
    }
    
    config = configs.get(experiment_name, {})
    
    # Apply debug mode modifications
    if args.debug:
        config["name"] += "_debug"
    
    return config


def run_single_experiment(experiment_name: str, config: Dict[str, Any], args) -> Dict[str, Any]:
    """Run a single experiment"""
    console.print(f"[bold blue]🚀 Starting experiment: {experiment_name}[/bold blue]")
    console.print(f"Description: {config.get('description', 'N/A')}")
    
    # Build command
    cmd = [
        sys.executable, "train.py",
        "--config", config["config"],
        "--experiment-name", config["name"]
    ]
    
    if args.debug:
        cmd.append("--debug")

    if args.no_wandb:
        cmd.append("--no-wandb")

    # Run experiment
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        return {
            "experiment": experiment_name,
            "name": config["name"],
            "description": config["description"],
            "status": "success",
            "duration": duration,
            "output": result.stdout[-1000:],  # Last 1000 chars
            "error": None
        }
    
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        duration = end_time - start_time
        
        console.print(f"[red]❌ Experiment {experiment_name} failed![/red]")
        console.print(f"Error: {e.stderr[-500:]}")  # Last 500 chars of error
        
        return {
            "experiment": experiment_name,
            "name": config["name"],
            "description": config["description"],
            "status": "failed",
            "duration": duration,
            "output": e.stdout[-1000:] if e.stdout else "",
            "error": e.stderr[-500:] if e.stderr else str(e)
        }


def collect_experiment_results(experiment_results: List[Dict[str, Any]], args) -> Dict[str, Any]:
    """Collect and compare results from all experiments"""
    console.print("[bold blue]📊 Collecting experiment results...[/bold blue]")
    
    comparison = {
        "experiments": [],
        "summary": {
            "total_experiments": len(experiment_results),
            "successful": 0,
            "failed": 0,
            "total_duration": 0
        }
    }
    
    for result in experiment_results:
        comparison["summary"]["total_duration"] += result["duration"]
        
        if result["status"] == "success":
            comparison["summary"]["successful"] += 1
            
            # Try to load metrics from experiment directory
            experiment_dir = Path("experiments") / result["name"]
            metrics_file = experiment_dir / "test_results.json"
            
            metrics = {}
            if metrics_file.exists():
                try:
                    with open(metrics_file, 'r') as f:
                        metrics = json.load(f)
                except:
                    pass
            
            comparison["experiments"].append({
                "name": result["name"],
                "experiment": result["experiment"],
                "description": result["description"],
                "duration": result["duration"],
                "metrics": metrics
            })
        else:
            comparison["summary"]["failed"] += 1
    
    return comparison


def save_comparison_results(comparison: Dict[str, Any], output_dir: str):
    """Save comparison results"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save full comparison
    comparison_file = output_path / "experiments_comparison.json"
    with open(comparison_file, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    
    console.print(f"[green]Full comparison saved to {comparison_file}[/green]")
    
    # Create summary table
    create_summary_table(comparison, output_path)


def create_summary_table(comparison: Dict[str, Any], output_path: Path):
    """Create and save summary table"""
    
    # Console table
    table = Table(title="Experiments Comparison Summary")
    table.add_column("Experiment", style="cyan")
    table.add_column("Model", style="blue")
    table.add_column("Duration", style="magenta")
    table.add_column("WER", style="red")
    table.add_column("CER", style="yellow")
    table.add_column("Parameters", style="green")
    
    for exp in comparison["experiments"]:
        metrics = exp.get("metrics", {})
        wer = metrics.get("eval_wer", "N/A")
        cer = metrics.get("eval_cer", "N/A")
        
        # Format metrics
        wer_str = f"{wer:.4f}" if isinstance(wer, (int, float)) else str(wer)
        cer_str = f"{cer:.4f}" if isinstance(cer, (int, float)) else str(cer)
        
        # Estimate parameters based on model type
        param_map = {
            "whisper_tiny": "39M",
            "whisper_base": "74M", 
            "whisper_small": "244M",
            "wav2vec2": "300M",
            "custom": "50M"
        }
        params = param_map.get(exp["experiment"], "Unknown")
        
        table.add_row(
            exp["experiment"],
            exp["name"],
            f"{exp['duration']:.0f}s",
            wer_str,
            cer_str,
            params
        )
    
    console.print(table)
    
    # Save CSV summary
    import pandas as pd
    
    summary_data = []
    for exp in comparison["experiments"]:
        metrics = exp.get("metrics", {})
        summary_data.append({
            "experiment": exp["experiment"],
            "name": exp["name"],
            "description": exp["description"],
            "duration_seconds": exp["duration"],
            "wer": metrics.get("eval_wer", None),
            "cer": metrics.get("eval_cer", None),
            "train_loss": metrics.get("train_loss", None),
            "eval_loss": metrics.get("eval_loss", None)
        })
    
    df = pd.DataFrame(summary_data)
    csv_file = output_path / "experiments_summary.csv"
    df.to_csv(csv_file, index=False)
    console.print(f"[green]Summary CSV saved to {csv_file}[/green]")


def print_experiment_plan(experiments: List[str], args):
    """Print experiment execution plan"""
    table = Table(title="Experiment Execution Plan")
    table.add_column("Order", style="cyan")
    table.add_column("Experiment", style="blue")
    table.add_column("Config", style="magenta")
    table.add_column("Description", style="green")
    
    for i, exp_name in enumerate(experiments, 1):
        config = get_experiment_config(exp_name, args)
        table.add_row(
            str(i),
            exp_name,
            config.get("config", "N/A"),
            config.get("description", "N/A")
        )
    
    console.print(table)
    
    # Estimate total time
    if args.debug:
        est_time_per_exp = 300  # 5 minutes in debug mode
    else:
        time_map = {
            "whisper_tiny": 3600,    # 1 hour
            "whisper_base": 7200,    # 2 hours
            "whisper_small": 14400,  # 4 hours
            "wav2vec2": 10800,       # 3 hours
            "custom": 18000          # 5 hours (training from scratch)
        }
        est_time_per_exp = sum(time_map.get(exp, 7200) for exp in experiments)
    
    console.print(f"\n[yellow]Estimated total time: {est_time_per_exp/3600:.1f} hours[/yellow]")
    console.print(f"[yellow]Debug mode: {args.debug}[/yellow]")


def main():
    """Main function to run experiments"""
    args = parse_args()
    
    console.print("[bold green]🧪 Speech-to-Text Experiments Runner[/bold green]")
    console.print("=" * 60)
    
    # Print execution plan
    print_experiment_plan(args.experiments, args)
    
    # Confirm execution
    if not args.debug:
        console.print("\n[yellow]⚠️  This will run full experiments and may take several hours![/yellow]")
        response = input("Do you want to continue? (y/N): ")
        if response.lower() != 'y':
            console.print("Experiments cancelled.")
            return
    
    # Run experiments
    experiment_results = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        for exp_name in args.experiments:
            config = get_experiment_config(exp_name, args)
            
            if args.parallel:
                # TODO: Implement parallel execution
                console.print("[yellow]Parallel execution not yet implemented, running sequentially[/yellow]")
            
            task = progress.add_task(f"Running {exp_name}...", total=None)
            result = run_single_experiment(exp_name, config, args)
            experiment_results.append(result)
            progress.remove_task(task)
            
            # Print immediate result
            if result["status"] == "success":
                console.print(f"[green]✅ {exp_name} completed in {result['duration']:.0f}s[/green]")
            else:
                console.print(f"[red]❌ {exp_name} failed after {result['duration']:.0f}s[/red]")
    
    # Collect and save results
    comparison = collect_experiment_results(experiment_results, args)
    save_comparison_results(comparison, args.output_dir)
    
    # Final summary
    console.print("\n[bold green]🎉 All experiments completed![/bold green]")
    console.print(f"Results saved to: {args.output_dir}")
    console.print(f"Successful experiments: {comparison['summary']['successful']}/{comparison['summary']['total_experiments']}")
    console.print(f"Total duration: {comparison['summary']['total_duration']/3600:.1f} hours")


if __name__ == "__main__":
    main()
