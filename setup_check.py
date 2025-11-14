"""
Environment setup and health check script for speech-to-text project.
Verifies installation, dependencies, and system compatibility.
"""

import sys
import subprocess
import importlib
import platform
import os
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    
if RICH_AVAILABLE:
    console = Console()
else:
    class MockConsole:
        def print(self, text, **kwargs):
            print(text)
    console = MockConsole()


def check_python_version() -> Tuple[bool, str]:
    """Check Python version compatibility"""
    version = sys.version_info
    required = (3, 8)
    
    if version >= required:
        return True, f"✅ Python {version.major}.{version.minor}.{version.micro}"
    else:
        return False, f"❌ Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)"


def check_package(package_name: str, import_name: Optional[str] = None) -> Tuple[bool, str]:
    """Check if a package is installed and importable"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        return True, f"✅ {package_name} {version}"
    except ImportError:
        return False, f"❌ {package_name} not found"


def check_cuda_pytorch() -> Tuple[bool, str]:
    """Check PyTorch CUDA availability"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            try:
                cuda_version = torch.version.cuda  # type: ignore
            except AttributeError:
                cuda_version = "unknown"
            return True, f"✅ CUDA {cuda_version}, {gpu_name} ({gpu_memory:.1f}GB)"
        else:
            return False, "⚠️  CUDA not available (CPU-only mode)"
    except ImportError:
        return False, "❌ PyTorch not installed"


def check_disk_space() -> Tuple[bool, str]:
    """Check available disk space"""
    try:
        import shutil
        free_bytes = shutil.disk_usage('.').free
        free_gb = free_bytes / 1024**3
        
        if free_gb >= 50:
            return True, f"✅ {free_gb:.1f}GB available"
        elif free_gb >= 20:
            return False, f"⚠️  {free_gb:.1f}GB available (50GB+ recommended)"
        else:
            return False, f"❌ {free_gb:.1f}GB available (insufficient space)"
    except:
        return False, "❌ Could not check disk space"


def check_memory() -> Tuple[bool, str]:
    """Check system memory"""
    try:
        import psutil
        memory = psutil.virtual_memory()
        total_gb = memory.total / 1024**3
        available_gb = memory.available / 1024**3
        
        if total_gb >= 16:
            return True, f"✅ {total_gb:.1f}GB total, {available_gb:.1f}GB available"
        elif total_gb >= 8:
            return False, f"⚠️  {total_gb:.1f}GB total (16GB+ recommended)"
        else:
            return False, f"❌ {total_gb:.1f}GB total (insufficient memory)"
    except ImportError:
        return False, "⚠️  psutil not available (can't check memory)"


def check_project_structure() -> Tuple[bool, str]:
    """Check project directory structure"""
    required_dirs = [
        "src", "configs", "notebooks", "data", "experiments", "tests"
    ]
    required_files = [
        "requirements.txt", "train.py", "README.md",
        "src/__init__.py", "src/config.py", "src/data.py", 
        "src/models.py", "src/metrics.py", "src/utils.py",
        "configs/default.yaml"
    ]
    
    missing = []
    
    for dir_name in required_dirs:
        if not Path(dir_name).exists():
            missing.append(f"Directory: {dir_name}")
    
    for file_name in required_files:
        if not Path(file_name).exists():
            missing.append(f"File: {file_name}")
    
    if not missing:
        return True, "✅ All required files and directories present"
    else:
        return False, f"❌ Missing: {', '.join(missing[:3])}{'...' if len(missing) > 3 else ''}"


def check_config_files() -> Tuple[bool, str]:
    """Check configuration files"""
    try:
        sys.path.append('src')
        from src.config import load_config
        
        config = load_config("configs/default.yaml")
        return True, "✅ Configuration loaded successfully"
    except Exception as e:
        return False, f"❌ Config error: {str(e)[:50]}..."


def check_huggingface_cache() -> Tuple[bool, str]:
    """Check HuggingFace cache and login"""
    try:
        from huggingface_hub import HfFolder
        token = HfFolder.get_token()
        
        if token:
            return True, "✅ HuggingFace token found"
        else:
            return False, "⚠️  No HuggingFace token (may limit dataset access)"
    except ImportError:
        return False, "⚠️  huggingface_hub not available"


def check_local_dataset() -> Tuple[bool, str]:
    """Check local dataset availability"""
    try:
        sys.path.append('src')
        from src.config import load_config
        from src.data import DataManager
        
        config = load_config("configs/default.yaml")
        data_manager = DataManager(config)
        info = data_manager.get_dataset_info()
        
        if info["status"] == "available":
            return True, f"✅ Local dataset available ({info['total_samples']:,} samples)"
        elif info["status"] == "not_downloaded":
            return False, "⚠️  Local dataset not downloaded (run: python download_data.py)"
        else:
            return False, f"❌ Dataset error: {info.get('error', 'Unknown')[:30]}..."
    except Exception as e:
        return False, f"❌ Check failed: {str(e)[:50]}..."

def run_basic_imports_test() -> Tuple[bool, str]:
    """Test basic imports from project modules"""
    try:
        sys.path.append('src')
        from src.config import ProjectConfig, load_config
        from src.data import DataManager
        from src.models import ModelFactory
        from src.metrics import STTMetrics
        from src.utils import setup_logging
        
        return True, "✅ All project modules importable"
    except Exception as e:
        return False, f"❌ Import error: {str(e)[:50]}..."


def install_missing_packages() -> bool:
    """Install missing packages from requirements.txt"""
    console.print("\n[yellow]Attempting to install missing packages...[/yellow]")
    
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True, capture_output=True)
        console.print("[green]✅ Packages installed successfully[/green]")
        return True
    except subprocess.CalledProcessError as e:
        console.print(f"[red]❌ Installation failed: {e}[/red]")
        return False


def main():
    """Main health check function"""
    if RICH_AVAILABLE:
        console.print(Panel.fit(
            "[bold blue]Speech-to-Text Project Environment Check[/bold blue]",
            border_style="blue"
        ))
    else:
        print("=== Speech-to-Text Project Environment Check ===")
    
    checks = [
        ("Python Version", check_python_version),
        ("Project Structure", check_project_structure),
        ("Configuration Files", check_config_files),
        ("System Memory", check_memory),
        ("Disk Space", check_disk_space),
        ("PyTorch", lambda: check_package("torch")),
        ("PyTorch CUDA", check_cuda_pytorch),
        ("Transformers", lambda: check_package("transformers")),
        ("Datasets", lambda: check_package("datasets")),
        ("OmegaConf", lambda: check_package("omegaconf")),
        ("TorchaAudio", lambda: check_package("torchaudio")),
        ("JIWER", lambda: check_package("jiwer")),
        ("Rich", lambda: check_package("rich")),
        ("HuggingFace Hub", check_huggingface_cache),
        ("Local Dataset", check_local_dataset),
        ("Project Imports", run_basic_imports_test)
    ]
    
    results = []
    failed_checks = []
    
    for check_name, check_func in checks:
        try:
            success, message = check_func()
            results.append((check_name, success, message))
            if not success:
                failed_checks.append(check_name)
        except Exception as e:
            results.append((check_name, False, f"❌ Error: {str(e)[:30]}..."))
            failed_checks.append(check_name)
    
    # Display results
    if RICH_AVAILABLE:
        table = Table(title="Environment Check Results")
        table.add_column("Check", style="cyan")
        table.add_column("Status", style="magenta")
        
        for check_name, success, message in results:
            table.add_row(check_name, message)
        
        console.print(table)
    else:
        print("\nCheck Results:")
        print("-" * 50)
        for check_name, success, message in results:
            print(f"{check_name:20} {message}")
    
    # Summary
    total_checks = len(results)
    passed_checks = sum(1 for _, success, _ in results if success)
    
    if RICH_AVAILABLE:
        if passed_checks == total_checks:
            console.print(Panel.fit(
                f"[bold green]🎉 All {total_checks} checks passed! Environment ready.[/bold green]",
                border_style="green"
            ))
        else:
            console.print(Panel.fit(
                f"[bold yellow]⚠️  {passed_checks}/{total_checks} checks passed.[/bold yellow]\n"
                f"Failed: {', '.join(failed_checks)}",
                border_style="yellow"
            ))
    else:
        print(f"\nSummary: {passed_checks}/{total_checks} checks passed")
        if failed_checks:
            print(f"Failed: {', '.join(failed_checks)}")
    
    # Suggestions
    if failed_checks:
        console.print("\n[blue]Suggestions:[/blue]")
        
        if any("torch" in check.lower() for check in failed_checks):
            console.print("• Install PyTorch: pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118")
        
        if any(pkg in failed_checks for pkg in ["Transformers", "Datasets", "OmegaConf", "Librosa", "JIWER", "Rich"]):
            console.print("• Install missing packages: pip install -r requirements.txt")
            
            response = input("\nAttempt automatic installation? (y/N): ")
            if response.lower() == 'y':
                if install_missing_packages():
                    console.print("\n[green]Re-run this script to verify installation[/green]")
        
        if "Local Dataset" in failed_checks:
            console.print("• Download dataset locally: python download_data.py")
            console.print("• This enables offline usage and faster data loading")
        
        if "CUDA" in failed_checks:
            console.print("• Check NVIDIA drivers and CUDA installation")
        
        if "memory" in failed_checks:
            console.print("• Consider using smaller models or adjusting batch sizes")
        
        if "disk space" in failed_checks:
            console.print("• Free up disk space or use external storage for datasets")
    
    # Quick start instructions
    if passed_checks >= total_checks * 0.8:  # 80% success rate
        console.print(f"\n[blue]Quick Start Commands:[/blue]")
        if "Local Dataset" in failed_checks:
            console.print("• Download data: python download_data.py")
        console.print("• Explore data: jupyter notebook notebooks/01_eda.ipynb")
        console.print("• Train model: python train.py --debug")
        console.print("• Run experiments: python run_experiments.py --debug")
    
    return passed_checks == total_checks


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
