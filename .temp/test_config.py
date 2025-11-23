import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import load_config

try:
    config = load_config('configs/whisper_small.yaml')
    print('Config loaded successfully!')
    print(f'Model: {config.model.model_name}')
    print(f'Max grad norm: {config.training.max_grad_norm}')
except Exception as e:
    print(f'Error: {type(e).__name__}: {e}')
    import traceback
    traceback.print_exc()
