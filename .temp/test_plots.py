import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from src.utils import plot_training_metrics

metrics_file = Path('experiments/s2t-cross-lingual_train_1_decoder_3e-3/metrics_on_all_epochs.json')
print(f'Metrics file: {metrics_file}')
print(f'Exists: {metrics_file.exists()}')

if metrics_file.exists():
    plot_training_metrics(str(metrics_file))
    print('✅ Plots generated!')
    print(f"Check: {metrics_file.parent / 'graphs'}")
else:
    print('❌ File not found')
