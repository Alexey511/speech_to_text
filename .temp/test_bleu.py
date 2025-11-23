"""Quick test for BLEU metric"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics import STTMetrics
import logging

logging.basicConfig(level=logging.INFO)

# Create metrics computer with BLEU enabled
print("Creating STTMetrics with use_bleu=True...")
metrics = STTMetrics(language='ru', use_bleu=True)
print(f'use_bleu: {metrics.use_bleu}')
print(f'bleu_metric loaded: {metrics.bleu_metric is not None}')

# Test BLEU with sample data
predictions = ['привет как дела', 'это тест']
references = ['привет как дела', 'это тестовая строка']

print("\nComputing BLEU...")
bleu = metrics.compute_bleu(predictions, references)
print(f'BLEU score: {bleu}')
