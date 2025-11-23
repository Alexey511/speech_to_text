"""Debug test for BLEU metric to understand data format"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics import STTMetrics
import evaluate
import logging

logging.basicConfig(level=logging.DEBUG)

# Test with raw evaluate library first
print("="*60)
print("Test 1: Direct evaluate.load('bleu') with keep_in_memory=True")
print("="*60)

bleu = evaluate.load("bleu", keep_in_memory=True)

# Test different formats
predictions = ['привет как дела', 'это тест']
references = [['привет как дела'], ['это тестовая строка']]

print(f"Predictions: {predictions}")
print(f"References: {references}")

result = bleu.compute(predictions=predictions, references=references)
print(f"Result: {result}")
print(f"BLEU score: {result['bleu']}")

print("\n" + "="*60)
print("Test 2: Using STTMetrics class")
print("="*60)

metrics = STTMetrics(language='ru', use_bleu=True)
predictions_test = ['привет как дела', 'это тест']
references_test = ['привет как дела', 'это тестовая строка']

bleu_score = metrics.compute_bleu(predictions_test, references_test)
print(f'BLEU score from STTMetrics: {bleu_score}')
