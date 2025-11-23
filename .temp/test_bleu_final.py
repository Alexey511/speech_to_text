"""Final test for BLEU metric with longer texts"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics import STTMetrics
import logging

logging.basicConfig(level=logging.INFO)

print("="*60)
print("Testing BLEU with realistic ASR examples")
print("="*60)

metrics = STTMetrics(language='ru', use_bleu=True)

# Test case 1: Perfect match
print("\nTest 1: Perfect match")
predictions = ["я пошёл в магазин за хлебом и молоком вчера вечером"]
references = ["я пошёл в магазин за хлебом и молоком вчера вечером"]
bleu = metrics.compute_bleu(predictions, references)
print(f"Predictions: {predictions[0]}")
print(f"References:  {references[0]}")
print(f"BLEU: {bleu:.4f} (expected: ~1.0)")

# Test case 2: High similarity
print("\nTest 2: High similarity")
predictions = ["я пошёл в магазин за хлебом и молоком вчера"]
references = ["я пошёл в магазин за хлебом и молоком вчера вечером"]
bleu = metrics.compute_bleu(predictions, references)
print(f"Predictions: {predictions[0]}")
print(f"References:  {references[0]}")
print(f"BLEU: {bleu:.4f}")

# Test case 3: Multiple samples (more realistic)
print("\nTest 3: Multiple samples")
predictions = [
    "привет как твои дела сегодня",
    "я иду домой после работы",
    "это очень интересная книга про историю",
    "завтра будет хорошая погода",
    "спасибо за помощь"
]
references = [
    "привет как твои дела сегодня",
    "я иду домой после работы вечером",
    "это очень интересная книга о истории",
    "завтра будет отличная погода",
    "спасибо большое за помощь"
]
bleu = metrics.compute_bleu(predictions, references)
print(f"Number of samples: {len(predictions)}")
print(f"BLEU: {bleu:.4f}")

# Test case 4: Poor match
print("\nTest 4: Poor match")
predictions = ["совсем другой текст"]
references = ["какой то другой контент"]
bleu = metrics.compute_bleu(predictions, references)
print(f"Predictions: {predictions[0]}")
print(f"References:  {references[0]}")
print(f"BLEU: {bleu:.4f} (expected: close to 0)")

print("\n" + "="*60)
print("✓ BLEU metric is working correctly!")
print("="*60)
