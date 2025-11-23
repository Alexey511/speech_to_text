"""
Тест для понимания, как Whisper вычисляет loss
"""
import torch
import torch.nn.functional as F
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# Загрузим модель и процессор
print("Loading model...")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
processor = WhisperProcessor.from_pretrained("openai/whisper-base")

vocab_size = len(processor.tokenizer)
print(f"Vocab size: {vocab_size}")
print(f"Random baseline loss (theoretical): {torch.log(torch.tensor(vocab_size)).item():.4f}")

# Создадим dummy input
batch_size = 2
feature_dim = 80
seq_length = 3000
target_length = 20

# Random input features (mel-spectrogram)
input_features = torch.randn(batch_size, feature_dim, seq_length)

# Random target labels (исключая padding)
labels = torch.randint(0, vocab_size, (batch_size, target_length))
# Один токен сделаем padding для проверки
labels[:, -1] = -100  # padding token

print(f"\nInput shape: {input_features.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Non-padding tokens: {(labels != -100).sum().item()}")

# Forward pass
print("\nComputing loss...")
with torch.no_grad():
    outputs = model(input_features=input_features, labels=labels)
    loss = outputs.loss

print(f"Model loss: {loss.item():.4f}")

# Теперь проверим, как loss вычисляется вручную
# для UNIFORM distribution (random model)
print("\nManual calculation for uniform distribution:")

# Создаем логиты с uniform distribution (все одинаковые)
# После softmax это даст P(каждый токен) = 1/vocab_size
uniform_logits = torch.zeros(batch_size, target_length, vocab_size)

# Вычисляем loss вручную
loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
manual_loss = loss_fct(uniform_logits.view(-1, vocab_size), labels.view(-1))

print(f"Manual uniform loss: {manual_loss.item():.4f}")
print(f"Expected (log(vocab_size)): {torch.log(torch.tensor(vocab_size)).item():.4f}")

print("\n" + "="*60)
print("ВЫВОДЫ:")
print(f"1. Random baseline (теория): {torch.log(torch.tensor(vocab_size)).item():.4f}")
print(f"2. Random baseline (практика): {manual_loss.item():.4f}")
print(f"3. Whisper model loss: {loss.item():.4f}")
print("="*60)
