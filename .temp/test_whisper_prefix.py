"""
Тест: нужен ли пробел в начале для Whisper labels
"""
from transformers import WhisperProcessor

processor = WhisperProcessor.from_pretrained("openai/whisper-base")

text1 = "Она удивительная женщина."
text2 = " Она удивительная женщина."  # С пробелом

print("=" * 80)
print("БЕЗ пробела в начале:")
tokens1 = processor.tokenizer(text1, return_tensors="pt", add_special_tokens=False).input_ids[0]
print(f"Токены: {tokens1.tolist()}")
print(f"Первые 3: {tokens1[:3].tolist()}")
decoded1 = processor.tokenizer.decode(tokens1)
print(f"Декодировано: {repr(decoded1)}")

print("\n" + "=" * 80)
print("С пробелом в начале:")
tokens2 = processor.tokenizer(text2, return_tensors="pt", add_special_tokens=False).input_ids[0]
print(f"Токены: {tokens2.tolist()}")
print(f"Первые 3: {tokens2[:3].tolist()}")
decoded2 = processor.tokenizer.decode(tokens2)
print(f"Декодировано: {repr(decoded2)}")

print("\n" + "=" * 80)
print("ВЫВОД:")
print("Если Whisper generate() добавляет пробел, то labels тоже должны начинаться с пробела!")
