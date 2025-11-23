"""
Тест правильной токенизации для Whisper labels
"""
from transformers import WhisperProcessor

processor = WhisperProcessor.from_pretrained("openai/whisper-base")

text = "Она удивительная женщина."

# Способ 1: Обычная токенизация (НЕПРАВИЛЬНО - добавляет спец токены)
print("=" * 80)
print("СПОСОБ 1: Обычная токенизация")
tokens_wrong = processor.tokenizer(text, return_tensors="pt")
print(f"Токены: {tokens_wrong.input_ids[0].tolist()}")
decoded = processor.tokenizer.decode(tokens_wrong.input_ids[0], skip_special_tokens=False)
print(f"Декодировано: {repr(decoded)}")

# Способ 2: БЕЗ спец токенов (ПРАВИЛЬНО для labels)
print("\n" + "=" * 80)
print("СПОСОБ 2: БЕЗ спец токенов (add_special_tokens=False)")
tokens_right = processor.tokenizer(text, return_tensors="pt", add_special_tokens=False)
print(f"Токены: {tokens_right.input_ids[0].tolist()}")
decoded = processor.tokenizer.decode(tokens_right.input_ids[0], skip_special_tokens=False)
print(f"Декодировано: {repr(decoded)}")

# Способ 3: Используя processor.tokenizer напрямую для текста
print("\n" + "=" * 80)
print("СПОСОБ 3: Через tokenizer.encode")
tokens_encode = processor.tokenizer.encode(text, add_special_tokens=False)
print(f"Токены: {tokens_encode}")
decoded = processor.tokenizer.decode(tokens_encode, skip_special_tokens=False)
print(f"Декодировано: {repr(decoded)}")

print("\n" + "=" * 80)
print("СПЕЦ ТОКЕНЫ:")
print(f"<|startoftranscript|>: {processor.tokenizer.convert_tokens_to_ids('<|startoftranscript|>')}")
print(f"<|notimestamps|>: {processor.tokenizer.convert_tokens_to_ids('<|notimestamps|>')}")
print(f"<|endoftext|>: {processor.tokenizer.eos_token_id}")
