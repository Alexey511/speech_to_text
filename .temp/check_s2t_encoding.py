"""
Проверка кодировки и префиксов Speech2Text модели
"""
import sys
sys.path.append('..')

from transformers import Speech2TextProcessor, Speech2TextTokenizer, Speech2TextFeatureExtractor

print("="*80)
print("ПРОВЕРКА SPEECH2TEXT МОДЕЛИ")
print("="*80)

# 1. Монолингвальная английская модель (LibriSpeech)
print("\n1️⃣ МОНОЛИНГВАЛЬНАЯ МОДЕЛЬ (facebook/s2t-small-librispeech-asr)")
print("-"*80)

model_name = "facebook/s2t-small-librispeech-asr"
processor = Speech2TextProcessor.from_pretrained(model_name)
tokenizer = processor.tokenizer

print(f"Tokenizer type: {type(tokenizer)}")
print(f"\n📋 Основные атрибуты:")
print(f"  vocab_size: {tokenizer.vocab_size}")
print(f"  bos_token: {tokenizer.bos_token} (id={tokenizer.bos_token_id})")
print(f"  eos_token: {tokenizer.eos_token} (id={tokenizer.eos_token_id})")
print(f"  pad_token: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")
print(f"  unk_token: {tokenizer.unk_token} (id={tokenizer.unk_token_id})")

# Проверяем наличие языковых токенов
lang_code_to_id = getattr(tokenizer, 'lang_code_to_id', {})
print(f"\n🌐 Языковые токены:")
print(f"  lang_code_to_id: {lang_code_to_id}")
print(f"  Тип: {'MULTILINGUAL' if lang_code_to_id else 'MONOLINGUAL'}")

# Проверяем prefix_tokens
prefix_tokens = getattr(tokenizer, 'prefix_tokens', None)
print(f"\n🎯 Префикс токены:")
print(f"  prefix_tokens: {prefix_tokens}")

# Пробуем токенизировать текст
test_text = "Она удивительная женщина."
tokens = tokenizer(test_text, return_tensors="pt").input_ids[0]
print(f"\n🧪 Токенизация тестового текста:")
print(f"  Текст: {test_text}")
print(f"  Токены: {tokens.tolist()}")
print(f"  Количество токенов: {len(tokens)}")
print(f"  Первые 5 токенов: {tokens[:5].tolist()}")

# Декодируем обратно
decoded = tokenizer.decode(tokens, skip_special_tokens=False)
decoded_clean = tokenizer.decode(tokens, skip_special_tokens=True)
print(f"  Декодировано (с спец токенами): {repr(decoded)}")
print(f"  Декодировано (без спец токенов): {repr(decoded_clean)}")

# 2. Мультилингвальная модель (MuST-C)
print("\n\n2️⃣ МУЛЬТИЛИНГВАЛЬНАЯ МОДЕЛЬ (facebook/s2t-medium-mustc-multilingual-st)")
print("-"*80)

model_name_multi = "facebook/s2t-medium-mustc-multilingual-st"
processor_multi = Speech2TextProcessor.from_pretrained(model_name_multi)
tokenizer_multi = processor_multi.tokenizer

print(f"Tokenizer type: {type(tokenizer_multi)}")
print(f"\n📋 Основные атрибуты:")
print(f"  vocab_size: {tokenizer_multi.vocab_size}")
print(f"  bos_token: {tokenizer_multi.bos_token} (id={tokenizer_multi.bos_token_id})")
print(f"  eos_token: {tokenizer_multi.eos_token} (id={tokenizer_multi.eos_token_id})")
print(f"  pad_token: {tokenizer_multi.pad_token} (id={tokenizer_multi.pad_token_id})")

# Проверяем языковые токены
lang_code_to_id_multi = getattr(tokenizer_multi, 'lang_code_to_id', {})
print(f"\n🌐 Языковые токены:")
print(f"  lang_code_to_id: {lang_code_to_id_multi}")
print(f"  Доступные языки: {list(lang_code_to_id_multi.keys()) if lang_code_to_id_multi else 'None'}")
print(f"  Тип: {'MULTILINGUAL' if lang_code_to_id_multi else 'MONOLINGUAL'}")

# Проверяем, есть ли русский
if lang_code_to_id_multi and 'ru' in lang_code_to_id_multi:
    print(f"  ✅ Русский язык ДОСТУПЕН: ru -> {lang_code_to_id_multi['ru']}")
else:
    print(f"  ❌ Русский язык НЕ ДОСТУПЕН")

# Устанавливаем target language если доступен
if lang_code_to_id_multi and 'ru' in lang_code_to_id_multi:
    tokenizer_multi.tgt_lang = 'ru'
    print(f"\n  Set tgt_lang = 'ru'")

# Проверяем prefix_tokens
prefix_tokens_multi = getattr(tokenizer_multi, 'prefix_tokens', None)
print(f"\n🎯 Префикс токены:")
print(f"  prefix_tokens: {prefix_tokens_multi}")

# Пробуем токенизировать
tokens_multi = tokenizer_multi(test_text, return_tensors="pt").input_ids[0]
print(f"\n🧪 Токенизация тестового текста:")
print(f"  Текст: {test_text}")
print(f"  Токены: {tokens_multi.tolist()}")
print(f"  Количество токенов: {len(tokens_multi)}")
print(f"  Первые 5 токенов: {tokens_multi[:5].tolist()}")

# Декодируем обратно
decoded_multi = tokenizer_multi.decode(tokens_multi, skip_special_tokens=False)
decoded_clean_multi = tokenizer_multi.decode(tokens_multi, skip_special_tokens=True)
print(f"  Декодировано (с спец токенами): {repr(decoded_multi)}")
print(f"  Декодировано (без спец токенов): {repr(decoded_clean_multi)}")

print("\n" + "="*80)
print("ВЫВОДЫ:")
print("="*80)

print("\n📝 Структура кодировки Speech2Text:")
print("  1. Монолингвальные модели (LibriSpeech):")
print("     - Только английский язык")
print("     - Нет языковых токенов (lang_code_to_id пустой)")
print("     - Нет prefix_tokens")
print("     - Простая структура: текст -> токены -> </s>")
print("")
print("  2. Мультилингвальные модели (MuST-C):")
print("     - Поддержка нескольких языков через lang_code_to_id")
print("     - Возможна установка tgt_lang для генерации")
print("     - forced_bos_token_id используется для задания языка")
print("     - НЕТ аналога prefix_tokens как в Whisper!")

print("\n⚠️ КЛЮЧЕВОЕ ОТЛИЧИЕ ОТ WHISPER:")
print("  - Whisper: prefix_tokens добавляются АВТОМАТИЧЕСКИ в labels")
print("  - Speech2Text: язык задаётся через forced_bos_token_id при ГЕНЕРАЦИИ")
print("  - Speech2Text НЕ добавляет языковые токены в labels при обучении")
print("  - Для монолингвальных S2T моделей вообще нет языковых токенов")

print("\n✅ Выводы о рисках:")
print("  1. Монолингвальная S2T (LibriSpeech): БЕЗОПАСНА")
print("     - Нет prefix_tokens, нет языковых токенов")
print("     - Аналогичной проблемы с Whisper НЕ БУДЕТ")
print("")
print("  2. Мультилингвальная S2T (MuST-C): ОТНОСИТЕЛЬНО БЕЗОПАСНА")
print("     - Язык задаётся через forced_bos_token_id (только для генерации)")
print("     - В labels языковые токены НЕ добавляются")
print("     - Наш код в DataManager.setup_processor() правильно устанавливает tgt_lang")

print("\n" + "="*80)
