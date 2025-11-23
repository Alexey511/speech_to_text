"""
Проверка инициализации весов Speech2Text при cross-lingual transfer
"""
import torch
from transformers import (
    Speech2TextForConditionalGeneration,
    Speech2TextTokenizer,
    Speech2TextProcessor,
    Speech2TextFeatureExtractor
)

print("="*80)
print("Шаг 1: Создаём английскую модель (исходная)")
print("="*80)

english_model_name = "facebook/s2t-small-librispeech-asr"
model = Speech2TextForConditionalGeneration.from_pretrained(english_model_name)

print(f"\nАрхитектура модели:")
print(f"  - model.model.encoder: кодирует аудио в скрытые представления")
print(f"  - model.model.decoder: декодирует текст автореgrессивно")
print(f"  - model.lm_head: выходной слой (vocab_size)")

original_vocab_size = model.config.vocab_size
print(f"\nОригинальный vocab_size английской модели: {original_vocab_size}")

# Проверим размеры слоёв связанных с vocab
print(f"\nРазмеры слоёв, связанных с vocabulary:")
print(f"  - model.model.decoder.embed_tokens: {model.model.decoder.embed_tokens.weight.shape}")
print(f"  - model.lm_head: {model.lm_head.weight.shape}")

# Сохраним первые несколько весов для проверки
original_embed_weights = model.model.decoder.embed_tokens.weight[:5].clone()
original_lm_head_weights = model.lm_head.weight[:5].clone()

print("\nПервые 5 весов decoder.embed_tokens[0] (до resize):")
print(original_embed_weights[0, :10])

print("\n" + "="*80)
print("Шаг 2: Загружаем мультиязычный токенайзер")
print("="*80)

multilingual_tokenizer_name = "facebook/s2t-medium-mustc-multilingual-st"
multilingual_tokenizer = Speech2TextTokenizer.from_pretrained(multilingual_tokenizer_name)
multilingual_tokenizer.tgt_lang = "ru"

multilingual_vocab_size = len(multilingual_tokenizer)
print(f"\nVocab size мультиязычного токенайзера: {multilingual_vocab_size}")
print(f"Разница в vocab_size: {multilingual_vocab_size - original_vocab_size}")

print("\n" + "="*80)
print("Шаг 3: Выполняем resize_token_embeddings")
print("="*80)

# Resize embeddings
model.resize_token_embeddings(multilingual_vocab_size)

print(f"\nНовый vocab_size модели: {model.config.vocab_size}")
print(f"\nРазмеры слоёв после resize:")
print(f"  - model.model.decoder.embed_tokens: {model.model.decoder.embed_tokens.weight.shape}")
print(f"  - model.lm_head: {model.lm_head.weight.shape}")

# Проверим что произошло с весами
new_embed_weights = model.model.decoder.embed_tokens.weight[:5].clone()
new_lm_head_weights = model.lm_head.weight[:5].clone()

print("\nПервые 5 весов decoder.embed_tokens[0] (после resize):")
print(new_embed_weights[0, :10])

print("\nСравнение весов (старые == новые для первых токенов?):")
print(f"  - embed_tokens сохранены: {torch.allclose(original_embed_weights, new_embed_weights)}")
print(f"  - lm_head сохранены: {torch.allclose(original_lm_head_weights, new_lm_head_weights)}")

print("\nПроверка новых токенов (добавлены resize):")
new_tokens_start_idx = original_vocab_size
print(f"  - Индексы новых токенов: [{new_tokens_start_idx}, {multilingual_vocab_size})")
print(f"  - Количество новых токенов: {multilingual_vocab_size - original_vocab_size}")

# Проверим инициализацию новых токенов
new_token_embed_weights = model.model.decoder.embed_tokens.weight[new_tokens_start_idx:new_tokens_start_idx+3]
new_token_lm_head_weights = model.lm_head.weight[new_tokens_start_idx:new_tokens_start_idx+3]

print(f"\nПервые 3 новых токена в embed_tokens[{new_tokens_start_idx}] (должны быть случайными):")
print(new_token_embed_weights[0, :10])

print(f"\nПервые 3 новых токена в lm_head[{new_tokens_start_idx}] (должны быть случайными):")
print(new_token_lm_head_weights[0, :10])

# Проверим статистику новых весов
print(f"\nСтатистика новых весов в embed_tokens:")
print(f"  - Mean: {new_token_embed_weights.mean().item():.6f}")
print(f"  - Std: {new_token_embed_weights.std().item():.6f}")
print(f"  - Min: {new_token_embed_weights.min().item():.6f}")
print(f"  - Max: {new_token_embed_weights.max().item():.6f}")

print("\n" + "="*80)
print("Шаг 4: Проверка остальных слоёв модели")
print("="*80)

# Проверим несколько encoder и decoder слоёв
encoder_layer_0_weight = model.model.encoder.layers[0].self_attn.k_proj.weight
decoder_layer_0_weight = model.model.decoder.layers[0].self_attn.k_proj.weight

print(f"\nEncoder layer 0 self_attn.k_proj: {encoder_layer_0_weight.shape}")
print(f"  - Mean: {encoder_layer_0_weight.mean().item():.6f}")
print(f"  - Std: {encoder_layer_0_weight.std().item():.6f}")

print(f"\nDecoder layer 0 self_attn.k_proj: {decoder_layer_0_weight.shape}")
print(f"  - Mean: {decoder_layer_0_weight.mean().item():.6f}")
print(f"  - Std: {decoder_layer_0_weight.std().item():.6f}")

print("\n" + "="*80)
print("ВЫВОДЫ:")
print("="*80)
print("""
1. При создании модели через from_pretrained() загружаются ВСЕ предобученные веса
   - Encoder (feature extraction)
   - Decoder (все 6 слоёв)
   - Все attention, feedforward слои и т.д.

2. При resize_token_embeddings():
   - СОХРАНЯЮТСЯ оригинальные веса для первых {original_vocab} токенов
   - ДОБАВЛЯЮТСЯ новые веса для новых токенов (случайная инициализация)
   - Затронуты ТОЛЬКО 2 слоя:
     * model.model.decoder.embed_tokens (input embeddings)
     * model.lm_head (output projection)

3. Остальные слои модели (encoder, decoder layers) остаются с предобученными весами!

4. При fine-tuning:
   - Можно заморозить encoder (freeze_encoder=true) - acoustic features универсальны
   - Decoder учится новому языку через новые embeddings
   - lm_head учится предсказывать новые токены
""".format(original_vocab=original_vocab_size))
