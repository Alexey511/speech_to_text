"""
Проверка заморозки embeddings для Whisper и Speech2Text
"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import ModelConfig
from src.models import WhisperSTT, Speech2TextSTT
from transformers import WhisperProcessor, Speech2TextProcessor

print("="*80)
print("1. WHISPER MODEL - Структура и заморозка")
print("="*80)

# Создаём Whisper модель
whisper_config = ModelConfig(
    model_name="openai/whisper-tiny",
    model_type="whisper",
    freeze_encoder=False,
    freeze_decoder=False
)

from transformers import WhisperTokenizer, WhisperFeatureExtractor
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny", language="ru", task="transcribe")
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")
whisper_processor = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

whisper_model = WhisperSTT(whisper_config, processor=whisper_processor)

print("\nСтруктура Whisper модели:")
print("  - model.model.encoder")
print("    └─ conv1, conv2 (feature extraction)")
print("    └─ embed_positions (positional encoding)")
print("    └─ layers[0..N] (transformer layers)")
print("  - model.model.decoder")
print("    └─ embed_tokens (token embeddings) ⚠️")
print("    └─ embed_positions (positional encoding)")
print("    └─ layers[0..N] (transformer layers)")
print("  - model.proj_out (output projection to vocab) ⚠️")

# Проверим, где находятся embeddings
print("\n🔍 Где находятся embeddings в Whisper:")
print(f"  - Decoder input embeddings: model.model.decoder.embed_tokens")
print(f"    Shape: {whisper_model.model.model.decoder.embed_tokens.weight.shape}")
print(f"  - Output projection: model.proj_out")
print(f"    Shape: {whisper_model.model.proj_out.weight.shape}")

# Проверим, что все слои обучаемые (до заморозки)
print("\n✅ До заморозки - все слои обучаемые:")
encoder_trainable = sum(p.requires_grad for p in whisper_model.model.model.encoder.parameters())
decoder_trainable = sum(p.requires_grad for p in whisper_model.model.model.decoder.parameters())
print(f"  - Encoder trainable params: {encoder_trainable}")
print(f"  - Decoder trainable params: {decoder_trainable}")

# Проверяем embeddings до заморозки
embed_tokens_trainable = whisper_model.model.model.decoder.embed_tokens.weight.requires_grad
proj_out_trainable = whisper_model.model.proj_out.weight.requires_grad
print(f"  - embed_tokens.requires_grad: {embed_tokens_trainable}")
print(f"  - proj_out.requires_grad: {proj_out_trainable}")

print("\n" + "-"*80)
print("Замораживаем ENCODER (freeze_encoder)")
print("-"*80)
whisper_model.freeze_encoder()

encoder_trainable_after = sum(p.requires_grad for p in whisper_model.model.model.encoder.parameters())
decoder_trainable_after = sum(p.requires_grad for p in whisper_model.model.model.decoder.parameters())
print(f"  - Encoder trainable params: {encoder_trainable_after} (было {encoder_trainable})")
print(f"  - Decoder trainable params: {decoder_trainable_after} (было {decoder_trainable})")

embed_tokens_trainable_after = whisper_model.model.model.decoder.embed_tokens.weight.requires_grad
proj_out_trainable_after = whisper_model.model.proj_out.weight.requires_grad
print(f"  - embed_tokens.requires_grad: {embed_tokens_trainable_after} ✅ НЕ замерз!")
print(f"  - proj_out.requires_grad: {proj_out_trainable_after} ✅ НЕ замерз!")

print("\n" + "-"*80)
print("Замораживаем DECODER (freeze_decoder)")
print("-"*80)
whisper_model.freeze_decoder()

decoder_trainable_after2 = sum(p.requires_grad for p in whisper_model.model.model.decoder.parameters())
print(f"  - Decoder trainable params: {decoder_trainable_after2} (было {decoder_trainable_after})")

embed_tokens_trainable_after2 = whisper_model.model.model.decoder.embed_tokens.weight.requires_grad
proj_out_trainable_after2 = whisper_model.model.proj_out.weight.requires_grad
print(f"  - embed_tokens.requires_grad: {embed_tokens_trainable_after2} ❌ ЗАМЕРЗ!")
print(f"  - proj_out.requires_grad: {proj_out_trainable_after2} ✅ НЕ замерз (не часть decoder)")

print("\n" + "="*80)
print("2. SPEECH2TEXT MODEL - Структура и заморозка")
print("="*80)

# Создаём Speech2Text модель
s2t_config = ModelConfig(
    model_name="facebook/s2t-small-librispeech-asr",
    model_type="speech2text",
    freeze_encoder=False,
    freeze_decoder=False
)

from transformers import Speech2TextFeatureExtractor, Speech2TextTokenizer
s2t_feature_extractor = Speech2TextFeatureExtractor.from_pretrained("facebook/s2t-small-librispeech-asr")
s2t_tokenizer = Speech2TextTokenizer.from_pretrained("facebook/s2t-small-librispeech-asr")
s2t_processor = Speech2TextProcessor(feature_extractor=s2t_feature_extractor, tokenizer=s2t_tokenizer)

s2t_model = Speech2TextSTT(s2t_config, processor=s2t_processor)

print("\nСтруктура Speech2Text модели:")
print("  - model.model.encoder")
print("    └─ conv layers (feature extraction)")
print("    └─ embed_positions (positional encoding)")
print("    └─ layers[0..N] (transformer layers)")
print("  - model.model.decoder")
print("    └─ embed_tokens (token embeddings) ⚠️")
print("    └─ embed_positions (positional encoding)")
print("    └─ layers[0..N] (transformer layers)")
print("  - model.lm_head (output projection to vocab) ⚠️")

# Проверим, где находятся embeddings
print("\n🔍 Где находятся embeddings в Speech2Text:")
print(f"  - Decoder input embeddings: model.model.decoder.embed_tokens")
print(f"    Shape: {s2t_model.model.model.decoder.embed_tokens.weight.shape}")
print(f"  - Output projection: model.lm_head")
print(f"    Shape: {s2t_model.model.lm_head.weight.shape}")

# Проверим, что все слои обучаемые (до заморозки)
print("\n✅ До заморозки - все слои обучаемые:")
encoder_trainable = sum(p.requires_grad for p in s2t_model.model.model.encoder.parameters())
decoder_trainable = sum(p.requires_grad for p in s2t_model.model.model.decoder.parameters())
lm_head_trainable = sum(p.requires_grad for p in s2t_model.model.lm_head.parameters())
print(f"  - Encoder trainable params: {encoder_trainable}")
print(f"  - Decoder trainable params: {decoder_trainable}")
print(f"  - lm_head trainable params: {lm_head_trainable}")

# Проверяем embeddings до заморозки
embed_tokens_trainable = s2t_model.model.model.decoder.embed_tokens.weight.requires_grad
lm_head_trainable_flag = s2t_model.model.lm_head.weight.requires_grad
print(f"  - embed_tokens.requires_grad: {embed_tokens_trainable}")
print(f"  - lm_head.requires_grad: {lm_head_trainable_flag}")

print("\n" + "-"*80)
print("Замораживаем ENCODER (freeze_encoder)")
print("-"*80)
s2t_model.freeze_encoder()

encoder_trainable_after = sum(p.requires_grad for p in s2t_model.model.model.encoder.parameters())
decoder_trainable_after = sum(p.requires_grad for p in s2t_model.model.model.decoder.parameters())
lm_head_trainable_after = sum(p.requires_grad for p in s2t_model.model.lm_head.parameters())
print(f"  - Encoder trainable params: {encoder_trainable_after} (было {encoder_trainable})")
print(f"  - Decoder trainable params: {decoder_trainable_after} (было {decoder_trainable})")
print(f"  - lm_head trainable params: {lm_head_trainable_after} (было {lm_head_trainable})")

embed_tokens_trainable_after = s2t_model.model.model.decoder.embed_tokens.weight.requires_grad
lm_head_trainable_flag_after = s2t_model.model.lm_head.weight.requires_grad
print(f"  - embed_tokens.requires_grad: {embed_tokens_trainable_after} ✅ НЕ замерз!")
print(f"  - lm_head.requires_grad: {lm_head_trainable_flag_after} ✅ НЕ замерз!")

print("\n" + "-"*80)
print("Замораживаем DECODER (freeze_decoder)")
print("-"*80)
s2t_model.freeze_decoder()

decoder_trainable_after2 = sum(p.requires_grad for p in s2t_model.model.model.decoder.parameters())
lm_head_trainable_after2 = sum(p.requires_grad for p in s2t_model.model.lm_head.parameters())
print(f"  - Decoder trainable params: {decoder_trainable_after2} (было {decoder_trainable_after})")
print(f"  - lm_head trainable params: {lm_head_trainable_after2} (было {lm_head_trainable_after})")

embed_tokens_trainable_after2 = s2t_model.model.model.decoder.embed_tokens.weight.requires_grad
lm_head_trainable_flag_after2 = s2t_model.model.lm_head.weight.requires_grad
print(f"  - embed_tokens.requires_grad: {embed_tokens_trainable_after2} ❌ ЗАМЕРЗ!")
print(f"  - lm_head.requires_grad: {lm_head_trainable_flag_after2} ✅ НЕ замерз (не часть decoder)")

print("\n" + "="*80)
print("РЕЗЮМЕ")
print("="*80)
print("""
📊 WHISPER:
  - freeze_encoder(): замораживает encoder (conv, embed_positions, layers)
    └─ embed_tokens (decoder) ОСТАЁТСЯ ОБУЧАЕМЫМ ✅
    └─ proj_out ОСТАЁТСЯ ОБУЧАЕМЫМ ✅

  - freeze_decoder(): замораживает decoder (embed_tokens, embed_positions, layers)
    └─ embed_tokens ЗАМОРАЖИВАЕТСЯ ❌
    └─ proj_out ОСТАЁТСЯ ОБУЧАЕМЫМ ✅ (не часть model.model.decoder)

📊 SPEECH2TEXT:
  - freeze_encoder(): замораживает encoder (conv, embed_positions, layers)
    └─ embed_tokens (decoder) ОСТАЁТСЯ ОБУЧАЕМЫМ ✅
    └─ lm_head ОСТАЁТСЯ ОБУЧАЕМЫМ ✅

  - freeze_decoder(): замораживает decoder (embed_tokens, embed_positions, layers)
    └─ embed_tokens ЗАМОРАЖИВАЕТСЯ ❌
    └─ lm_head ОСТАЁТСЯ ОБУЧАЕМЫМ ✅ (не часть model.model.decoder)

⚠️ ВАЖНО:
1. При freeze_decoder() embeddings (embed_tokens) ЗАМОРАЖИВАЮТСЯ!
2. Output projection (proj_out/lm_head) НЕ замораживается (не часть decoder)
3. Для cross-lingual transfer нужны ОБУЧАЕМЫЕ embeddings!
4. Поэтому наша стратегия: freeze_encoder=true, freeze_decoder=FALSE
""")
