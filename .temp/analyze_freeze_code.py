"""
Анализ кода заморозки - что именно замораживается
"""
print("="*80)
print("АНАЛИЗ КОДА ЗАМОРОЗКИ ИЗ src/models.py")
print("="*80)

print("\n1️⃣ WHISPER - метод freeze_encoder() (строки 293-296):")
print("-"*80)
print("""
def freeze_encoder(self):
    for param in self.model.model.encoder.parameters():
        param.requires_grad = False
    logger.info("Encoder frozen")
""")
print("\n📌 Замораживает: self.model.model.encoder")
print("   Включает:")
print("   - conv1, conv2 (feature extraction)")
print("   - embed_positions (positional encoding)")
print("   - layers[0..N] (все transformer layers)")
print("\n📌 НЕ затрагивает:")
print("   ✅ self.model.model.decoder.embed_tokens (input embeddings)")
print("   ✅ self.model.proj_out (output projection)")

print("\n" + "-"*80)
print("2️⃣ WHISPER - метод freeze_decoder() (строки 298-301):")
print("-"*80)
print("""
def freeze_decoder(self):
    for param in self.model.model.decoder.parameters():
        param.requires_grad = False
    logger.info("Decoder frozen")
""")
print("\n📌 Замораживает: self.model.model.decoder")
print("   Включает:")
print("   - embed_tokens (token embeddings) ❌ ЗАМЕРЗ!")
print("   - embed_positions (positional encoding)")
print("   - layers[0..N] (все transformer layers)")
print("\n📌 НЕ затрагивает:")
print("   ✅ self.model.proj_out (output projection, не часть decoder)")

print("\n" + "="*80)
print("\n3️⃣ SPEECH2TEXT - метод freeze_encoder() (строки 499-502):")
print("-"*80)
print("""
def freeze_encoder(self):
    for param in self.model.model.encoder.parameters():
        param.requires_grad = False
    logger.info("Encoder frozen")
""")
print("\n📌 Замораживает: self.model.model.encoder")
print("   Включает:")
print("   - conv layers (feature extraction)")
print("   - embed_positions (positional encoding)")
print("   - layers[0..N] (все transformer layers)")
print("\n📌 НЕ затрагивает:")
print("   ✅ self.model.model.decoder.embed_tokens (input embeddings)")
print("   ✅ self.model.lm_head (output projection)")

print("\n" + "-"*80)
print("4️⃣ SPEECH2TEXT - метод freeze_decoder() (строки 504-507):")
print("-"*80)
print("""
def freeze_decoder(self):
    for param in self.model.model.decoder.parameters():
        param.requires_grad = False
    logger.info("Decoder frozen")
""")
print("\n📌 Замораживает: self.model.model.decoder")
print("   Включает:")
print("   - embed_tokens (token embeddings) ❌ ЗАМЕРЗ!")
print("   - embed_positions (positional encoding)")
print("   - layers[0..N] (все transformer layers)")
print("\n📌 НЕ затрагивает:")
print("   ✅ self.model.lm_head (output projection, не часть decoder)")

print("\n" + "="*80)
print("ОТВЕТ НА ВОПРОС")
print("="*80)
print("""
❓ Вопрос: "Когда мы размораживаем/замораживаем некоторые слои -
           эмбединги всегда остаются размороженными для s2t? А для whisper?"

✅ ОТВЕТ:

1. При freeze_encoder():
   - Whisper: embed_tokens ОСТАЮТСЯ РАЗМОРОЖЕННЫМИ ✅
   - Speech2Text: embed_tokens ОСТАЮТСЯ РАЗМОРОЖЕННЫМИ ✅

2. При freeze_decoder():
   - Whisper: embed_tokens ЗАМОРАЖИВАЮТСЯ ❌
   - Speech2Text: embed_tokens ЗАМОРАЖИВАЮТСЯ ❌

3. Output projection (proj_out/lm_head):
   - Whisper: proj_out ВСЕГДА остаётся размороженным ✅
   - Speech2Text: lm_head ВСЕГДА остаётся размороженным ✅
   - Причина: они НЕ являются частью model.model.decoder

📊 СТРУКТУРА МОДЕЛЕЙ:

Whisper:
  model.model.encoder          ← freeze_encoder() затрагивает
  model.model.decoder          ← freeze_decoder() затрагивает
    └─ embed_tokens ⚠️        ← ЧАСТЬ decoder, замерзает при freeze_decoder()!
    └─ embed_positions
    └─ layers[...]
  model.proj_out               ← НЕ затрагивается ни одним методом

Speech2Text:
  model.model.encoder          ← freeze_encoder() затрагивает
  model.model.decoder          ← freeze_decoder() затрагивает
    └─ embed_tokens ⚠️        ← ЧАСТЬ decoder, замерзает при freeze_decoder()!
    └─ embed_positions
    └─ layers[...]
  model.lm_head                ← НЕ затрагивается ни одним методом

⚠️ КРИТИЧЕСКОЕ ДЛЯ CROSS-LINGUAL TRANSFER:

Наш конфиг s2t_cross_lingual.yaml:
  freeze_encoder: true    ✅ Правильно - acoustic features универсальны
  freeze_decoder: FALSE   ✅ ВАЖНО! Иначе embed_tokens замерзнут!

Если freeze_decoder=true:
  ❌ embed_tokens замёрзнут
  ❌ Новые русские токены не обучатся
  ❌ Модель не сможет научиться русскому языку

Если freeze_decoder=false (текущая настройка):
  ✅ embed_tokens обучаются
  ✅ Decoder layers обучаются
  ✅ lm_head обучается
  ✅ Модель учится генерировать русский текст!
""")
