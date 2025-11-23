"""Скрипт для обновления конфигов с новыми параметрами разморозки"""
from pathlib import Path

# Текст для вставки
new_params = """
  # Fine-grained unfreezing for critical components
  unfreeze_embed_tokens: false
  unfreeze_embed_positions_decoder: false
  unfreeze_lm_head: false
  unfreeze_layer_norm_decoder: false
"""

configs_to_update = [
    "configs/whisper_base.yaml",
    "configs/whisper_small.yaml",
    "configs/debug.yaml"
]

for config_path in configs_to_update:
    path = Path(config_path)
    if not path.exists():
        print(f"❌ Файл не найден: {config_path}")
        continue

    # Читаем содержимое
    content = path.read_text(encoding='utf-8')

    # Проверяем, что параметры ещё не добавлены
    if "unfreeze_embed_tokens" in content:
        print(f"⏭️  Пропускаем {config_path} - параметры уже добавлены")
        continue

    # Ищем строку с unfreeze_last_n_decoder_layers и добавляем после неё
    lines = content.split('\n')
    new_lines = []

    for line in lines:
        new_lines.append(line)
        if "unfreeze_last_n_decoder_layers:" in line:
            # Добавляем новые параметры
            new_lines.append(new_params.rstrip())

    # Записываем обратно
    new_content = '\n'.join(new_lines)
    path.write_text(new_content, encoding='utf-8')
    print(f"✅ Обновлён: {config_path}")

print("\n✅ Все конфиги обновлены!")
