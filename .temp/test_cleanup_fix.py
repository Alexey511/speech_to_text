"""Тест для проверки исправления cleanup_old_checkpoints"""
import sys
from pathlib import Path
import tempfile
import shutil

# Добавляем путь к проекту
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from train import cleanup_old_checkpoints
import logging

# Создаём временную директорию для теста
temp_dir = Path(tempfile.mkdtemp())
checkpoints_dir = temp_dir / "checkpoints"
checkpoints_dir.mkdir()

try:
    # Создаём директории эпох (как в реальном сценарии)
    for epoch in [7, 8, 9, 10, 11, 12, 13, 14]:
        (checkpoints_dir / f"epoch_{epoch}").mkdir()

    print("До cleanup_old_checkpoints:")
    epoch_dirs_before = sorted([d.name for d in checkpoints_dir.glob("epoch_*")],
                               key=lambda x: int(x.split('_')[1]))
    print(f"  Существующие директории: {epoch_dirs_before}")

    # Настраиваем logger
    logger = logging.getLogger("test")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(handler)

    # Вызываем cleanup с keep_last=3
    print("\nВызываем cleanup_old_checkpoints(keep_last=3)...")
    cleanup_old_checkpoints(checkpoints_dir, keep_last=3, logger=logger)

    # Проверяем результат
    print("\nПосле cleanup_old_checkpoints:")
    epoch_dirs_after = sorted([d.name for d in checkpoints_dir.glob("epoch_*")],
                              key=lambda x: int(x.split('_')[1]))
    print(f"  Оставшиеся директории: {epoch_dirs_after}")

    # Проверка корректности
    expected = ['epoch_12', 'epoch_13', 'epoch_14']
    if epoch_dirs_after == expected:
        print(f"\n✅ ТЕСТ ПРОЙДЕН! Оставлены последние 3 эпохи: {expected}")
    else:
        print(f"\n❌ ТЕСТ ПРОВАЛЕН!")
        print(f"  Ожидалось: {expected}")
        print(f"  Получено: {epoch_dirs_after}")
        sys.exit(1)

finally:
    # Очистка
    shutil.rmtree(temp_dir)
    print(f"\nВременная директория удалена: {temp_dir}")
