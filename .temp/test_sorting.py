from pathlib import Path

# Симуляция директорий
dirs = [
    Path("checkpoints/epoch_7"),
    Path("checkpoints/epoch_8"),
    Path("checkpoints/epoch_9"),
    Path("checkpoints/epoch_10"),
    Path("checkpoints/epoch_11"),
    Path("checkpoints/epoch_12"),
    Path("checkpoints/epoch_13"),
    Path("checkpoints/epoch_14"),
]

print("Лексикографическая сортировка (текущая реализация):")
sorted_dirs = sorted(dirs)
for d in sorted_dirs:
    print(f"  {d.name}")

print(f"\nПоследние 3 (что ОСТАВЛЯЕМ): {[d.name for d in sorted_dirs[-3:]]}")
print(f"Удаляем (все кроме последних 3): {[d.name for d in sorted_dirs[:-3]]}")

print("\n" + "="*60)
print("Правильная сортировка (по номеру эпохи):")

def extract_epoch(path):
    return int(path.name.split('_')[1])

sorted_dirs_correct = sorted(dirs, key=extract_epoch)
for d in sorted_dirs_correct:
    print(f"  {d.name}")

print(f"\nПоследние 3 (что ОСТАВЛЯЕМ): {[d.name for d in sorted_dirs_correct[-3:]]}")
print(f"Удаляем (все кроме последних 3): {[d.name for d in sorted_dirs_correct[:-3]]}")
