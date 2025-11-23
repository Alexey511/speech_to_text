import torch
from torch.nn.utils.rnn import pad_sequence

# Симуляция входных данных от Dataset
batch = [
    torch.randn(80, 434),  # (freq, time) - SHORT
    torch.randn(80, 794),  # (freq, time) - LONG
    torch.randn(80, 521),  # (freq, time) - MEDIUM
]

print('Входные формы:', [f.shape for f in batch])

# Step 1: Транспонируем
features_transposed = [f.T for f in batch]
print('После транспонирования:', [f.shape for f in features_transposed])

# Step 2: Паддим
input_features = pad_sequence(features_transposed, batch_first=True)
print('После паддинга:', input_features.shape)

# Step 3: Транспонируем обратно
input_features = input_features.transpose(1, 2)
print('После обратного транспонирования:', input_features.shape)

# Step 4: Финальное транспонирование для Speech2Text
input_features_final = input_features.transpose(1, 2)
print('Финальная форма для Speech2Text:', input_features_final.shape)
print('Ожидаемая форма: (batch=3, time=794, freq=80)')
print()

# Проверка attention_mask
time_lengths = [f.shape[0] for f in features_transposed]
max_len = input_features.shape[2]
print('Длины времени:', time_lengths)
print('Максимальная длина:', max_len)

attention_mask = torch.ones(len(batch), max_len, dtype=torch.long)
for i, length in enumerate(time_lengths):
    if length < max_len:
        attention_mask[i, length:] = 0

print('Форма attention_mask:', attention_mask.shape)
print('Attention mask суммы (реальные данные):')
for i in range(len(batch)):
    print(f'  Sample {i}: {attention_mask[i].sum().item()} / {max_len} (expected {time_lengths[i]})')
