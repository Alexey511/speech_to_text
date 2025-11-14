# Baseline Evaluation - Инструкция

Пошаговая инструкция для оценки baseline моделей (без дообучения) на русском датасете Common Voice 22.0.

## 📁 Созданные файлы

### 1. Jupyter ноутбук
- **`notebooks/02_prepare_baseline_models.ipynb`** - подготовка baseline моделей

### 2. Конфиги для evaluation
- **`configs/eval_baseline_whisper_small.yaml`** - конфиг для Whisper Small
- **`configs/eval_baseline_whisper_base.yaml`** - конфиг для Whisper Base
- **`configs/eval_baseline_s2t.yaml`** - конфиг для Speech2Text Small

## 🚀 Шаги для baseline evaluation

### Шаг 1: Подготовка baseline моделей

Запустите Jupyter ноутбук для загрузки и сохранения моделей:

```bash
jupyter notebook notebooks/02_prepare_baseline_models.ipynb
```

Ноутбук выполнит:
1. Загрузку моделей из HuggingFace:
   - `openai/whisper-small`
   - `openai/whisper-base`
   - `facebook/s2t-small-librispeech-asr`

2. Заморозку всех параметров (для чистоты эксперимента)

3. Сохранение в custom checkpoint формате с полным конфигом:
   - `experiments/baselines/whisper-small-no-finetune/`
     - `model_weights.pt` - веса модели
     - `metadata.json` - метаданные чекпоинта
     - `config.yaml` - полный конфиг проекта
   - `experiments/baselines/whisper-base-no-finetune/`
   - `experiments/baselines/s2t-small-no-finetune/`

### Шаг 2: Оценка на тестовом датасете

После подготовки моделей запустите evaluation.

**Способ 1: Через VSCode "Run" кнопку (рекомендуется)**

1. Открой [evaluation.py](evaluation.py)
2. В начале файла найди секцию `DEFAULT_CONFIG`
3. Измени `model_path` на нужную модель:
   ```python
   DEFAULT_CONFIG = {
       "model_path": "experiments/baselines/whisper-small-no-finetune",
       "config": None,  # Автоматически найдётся в директории модели
       "dataset_split": "test",
       "experiment_name": None,
   }
   ```
4. Нажми "Run" кнопку в правом верхнем углу VSCode

**Способ 2: Через командную строку**

Конфиг теперь **не обязательно** указывать - он автоматически найдётся в директории модели!

#### Whisper Small Baseline
```bash
python evaluation.py --model-path experiments/baselines/whisper-small-no-finetune
```

#### Whisper Base Baseline
```bash
python evaluation.py --model-path experiments/baselines/whisper-base-no-finetune
```

#### Speech2Text Small Baseline
```bash
python evaluation.py --model-path experiments/baselines/s2t-small-no-finetune
```

*(Опционально: можно добавить `--dataset-split validation` для оценки на dev сете)*

## 📊 Результаты

Для каждой модели будут созданы:

```
experiments/
└── <model_name>_baseline_eval/
    ├── logs/
    │   └── evaluation.log
    ├── config.yaml
    ├── test_results.json          # Метрики (WER, CER, BLEU)
    ├── test_predictions.csv       # Предсказания модели
    └── test_detailed_analysis.json # Детальный анализ ошибок
```

### Основные метрики
- **WER** (Word Error Rate) - процент ошибочных слов
- **CER** (Character Error Rate) - процент ошибочных символов
- **BLEU** - метрика качества перевода/транскрипции
- **Substitutions, Deletions, Insertions** - типы ошибок

## 🔍 Анализ результатов

После evaluation проверьте:

1. **test_results.json** - основные метрики
2. **test_predictions.csv** - сравнение reference vs prediction
3. **test_detailed_analysis.json** - анализ ошибок по длине текста
4. **logs/evaluation.log** - полный лог процесса

## ⚙️ Особенности конфигов

Все evaluation конфиги настроены для baseline оценки:
- ✅ Все параметры модели заморожены (`freeze_*: true`)
- ✅ Аугментация отключена (`augmentation.enabled: false`)
- ✅ Фильтрация по длительности отключена (`filter_by_duration: false`)
- ✅ BLEU включен для полноты оценки (`calculate_bleu: true`)
- ✅ WandB отключен (`use_wandb: false`)
- ✅ Больший batch size для evaluation (`per_device_eval_batch_size: 16`)

## 📝 Следующие шаги

После baseline evaluation:
1. Проанализировать результаты
2. Обсудить стратегию fine-tuning
3. Создать конфиги для обучения с разной заморозкой слоев
4. Запустить серию экспериментов fine-tuning

## ❓ Проверка готовности

Перед запуском убедитесь:
- [ ] Датасет Common Voice 22.0 загружен в `data/cv-corpus-22.0-2025-06-20/ru/`
- [ ] Conda окружение `basenn` активировано
- [ ] Достаточно места на диске (~2GB для baseline моделей)
- [ ] GPU доступна (опционально, но рекомендуется)

## 🐛 Troubleshooting

**Ошибка "Dataset not found":**
- Проверьте путь к датасету в конфиге: `data.dataset_path`

**Ошибка "Out of memory":**
- Уменьшите `per_device_eval_batch_size` в конфиге

**Модель не загружается:**
- Проверьте что ноутбук успешно сохранил модели в `experiments/baselines/`
