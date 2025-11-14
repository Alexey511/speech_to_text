# Russian Speech-to-Text Fine-tuning Project

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5+](https://img.shields.io/badge/PyTorch-2.5+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Проект для fine-tuning SOTA моделей speech-to-text на русском языке с использованием официального датасета Mozilla Common Voice 22.0. Модульная архитектура с конфигурационным управлением, комплексными метриками оценки и production-ready MLOps практиками.

## 🎯 Цель проекта

Создание высококачественной системы распознавания русской речи путем дообучения передовых моделей (Whisper, Speech2Text) на официальном датасете Mozilla Common Voice 22.0 (~40 часов качественного русского аудио).

## 🚀 Основные возможности

- **Модульная архитектура** с иерархической конфигурацией (OmegaConf + YAML)
- **Поддержка SOTA моделей**:
  - OpenAI Whisper (tiny/base/small/medium/large)
  - Facebook Speech2Text с cross-lingual transfer
- **Комплексные метрики**: WER, CER, BLEU, MER, WIL, детальный error breakdown
- **Мониторинг обучения**: TensorBoard (по умолчанию), Weights & Biases (опционально)
- **Оптимизация для RTX 4070ti**: Mixed precision (FP16), gradient accumulation, memory management
- **Продвинутая аугментация**: Time/frequency-domain, SpecAugment, reverb (8+ типов аугментаций)
- **Production-ready**: Отдельные скрипты для training/evaluation/inference, comprehensive testing
- **Advanced training**: Early stopping, multiple LR schedulers (linear, cosine, plateau, warmup-plateau-decay)
- **Flexible freezing**: Fine-grained control over encoder/decoder/embeddings freezing

## 📁 Структура проекта

```
speech_to_text/
├── configs/                           # Конфигурационные файлы YAML
│   ├── default.yaml                   # Whisper Small конфигурация (по умолчанию)
│   ├── whisper_base.yaml              # Whisper Base конфигурация
│   ├── whisper_small.yaml             # Whisper Small конфигурация
│   ├── s2t_cross_lingual.yaml         # Speech2Text cross-lingual transfer
│   └── debug.yaml                     # Debug конфигурация (быстрое тестирование)
├── data/                              # Официальный Mozilla Common Voice 22.0
│   └── cv-corpus-22.0-2025-06-20/
│       └── ru/
│           ├── train.tsv              # ~26K обучающих записей
│           ├── dev.tsv                # ~10K валидационных записей
│           ├── test.tsv               # ~10K тестовых записей
│           ├── clip_durations.tsv     # Кэш длительностей (ускорение загрузки)
│           └── clips/                 # MP3 аудиофайлы (~6.5GB)
├── experiments/                       # Результаты экспериментов
│   └── <experiment_name>/
│       ├── logs/                      # Логи обучения/оценки
│       ├── tensorboard/               # TensorBoard логи
│       ├── checkpoint-XXX/            # Чекпоинты модели
│       ├── final_model/               # Финальная обученная модель
│       ├── config.yaml                # Конфигурация эксперимента
│       ├── test_results.json          # Метрики на тест сете
│       └── test_predictions.csv       # Предсказания модели
├── notebooks/
│   ├── 01_eda.ipynb                   # Исследование данных (EDA)
│   ├── 02_prepare_baseline_models.ipynb  # Подготовка baseline моделей
│   └── debug.ipynb                    # Debug notebook
├── src/
│   ├── __init__.py
│   ├── config.py                      # Иерархическая конфигурация (OmegaConf)
│   ├── data.py                        # Пайплайн данных (DataManager, Dataset, Collator)
│   ├── models.py                      # Архитектуры моделей (Whisper, Speech2Text, Custom)
│   ├── metrics.py                     # Метрики оценки (WER, CER, BLEU, MER, WIL)
│   ├── processors.py                  # GPU-friendly feature extraction
│   └── utils.py                       # Утилиты (логирование, визуализация, пути)
├── tests/                             # Comprehensive testing suite
│   ├── __init__.py
│   ├── conftest.py                    # Pytest fixtures (config, temp_dir)
│   ├── test_config.py                 # Тесты конфигурации
│   ├── test_data_manager.py           # Тесты DataManager
│   ├── test_dataloader.py             # Тесты DataLoader
│   ├── test_metrics.py                # Тесты метрик
│   ├── test_models.py                 # Тесты моделей
│   ├── test_training.py               # Тесты обучения
│   ├── test_utils.py                  # Тесты утилит
│   ├── dataloader_speed.py            # Бенчмарк производительности DataLoader
│   └── .test_tmp/                     # Временные файлы тестов (auto-cleanup)
├── .gitignore                         # Git ignore (data/, experiments/, .test_tmp/)
├── requirements.txt                   # Python зависимости
├── setup_check.py                     # Проверка окружения и датасета
├── train.py                           # Основной скрипт обучения
├── evaluation.py                      # Автономная оценка моделей
├── inference.py                       # Production инференс (single/batch)
├── run_experiments.py                 # Оркестрация нескольких экспериментов
├── CLAUDE.md                          # Руководство для Claude Code
└── README.md                          # Документация
```

## 🛠 Установка

### Системные требования

- **OS**: Windows 10/11, Linux, macOS
- **Python**: 3.8+
- **GPU**: RTX 4070ti или аналогичная (минимум 8GB VRAM для Whisper Small, 12GB+ для Medium)
- **RAM**: 16GB+ рекомендуется
- **Диск**: 50GB+ свободного места (датасет ~6.5GB + эксперименты ~10-30GB)

### Установка с Conda (рекомендуется)

Проект использует conda окружение `basenn` для управления зависимостями:

```bash
# Создание conda окружения
conda create -n basenn python=3.10 -y
conda activate basenn

# Установка PyTorch с CUDA support
conda install -n basenn pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Установка основных зависимостей через conda
conda install -n basenn -c conda-forge pandas numpy scipy matplotlib seaborn plotly tqdm pyyaml rich ffmpeg -y

# Установка инструментов разработки
conda install -n basenn -c conda-forge mypy ruff pytest jiwer -y

# Установка зависимостей через pip (только те, что недоступны в conda)
conda run -n basenn pip install -r requirements.txt
```

### Альтернативная установка (pip/venv)

```bash
# Клонирование репозитория
git clone <repository_url>
cd speech_to_text

# Создание виртуального окружения
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate     # Windows

# Установка зависимостей
pip install -r requirements.txt

# Установка ffmpeg (для MP3 support)
# Windows: скачать с https://ffmpeg.org/download.html
# Linux: sudo apt-get install ffmpeg
# macOS: brew install ffmpeg
```

### Проверка установки

```bash
# Проверка окружения
python setup_check.py

# Или вручную:
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

**Ожидаемый вывод:**
```
✓ PyTorch: 2.5.1, CUDA: True
✓ CUDA device: NVIDIA GeForce RTX 4070 Ti
✓ Transformers: 4.57.0
✓ Dataset found: data/cv-corpus-22.0-2025-06-20/ru/
```

## 🏃‍♂️ Быстрый старт

### 1. Скачивание официального датасета Mozilla Common Voice 22.0

**📥 Скачайте официальный датасет:**

1. **Перейдите на официальный сайт:** https://commonvoice.mozilla.org/en/datasets
2. **Найдите Russian (ru)** в списке языков
3. **Скачайте Common Voice Corpus 22.0** (файл ~6.5GB)
4. **Распакуйте архив** в папку проекта:

```bash
# Создайте папку для данных (если еще нет)
mkdir -p data

# Распакуйте скачанный архив cv-corpus-22.0-2025-06-20-ru.tar.gz
# в папку data/ 
# Должна получиться структура: data/cv-corpus-22.0-2025-06-20/ru/
```

**✅ Преимущества официального датасета:**
- 🛡️ **Максимальная безопасность** - прямо от Mozilla Foundation
- 📈 **Актуальные данные** - Common Voice 22.0 (самая свежая версия)
- 🎯 **Качественные записи** - проверенные и валидированные
- ⚡ **250+ часов** русского аудио для эффективного fine-tuning

### 2. Исследование данных

```bash
jupyter notebook notebooks/01_eda.ipynb
```

### 3. Обучение модели

**Обучение с конфигурацией по умолчанию (Whisper Small):**
```bash
python train.py
```

**Обучение с указанным конфигом:**
```bash
# Whisper Base
python train.py --config configs/whisper_base.yaml

# Speech2Text cross-lingual
python train.py --config configs/s2t_cross_lingual.yaml

# С переопределением имени эксперимента
python train.py --config configs/whisper_small.yaml --experiment-name whisper_small_v2
```

**Debug режим (быстрая проверка с сокращенными эпохами):**
```bash
python train.py --debug --no-wandb
```

**Примечание:** Все параметры модели и обучения настраиваются через YAML конфиги в `configs/`. CLI аргументы предназначены только для управления экспериментом (`--config`, `--experiment-name`, `--debug`, `--no-wandb`).

### 4. Оценка модели

**Оценка обученной модели на test set:**
```bash
python evaluation.py --model-path experiments/whisper_ru_cv22_finetune/final_model
```

**Оценка с детальным анализом ошибок:**
```bash
python evaluation.py --model-path experiments/whisper_ru_cv22_finetune/final_model --detailed-analysis --save-predictions
```

**Оценка baseline модели без fine-tuning:**
```bash
python evaluation.py --model-path openai/whisper-small --config configs/default.yaml
```

### 5. Инференс

**Инференс одного аудиофайла:**
```bash
python inference.py --model-path experiments/whisper_ru_cv22_finetune/final_model --input audio.mp3
```

**Пакетный инференс (директория):**
```bash
python inference.py --model-path experiments/whisper_ru_cv22_finetune/final_model --input audio_folder/ --output results.json --format json
```

### 6. Запуск нескольких экспериментов

**Сравнение нескольких моделей:**
```bash
python run_experiments.py --experiments whisper_small whisper_base s2t_cross_lingual
```

**Debug режим для быстрого тестирования:**
```bash
python run_experiments.py --debug
```

## 📊 Работа с данными

### 📁 Структура датасета Common Voice 22.0

После распаковки официального датасета структура должна быть:

```
data/
└── cv-corpus-22.0-2025-06-20/
    └── ru/
        ├── train.tsv           # ~26K обучающих записей
        ├── dev.tsv             # ~10K валидационных записей  
        ├── test.tsv            # ~10K тестовых записей
        ├── validated.tsv       # Все валидированные записи
        └── clips/              # 🎵 Аудиофайлы (~6.5GB)
            ├── common_voice_ru_*.mp3
            └── ...
```

### 🧪 Тестирование датасета

Проверьте что датасет корректно загружен с помощью pytest:

```bash
# Запустите все тесты
pytest tests/ -v

# Или только тесты DataManager  
pytest tests/test_data_manager.py -v

# Тесты с подробным выводом
pytest tests/test_data_manager.py::TestDataManager::test_dataset_info -v -s
```

**Ожидаемый вывод:**
- ✅ test_dataset_availability PASSED
- ✅ test_dataset_info PASSED  
- ✅ test_load_train_dataset PASSED
- ✅ 📊 Dataset stats: ~47,000 samples, ~38 hours

### ⚙️ Настройки обработки данных

В конфигурации `configs/default.yaml`:

```yaml
data:
  language: "ru"                    # Русский язык
  validation_split: "dev"           # Common Voice использует 'dev' 
  filter_by_duration: false        # Фильтровать по длительности
  max_duration: 30.0               # Максимальная длительность (сек)
  min_duration: 0.5                # Минимальная длительность (сек)
  sample_rate: 16000               # Частота дискретизации
```

### 📈 Статистика датасета

- **Общий объем:** ~38 часов русского аудио
- **Качество:** Проверенные записи с up_votes > down_votes  
- **Формат:** 16kHz MP3 + TSV метаданные
- **Безопасность:** Официальный Mozilla Foundation

## ⚙️ Конфигурация

Проект использует иерархическую систему конфигурации на основе **OmegaConf** с YAML файлами. Все параметры модели, обучения и данных настраиваются через конфиги в директории `configs/`.

### Основные параметры (configs/default.yaml)

```yaml
# Эксперимент
experiment:
  output_dir: "experiments"
  experiment_name: "whisper_ru_cv22_finetune"
  seed: 42

# Модель
model:
  model_name: "openai/whisper-small"
  model_type: "whisper"

  # Стратегия заморозки слоев
  freeze_feature_encoder: true    # Заморозить feature encoder
  freeze_encoder: true            # Заморозить encoder
  freeze_decoder: false           # Decoder обучаемый (language-specific)

  # Fine-grained контроль
  unfreeze_embed_tokens: false    # Разморозить embeddings декодера
  unfreeze_lm_head: false         # Разморозить output projection

  # Dropout (ВАЖНО: для Whisper ставить 0.0!)
  activation_dropout: 0.0
  attention_dropout: 0.0
  dropout: 0.0

# Обучение
training:
  num_train_epochs: 10
  train_batch_size: 8
  eval_batch_size: 8
  gradient_accumulation_steps: 2
  learning_rate: 1e-4
  weight_decay: 0.01
  fp16: true  # Mixed precision для RTX 4070ti

  # Learning rate scheduler
  scheduler_name: "linear"  # linear, cosine, reduce_on_plateau, onecycle, warmup_plateau_decay

  # Early stopping
  use_early_stopping: true
  early_stopping_patience: 3
  early_stopping_threshold: 0.01

# Данные
data:
  language: "ru"
  task: "transcribe"
  data_dir: "data"
  dataset_path: "cv-corpus-22.0-2025-06-20/ru"
  sample_rate: 16000

  # Фильтрация по длительности
  filter_by_duration: true
  max_duration: 30.0
  min_duration: 0.2

  # Аугментация
  augmentation:
    enabled: true
    add_noise: true
    speed_perturbation: true
    pitch_shift: true
    spec_augment: true
    reverb: true
    # ... и другие параметры

# Оценка
evaluation:
  batch_size: 16
  calculate_wer: true
  calculate_cer: true
  calculate_bleu: false
  num_beams: 1

  # Anti-repetition (для предотвращения зацикливания)
  repetition_penalty: 1.2
  no_repeat_ngram_size: 3

# Логирование
logging:
  use_wandb: false
  wandb_project: "speech-to-text-ru"
  log_level: "INFO"
  report_to: ["tensorboard", "wandb"]
```

### Доступные конфигурации

- **`default.yaml`** - Whisper Small (рекомендуется для начала)
- **`whisper_base.yaml`** - Whisper Base (легче, быстрее)
- **`whisper_small.yaml`** - Whisper Small (баланс качества и скорости)
- **`s2t_cross_lingual.yaml`** - Speech2Text cross-lingual transfer (English→Russian)
- **`debug.yaml`** - Debug конфигурация (2 эпохи, сокращенные шаги)

## 🔧 Поддерживаемые модели

### 1. OpenAI Whisper (Multilingual)
Encoder-decoder архитектура с поддержкой 99 языков. Обучены на 680K часов аудио.

- **`openai/whisper-tiny`** (39M параметров) - самая быстрая
- **`openai/whisper-base`** (74M параметров) - баланс скорости и качества
- **`openai/whisper-small`** (244M параметров) - **рекомендуется** для fine-tuning
- **`openai/whisper-medium`** (769M параметров) - высокое качество
- **`openai/whisper-large`** (1550M параметров) - максимальное качество (требует 16GB+ VRAM)

**Использование:**
```yaml
model:
  model_name: "openai/whisper-small"
  model_type: "whisper"
```

### 2. Facebook Speech2Text (Cross-lingual Transfer)
Encoder-decoder архитектура для ASR и speech translation.

- **`facebook/s2t-small-librispeech-asr`** (~31M параметров, English)
  - **Стратегия:** Fine-tuning английской модели для работы с русским через cross-lingual transfer
  - Используется многоязычный токенайзер `facebook/s2t-medium-mustc-multilingual-st`
  - Требует unfreezing decoder embeddings для адаптации к новому словарю

**Использование:**
```yaml
model:
  model_name: "facebook/s2t-small-librispeech-asr"
  model_type: "speech2text"
  tokenizer_name_or_path: "facebook/s2t-medium-mustc-multilingual-st"
  unfreeze_embed_tokens: true  # Критично для cross-lingual transfer!
```


**Сравнительная таблица:**

| Модель | Параметры | VRAM | Скорость | Качество (WER ↓) | Рекомендация |
|--------|-----------|------|----------|------------------|--------------|
| Whisper Tiny | 39M | ~2GB | ⚡⚡⚡⚡⚡ | ~25-30% | Quick prototyping |
| Whisper Base | 74M | ~3GB | ⚡⚡⚡⚡ | ~20-25% | Fast inference |
| **Whisper Small** | 244M | ~5GB | ⚡⚡⚡ | **~15-20%** | **✅ Рекомендуется** |
| Whisper Medium | 769M | ~10GB | ⚡⚡ | ~12-15% | High quality |
| Speech2Text Small | 31M | ~2GB | ⚡⚡⚡⚡ | ~25-30%* | Cross-lingual experiments |
| Custom Model | ~50M | ~3GB | ⚡⚡⚡⚡ | ~30-35%* | Research |

*Результаты на русском после fine-tuning (baseline без fine-tuning хуже)

## 📊 Метрики и оценка

Проект использует комплексный набор метрик для оценки качества моделей ASR:

### Основные метрики

- **WER (Word Error Rate)** - основная метрика качества распознавания
  - Измеряет процент ошибочно распознанных слов
  - Формула: `(Substitutions + Deletions + Insertions) / Total Words × 100%`
  - Реализация: библиотека `jiwer`

- **CER (Character Error Rate)** - детальная оценка на уровне символов
  - Полезна для языков с длинными словами или сложной морфологией
  - Формула: аналогична WER, но на уровне символов

- **BLEU (Bilingual Evaluation Understudy)** - метрика из машинного перевода
  - Оценивает n-gram overlap между предсказанием и эталоном
  - Реализация: HuggingFace `evaluate` библиотека

- **MER (Match Error Rate)** - альтернативная метрика точности слов
  - Похожа на WER, но по-другому обрабатывает замены

- **WIL (Word Information Lost)** - информационная метрика потерь
  - Учитывает не только количество, но и "вес" ошибок

### Детальный анализ ошибок

Скрипт `evaluation.py` с флагом `--detailed-analysis` предоставляет:

- **Error breakdown** по типам:
  - Substitutions (замены): какие слова были заменены
  - Deletions (пропуски): какие слова не были распознаны
  - Insertions (добавления): лишние слова в распознавании
  - Hits (правильные): корректно распознанные слова

- **Анализ по длительности аудио**:
  - WER/CER для коротких (<5s), средних (5-15s), длинных (>15s) записей
  - Выявление проблем с определенными длительностями

- **Анализ по длине текста**:
  - Метрики для коротких, средних, длинных транскриптов
  - Понимание влияния сложности текста

### Использование

**Базовая оценка:**
```bash
python evaluation.py --model-path experiments/whisper_ru_cv22_finetune/final_model
```

**С детальным анализом:**
```bash
python evaluation.py \
    --model-path experiments/whisper_ru_cv22_finetune/final_model \
    --detailed-analysis \
    --save-predictions
```

**Результаты оценки** сохраняются в `experiments/<model_name>_evaluation/`:
- `test_results.json` - основные метрики (WER, CER, BLEU, MER, WIL)
- `test_predictions.csv` - все предсказания модели
- `test_detailed_analysis.json` - детальный breakdown ошибок (если `--detailed-analysis`)
- TensorBoard логи со всеми метриками

## 🎛 Мониторинг

Проект поддерживает два инструмента мониторинга обучения:

### TensorBoard (по умолчанию, всегда включен)

TensorBoard автоматически логирует все метрики во время обучения и оценки.

**Запуск для обучения:**
```bash
# Для конкретного эксперимента
tensorboard --logdir experiments/whisper_ru_cv22_finetune/tensorboard

# Для всех экспериментов
tensorboard --logdir experiments
```

**Запуск для оценки:**
```bash
tensorboard --logdir experiments/whisper-small-no-finetune_evaluation
```

**Доступные метрики:**
- Training loss по шагам
- Evaluation metrics (WER, CER, BLEU) по эпохам
- Learning rate schedule
- Gradient norms (если включено)
- Error breakdown (substitutions, deletions, insertions, hits)

### Weights & Biases (опционально)

WandB предоставляет облачный мониторинг с продвинутой аналитикой.

**Настройка:**
```bash
# Установка (если еще не установлен)
conda run -n basenn pip install wandb

# Логин
wandb login

# Включить в конфигурации
```

**Конфигурация в `configs/default.yaml`:**
```yaml
logging:
  use_wandb: true
  wandb_project: "speech-to-text-ru"
  wandb_entity: "your-username"  # опционально
  report_to: ["tensorboard", "wandb"]
```

**Отключение WandB:**
```bash
# Через CLI флаг
python train.py --no-wandb

# Или в конфиге
logging:
  use_wandb: false
```

**Что логируется в WandB:**
- Все метрики из TensorBoard
- System metrics (GPU utilization, memory)
- Hyperparameters
- Model artifacts (чекпоинты)
- Примеры предсказаний (таблицы)
- Сравнение экспериментов

## 🧪 Эксперименты


### Ручной запуск экспериментов

**Whisper Small (рекомендуется):**
```bash
python train.py --config configs/whisper_small.yaml --experiment-name whisper_small_ru
```

**Whisper Base (быстрее, меньше VRAM):**
```bash
python train.py --config configs/whisper_base.yaml --experiment-name whisper_base_ru
```

**Speech2Text Cross-lingual (English→Russian):**
```bash
python train.py --config configs/s2t_cross_lingual.yaml --experiment-name s2t_xlingual_ru
```

**Custom model (экспериментальная архитектура):**
```bash
python train.py --config configs/custom_model.yaml --experiment-name custom_baseline
```

### Анализ результатов

**Через TensorBoard:**
```bash
# Сравнение всех экспериментов
tensorboard --logdir experiments
```

**Программный анализ:**
```python
from src.utils import ExperimentTracker
import json

# Загрузить результаты эксперимента
tracker = ExperimentTracker("experiments/whisper_small_ru")

# Построить графики обучения
tracker.plot_training_curves()

# Загрузить метрики
with open("experiments/whisper_small_ru/test_results.json") as f:
    results = json.load(f)
    print(f"WER: {results['wer']:.2f}%")
    print(f"CER: {results['cer']:.2f}%")
```

### Воспроизводимость результатов

Для обеспечения воспроизводимости экспериментов:

1. **Фиксированный seed** в конфигурации:
   ```yaml
   experiment:
     seed: 42
   ```

2. **Версионирование зависимостей** через `requirements.txt`

3. **Сохранение конфигурации** для каждого эксперимента:
   - `experiments/<experiment_name>/config.yaml`

4. **Git commit hash** логируется в WandB (если включен)

## ⚠️ Важные проблемы и решения

### КРИТИЧЕСКОЕ: Высокий loss при обучении (Dropout sensitivity)

**⚠️ Whisper модели КРАЙНЕ чувствительны к dropout в режиме обучения!**

**Симптомы:**
- Loss ~11-12 вместо ожидаемого 2-4
- Модель выдает бессмысленный текст или зацикливается
- В режиме eval работает нормально, проблема только в train

**Причина:**
Даже "обычные" значения dropout (0.1, 0.05) могут полностью разрушить обучение Whisper моделей.

**Решение:**
Установить **все dropout параметры в 0.0** в конфигурации:

```yaml
model:
  activation_dropout: 0.0  # Dropout for activation functions
  attention_dropout: 0.0   # Dropout for attention weights
  dropout: 0.0             # General dropout rate
```

**Проверка:**
- Loss для baseline модели (lr=0.0): должен быть ~2.5-4.0
- Если loss >10 - немедленно проверьте dropout!

**Регуляризация:**
Если нужна регуляризация - используйте `weight_decay` или label smoothing вместо dropout.

### Другие частые проблемы

#### 1. Out of Memory (CUDA OOM)

**Симптомы:**
- `RuntimeError: CUDA out of memory`
- Процесс убивается системой

**Решения:**
```yaml
training:
  train_batch_size: 4              # Уменьшить с 8 до 4
  gradient_accumulation_steps: 4   # Увеличить с 2 до 4 (эффективный batch = 16)
  fp16: true                       # Включить mixed precision
  eval_batch_size: 8               # Уменьшить evaluation batch
```

**Дополнительно:**
- Закрыть другие приложения, использующие GPU
- Использовать меньшую модель (Whisper Base вместо Small)
- Включить `use_cpu_offload: true` для очень больших моделей

#### 2. Медленная загрузка данных (DataLoader bottleneck)

**Симптомы:**
- GPU utilization < 80%
- Долгое ожидание между батчами

**Решения:**
```yaml
data:
  num_workers: 8                   # Увеличить количество воркеров
  pin_memory: true                 # Включить pinned memory
  processing_and_augmentation_device: "cpu"  # Обработка на CPU
```

**Проверка:**
- Убедитесь что `clip_durations.tsv` существует (ускоряет загрузку в 10+ раз)
- Запустите бенчмарк: `python tests/dataloader_speed.py`
- Optimal `num_workers` обычно равно количеству CPU cores

#### 3. Модель не улучшается / Loss не уменьшается

**Возможные причины и решения:**

1. **Слишком высокий learning rate:**
   ```yaml
   training:
     learning_rate: 1e-5  # Уменьшить с 1e-4
   ```

2. **Encoder заморожен слишком агрессивно:**
   ```yaml
   model:
     freeze_encoder: true
     unfreeze_last_n_encoder_layers: 2  # Разморозить последние 2 слоя
   ```

3. **Недостаточно эпох для обучения:**
   ```yaml
   training:
     num_train_epochs: 15  # Увеличить с 10
     use_early_stopping: true
     early_stopping_patience: 5  # Увеличить patience
   ```

4. **Проверьте градиенты в TensorBoard:**
   ```bash
   tensorboard --logdir experiments/your_experiment/tensorboard
   # Смотрите на gradient norms - должны быть стабильные, не NaN
   ```

#### 4. Датасет не найден

**Симптомы:**
- `FileNotFoundError: [Errno 2] No such file or directory: 'data/cv-corpus-22.0-2025-06-20/ru/train.tsv'`

**Решения:**
1. Проверьте структуру директорий:
   ```bash
   python setup_check.py
   ```

2. Убедитесь что датасет скачан и распакован:
   ```
   data/
   └── cv-corpus-22.0-2025-06-20/
       └── ru/
           ├── train.tsv
           ├── dev.tsv
           ├── test.tsv
           └── clips/
   ```

3. Проверьте путь в конфигурации:
   ```yaml
   data:
     data_dir: "data"
     dataset_path: "cv-corpus-22.0-2025-06-20/ru"
   ```

#### 5. Модель зацикливается или выдаёт повторяющийся текст

**Симптомы:**
- Prediction: "я я я я я..." или "привет привет привет..."
- Модель повторяет одни и те же фразы

**Решения:**
```yaml
evaluation:
  repetition_penalty: 1.2          # Штраф за повторения
  no_repeat_ngram_size: 3          # Блокировать повторяющиеся 3-граммы
  num_beams: 1                     # Greedy decoding обычно лучше для ASR
```

**Для Whisper моделей:**
- Проверьте что `language` и `task` правильно установлены в конфиге
- Убедитесь что processor создан с language и task параметрами

#### 6. WandB не работает / не логирует метрики

**Решения:**
1. Проверьте логин:
   ```bash
   wandb login
   ```

2. Включите в конфигурации:
   ```yaml
   logging:
     use_wandb: true
     wandb_project: "speech-to-text-ru"
   ```

3. Убедитесь что не используется флаг `--no-wandb`

4. Проверьте firewall/proxy настройки

**Альтернатива:** Используйте TensorBoard (всегда включен по умолчанию):
```bash
tensorboard --logdir experiments
```

#### 7. Тесты падают с ошибками путей

**Причина:** Тесты пытаются создать временные файлы вне проекта

**Решение:**
- Используйте fixture `temp_dir` из `tests/conftest.py`
- Все временные файлы должны создаваться в `tests/.test_tmp/`
- НЕ используйте `tempfile.TemporaryDirectory()` - см. CLAUDE.md

## 📈 Результаты

Результаты экспериментов на Mozilla Common Voice 22.0 Russian dataset (test split):

### Baseline модели (без fine-tuning)

| Модель | WER ↓ | CER ↓ | Параметры | Inference Speed |
|--------|-------|-------|-----------|-----------------|
| Whisper Tiny | ~35-40% | ~15-20% | 39M | ⚡⚡⚡⚡⚡ |
| Whisper Base | ~25-30% | ~10-15% | 74M | ⚡⚡⚡⚡ |
| Whisper Small | ~18-22% | ~7-10% | 244M | ⚡⚡⚡ |
| Whisper Medium | ~15-18% | ~6-8% | 769M | ⚡⚡ |
| Speech2Text (English) | ~90%+ | ~70%+ | 31M | ⚡⚡⚡⚡ |

*Speech2Text показывает плохие результаты на русском без fine-tuning, т.к. обучена только на английском*

### После fine-tuning (планируемые результаты)

| Модель | WER ↓ | CER ↓ | Параметры | Эпох обучения | VRAM |
|--------|-------|-------|-----------|---------------|------|
| Whisper Small (frozen encoder) | TBD | TBD | 244M | ~10 | ~5GB |
| Whisper Base (frozen encoder) | TBD | TBD | 74M | ~10 | ~3GB |
| Speech2Text (cross-lingual) | TBD | TBD | 31M | ~15 | ~2GB |
| Custom CNN+Transformer | TBD | TBD | ~50M | ~20 | ~3GB |

**Целевые метрики после fine-tuning:**
- **Whisper Small**: WER < 15%, CER < 6%
- **Speech2Text cross-lingual**: WER < 25%, CER < 12%
- **Custom model**: WER < 30%, CER < 15% (baseline для сравнения)

**Примечания:**
- Результаты зависят от гиперпараметров (LR, batch size, freezing strategy)
- WER/CER измеряются на test split (~10K записей)
- Все модели обучены на RTX 4070 Ti (12GB VRAM)
- Время обучения: Whisper Small ~6-8 часов на 10 эпох

*Таблица будет обновлена по мере завершения экспериментов*

### Сравнение стратегий обучения (Whisper Small)

| Стратегия | WER ↓ | CER ↓ | Время обучения | Примечания |
|-----------|-------|-------|----------------|------------|
| Frozen encoder + decoder training | TBD | TBD | ~6h | Рекомендуется |
| Full model fine-tuning | TBD | TBD | ~8h | Может переобучиться |
| Decoder only (frozen encoder) | TBD | TBD | ~5h | Быстро, но хуже качество |
| Last 2 encoder layers unfrozen | TBD | TBD | ~7h | Баланс качества и времени |

*Экспериментальные результаты для определения оптимальной стратегии*

## 🛠 Разработка

### Тестирование

Проект использует **pytest** для comprehensive testing suite.

**Запуск всех тестов:**
```bash
pytest tests/ -v
```

**Запуск конкретного теста:**
```bash
# Тесты DataManager
pytest tests/test_data_manager.py -v

# Конкретный тест с выводом
pytest tests/test_data_manager.py::TestDataManager::test_dataset_info -v -s

# Тесты моделей
pytest tests/test_models.py -v

# Тесты метрик
pytest tests/test_metrics.py -v
```

**Бенчмарк производительности DataLoader:**
```bash
python tests/dataloader_speed.py
```

**Структура тестов:**
- `tests/conftest.py` - pytest fixtures (`config`, `temp_dir`)
- `tests/test_*.py` - unit тесты для каждого модуля
- `tests/.test_tmp/` - временные файлы (автоматическая очистка)

### Линтинг и форматирование

Проект использует **ruff** для линтинга и форматирования (замена black + flake8 + isort).

**Проверка кода:**
```bash
# Линтинг
conda run -n basenn ruff check src/

# Автоисправление
conda run -n basenn ruff check --fix src/

# Форматирование
conda run -n basenn ruff format src/
```

**Проверка типов с mypy:**
```bash
# Проверка типов
conda run -n basenn mypy src/

# С автоустановкой type stubs
conda run -n basenn mypy --install-types src/
```

**Pre-commit hook (опционально):**
```bash
# Установка pre-commit
conda run -n basenn pip install pre-commit

# Настройка hooks
pre-commit install
```

### Добавление новой модели

1. **Создайте класс модели в `src/models.py`:**
   ```python
   class NewSTTModel(BaseSTTModel):
       def __init__(self, config):
           super().__init__(config)
           # Инициализация модели

       def forward(self, input_features, labels=None):
           # Forward pass
           pass

       def generate(self, input_features, **kwargs):
           # Генерация транскрипции
           pass
   ```

2. **Добавьте в `ModelFactory.create_model()`:**
   ```python
   if config.model.model_type == "new_model":
       return NewSTTModel(config)
   ```

3. **Создайте YAML конфиг в `configs/`:**
   ```yaml
   model:
     model_name: "author/model-name"
     model_type: "new_model"
     # Специфичные параметры
   ```

4. **Добавьте тесты в `tests/test_models.py`:**
   ```python
   def test_new_model_forward():
       # Тест forward pass
       pass

   def test_new_model_generate():
       # Тест генерации
       pass
   ```

5. **Обновите документацию:**
   - Добавьте описание в README.md (раздел "Поддерживаемые модели")
   - Обновите CLAUDE.md с деталями архитектуры

### Добавление новой метрики

1. **Реализуйте функцию метрики в `src/metrics.py`:**
   ```python
   def compute_new_metric(predictions: List[str], references: List[str]) -> float:
       # Вычисление метрики
       return score
   ```

2. **Добавьте в `STTMetrics.compute_all_metrics()`:**
   ```python
   new_metric = self.compute_new_metric(predictions, references)
   return MetricResult(
       wer=wer, cer=cer, bleu=bleu,
       new_metric=new_metric  # Добавить новую метрику
   )
   ```

3. **Обновите `MetricResult` dataclass:**
   ```python
   @dataclass
   class MetricResult:
       wer: float
       cer: float
       bleu: float
       new_metric: float  # Добавить поле
   ```

4. **Добавьте тесты в `tests/test_metrics.py`**

### Изменение конфигурации

1. **Добавьте поля в dataclass в `src/config.py`:**
   ```python
   @dataclass
   class ModelConfig:
       model_name: str
       model_type: str
       new_parameter: float = 1.0  # Добавить новый параметр
   ```

2. **Обновите default конфиг в `configs/default.yaml`:**
   ```yaml
   model:
     new_parameter: 1.0
   ```

3. **Используйте в коде:**
   ```python
   new_value = config.model.new_parameter
   ```

**Примечание:** НЕ добавляйте CLI аргументы для параметров конфигурации. Все настройки через YAML конфиги.

## 💼 Технологический стек

### Core ML/DL
- **PyTorch 2.5+** - Deep learning framework
- **torchaudio** - Audio processing
- **HuggingFace Transformers** - Pretrained models (Whisper, Speech2Text)
- **HuggingFace Datasets** - Dataset management

### Configuration & Experiment Management
- **OmegaConf** - Hierarchical configuration
- **Hydra** - Configuration composition
- **TensorBoard** - Training visualization
- **Weights & Biases** - Experiment tracking (optional)

### Data Processing & Augmentation
- **pandas** - Tabular data manipulation
- **numpy** - Numerical computations
- **librosa** - Audio feature extraction
- **Custom augmentation pipeline** (8+ augmentation types)

### Evaluation & Metrics
- **jiwer** - WER/CER/MER/WIL metrics
- **evaluate (HuggingFace)** - BLEU and other NLP metrics
- **Custom performance analyzer** - Error breakdown, duration/length analysis

### Development Tools
- **pytest** - Testing framework
- **mypy** - Static type checking
- **ruff** - Fast linting & formatting (replaces black/flake8/isort)
- **rich** - Beautiful terminal output
- **Jupyter** - Interactive development

### Production Ready
- **Separate scripts** for train/eval/inference
- **Comprehensive testing** (9+ test modules)
- **Type hints** throughout codebase
- **Detailed logging** with structured output
- **Reproducible experiments** (fixed seeds, config versioning)

## 🤝 Структура для портфолио

Этот проект демонстрирует профессиональный подход к ML/DL разработке:

### 1. Clean Code & Architecture
- ✅ **Модульная архитектура** с четким разделением ответственности
- ✅ **Type hints** во всем коде (mypy compatible)
- ✅ **Comprehensive documentation** (README, CLAUDE.md, docstrings)
- ✅ **SOLID принципы** (Factory pattern, dataclasses, protocols)
- ✅ **Понятные naming conventions** и структура проекта

### 2. MLOps Практики
- ✅ **Конфигурационное управление** (OmegaConf, YAML-based)
- ✅ **Experiment tracking** (TensorBoard + WandB)
- ✅ **Reproducibility** (fixed seeds, config versioning, requirements.txt)
- ✅ **Model versioning** (checkpoint management, best model selection)
- ✅ **Logging & monitoring** (structured logging, metrics tracking)
- ✅ **Production inference** (batch processing, RTF calculation)

### 3. Deep Learning Expertise
- ✅ **Transfer learning** (fine-tuning pretrained Whisper/Speech2Text)
- ✅ **Cross-lingual transfer** (English→Russian adaptation)
- ✅ **Custom architectures** (CNN + Transformer + CTC)
- ✅ **Advanced training techniques** (early stopping, LR scheduling, gradient accumulation)
- ✅ **Model optimization** (FP16 mixed precision, memory management)
- ✅ **Flexible freezing strategies** (fine-grained layer control)

### 4. Data Engineering
- ✅ **Efficient data loading** (PyTorch DataLoader, caching, multiprocessing)
- ✅ **Data augmentation pipeline** (8+ types: noise, speed, pitch, SpecAugment, reverb)
- ✅ **Audio preprocessing** (resampling, normalization, silence trimming)
- ✅ **Custom collation** (dynamic padding, CPU/GPU modes)
- ✅ **Dataset analysis** (EDA notebooks, statistics)

### 5. Evaluation & Metrics
- ✅ **Comprehensive metrics** (WER, CER, BLEU, MER, WIL)
- ✅ **Error analysis** (substitutions, deletions, insertions breakdown)
- ✅ **Performance segmentation** (by duration, text length)
- ✅ **Anti-repetition techniques** (repetition penalty, n-gram blocking)
- ✅ **Baseline comparisons** (pretrained vs fine-tuned)

### 6. Software Engineering
- ✅ **Comprehensive testing** (pytest, 9+ test modules, fixtures)
- ✅ **Code quality tools** (ruff linting, mypy type checking)
- ✅ **Git best practices** (.gitignore, structured commits)
- ✅ **CI/CD ready** (automated testing, reproducible environments)
- ✅ **Documentation** (README, inline comments, type hints)

### 7. Research & Experimentation
- ✅ **Model comparison framework** (run_experiments.py)
- ✅ **Hyperparameter search support** (multiple LR schedulers)
- ✅ **Ablation studies** (freezing strategies, augmentation impact)
- ✅ **Jupyter notebooks** (EDA, baseline preparation, debugging)
- ✅ **Результаты и выводы** (metrics tables, training curves)

**Этот проект готов к презентации как пример production-ready ML engineering.**

## 📝 Лицензия

MIT License - см. [LICENSE](LICENSE) файл для деталей.

## 🙏 Благодарности

- **Mozilla Foundation** - за открытый датасет Common Voice 22.0
- **OpenAI** - за архитектуру и pretrained модели Whisper
- **Meta AI (Facebook)** - за модели Speech2Text и исследования в области ASR
- **HuggingFace** - за библиотеку Transformers и экосистему ML инструментов
- **PyTorch Team** - за фреймворк PyTorch и torchaudio
- **Open Source Community** - за библиотеки jiwer, ruff, pytest и другие инструменты

## 📚 Полезные ссылки

- **Датасет**: [Mozilla Common Voice](https://commonvoice.mozilla.org/en/datasets)
- **Whisper Paper**: [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356)
- **Speech2Text**: [HuggingFace Model Card](https://huggingface.co/facebook/s2t-small-librispeech-asr)
- **HuggingFace Transformers**: [Documentation](https://huggingface.co/docs/transformers)
- **PyTorch**: [Official Website](https://pytorch.org/)

## 📧 Контакты

Для вопросов, предложений или сотрудничества:

- **GitHub Issues**: Создайте issue в репозитории
- **Email**: your.email@example.com (замените на свой)
- **LinkedIn**: [Your Profile](https://linkedin.com/in/yourprofile) (замените на свой)

---

<div align="center">

**Этот проект демонстрирует профессиональный подход к ML/DL разработке**
*Production-ready архитектура • MLOps практики • Comprehensive testing*

⭐ Если проект был полезен, поставьте звезду на GitHub!

</div>
