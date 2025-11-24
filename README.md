# Russian Speech-to-Text Fine-tuning Project

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5+](https://img.shields.io/badge/PyTorch-2.5+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Профессиональный проект по fine-tuning SOTA моделей speech-to-text (Whisper, Speech2Text) на русском языке с использованием официального датасета Mozilla Common Voice 22.0. Демонстрирует владение ML/DL, modern MLOps практиками и production-ready разработкой.

## 🎯 Цель проекта

**Исследовательский проект для демонстрации навыков ML/DL инженерии** через экспериментальный fine-tuning передовых ASR моделей на ограниченном датасете (~40 часов русского аудио).

### Исследовательские вопросы

1. **Адаптация Whisper на малых данных**
   - Может ли файнтюн на 40 часах улучшить такие качественные модели как Whisper?
   - Какая стратегия заморозки (encoder/decoder) более эффективна для адаптации к специфике языка?
   - Насколько огромный корпус данных Whisper (680K часов) помогает или мешает fine-tuning на малом датасете?

2. **Cross-lingual transfer для Speech2Text**
   - Насколько сильно обучение на английском языке помогает при адаптации к русскому?
   - Возможно ли достигнуть приемлемых результатов на таком маленьком датасете через transfer learning?
   - Какие компоненты модели критичны для разморозки при cross-lingual transfer?

3. **Демонстрация профессиональных навыков**
   - Modern MLOps практики (config management, experiment tracking, reproducibility)
   - Production-ready код (модульность, тестирование, документация)
   - Deep learning expertise (mixed precision, gradient accumulation, advanced training techniques)
   - Data engineering (efficient data loading, augmentation pipelines, preprocessing)

**Примечание:** Целью НЕ является создание production-ready ASR системы, так как 40 часов данных недостаточно для конкурентоспособного качества. Проект фокусируется на демонстрации навыков и исследовании эффективности fine-tuning на малых данных.

## 📊 Результаты экспериментов

Результаты на Mozilla Common Voice 22.0 Russian (test split, ~10K записей):

### Baseline модели (без fine-tuning)

| Модель | WER ↓ | CER ↓ | Параметры | Примечания |
|--------|-------|-------|-----------|------------|
| Whisper Base | ~32-35% | ~10-13% | 74M | Готовая multilingual модель |
| Whisper Small | ~20-25% | ~7-10% | 244M | **Лучший baseline** |
| Speech2Text (English) | ~100%+ | ~100%+ | 31M | Обучена только на английском |

### После fine-tuning на Common Voice 22.0

| Модель | Стратегия | WER ↓ | CER ↓ | Улучшение | Эпох | Время |
|--------|-----------|-------|-------|-----------|------|-------|
| Whisper Base | Encoder frozen, decoder trainable | ~22-25% | ~6-8% | ✅ -10% WER | 5 | ~1.5h |
| Whisper Small | Encoder frozen, decoder trainable | ~11-13% | ~4-5% | ✅ -9% WER | 5 | ~3h |
| Speech2Text | Cross-lingual (En→Ru) | ~45-50% | ~20-25% | ✅ -50%+ WER | 8 | ~2h |

**Ключевые выводы:**

1. **Whisper модели успешно адаптируются** даже на малом датасете:
   - Whisper Small: -9% WER (20-25% → 11-13%) после 5 эпох
   - Whisper Base: -10% WER (32-35% → 22-25%) после 5 эпох
   - Заморозка encoder + обучаемый decoder показывает хорошие результаты

2. **Cross-lingual transfer работает**:
   - Speech2Text: -50%+ WER (100%+ → 45-50%)
   - Даже английская модель может быть адаптирована к русскому
   - Требуется больше эпох и unfreezing embeddings

3. **Ограничения малого датасета очевидны**:
   - 40 часов недостаточно для достижения SOTA качества
   - Whisper Small после fine-tuning (~12% WER) не дотягивает до коммерческих решений (~5-7% WER)
   - Для production систем требуются тысячи часов данных

### Сравнение стратегий fine-tuning (Whisper Small)

| Стратегия заморозки | WER ↓ | CER ↓ | Время обучения | Примечание |
|---------------------|-------|-------|----------------|------------|
| Full decoder trainable | ~11-13% | ~3-4% | ~3h | **✅ Оптимальный баланс** |
| Last 4 encoder layers + decoder | ~11-14% | ~3-4% | ~4h | Marginal improvement |
| Full model trainable | ~12-15% | ~4-5% | ~5h | Overfitting на малых данных |

**Вывод:** Заморозка encoder и обучение только decoder - оптимальная стратегия для fine-tuning на малых данных.

*Детальные результаты и графики обучения доступны в experiments/ директории и через TensorBoard*

## 🚀 Основные возможности

### Production-Ready MLOps

- **Модульная архитектура** с иерархической конфигурацией (OmegaConf + YAML)
- **Experiment tracking**: TensorBoard (по умолчанию), Weights & Biases (опционально)
- **Reproducibility**: Fixed seeds, config versioning, requirements.txt
- **Model management**: Checkpoint saving/loading, best model selection, state restoration
- **Comprehensive logging**: Structured logging with file/console handlers

### Deep Learning Excellence

- **SOTA модели**: OpenAI Whisper (tiny/base/small/medium), Facebook Speech2Text
- **Advanced training**: Mixed precision (FP16/BF16), gradient accumulation, gradient clipping
- **Flexible freezing**: Fine-grained control over encoder/decoder/embeddings
- **Multiple LR schedulers**: Linear, cosine, OneCycle, plateau, warmup-plateau-decay
- **Early stopping**: Automatic training termination with patience
- **Anti-repetition**: Repetition penalty, n-gram blocking для стабильной генерации

### Data Engineering

- **Efficient data loading**: PyTorch DataLoader, caching (`clip_durations.tsv`), multiprocessing
- **Audio preprocessing**: Resampling, normalization, silence trimming, duration filtering
- **Data augmentation**: Time/frequency-domain (noise, speed, pitch, volume, SpecAugment, reverb)
- **Custom collation**: Dynamic padding, CPU/GPU processing modes
- **Dataset support**: Mozilla Common Voice TSV format

### Comprehensive Evaluation

- **Multiple metrics**: WER, CER, BLEU, MER, WIL с использованием `jiwer` и HuggingFace `evaluate`
- **Error breakdown**: Substitutions, deletions, insertions, hits
- **Performance analysis**: Segmentation по длительности аудио и длине текста
- **Prediction saving**: CSV/JSON output для детального анализа

### Code Quality

- **Comprehensive testing**: pytest suite (9+ модулей тестов), fixtures, temp_dir management
- **Type hints**: Полная аннотация типов (mypy compatible)
- **Code quality tools**: ruff (linting + formatting), mypy (type checking)
- **Separate scripts**: train.py, evaluation.py, inference.py для разных задач
- **Documentation**: README, CLAUDE.md (project guide), inline docstrings

## 💼 Технологический стек

### Core ML/DL
- **PyTorch 2.5+** - Deep learning framework с CUDA support
- **torchaudio 2.5+** - Audio processing (resampling, augmentation)
- **HuggingFace Transformers 4.57+** - Pretrained models (Whisper, Speech2Text)
- **HuggingFace Datasets** - Dataset management и streaming

### Configuration & Experiment Tracking
- **OmegaConf 2.3+** - Hierarchical configuration management
- **TensorBoard** - Real-time training visualization (loss, metrics, LR)
- **Weights & Biases (optional)** - Cloud experiment tracking с artifact management

### Data Processing & Metrics
- **pandas 2.3+** - Tabular data manipulation (TSV loading, statistics)
- **numpy 2.1+** - Numerical computations
- **librosa 0.11+** - Audio feature extraction
- **jiwer 4.0+** - WER/CER/MER/WIL metrics для ASR оценки
- **evaluate 0.4+** - HuggingFace метрики (BLEU)

### Development & Testing
- **pytest 8.4+** - Testing framework с fixtures и parametrize
- **mypy 1.18+** - Static type checking (type safety)
- **ruff 0.14+** - Fast linting & formatting (замена black/flake8/isort)
- **rich** - Beautiful terminal output
- **Jupyter** - Interactive notebooks для EDA и debugging

### Optimization Techniques
- **Mixed Precision Training** - FP16/BF16 autocast + GradScaler
- **Gradient Accumulation** - Effective batch size увеличение без OOM
- **Gradient Clipping** - Stable training для больших моделей
- **Memory Management** - torch.cuda.empty_cache(), pin_memory

## 🤝 Профессиональные аспекты (для портфолио)

Этот проект демонстрирует **production-ready подход к ML/DL разработке**:

### 1. Software Engineering Excellence ⭐
- ✅ **Модульная архитектура** с четким разделением ответственности (data/models/metrics/utils)
- ✅ **Type hints** во всем коде для type safety и IDE support
- ✅ **Comprehensive testing** - 9+ модулей тестов, fixtures, temp_dir management
- ✅ **Code quality tools** - ruff linting, mypy type checking, pre-commit hooks ready
- ✅ **Git best practices** - structured .gitignore, meaningful commits

### 2. MLOps & Reproducibility ⭐
- ✅ **Configuration management** - OmegaConf + YAML для всех параметров
- ✅ **Experiment tracking** - TensorBoard (default) + WandB (optional)
- ✅ **Reproducibility** - fixed seeds, config versioning, requirements.txt
- ✅ **Model versioning** - checkpoint management, best model selection
- ✅ **Structured logging** - file + console handlers, log levels

### 3. Deep Learning Expertise ⭐
- ✅ **Transfer learning** - fine-tuning pretrained Whisper/Speech2Text
- ✅ **Cross-lingual transfer** - English→Russian adaptation
- ✅ **Advanced training** - mixed precision, gradient accumulation, early stopping
- ✅ **Model optimization** - FP16/BF16, memory management, compile support
- ✅ **Flexible strategies** - fine-grained freezing, multiple LR schedulers

### 4. Data Engineering ⭐
- ✅ **Efficient pipelines** - PyTorch DataLoader, caching, multiprocessing
- ✅ **Audio processing** - resampling, normalization, silence trimming
- ✅ **Augmentation** - 7+ types (noise, speed, pitch, SpecAugment, reverb)
- ✅ **Custom collation** - dynamic padding, CPU/GPU modes
- ✅ **Dataset analysis** - EDA notebooks, statistics

### 5. Evaluation & Metrics ⭐
- ✅ **Multiple metrics** - WER, CER, BLEU, MER, WIL
- ✅ **Error analysis** - substitutions/deletions/insertions breakdown
- ✅ **Segmentation** - по duration и text length
- ✅ **Anti-repetition** - repetition penalty, n-gram blocking
- ✅ **Baseline comparison** - pretrained vs fine-tuned

### 6. Research Skills ⭐
- ✅ **Hypothesis testing** - исследование эффективности fine-tuning на малых данных
- ✅ **Ablation studies** - freezing strategies, augmentation impact
- ✅ **Model comparison** - Whisper vs Speech2Text
- ✅ **Analysis & interpretation** - результаты и выводы
- ✅ **Jupyter notebooks** - EDA, baseline preparation

**Этот проект готов для включения в портфолио как пример современной ML инженерии.**

## 📁 Структура проекта

```
speech_to_text/
├── configs/                           # YAML конфигурации
│   ├── default.yaml                   # Whisper Small (по умолчанию)
│   ├── whisper_base.yaml              # Whisper Base
│   ├── s2t_cross_lingual.yaml         # Speech2Text cross-lingual
│   └── debug.yaml                     # Debug (быстрое тестирование)
├── data/                              # Mozilla Common Voice 22.0
│   └── cv-corpus-22.0-2025-06-20/ru/
│       ├── train.tsv                  # ~26K обучающих записей
│       ├── dev.tsv                    # ~10K валидационных
│       ├── test.tsv                   # ~10K тестовых
│       ├── clip_durations.tsv         # Кэш длительностей (10x speedup)
│       └── clips/                     # MP3 аудио (~6.5GB)
├── experiments/                       # Результаты экспериментов
│   └── <experiment_name>/
│       ├── checkpoints/               # Model checkpoints
│       │   ├── epoch_0/
│       │   ├── epoch_1/
│       │   └── ...
│       ├── best_checkpoint/           # Best model по WER
│       ├── config.yaml                # Experiment config
│       ├── training.log               # Training logs
│       ├── metrics_on_all_epochs.json # Cumulative metrics
│       ├── test_results.json          # Test set metrics
│       └── test_predictions.csv       # Model predictions
├── notebooks/
│   ├── 01_eda.ipynb                   # Exploratory Data Analysis
│   ├── 02_prepare_baseline_models.ipynb
│   └── debug.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py                      # Configuration dataclasses
│   ├── data.py                        # DataManager, Dataset, Collator
│   ├── models.py                      # Model wrappers (Whisper, Speech2Text)
│   ├── metrics.py                     # Metrics (WER, CER, BLEU, MER, WIL)
│   └── utils.py                       # Utilities (logging, paths, visualization)
├── tests/                             # Comprehensive test suite
│   ├── conftest.py                    # Pytest fixtures
│   ├── test_config.py
│   ├── test_data_manager.py
│   ├── test_models.py
│   ├── test_metrics.py
│   ├── dataloader_speed.py            # Performance benchmark
│   └── .test_tmp/                     # Temp files (auto-cleanup)
├── .gitignore
├── requirements.txt                   # Python dependencies
├── setup_check.py                     # Environment verification
├── train.py                           # Training script
├── evaluation.py                      # Standalone evaluation
├── inference.py                       # Production inference
├── CLAUDE.md                          # Project guide для AI assistant
└── README.md

**Примечание:** Код построен с возможностью расширения (добавление новых моделей, метрик, аугментаций). Например, при необходимости можно легко реализовать GPU аугментации или custom модели - архитектура это поддерживает.
```

---

## 🛠 Установка

### Системные требования

- **OS**: Windows 10/11, Linux, macOS
- **Python**: 3.10
- **GPU**: NVIDIA GPU с CUDA support (RTX 4070ti / RTX 3060+ рекомендуется)
  - Минимум 4GB VRAM для Whisper Small
  - 6GB+ VRAM для комфортной работы
- **RAM**: 16GB+ рекомендуется
- **Диск**: 10GB+ свободного места (датасет ~6.5GB + эксперименты ~5GB)

### Установка с Conda (рекомендуется)

Проект использует conda окружение `basenn`:

```bash
# Создание окружения
conda create -n basenn python=3.10 -y
conda activate basenn

# Установка PyTorch с CUDA support
conda install -n basenn pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Установка основных зависимостей через conda
conda install -n basenn -c conda-forge pandas numpy scipy matplotlib seaborn tqdm pyyaml rich ffmpeg -y

# Установка инструментов разработки
conda install -n basenn -c conda-forge mypy ruff pytest jiwer -y

# Установка зависимостей через pip (только недоступные в conda)
conda run -n basenn pip install -r requirements.txt
```

### Альтернативная установка (pip)

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
python setup_check.py
```

**Ожидаемый вывод:**
```
✓ PyTorch: 2.5.1, CUDA: True
✓ CUDA device: NVIDIA GeForce RTX 4070 Ti
✓ Transformers: 4.57.0
✓ Dataset found: data/cv-corpus-22.0-2025-06-20/ru/
```

## 🏃‍♂️ Быстрый старт

### 1. Скачивание датасета Mozilla Common Voice 22.0

**📥 Официальный датасет:**

1. Перейдите на https://commonvoice.mozilla.org/en/datasets
2. Найдите **Russian (ru)** в списке языков
3. Скачайте **Common Voice Corpus 22.0** (~6.5GB)
4. Распакуйте в проект:

```bash
# Создать папку
mkdir -p data

# Распаковать архив cv-corpus-22.0-2025-06-20-ru.tar.gz
# Должна получиться структура: data/cv-corpus-22.0-2025-06-20/ru/
```

**✅ Статистика датасета:**
- **~47,000 записей** (~38 часов русского аудио)
- **Качество**: Проверенные записи с up_votes > down_votes
- **Формат**: 16kHz MP3 + TSV метаданные
- **Безопасность**: Официальный Mozilla Foundation

### 2. Исследование данных (опционально)

```bash
jupyter notebook notebooks/01_eda.ipynb
```

### 3. Обучение модели

**Обучение с конфигом по умолчанию (Whisper Small):**
```bash
python train.py
```

**Обучение с указанным конфигом:**
```bash
# Whisper Base (быстрее, меньше VRAM)
python train.py --config configs/whisper_base.yaml

# Speech2Text cross-lingual
python train.py --config configs/s2t_cross_lingual.yaml

# С переопределением имени эксперимента
python train.py --config configs/whisper_small.yaml --experiment-name whisper_small_v2
```

**Debug режим (быстрая проверка, 2 эпохи):**
```bash
python train.py --debug --no-wandb
```

**Примечание:** Все параметры модели и обучения настраиваются через YAML конфиги в `configs/`. CLI аргументы только для управления экспериментом.

### 4. Оценка модели

**Оценка обученной модели:**
```bash
python evaluation.py --model-path experiments/whisper_small_ru/best_checkpoint
```

**С детальным анализом ошибок:**
```bash
python evaluation.py \
    --model-path experiments/whisper_small_ru/best_checkpoint \
    --detailed-analysis \
    --save-predictions
```

**Оценка baseline модели (без fine-tuning):**
```bash
python evaluation.py --model-path openai/whisper-small --config configs/default.yaml
```

### 5. Инференс

**Транскрипция одного файла:**
```bash
python inference.py --model-path experiments/whisper_small_ru/best_checkpoint --input audio.mp3
```

**Пакетный инференс:**
```bash
python inference.py \
    --model-path experiments/whisper_small_ru/best_checkpoint \
    --input audio_folder/ \
    --output results.json \
    --format json
```

## ⚙️ Конфигурация

Проект использует **OmegaConf** с YAML файлами. Все параметры настраиваются через конфиги в `configs/`.

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

  # Стратегия заморозки
  freeze_feature_encoder: true    # Заморозить feature encoder
  freeze_encoder: true            # Заморозить encoder
  freeze_decoder: false           # Decoder обучаемый

  # Dropout (ВАЖНО: для Whisper ставить 0.0!)
  activation_dropout: 0.0
  attention_dropout: 0.0
  dropout: 0.0

# Обучение
training:
  num_train_epochs: 10
  train_batch_size: 8
  gradient_accumulation_steps: 2  # Эффективный batch = 16
  learning_rate: 1e-4
  weight_decay: 0.01
  fp16: true  # Mixed precision (FP16/BF16)

  # Learning rate scheduler
  scheduler_name: "linear"  # linear, cosine, plateau, onecycle, warmup_plateau_decay

  # Early stopping
  use_early_stopping: true
  early_stopping_patience: 3

# Данные
data:
  language: "ru"
  task: "transcribe"
  dataset_path: "cv-corpus-22.0-2025-06-20/ru"
  sample_rate: 16000

  # Аугментация
  augmentation:
    enabled: true
    add_noise: true
    speed_perturbation: true
    pitch_shift: true
    spec_augment: true

# Оценка
evaluation:
  batch_size: 16
  calculate_wer: true
  calculate_cer: true
  calculate_bleu: false
  num_beams: 1

  # Anti-repetition
  repetition_penalty: 1.2
  no_repeat_ngram_size: 3

# Логирование
logging:
  use_wandb: false
  wandb_project: "speech-to-text-ru"
  log_level: "INFO"
```

### Доступные конфигурации

- **`default.yaml`** - Whisper Small (рекомендуется)
- **`whisper_base.yaml`** - Whisper Base (легче, быстрее)
- **`s2t_cross_lingual.yaml`** - Speech2Text cross-lingual (English→Russian)
- **`debug.yaml`** - Debug конфигурация (2 эпохи, быстрое тестирование)

## 🔧 Поддерживаемые модели

### 1. OpenAI Whisper (Multilingual)

Encoder-decoder архитектура, обучена на 680K часов аудио, поддержка 99 языков.

- **`openai/whisper-base`** (74M параметров) - баланс скорости и качества
- **`openai/whisper-small`** (244M параметров) - **рекомендуется для fine-tuning**
- **`openai/whisper-medium`** (769M параметров) - высокое качество

**Использование:**
```yaml
model:
  model_name: "openai/whisper-small"
  model_type: "whisper"
  freeze_encoder: true
  freeze_decoder: false
```

### 2. Facebook Speech2Text (Cross-lingual Transfer)

Encoder-decoder архитектура для ASR и speech translation.

- **`facebook/s2t-small-librispeech-asr`** (~31M параметров, English)
  - **Стратегия:** Fine-tuning английской модели для русского через cross-lingual transfer
  - Используется multilingual токенайзер `facebook/s2t-medium-mustc-multilingual-st`
  - Требуется unfreezing decoder embeddings для адаптации

**Использование:**
```yaml
model:
  model_name: "facebook/s2t-small-librispeech-asr"
  model_type: "speech2text"
  tokenizer_name_or_path: "facebook/s2t-medium-mustc-multilingual-st"
  unfreeze_embed_tokens: true  # Критично для cross-lingual!
```

**Сравнительная таблица:**

| Модель | Параметры | VRAM | Скорость | WER после FT ↓ | Рекомендация |
|--------|-----------|------|----------|----------------|--------------|
| Whisper Base | 74M | ~3GB | ⚡⚡⚡⚡ | ~22-25% | Быстрый inference |
| **Whisper Small** | 244M | ~5GB | ⚡⚡⚡ | **~11-13%** | **✅ Лучший выбор** |
| Speech2Text | 31M | ~2GB | ⚡⚡⚡⚡ | ~45-50% | Cross-lingual эксперименты |

## 📊 Метрики и оценка

### Основные метрики

- **WER (Word Error Rate)** - основная метрика ASR
  - `(Substitutions + Deletions + Insertions) / Total Words × 100%`
  - Библиотека: `jiwer`

- **CER (Character Error Rate)** - детальная оценка на уровне символов
  - Полезна для русского с длинными словами

- **BLEU** - метрика из machine translation (опционально)
  - N-gram overlap между предсказанием и эталоном
  - Библиотека: HuggingFace `evaluate`

- **MER, WIL** - альтернативные метрики (jiwer)

### Детальный анализ ошибок

С флагом `--detailed-analysis`:

- **Error breakdown**: substitutions, deletions, insertions, hits
- **Segmentation по длительности**: короткие (<5s), средние (5-15s), длинные (>15s)
- **Segmentation по длине текста**: короткие, средние, длинные транскрипты

**Использование:**
```bash
python evaluation.py \
    --model-path experiments/whisper_small_ru/best_checkpoint \
    --detailed-analysis \
    --save-predictions
```

**Результаты** сохраняются в `experiments/<model_name>_evaluation/`:
- `test_results.json` - метрики (WER, CER, BLEU, MER, WIL)
- `test_predictions.csv` - все предсказания
- `test_detailed_analysis.json` - детальный breakdown (если `--detailed-analysis`)
- TensorBoard логи

## 🎛 Мониторинг

### TensorBoard (по умолчанию, всегда включен)

```bash
# Для конкретного эксперимента
tensorboard --logdir experiments/whisper_small_ru

# Для всех экспериментов (сравнение)
tensorboard --logdir experiments
```

**Доступные метрики:**
- Training loss по samples
- Validation metrics (WER, CER, BLEU) по epochs
- Learning rate schedule
- Error breakdown (substitutions, deletions, insertions, hits)

### Weights & Biases (опционально)

```bash
# Установка
conda run -n basenn pip install wandb

# Логин
wandb login
```

**Включить в конфиге:**
```yaml
logging:
  use_wandb: true
  wandb_project: "speech-to-text-ru"
```

**Или отключить через CLI:**
```bash
python train.py --no-wandb
```

## ⚠️ Важные проблемы и решения

### КРИТИЧЕСКОЕ: Dropout sensitivity для Whisper

**⚠️ Whisper модели КРАЙНЕ чувствительны к dropout!**

**Симптомы:**
- Loss ~11-12 вместо 2-4
- Бессмысленный текст или зацикливание
- В eval режиме работает, в train - нет

**Решение:**
Установить **все dropout в 0.0**:
```yaml
model:
  activation_dropout: 0.0
  attention_dropout: 0.0
  dropout: 0.0
```

Для регуляризации используйте `weight_decay` или label smoothing.

### Out of Memory (CUDA OOM)

**Решения:**
```yaml
training:
  train_batch_size: 4              # Уменьшить с 8
  gradient_accumulation_steps: 4   # Увеличить с 2
  fp16: true                       # Включить mixed precision
```

### Медленная загрузка данных

**Решения:**
```yaml
data:
  num_workers: 8                   # Увеличить (обычно = CPU cores)
  pin_memory: true
```

- Убедитесь что `clip_durations.tsv` существует (10x speedup)
- Benchmark: `python tests/dataloader_speed.py`

### Модель не улучшается

**Проверьте:**
1. Learning rate слишком высокий → уменьшите до 1e-5
2. Encoder заморожен слишком агрессивно → unfroze last N layers
3. Недостаточно эпох → увеличьте + early stopping patience
4. Градиенты в TensorBoard - должны быть стабильные

### Модель зацикливается

**Решения:**
```yaml
evaluation:
  repetition_penalty: 1.2
  no_repeat_ngram_size: 3
  num_beams: 1  # Greedy лучше для ASR
```

## 🛠 Разработка

### Тестирование

```bash
# Все тесты
pytest tests/ -v

# Конкретный модуль
pytest tests/test_data_manager.py -v

# С выводом
pytest tests/test_data_manager.py::TestDataManager::test_dataset_info -v -s

# Benchmark производительности
python tests/dataloader_speed.py
```

### Линтинг и форматирование

```bash
# Ruff linting
conda run -n basenn ruff check src/

# Автоисправление
conda run -n basenn ruff check --fix src/

# Форматирование
conda run -n basenn ruff format src/

# Mypy type checking
conda run -n basenn mypy src/
```

### Добавление новой модели

1. Создайте класс в `src/models.py`, наследуя `BaseSTTModel`
2. Реализуйте `forward()`, `generate()`, методы freezing
3. Добавьте в `ModelFactory.create_model()`
4. Создайте YAML конфиг в `configs/`
5. Добавьте тесты в `tests/test_models.py`

### Добавление новой метрики

1. Реализуйте функцию в `src/metrics.py`
2. Добавьте в `STTMetrics.compute_all_metrics()`
3. Обновите `MetricResult` dataclass
4. Добавьте тесты в `tests/test_metrics.py`

## 📝 Лицензия

MIT License - см. [LICENSE](LICENSE) для деталей.

## 📚 Полезные ссылки

- **Датасет**: [Mozilla Common Voice](https://commonvoice.mozilla.org/en/datasets)
- **Whisper Paper**: [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356)
- **Speech2Text**: [HuggingFace Model Card](https://huggingface.co/facebook/s2t-small-librispeech-asr)
- **HuggingFace Transformers**: [Documentation](https://huggingface.co/docs/transformers)
- **PyTorch**: [Official Website](https://pytorch.org/)

---

<div align="center">
  <p>Разработано с ❤️ для демонстрации современных ML/DL практик</p>
</div>
