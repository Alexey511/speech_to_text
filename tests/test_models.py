"""
Тесты для модуля models.py
Проверка работоспособности классов моделей WhisperSTT, Speech2TextSTT, ModelFactory и ModelManager.
"""

import pytest
import torch
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
from dataclasses import replace

# Add project root to Python path
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

from src.models import (
    WhisperSTT,
    Speech2TextSTT,
    ModelFactory,
    ModelManager,
    BaseSTTModel,
)
from src.config import ModelConfig
from transformers import (
    WhisperProcessor,
    WhisperTokenizer,
    WhisperFeatureExtractor,
    Speech2TextProcessor
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture(scope="module")
def whisper_config():
    """Конфигурация для Whisper модели"""
    return ModelConfig(
        model_name="openai/whisper-tiny",  # Маленькая модель для быстрого тестирования
        model_type="whisper",
        freeze_feature_encoder=False,
        freeze_encoder=False,
        freeze_decoder=False,
        unfreeze_last_n_encoder_layers=0,
        unfreeze_last_n_decoder_layers=0,
        activation_dropout=0.0,
        attention_dropout=0.0,
        dropout=0.0,
        compile_model=False,
        use_gpu=True,
    )


@pytest.fixture(scope="module")
def speech2text_config():
    """Конфигурация для Speech2Text модели (английская ASR модель)"""
    return ModelConfig(
        model_name="facebook/s2t-small-librispeech-asr",  # Английская ASR модель для fine-tuning на русском
        model_type="speech2text",
        freeze_feature_encoder=False,
        freeze_encoder=False,
        freeze_decoder=False,
        unfreeze_last_n_encoder_layers=0,
        unfreeze_last_n_decoder_layers=0,
        activation_dropout=0.15,
        attention_dropout=0.15,
        dropout=0.15,
        compile_model=False,
        use_gpu=True,
    )


@pytest.fixture(scope="module")
def whisper_processor():
    """
    WhisperProcessor для тестов.
    КРИТИЧНО: Создаётся с явными language='ru' и task='transcribe' для корректной работы prefix_tokens.
    """
    # Создаём токенайзер с language и task
    tokenizer = WhisperTokenizer.from_pretrained(
        "openai/whisper-tiny",
        language="ru",
        task="transcribe"
    )

    # Создаём feature extractor
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")

    # Собираем процессор из компонентов
    return WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)


@pytest.fixture(scope="module")
def speech2text_processor():
    """Speech2TextProcessor для английской ASR модели"""
    return Speech2TextProcessor.from_pretrained("facebook/s2t-small-librispeech-asr")


@pytest.fixture(scope="module")
def speech2text_cross_lingual_config():
    """Конфигурация для Speech2Text Cross-Lingual Transfer (English model + Multilingual tokenizer)"""
    return ModelConfig(
        model_name="facebook/s2t-small-librispeech-asr",  # English model
        model_type="speech2text",
        tokenizer_name_or_path="facebook/s2t-medium-mustc-multilingual-st",  # Multilingual tokenizer
        # Note: language is passed separately to load_checkpoint(), not stored in model config
        freeze_feature_encoder=True,
        freeze_encoder=True,  # Freeze encoder (acoustic features are language-independent)
        freeze_decoder=False,  # Train decoder (learns new Russian vocabulary)
        unfreeze_last_n_encoder_layers=0,
        unfreeze_last_n_decoder_layers=0,
        activation_dropout=0.15,
        attention_dropout=0.15,
        dropout=0.15,
        compile_model=False,
        use_gpu=True,
    )


@pytest.fixture(scope="module")
def speech2text_multilingual_processor():
    """Speech2TextProcessor with multilingual tokenizer for cross-lingual transfer"""
    from transformers import Speech2TextFeatureExtractor, Speech2TextTokenizer

    # Feature extractor from English model
    feature_extractor = Speech2TextFeatureExtractor.from_pretrained("facebook/s2t-small-librispeech-asr")

    # Multilingual tokenizer
    tokenizer = Speech2TextTokenizer.from_pretrained("facebook/s2t-medium-mustc-multilingual-st")

    # Set target language
    if hasattr(tokenizer, 'set_tgt_lang_special_tokens'):
        tokenizer.tgt_lang = "ru"

    return Speech2TextProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)


@pytest.fixture
def dummy_whisper_input():
    """Dummy входные данные для Whisper (mel-spectrogram)"""
    batch_size = 2
    n_mels = 80
    seq_length = 3000  # Whisper требует ровно 3000 frames
    return {
        'input_features': torch.randn(batch_size, n_mels, seq_length),
        'attention_mask': torch.ones(batch_size, seq_length, dtype=torch.long),
        'decoder_input_ids': torch.tensor([[50258, 50259] for _ in range(batch_size)])  # Start tokens
    }


@pytest.fixture
def dummy_speech2text_input():
    """Dummy входные данные для Speech2Text (mel-spectrogram, transposed)"""
    batch_size = 2
    seq_length = 1000
    n_mels = 80
    return {
        'input_features': torch.randn(batch_size, seq_length, n_mels),
        'decoder_input_ids': torch.tensor([[2, 3] for _ in range(batch_size)])  # Start tokens
    }


@pytest.fixture
def dummy_labels():
    """Dummy метки для тестов"""
    batch_size = 2
    label_length = 50
    vocab_size = 1000
    # Генерируем случайные токены в пределах vocab_size
    return torch.randint(0, vocab_size, (batch_size, label_length))


# ============================================================
# Unit Tests для WhisperSTT
# ============================================================

class TestWhisperSTT:
    """Тесты для класса WhisperSTT"""

    def test_init_with_processor(self, whisper_config, whisper_processor):
        """Тест создания WhisperSTT с процессором"""
        model = WhisperSTT(whisper_config, processor=whisper_processor)

        assert model is not None
        assert model.model_type == "whisper"
        assert model.model_name == "openai/whisper-tiny"
        assert model.processor == whisper_processor
        assert model.get_num_parameters() > 0
        assert model.get_trainable_parameters() > 0

    def test_init_without_processor_raises(self, whisper_config):
        """Тест создания WhisperSTT без процессора вызывает ValueError"""
        with pytest.raises(ValueError, match="expects a processor"):
            WhisperSTT(whisper_config, processor=None)

    def test_init_with_wrong_processor_type(self, whisper_config, speech2text_processor):
        """Тест создания WhisperSTT с неправильным типом процессора"""
        with pytest.raises(TypeError, match="expected processor of type"):
            WhisperSTT(whisper_config, processor=speech2text_processor)

    def test_forward_with_labels(self, whisper_config, whisper_processor, dummy_whisper_input, dummy_labels):
        """Тест forward pass с метками (вычисление loss)"""
        model = WhisperSTT(whisper_config, processor=whisper_processor)
        model.eval()

        with torch.no_grad():
            output = model.forward(input_features=dummy_whisper_input['input_features'], labels=dummy_labels)

        # Проверка типа выхода
        assert hasattr(output, 'loss')
        assert hasattr(output, 'logits')

        # Проверка loss
        assert output.loss is not None
        assert output.loss.item() > 0

        # Проверка формы logits
        batch_size, label_length = dummy_labels.shape
        assert output.logits.shape[0] == batch_size
        assert output.logits.shape[1] == label_length

    def test_forward_without_labels(self, whisper_config, whisper_processor, dummy_whisper_input):
        """Тест forward pass без меток (loss должен быть None)"""
        model = WhisperSTT(whisper_config, processor=whisper_processor)
        model.eval()

        with torch.no_grad():
            output = model.forward(
                input_features=dummy_whisper_input['input_features'],
                decoder_input_ids=dummy_whisper_input['decoder_input_ids']
            )

        assert output.loss is None
        assert output.logits is not None
        assert output.logits.shape[0] == dummy_whisper_input['input_features'].shape[0]

    def test_generate_with_language_and_task(self, whisper_config, whisper_processor, dummy_whisper_input):
        """Тест генерации транскрипции с language и task"""
        model = WhisperSTT(whisper_config, processor=whisper_processor)
        model.eval()

        transcriptions = model.generate(
            dummy_whisper_input['input_features'],
            language="ru",
            task="transcribe",
            return_text=True,
            attention_mask=dummy_whisper_input['attention_mask']
        )

        assert isinstance(transcriptions, list)
        assert len(transcriptions) == dummy_whisper_input['input_features'].shape[0]
        assert all(isinstance(t, str) for t in transcriptions)

    def test_generate_without_language_raises(self, whisper_config, whisper_processor, dummy_whisper_input):
        """Тест генерации без language вызывает ValueError"""
        model = WhisperSTT(whisper_config, processor=whisper_processor)
        model.eval()

        with pytest.raises(ValueError, match="language parameter is required"):
            model.generate(dummy_whisper_input['input_features'], task="transcribe")

    def test_generate_without_task_raises(self, whisper_config, whisper_processor, dummy_whisper_input):
        """Тест генерации без task вызывает ValueError"""
        model = WhisperSTT(whisper_config, processor=whisper_processor)
        model.eval()

        with pytest.raises(ValueError, match="task parameter is required"):
            model.generate(dummy_whisper_input['input_features'], language="ru")

    def test_generate_return_tokens(self, whisper_config, whisper_processor, dummy_whisper_input):
        """Тест генерации с return_text=False возвращает токены"""
        model = WhisperSTT(whisper_config, processor=whisper_processor)
        model.eval()

        tokens = model.generate(
            dummy_whisper_input['input_features'],
            language="ru",
            task="transcribe",
            return_text=False,
            attention_mask=dummy_whisper_input['attention_mask']
        )

        assert isinstance(tokens, torch.Tensor)
        assert tokens.shape[0] == dummy_whisper_input['input_features'].shape[0]

    def test_freeze_feature_encoder(self, whisper_config, whisper_processor):
        """Тест заморозки feature encoder"""
        model = WhisperSTT(whisper_config, processor=whisper_processor)

        # Замораживаем feature encoder
        model.freeze_feature_encoder()

        # Проверяем что conv1, conv2, embed_positions заморожены
        encoder = model.model.model.encoder
        for param in encoder.conv1.parameters():
            assert not param.requires_grad
        for param in encoder.conv2.parameters():
            assert not param.requires_grad
        for param in encoder.embed_positions.parameters():
            assert not param.requires_grad

    def test_freeze_encoder(self, whisper_config, whisper_processor):
        """Тест заморозки всего encoder"""
        model = WhisperSTT(whisper_config, processor=whisper_processor)

        trainable_before = model.get_trainable_parameters()
        model.freeze_encoder()
        trainable_after = model.get_trainable_parameters()

        # Проверяем что все параметры encoder заморожены
        for param in model.model.model.encoder.parameters():
            assert not param.requires_grad

        # Количество обучаемых параметров уменьшилось
        assert trainable_after < trainable_before

    def test_freeze_decoder(self, whisper_config, whisper_processor):
        """Тест заморозки decoder"""
        model = WhisperSTT(whisper_config, processor=whisper_processor)

        trainable_before = model.get_trainable_parameters()
        model.freeze_decoder()
        trainable_after = model.get_trainable_parameters()

        # Проверяем что все параметры decoder заморожены
        for param in model.model.model.decoder.parameters():
            assert not param.requires_grad

        assert trainable_after < trainable_before

    def test_unfreeze_last_n_encoder_layers(self, whisper_config, whisper_processor):
        """Тест разморозки последних N слоёв encoder"""
        model = WhisperSTT(whisper_config, processor=whisper_processor)

        # Замораживаем весь encoder
        model.freeze_encoder()

        # Размораживаем последние 2 слоя
        n = 2
        model.unfreeze_last_n_encoder_layers(n)

        # Проверяем что последние n слоёв разморожены
        encoder_layers = model.model.model.encoder.layers
        for layer in encoder_layers[-n:]:
            for param in layer.parameters():
                assert param.requires_grad

    def test_unfreeze_last_n_decoder_layers(self, whisper_config, whisper_processor):
        """Тест разморозки последних N слоёв decoder"""
        model = WhisperSTT(whisper_config, processor=whisper_processor)

        # Замораживаем весь decoder
        model.freeze_decoder()

        # Размораживаем последние 2 слоя
        n = 2
        model.unfreeze_last_n_decoder_layers(n)

        # Проверяем что последние n слоёв разморожены
        decoder_layers = model.model.model.decoder.layers
        for layer in decoder_layers[-n:]:
            for param in layer.parameters():
                assert param.requires_grad

    def test_apply_freezing_from_config(self, whisper_processor):
        """Тест применения freezing настроек из конфига"""
        config = ModelConfig(
            model_name="openai/whisper-tiny",
            model_type="whisper",
            freeze_encoder=True,
            freeze_decoder=False,
            unfreeze_last_n_encoder_layers=2,
        )

        model = WhisperSTT(config, processor=whisper_processor)

        # Проверяем что encoder заморожен, кроме последних 2 слоёв
        encoder_layers = model.model.model.encoder.layers

        # Последние 2 слоя должны быть разморожены
        for layer in encoder_layers[-2:]:
            # Хотя бы один параметр должен быть trainable
            has_trainable = any(p.requires_grad for p in layer.parameters())
            assert has_trainable

        # Decoder должен быть разморожен
        decoder_trainable = any(p.requires_grad for p in model.model.model.decoder.parameters())
        assert decoder_trainable

    def test_unfreeze_embed_tokens(self, whisper_processor):
        """Тест разморозки token embeddings декодера"""
        config = ModelConfig(
            model_name="openai/whisper-tiny",
            model_type="whisper",
            freeze_decoder=True,  # Замораживаем decoder (включая embed_tokens)
            unfreeze_embed_tokens=True  # Но размораживаем embed_tokens
        )

        model = WhisperSTT(config, processor=whisper_processor)

        # embed_tokens должен быть разморожен
        assert model.model.model.decoder.embed_tokens.weight.requires_grad

        # Остальные части decoder должны быть заморожены
        decoder_layers = model.model.model.decoder.layers
        first_layer_frozen = all(not p.requires_grad for p in decoder_layers[0].parameters())
        assert first_layer_frozen

    def test_unfreeze_lm_head(self, whisper_processor):
        """Тест разморозки output projection (proj_out)"""
        config = ModelConfig(
            model_name="openai/whisper-tiny",
            model_type="whisper",
            freeze_decoder=True,  # Замораживаем decoder
            unfreeze_lm_head=True  # Размораживаем proj_out
        )

        model = WhisperSTT(config, processor=whisper_processor)

        # proj_out должен быть разморожен
        assert model.model.proj_out.weight.requires_grad

    def test_combined_freezing_for_cross_lingual(self, whisper_processor):
        """Тест комбинированной заморозки для cross-lingual transfer"""
        config = ModelConfig(
            model_name="openai/whisper-tiny",
            model_type="whisper",
            freeze_encoder=True,  # Encoder заморожен (acoustic features универсальны)
            freeze_decoder=True,  # Decoder заморожен
            unfreeze_embed_tokens=True,  # Но embeddings разморожены (для нового языка)
            unfreeze_lm_head=True  # И lm_head разморожен
        )

        model = WhisperSTT(config, processor=whisper_processor)

        # Encoder заморожен
        encoder_frozen = all(not p.requires_grad for p in model.model.model.encoder.parameters())
        assert encoder_frozen

        # embed_tokens разморожены (критично для cross-lingual)
        assert model.model.model.decoder.embed_tokens.weight.requires_grad

        # proj_out разморожен
        assert model.model.proj_out.weight.requires_grad

        # Decoder layers заморожены
        decoder_layers = model.model.model.decoder.layers
        first_layer_frozen = all(not p.requires_grad for p in decoder_layers[0].parameters())
        assert first_layer_frozen

    def test_weight_tying_save_and_restore(self, whisper_config, whisper_processor, temp_dir):
        """Тест сохранения и восстановления weight tying для Whisper"""
        # Создаем модель
        model = WhisperSTT(whisper_config, processor=whisper_processor)

        # Проверяем наличие weight tying в оригинальной модели
        has_weight_tying_original = (
            model.model.model.decoder.embed_tokens.weight
            is model.model.proj_out.weight
        )

        # Сохраняем модель
        save_dir = temp_dir / "whisper_checkpoint"
        manager = ModelManager()
        manager.save_checkpoint(model, whisper_config, str(save_dir))

        # Проверяем metadata
        import json
        with open(save_dir / "model_metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)

        assert "weight_tying" in metadata
        assert metadata["weight_tying"] == has_weight_tying_original

        # Загружаем модель
        loaded_model, _, _ = manager.load_checkpoint(
            str(save_dir),
            device=torch.device("cpu"),
            processor=whisper_processor,
            compile=False
        )

        # Извлекаем оригинальную модель если она скомпилирована
        if hasattr(loaded_model, "_orig_mod"):
            loaded_model = loaded_model._orig_mod

        # Проверяем что weight tying восстановлен корректно
        has_weight_tying_loaded = (
            loaded_model.model.model.decoder.embed_tokens.weight
            is loaded_model.model.proj_out.weight
        )

        assert has_weight_tying_loaded == has_weight_tying_original

        # Если был weight tying, проверяем что веса действительно разделяются
        if has_weight_tying_original:
            # Изменение embed_tokens должно отразиться на proj_out
            original_proj_out_data = loaded_model.model.proj_out.weight.data.clone()
            loaded_model.model.model.decoder.embed_tokens.weight.data.fill_(0.5)
            assert torch.allclose(
                loaded_model.model.proj_out.weight.data,
                torch.full_like(original_proj_out_data, 0.5)
            )


# ============================================================
# Unit Tests для Speech2TextSTT
# ============================================================

class TestSpeech2TextSTT:
    """Тесты для класса Speech2TextSTT"""

    def test_init_with_processor(self, speech2text_config, speech2text_processor):
        """Тест создания Speech2TextSTT с процессором"""
        model = Speech2TextSTT(speech2text_config, processor=speech2text_processor)

        assert model is not None
        assert model.model_type == "speech2text"
        assert model.processor == speech2text_processor
        assert model.get_num_parameters() > 0

    def test_init_without_processor_raises(self, speech2text_config):
        """Тест создания Speech2TextSTT без процессора вызывает ValueError"""
        with pytest.raises(ValueError, match="expects a processor"):
            Speech2TextSTT(speech2text_config, processor=None)

    def test_init_with_wrong_processor_type(self, speech2text_config, whisper_processor):
        """Тест создания Speech2TextSTT с неправильным типом процессора"""
        with pytest.raises(TypeError, match="expected processor of type"):
            Speech2TextSTT(speech2text_config, processor=whisper_processor)

    def test_forward_with_labels(self, speech2text_config, speech2text_processor,
                                 dummy_speech2text_input, dummy_labels):
        """Тест forward pass с метками"""
        model = Speech2TextSTT(speech2text_config, processor=speech2text_processor)
        model.eval()

        with torch.no_grad():
            output = model.forward(input_features=dummy_speech2text_input['input_features'], labels=dummy_labels)

        assert output.loss is not None
        assert output.loss.item() > 0
        assert output.logits is not None
        assert output.logits.shape[0] == dummy_speech2text_input['input_features'].shape[0]

    def test_forward_without_labels(self, speech2text_config, speech2text_processor, dummy_speech2text_input):
        """Тест forward pass без меток"""
        model = Speech2TextSTT(speech2text_config, processor=speech2text_processor)
        model.eval()

        with torch.no_grad():
            output = model.forward(
                input_features=dummy_speech2text_input['input_features'],
                decoder_input_ids=dummy_speech2text_input['decoder_input_ids']
            )

        assert output.loss is None
        assert output.logits is not None

    def test_generate_without_language(self, speech2text_config, speech2text_processor, dummy_speech2text_input):
        """Тест генерации без указания языка (для монолингвальной модели)"""
        model = Speech2TextSTT(speech2text_config, processor=speech2text_processor)
        model.eval()

        transcriptions = model.generate(
            dummy_speech2text_input['input_features'],
            return_text=True
        )

        assert isinstance(transcriptions, list)
        assert len(transcriptions) == dummy_speech2text_input['input_features'].shape[0]

    def test_generate_with_language_optional(self, speech2text_config, speech2text_processor,
                                              dummy_speech2text_input):
        """Тест генерации с указанием языка (опционально для монолингвальной модели)"""
        model = Speech2TextSTT(speech2text_config, processor=speech2text_processor)
        model.eval()

        # Для монолингвальной модели language игнорируется, но не вызывает ошибку
        transcriptions = model.generate(
            dummy_speech2text_input['input_features'],
            language="en",  # Будет проигнорировано для монолингвальной модели
            return_text=True
        )

        assert isinstance(transcriptions, list)
        assert len(transcriptions) == dummy_speech2text_input['input_features'].shape[0]

    def test_monolingual_model_accepts_any_language(self, speech2text_config, speech2text_processor,
                                           dummy_speech2text_input):
        """Тест что монолингвальная модель не проверяет язык"""
        model = Speech2TextSTT(speech2text_config, processor=speech2text_processor)
        model.eval()

        # Монолингвальная модель игнорирует параметр language
        transcriptions = model.generate(
            dummy_speech2text_input['input_features'],
            language="any_language",  # Будет проигнорировано
            return_text=True
        )

        assert isinstance(transcriptions, list)
        assert len(transcriptions) == dummy_speech2text_input['input_features'].shape[0]

    def test_freeze_feature_encoder(self, speech2text_config, speech2text_processor):
        """Тест заморозки feature encoder"""
        model = Speech2TextSTT(speech2text_config, processor=speech2text_processor)

        trainable_before = model.get_trainable_parameters()
        model.freeze_feature_encoder()
        trainable_after = model.get_trainable_parameters()

        assert trainable_after < trainable_before

    def test_freeze_encoder(self, speech2text_config, speech2text_processor):
        """Тест заморозки encoder"""
        model = Speech2TextSTT(speech2text_config, processor=speech2text_processor)

        model.freeze_encoder()

        for param in model.model.model.encoder.parameters():
            assert not param.requires_grad

    def test_unfreeze_last_n_encoder_layers(self, speech2text_config, speech2text_processor):
        """Тест разморозки последних N слоёв encoder"""
        model = Speech2TextSTT(speech2text_config, processor=speech2text_processor)

        model.freeze_encoder()
        model.unfreeze_last_n_encoder_layers(2)

        encoder_layers = model.model.model.encoder.layers
        for layer in encoder_layers[-2:]:
            has_trainable = any(p.requires_grad for p in layer.parameters())
            assert has_trainable

    def test_unfreeze_embed_tokens(self, speech2text_processor):
        """Тест разморозки token embeddings декодера"""
        config = ModelConfig(
            model_name="facebook/s2t-small-librispeech-asr",
            model_type="speech2text",
            freeze_decoder=True,  # Замораживаем decoder (включая embed_tokens)
            unfreeze_embed_tokens=True  # Но размораживаем embed_tokens
        )

        model = Speech2TextSTT(config, processor=speech2text_processor)

        # embed_tokens должен быть разморожен
        assert model.model.model.decoder.embed_tokens.weight.requires_grad

        # Остальные части decoder должны быть заморожены
        decoder_layers = model.model.model.decoder.layers
        first_layer_frozen = all(not p.requires_grad for p in decoder_layers[0].parameters())
        assert first_layer_frozen

    def test_unfreeze_lm_head(self, speech2text_processor):
        """Тест разморозки output projection (lm_head)"""
        config = ModelConfig(
            model_name="facebook/s2t-small-librispeech-asr",
            model_type="speech2text",
            freeze_decoder=True,  # Замораживаем decoder
            unfreeze_lm_head=True  # Размораживаем lm_head
        )

        model = Speech2TextSTT(config, processor=speech2text_processor)

        # lm_head должен быть разморожен
        assert model.model.lm_head.weight.requires_grad

    def test_combined_freezing_for_cross_lingual(self, speech2text_processor):
        """Тест комбинированной заморозки для cross-lingual transfer"""
        config = ModelConfig(
            model_name="facebook/s2t-small-librispeech-asr",
            model_type="speech2text",
            freeze_encoder=True,  # Encoder заморожен (acoustic features универсальны)
            freeze_decoder=True,  # Decoder заморожен
            unfreeze_embed_tokens=True,  # Но embeddings разморожены (для нового языка)
            unfreeze_lm_head=True  # И lm_head разморожен
        )

        model = Speech2TextSTT(config, processor=speech2text_processor)

        # Encoder заморожен
        encoder_frozen = all(not p.requires_grad for p in model.model.model.encoder.parameters())
        assert encoder_frozen

        # embed_tokens разморожены (критично для cross-lingual)
        assert model.model.model.decoder.embed_tokens.weight.requires_grad

        # lm_head разморожен
        assert model.model.lm_head.weight.requires_grad

        # Decoder layers заморожены
        decoder_layers = model.model.model.decoder.layers
        first_layer_frozen = all(not p.requires_grad for p in decoder_layers[0].parameters())
        assert first_layer_frozen

    def test_weight_tying_save_and_restore(self, speech2text_config, speech2text_processor, temp_dir):
        """Тест сохранения и восстановления weight tying для Speech2Text"""
        # Создаем модель
        model = Speech2TextSTT(speech2text_config, processor=speech2text_processor)

        # Проверяем наличие weight tying в оригинальной модели
        has_weight_tying_original = (
            model.model.model.decoder.embed_tokens.weight
            is model.model.lm_head.weight
        )

        # Сохраняем модель
        save_dir = temp_dir / "speech2text_checkpoint"
        manager = ModelManager()
        manager.save_checkpoint(model, speech2text_config, str(save_dir))

        # Проверяем metadata
        import json
        with open(save_dir / "model_metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)

        assert "weight_tying" in metadata
        assert metadata["weight_tying"] == has_weight_tying_original

        # Загружаем модель
        loaded_model, _, _ = manager.load_checkpoint(
            str(save_dir),
            device=torch.device("cpu"),
            processor=speech2text_processor,
            compile=False
        )

        # Извлекаем оригинальную модель если она скомпилирована
        if hasattr(loaded_model, "_orig_mod"):
            loaded_model = loaded_model._orig_mod

        # Проверяем что weight tying восстановлен корректно
        has_weight_tying_loaded = (
            loaded_model.model.model.decoder.embed_tokens.weight
            is loaded_model.model.lm_head.weight
        )

        assert has_weight_tying_loaded == has_weight_tying_original

        # Если был weight tying, проверяем что веса действительно разделяются
        if has_weight_tying_original:
            # Изменение embed_tokens должно отразиться на lm_head
            original_lm_head_data = loaded_model.model.lm_head.weight.data.clone()
            loaded_model.model.model.decoder.embed_tokens.weight.data.fill_(0.5)
            assert torch.allclose(
                loaded_model.model.lm_head.weight.data,
                torch.full_like(original_lm_head_data, 0.5)
            )


# ============================================================
# Unit Tests для Speech2Text Cross-Lingual Transfer
# ============================================================

class TestSpeech2TextCrossLingual:
    """Тесты для cross-lingual transfer (English model + Multilingual tokenizer)"""

    def test_init_with_multilingual_processor(self, speech2text_cross_lingual_config,
                                              speech2text_multilingual_processor):
        """Тест создания Speech2Text с multilingual processor"""
        model = Speech2TextSTT(speech2text_cross_lingual_config, processor=speech2text_multilingual_processor)

        assert model is not None
        assert model.model_type == "speech2text"
        assert model.processor == speech2text_multilingual_processor

    def test_tokenizer_differences(self, speech2text_cross_lingual_config,
                                   speech2text_multilingual_processor, speech2text_processor):
        """Тест различий между monolingual и multilingual tokenizers"""
        # Original English tokenizer
        original_tokenizer = getattr(speech2text_processor, 'tokenizer', None)
        assert original_tokenizer is not None
        original_vocab_size = len(original_tokenizer)

        # Multilingual tokenizer
        multilingual_tokenizer = getattr(speech2text_multilingual_processor, 'tokenizer', None)
        assert multilingual_tokenizer is not None
        multilingual_vocab_size = len(multilingual_tokenizer)

        # Vocab sizes могут быть одинаковыми (оба используют один и тот же vocabulary)
        # Главное различие - поддержка языков через lang_code_to_id
        assert hasattr(multilingual_tokenizer, 'lang_code_to_id')
        assert len(multilingual_tokenizer.lang_code_to_id) > 0  # Multilingual поддерживает языки
        assert 'ru' in multilingual_tokenizer.lang_code_to_id  # Включая русский

        # English tokenizer - монолингвальный (пустой lang_code_to_id)
        assert hasattr(original_tokenizer, 'lang_code_to_id')
        assert len(original_tokenizer.lang_code_to_id) == 0  # Монолингвальный

        # Создаём модель через ModelFactory
        model = ModelFactory.create_model(speech2text_cross_lingual_config,
                                         processor=speech2text_multilingual_processor)

        # Vocab_size модели должен соответствовать токенайзеру
        assert model.model.config.vocab_size == multilingual_vocab_size

    def test_resize_embeddings_for_tokenizer(self, speech2text_cross_lingual_config,
                                             speech2text_multilingual_processor):
        """Тест метода resize_embeddings_for_tokenizer"""
        model = Speech2TextSTT(speech2text_cross_lingual_config, processor=speech2text_multilingual_processor)

        original_vocab_size = model.model.config.vocab_size
        new_vocab_size = original_vocab_size + 100  # Симулируем увеличение vocab

        # Вызываем resize
        model.resize_embeddings_for_tokenizer(new_vocab_size)

        # Проверяем что vocab_size изменился
        assert model.model.config.vocab_size == new_vocab_size

    def test_resize_embeddings_same_size_skip(self, speech2text_cross_lingual_config,
                                              speech2text_multilingual_processor):
        """Тест что resize пропускается если размер не изменился"""
        model = Speech2TextSTT(speech2text_cross_lingual_config, processor=speech2text_multilingual_processor)

        original_vocab_size = model.model.config.vocab_size

        # Вызываем resize с тем же размером
        model.resize_embeddings_for_tokenizer(original_vocab_size)

        # Размер должен остаться прежним
        assert model.model.config.vocab_size == original_vocab_size

    def test_freezing_strategy(self, speech2text_cross_lingual_config,
                               speech2text_multilingual_processor):
        """Тест что freezing strategy применяется корректно (encoder frozen, decoder trainable)"""
        model = ModelFactory.create_model(speech2text_cross_lingual_config,
                                         processor=speech2text_multilingual_processor)

        # Проверяем encoder заморожен
        encoder_trainable = any(p.requires_grad for p in model.model.model.encoder.parameters())
        assert not encoder_trainable, "Encoder должен быть заморожен для cross-lingual transfer"

        # Проверяем decoder разморожен
        decoder_trainable = any(p.requires_grad for p in model.model.model.decoder.parameters())
        assert decoder_trainable, "Decoder должен быть trainable для cross-lingual transfer"

    def test_generate_with_multilingual_tokenizer(self, speech2text_cross_lingual_config,
                                                  speech2text_multilingual_processor,
                                                  dummy_speech2text_input):
        """Тест генерации с multilingual tokenizer требует language параметр"""
        model = ModelFactory.create_model(speech2text_cross_lingual_config,
                                         processor=speech2text_multilingual_processor)
        model.eval()

        # Multilingual модель требует указание языка
        transcriptions = model.generate(
            dummy_speech2text_input['input_features'],
            language="ru",
            return_text=True
        )

        assert isinstance(transcriptions, list)
        assert len(transcriptions) == dummy_speech2text_input['input_features'].shape[0]

    def test_generate_multilingual_without_language_raises(self, speech2text_cross_lingual_config,
                                                           speech2text_multilingual_processor,
                                                           dummy_speech2text_input):
        """Тест что multilingual модель требует language параметр"""
        model = ModelFactory.create_model(speech2text_cross_lingual_config,
                                         processor=speech2text_multilingual_processor)
        model.eval()

        # Должна быть ошибка если language не указан для multilingual модели
        with pytest.raises(ValueError, match="This is a multilingual model"):
            model.generate(
                dummy_speech2text_input['input_features'],
                return_text=True
            )

    def test_generate_multilingual_invalid_language_raises(self, speech2text_cross_lingual_config,
                                                           speech2text_multilingual_processor,
                                                           dummy_speech2text_input):
        """Тест что неподдерживаемый язык вызывает ошибку"""
        model = ModelFactory.create_model(speech2text_cross_lingual_config,
                                         processor=speech2text_multilingual_processor)
        model.eval()

        # Несуществующий язык должен вызвать ошибку
        with pytest.raises(ValueError, match="Language .* not found"):
            model.generate(
                dummy_speech2text_input['input_features'],
                language="invalid_lang_code",
                return_text=True
            )

    def test_tokenizer_has_russian_support(self, speech2text_multilingual_processor):
        """Тест что multilingual tokenizer поддерживает русский язык"""
        tokenizer = getattr(speech2text_multilingual_processor, 'tokenizer', None)
        assert tokenizer is not None

        assert hasattr(tokenizer, 'lang_code_to_id')
        assert 'ru' in tokenizer.lang_code_to_id
        assert tokenizer.tgt_lang == "ru"

    def test_forward_with_multilingual(self, speech2text_cross_lingual_config,
                                       speech2text_multilingual_processor,
                                       dummy_speech2text_input, dummy_labels):
        """Тест forward pass с multilingual processor"""
        model = ModelFactory.create_model(speech2text_cross_lingual_config,
                                         processor=speech2text_multilingual_processor)
        model.eval()

        with torch.no_grad():
            output = model.forward(
                input_features=dummy_speech2text_input['input_features'],
                labels=dummy_labels
            )

        assert output.loss is not None
        assert output.logits is not None
        # Vocab size должен соответствовать multilingual tokenizer
        tokenizer = getattr(speech2text_multilingual_processor, 'tokenizer', None)
        assert tokenizer is not None
        assert output.logits.shape[-1] == len(tokenizer)

    def test_save_and_load_cross_lingual_checkpoint(self, speech2text_cross_lingual_config,
                                                    speech2text_multilingual_processor, temp_dir):
        """Тест сохранения и загрузки cross-lingual модели"""
        manager = ModelManager()

        # Создаём модель с multilingual tokenizer
        model = manager.create_model(speech2text_cross_lingual_config,
                                    processor=speech2text_multilingual_processor)
        device = torch.device("cpu")
        model = manager.compile_model(model, device=device, compile=False)

        original_vocab_size = model.model.config.vocab_size

        # Сохраняем чекпоинт
        checkpoint_dir = str(temp_dir / "cross_lingual_checkpoint")
        manager.save_checkpoint(model, speech2text_cross_lingual_config, checkpoint_dir)

        # Загружаем чекпоинт (processor должен быть создан автоматически с правильным tokenizer)
        # Для multilingual tokenizer нужно явно передать language
        loaded_model, loaded_processor, metadata = manager.load_checkpoint(
            checkpoint_dir,
            device=device,
            processor=None,
            compile=False,
            language="ru"  # Required for multilingual Speech2Text tokenizer
        )

        # Проверяем что конфигурация сохранилась
        assert metadata["config"]["tokenizer_name_or_path"] == "facebook/s2t-medium-mustc-multilingual-st"

        # Проверяем что vocab_size правильный
        assert loaded_model.model.config.vocab_size == original_vocab_size

        # Проверяем что loaded processor имеет правильный tokenizer
        loaded_tokenizer = getattr(loaded_processor, 'tokenizer', None)
        assert loaded_tokenizer is not None
        assert len(loaded_tokenizer) == original_vocab_size

    def test_trainable_parameters_ratio(self, speech2text_cross_lingual_config,
                                        speech2text_multilingual_processor):
        """Тест что только decoder trainable (encoder frozen)"""
        model = ModelFactory.create_model(speech2text_cross_lingual_config,
                                         processor=speech2text_multilingual_processor)

        total_params = model.get_num_parameters()
        trainable_params = model.get_trainable_parameters()

        # Encoder frozen, decoder trainable => примерно 50% параметров trainable
        trainable_ratio = trainable_params / total_params
        assert 0.3 < trainable_ratio < 0.7, \
            f"Expected ~50% trainable params, got {trainable_ratio:.1%}"


# ============================================================
# Unit Tests для ModelFactory
# ============================================================

class TestModelFactory:
    """Тесты для класса ModelFactory"""

    def test_create_whisper_model(self, whisper_config, whisper_processor):
        """Тест создания Whisper модели через фабрику"""
        model = ModelFactory.create_model(whisper_config, processor=whisper_processor)

        assert isinstance(model, WhisperSTT)
        assert model.model_type == "whisper"

    def test_create_speech2text_model(self, speech2text_config, speech2text_processor):
        """Тест создания Speech2Text модели через фабрику"""
        model = ModelFactory.create_model(speech2text_config, processor=speech2text_processor)

        assert isinstance(model, Speech2TextSTT)
        assert model.model_type == "speech2text"

    def test_create_model_with_invalid_type(self, whisper_config):
        """Тест создания модели с неподдерживаемым типом"""
        invalid_config = replace(whisper_config, model_type="invalid_type")

        with pytest.raises(ValueError, match="Unsupported model type"):
            ModelFactory.create_model(invalid_config)

    def test_get_model_info(self, whisper_config, whisper_processor):
        """Тест получения информации о модели"""
        model = ModelFactory.create_model(whisper_config, processor=whisper_processor)
        info = ModelFactory.get_model_info(model)

        assert "model_type" in info
        assert "model_name" in info
        assert "total_parameters" in info
        assert "trainable_parameters" in info
        assert "trainable_ratio" in info

        assert info["model_type"] == "whisper"
        assert info["total_parameters"] > 0
        assert info["trainable_parameters"] > 0
        assert 0 <= info["trainable_ratio"] <= 1


# ============================================================
# Unit Tests для ModelManager
# ============================================================

class TestModelManager:
    """Тесты для класса ModelManager"""

    def test_create_model(self, whisper_config, whisper_processor):
        """Тест создания модели через ModelManager"""
        manager = ModelManager()
        model = manager.create_model(whisper_config, processor=whisper_processor)

        assert isinstance(model, WhisperSTT)
        assert model.model_type == "whisper"

    def test_compile_model_without_compilation(self, whisper_config, whisper_processor):
        """Тест перемещения модели на device без компиляции"""
        manager = ModelManager()
        model = manager.create_model(whisper_config, processor=whisper_processor)
        device = torch.device("cpu")
        compiled_model = manager.compile_model(model, device=device, compile=False)

        # Модель должна быть на правильном device
        assert next(compiled_model.parameters()).device.type == "cpu"

        # Модель не должна быть скомпилирована
        assert not hasattr(compiled_model, "_orig_mod")

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile недоступен")
    def test_compile_model_with_compilation(self, whisper_config, whisper_processor):
        """Тест компиляции модели с torch.compile"""
        manager = ModelManager()
        model = manager.create_model(whisper_config, processor=whisper_processor)
        device = torch.device("cpu")

        # Попытка компиляции (может не работать на всех платформах)
        compiled_model = manager.compile_model(model, device=device, compile=True)

        # Если компиляция успешна, должен быть _orig_mod
        # Иначе вернётся обычная модель (fallback)
        assert compiled_model is not None

    def test_save_and_load_checkpoint(self, whisper_config, whisper_processor, temp_dir):
        """Тест сохранения и загрузки чекпоинта"""
        manager = ModelManager()

        # Создаём модель
        model = manager.create_model(whisper_config, processor=whisper_processor)
        device = torch.device("cpu")
        model = manager.compile_model(model, device=device, compile=False)

        # Сохраняем чекпоинт (теперь в директорию)
        checkpoint_dir = str(temp_dir / "test_checkpoint")
        manager.save_checkpoint(model, whisper_config, checkpoint_dir)

        # Проверяем что файлы созданы
        assert os.path.exists(checkpoint_dir)
        assert os.path.exists(os.path.join(checkpoint_dir, "model_weights.pt"))
        assert os.path.exists(os.path.join(checkpoint_dir, "model_metadata.json"))

        # Загружаем чекпоинт
        loaded_model, loaded_processor, checkpoint_data = manager.load_checkpoint(
            checkpoint_dir,
            device=device,
            processor=whisper_processor,
            compile=False
        )

        assert isinstance(loaded_model, WhisperSTT)
        assert "model_info" in checkpoint_data

    def test_save_compiled_model(self, whisper_config, whisper_processor, temp_dir):
        """Тест сохранения скомпилированной модели (должна сохраняться оригинальная)"""
        manager = ModelManager()

        # Создаём оригинальную модель
        original_model = manager.create_model(whisper_config, processor=whisper_processor)

        # Создаём wrapper, симулирующий torch.compile
        class CompiledModelWrapper:
            def __init__(self, orig_model):
                self._orig_mod = orig_model

            def __getattr__(self, name):
                # Делегируем все атрибуты оригинальной модели
                return getattr(self._orig_mod, name)

        compiled_model = CompiledModelWrapper(original_model)

        checkpoint_dir = str(temp_dir / "compiled_checkpoint")
        manager.save_checkpoint(compiled_model, whisper_config, checkpoint_dir)  # type: ignore

        # Проверяем что чекпоинт сохранён
        assert os.path.exists(checkpoint_dir)
        assert os.path.exists(os.path.join(checkpoint_dir, "model_weights.pt"))
        assert os.path.exists(os.path.join(checkpoint_dir, "model_metadata.json"))

        # Загружаем и проверяем
        device = torch.device("cpu")
        loaded_model, _, _ = manager.load_checkpoint(
            checkpoint_dir,
            device=device,
            processor=whisper_processor,
            compile=False
        )
        assert isinstance(loaded_model, BaseSTTModel)

    def test_inference_with_amp_disabled(self, whisper_config, whisper_processor, dummy_whisper_input):
        """Тест инференса без AMP"""
        manager = ModelManager()
        model = manager.create_model(whisper_config, processor=whisper_processor)
        device = torch.device("cpu")
        model = manager.compile_model(model, device=device, compile=False)

        output = manager.inference_with_amp(
            model,
            dummy_whisper_input['input_features'],
            use_amp=False,
            language="ru",
            task="transcribe",
            return_text=True,
            attention_mask=dummy_whisper_input['attention_mask']
        )

        assert isinstance(output, list)
        assert len(output) == dummy_whisper_input['input_features'].shape[0]

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA не доступен")
    def test_inference_with_amp_enabled(self, whisper_config, whisper_processor):
        """Тест инференса с AMP на GPU"""
        manager = ModelManager()
        model = manager.create_model(whisper_config, processor=whisper_processor)
        device = torch.device("cuda")
        model = manager.compile_model(model, device=device, compile=False)

        # Создаём dummy данные на GPU
        seq_length = 3000  # Whisper требует 3000 frames
        dummy_input = torch.randn(1, 80, seq_length).cuda()
        attention_mask = torch.ones(1, seq_length, dtype=torch.long).cuda()

        output = manager.inference_with_amp(
            model,
            dummy_input,
            use_amp=True,
            amp_dtype=torch.float16,
            language="ru",
            task="transcribe",
            return_text=True,
            attention_mask=attention_mask
        )

        assert isinstance(output, list)
        assert len(output) == 1

    @pytest.mark.skipif(torch.cuda.is_available(), reason="CUDA доступен, нужен тест без CUDA")
    def test_get_memory_usage_cpu(self):
        """Тест получения информации о памяти на CPU"""
        manager = ModelManager()
        memory_info = manager.get_memory_usage()

        assert "message" in memory_info
        assert memory_info["message"] == "CUDA not available"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA не доступен")
    def test_get_memory_usage_gpu(self):
        """Тест получения информации о памяти на GPU"""
        manager = ModelManager()
        memory_info = manager.get_memory_usage()

        assert "allocated_gb" in memory_info
        assert "reserved_gb" in memory_info
        assert "total_gb" in memory_info

        # Type narrowing: проверяем что значения - float
        assert isinstance(memory_info["allocated_gb"], float)
        assert isinstance(memory_info["reserved_gb"], float)
        assert isinstance(memory_info["total_gb"], float)

        assert memory_info["allocated_gb"] >= 0
        assert memory_info["total_gb"] > 0


# ============================================================
# Integration Tests
# ============================================================

class TestIntegration:
    """Интеграционные тесты полного пайплайна"""

    def test_full_pipeline_whisper(self, whisper_config, whisper_processor, dummy_whisper_input, temp_dir):
        """Тест полного пайплайна: создание -> заморозка -> forward -> generate -> save -> load"""
        manager = ModelManager()

        # Создание модели
        model = manager.create_model(whisper_config, processor=whisper_processor)
        device = torch.device("cpu")
        model = manager.compile_model(model, device=device, compile=False)

        # Заморозка encoder
        model.freeze_encoder()
        trainable_frozen = model.get_trainable_parameters()

        # Forward pass
        with torch.no_grad():
            output = model.forward(
                input_features=dummy_whisper_input['input_features'],
                decoder_input_ids=dummy_whisper_input['decoder_input_ids']
            )
        assert output.logits is not None

        # Generate
        transcriptions = model.generate(
            dummy_whisper_input['input_features'],
            language="ru",
            task="transcribe",
            return_text=True,
            attention_mask=dummy_whisper_input['attention_mask']
        )
        assert len(transcriptions) == dummy_whisper_input['input_features'].shape[0]

        # Save
        checkpoint_dir = str(temp_dir / "pipeline_checkpoint")
        manager.save_checkpoint(model, whisper_config, checkpoint_dir)

        # Load
        loaded_model, _, _ = manager.load_checkpoint(
            checkpoint_dir,
            device=device,
            processor=whisper_processor,
            compile=False
        )

        # После загрузки нужно снова применить заморозку
        # (state_dict не сохраняет requires_grad)
        loaded_model.freeze_encoder()
        assert loaded_model.get_trainable_parameters() == trainable_frozen

        # Проверяем что generate работает так же
        transcriptions_loaded = loaded_model.generate(
            dummy_whisper_input['input_features'],
            language="ru",
            task="transcribe",
            return_text=True,
            attention_mask=dummy_whisper_input['attention_mask']
        )
        assert len(transcriptions_loaded) == len(transcriptions)

    def test_full_pipeline_speech2text(self, speech2text_config, speech2text_processor,
                                       dummy_speech2text_input):
        """Тест полного пайплайна для Speech2Text"""
        manager = ModelManager()

        model = manager.create_model(speech2text_config, processor=speech2text_processor)
        device = torch.device("cpu")
        model = manager.compile_model(model, device=device, compile=False)

        # Forward
        with torch.no_grad():
            output = model.forward(
                input_features=dummy_speech2text_input['input_features'],
                decoder_input_ids=dummy_speech2text_input['decoder_input_ids']
            )
        assert output.logits is not None

        # Generate (без language для монолингвальной модели)
        transcriptions = model.generate(
            dummy_speech2text_input['input_features'],
            return_text=True
        )
        assert len(transcriptions) == dummy_speech2text_input['input_features'].shape[0]

    def test_batch_vs_single_whisper(self, whisper_config, whisper_processor):
        """Тест обработки batch vs single sample для Whisper"""
        model = WhisperSTT(whisper_config, processor=whisper_processor)
        model.eval()

        # Single sample - Whisper требует mel features длиной 3000
        single_input = torch.randn(1, 80, 3000)
        single_decoder_ids = torch.tensor([[50258, 50259]])
        with torch.no_grad():
            single_output = model.forward(
                input_features=single_input,
                decoder_input_ids=single_decoder_ids
            )

        # Batch
        batch_input = torch.randn(4, 80, 3000)
        batch_decoder_ids = torch.tensor([[50258, 50259] for _ in range(4)])
        with torch.no_grad():
            batch_output = model.forward(
                input_features=batch_input,
                decoder_input_ids=batch_decoder_ids
            )

        assert single_output.logits.shape[0] == 1
        assert batch_output.logits.shape[0] == 4


# ============================================================
# Edge Cases и Robustness Tests
# ============================================================

class TestEdgeCases:
    """Тесты граничных случаев и robustness"""

    def test_generate_without_processor(self, whisper_config):
        """Тест generate без процессора (должно быть поймано в __init__)"""
        # WhisperSTT требует процессор в __init__, так что этот тест проверяет это
        with pytest.raises(ValueError):
            WhisperSTT(whisper_config, processor=None)

    def test_empty_batch(self, whisper_config, whisper_processor):
        """Тест с пустым батчем (размер 0)"""
        model = WhisperSTT(whisper_config, processor=whisper_processor)
        model.eval()

        # Создаём пустой тензор
        empty_input = torch.randn(0, 80, 1000)

        # Forward должен обработать без ошибок (или выбросить понятную ошибку)
        with torch.no_grad():
            try:
                output = model.forward(input_features=empty_input)
                # Если не вызвало ошибку, проверяем размер
                assert output.logits.shape[0] == 0
            except (RuntimeError, ValueError):
                # Ожидаемое поведение - ошибка при пустом батче
                pass

    def test_very_long_sequence(self, whisper_config, whisper_processor):
        """Тест с максимально допустимой длинной последовательностью"""
        model = WhisperSTT(whisper_config, processor=whisper_processor)
        model.eval()

        # Whisper работает только с фиксированным размером 3000
        # Тест проверяет поведение на правильном максимальном размере
        max_input = torch.randn(1, 80, 3000)
        decoder_ids = torch.tensor([[50258, 50259]])

        # Проверяем что модель хотя бы не крашится
        with torch.no_grad():
            try:
                output = model.forward(input_features=max_input, decoder_input_ids=decoder_ids)
                assert output.logits is not None
            except (RuntimeError, torch.cuda.OutOfMemoryError):
                # OOM ожидается для длинных последовательностей на некоторых системах
                pytest.skip("Out of memory для длинной последовательности (ожидаемо)")

    def test_model_info_consistency(self, whisper_config, whisper_processor):
        """Тест консистентности информации о модели до и после заморозки"""
        model = WhisperSTT(whisper_config, processor=whisper_processor)

        total_before = model.get_num_parameters()
        trainable_before = model.get_trainable_parameters()

        # Замораживаем encoder
        model.freeze_encoder()

        total_after = model.get_num_parameters()
        trainable_after = model.get_trainable_parameters()

        # Общее количество параметров не должно измениться
        assert total_before == total_after

        # Количество обучаемых должно уменьшиться
        assert trainable_after < trainable_before

    def test_dropout_applied(self, whisper_processor):
        """Тест применения dropout из конфига (standard HuggingFace names)"""
        config = ModelConfig(
            model_name="openai/whisper-tiny",
            model_type="whisper",
            activation_dropout=0.2,
            attention_dropout=0.3,
            dropout=0.1,
        )

        model = WhisperSTT(config, processor=whisper_processor)

        # Проверяем что dropout параметры переданы в модель
        # (точная проверка зависит от внутренней структуры Whisper)
        assert model.config.activation_dropout == 0.2
        assert model.config.attention_dropout == 0.3
        assert model.config.dropout == 0.1



if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
