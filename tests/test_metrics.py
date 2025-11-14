"""
Comprehensive tests for metrics.py module.
Tests all metrics computation, normalization, and analysis functionality.
"""

import pytest
import numpy as np
from typing import List, Dict
from unittest.mock import Mock, MagicMock
from transformers import EvalPrediction
import sys
from pathlib import Path

# Add project root to Python path
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))


# Import classes to test
from src.metrics import (
    TextNormalizer,
    STTMetrics,
    MetricResult,
    HuggingFaceMetricsComputer,
    PerformanceAnalyzer,
    create_metrics_summary
)


# ============================================================
# Test TextNormalizer
# ============================================================
class TestTextNormalizer:
    """Test text normalization for evaluation"""

    def test_normalize_russian_text(self):
        """Test Russian text normalization"""
        normalizer = TextNormalizer(language="ru")

        # Test lowercase
        assert normalizer.normalize("ПРИВЕТ МИР") == "привет мир"

        # Test punctuation removal
        assert normalizer.normalize("Привет, мир!") == "привет мир"
        assert normalizer.normalize("Привет. Мир?") == "привет мир"

        # Test multiple spaces
        assert normalizer.normalize("Привет    мир") == "привет мир"

        # Test leading/trailing spaces
        assert normalizer.normalize("  Привет мир  ") == "привет мир"

    def test_normalize_empty_string(self):
        """Test normalization of empty strings"""
        normalizer = TextNormalizer(language="ru")
        assert normalizer.normalize("") == ""
        assert normalizer.normalize("   ") == ""

    def test_normalize_batch(self):
        """Test batch normalization"""
        normalizer = TextNormalizer(language="ru")
        texts = ["ПРИВЕТ МИР", "Как дела?", "  Всё хорошо!  "]
        expected = ["привет мир", "как дела", "всё хорошо"]

        result = normalizer.normalize_batch(texts)
        assert result == expected

    def test_apostrophes_preserved(self):
        """Test that apostrophes are preserved"""
        normalizer = TextNormalizer(language="ru")
        # For English compatibility
        assert "'" in normalizer.normalize("don't worry")


# ============================================================
# Test MetricResult
# ============================================================
class TestMetricResult:
    """Test MetricResult dataclass"""

    def test_metric_result_to_dict(self):
        """Test conversion to dictionary"""
        result = MetricResult(
            wer=0.25,
            cer=0.15,
            bleu=0.65,
            mer=0.20,
            wil=0.30,
            insertions=5,
            deletions=3,
            substitutions=7,
            hits=85,
            total_predictions=100,
            empty_predictions=2
        )

        result_dict = result.to_dict()

        assert result_dict["wer"] == 0.25
        assert result_dict["cer"] == 0.15
        assert result_dict["bleu"] == 0.65
        assert result_dict["mer"] == 0.20
        assert result_dict["wil"] == 0.30
        assert result_dict["insertions"] == 5
        assert result_dict["deletions"] == 3
        assert result_dict["substitutions"] == 7
        assert result_dict["hits"] == 85
        assert result_dict["total_predictions"] == 100
        assert result_dict["empty_predictions"] == 2
        assert result_dict["empty_prediction_rate"] == 0.02

    def test_empty_prediction_rate_calculation(self):
        """Test empty prediction rate calculation"""
        result = MetricResult(
            total_predictions=10,
            empty_predictions=3
        )

        assert result.to_dict()["empty_prediction_rate"] == 0.3

    def test_empty_prediction_rate_zero_division(self):
        """Test empty prediction rate with zero total"""
        result = MetricResult(
            total_predictions=0,
            empty_predictions=0
        )

        # Should not raise ZeroDivisionError
        rate = result.to_dict()["empty_prediction_rate"]
        assert rate == 0.0


# ============================================================
# Test STTMetrics
# ============================================================
class TestSTTMetrics:
    """Test speech-to-text metrics computation"""

    def test_compute_wer_perfect_match(self):
        """Test WER computation with perfect predictions"""
        metrics = STTMetrics(language="ru", use_bleu=False)

        predictions = ["привет мир", "как дела"]
        references = ["привет мир", "как дела"]

        wer = metrics.compute_wer(predictions, references)
        assert wer == 0.0

    def test_compute_wer_with_errors(self):
        """Test WER computation with errors"""
        metrics = STTMetrics(language="ru", use_bleu=False)

        # Substitution error
        predictions = ["привет мир"]
        references = ["привет друг"]

        wer = metrics.compute_wer(predictions, references)
        assert wer > 0.0

    def test_compute_cer_perfect_match(self):
        """Test CER computation with perfect predictions"""
        metrics = STTMetrics(language="ru", use_bleu=False)

        predictions = ["привет"]
        references = ["привет"]

        cer = metrics.compute_cer(predictions, references)
        assert cer == 0.0

    def test_compute_cer_with_errors(self):
        """Test CER computation with character-level errors"""
        metrics = STTMetrics(language="ru", use_bleu=False)

        predictions = ["привет"]
        references = ["првет"]

        cer = metrics.compute_cer(predictions, references)
        assert cer > 0.0

    def test_compute_detailed_measures(self):
        """Test detailed measures computation"""
        metrics = STTMetrics(language="ru", use_bleu=False)

        predictions = ["привет мир", "как дела"]
        references = ["привет друг", "как дела"]

        detailed = metrics.compute_detailed_measures(predictions, references)

        assert "wer" in detailed
        assert "cer" in detailed
        assert "mer" in detailed
        assert "wil" in detailed
        assert "substitutions" in detailed
        assert "deletions" in detailed
        assert "insertions" in detailed
        assert "hits" in detailed
        assert "total_predictions" in detailed
        assert "empty_predictions" in detailed

        assert detailed["total_predictions"] == 2
        assert detailed["empty_predictions"] == 0

    def test_compute_detailed_measures_with_empty_predictions(self):
        """Test detailed measures with empty predictions"""
        metrics = STTMetrics(language="ru", use_bleu=False)

        predictions = ["привет мир", "", "как дела"]
        references = ["привет мир", "пустой", "как дела"]

        detailed = metrics.compute_detailed_measures(predictions, references)

        assert detailed["total_predictions"] == 3
        assert detailed["empty_predictions"] == 1

    def test_compute_all_metrics(self):
        """Test compute_all_metrics returns MetricResult"""
        metrics = STTMetrics(language="ru", use_bleu=False)

        predictions = ["привет мир"]
        references = ["привет мир"]

        result = metrics.compute_all_metrics(predictions, references)

        assert isinstance(result, MetricResult)
        assert result.wer == 0.0
        assert result.cer == 0.0
        assert result.total_predictions == 1
        assert result.empty_predictions == 0

    def test_compute_bleu_score(self):
        """Test BLEU score computation"""
        metrics = STTMetrics(language="ru", use_bleu=True)

        predictions = ["привет мир"]
        references = ["привет мир"]

        bleu = metrics.compute_bleu(predictions, references)
        assert bleu >= 0.0
        assert bleu <= 1.0

    def test_empty_predictions_list(self):
        """Test handling of empty predictions list"""
        metrics = STTMetrics(language="ru", use_bleu=False)

        predictions = []
        references = []

        wer = metrics.compute_wer(predictions, references)
        assert wer == 1.0

        detailed = metrics.compute_detailed_measures(predictions, references)
        assert detailed["wer"] == 1.0


# ============================================================
# Test HuggingFaceMetricsComputer
# ============================================================
class TestHuggingFaceMetricsComputer:
    """Test HuggingFace Trainer metrics computation"""

    def test_initialization_seq2seq(self):
        """Test initialization with seq2seq model type"""
        processor = Mock()
        processor.tokenizer = Mock()
        processor.tokenizer.pad_token_id = 0

        computer = HuggingFaceMetricsComputer(
            processor=processor,
            model_type="seq2seq",
            language="ru"
        )

        assert computer.model_type == "seq2seq"
        assert computer.processor == processor

    def test_initialization_ctc(self):
        """Test initialization with CTC model type"""
        processor = Mock()

        computer = HuggingFaceMetricsComputer(
            processor=processor,
            model_type="ctc",
            language="ru"
        )

        assert computer.model_type == "ctc"

    def test_initialization_invalid_model_type(self):
        """Test initialization with invalid model type"""
        processor = Mock()

        with pytest.raises(ValueError, match="model_type must be"):
            HuggingFaceMetricsComputer(
                processor=processor,
                model_type="invalid",
                language="ru"
            )

    def test_seq2seq_predictions_decoding(self):
        """Test seq2seq predictions decoding (Whisper/Speech2Text)"""
        # Mock processor
        processor = Mock()
        processor.tokenizer = Mock()
        processor.tokenizer.pad_token_id = 0
        processor.batch_decode = Mock(return_value=["привет мир", "как дела"])

        computer = HuggingFaceMetricsComputer(
            processor=processor,
            model_type="seq2seq",
            language="ru",
            use_bleu=False
        )

        # Create mock predictions (token IDs from generate())
        predictions = np.array([
            [1, 2, 3, 0, 0],  # Batch item 1
            [4, 5, 6, 7, 0]   # Batch item 2
        ])
        labels = np.array([
            [1, 2, 3, 0, 0],
            [4, 5, 6, 7, 0]
        ])

        eval_pred = EvalPrediction(predictions=predictions, label_ids=labels)

        result = computer(eval_pred)

        assert "eval_wer" in result
        assert "eval_cer" in result
        assert "eval_mer" in result
        assert "eval_wil" in result
        assert "eval_empty_prediction_rate" in result

        # Perfect match should give WER=0
        assert result["eval_wer"] == 0.0

    def test_ctc_predictions_decoding(self):
        """Test CTC predictions decoding (Custom model)"""
        # Mock processor
        processor = Mock()
        processor.batch_decode = Mock(return_value=["привет мир", "как дела"])

        computer = HuggingFaceMetricsComputer(
            processor=processor,
            model_type="ctc",
            language="ru",
            use_bleu=False
        )

        # Create mock predictions (logits - 3D array)
        batch_size = 2
        time_steps = 10
        vocab_size = 100

        predictions = np.random.randn(batch_size, time_steps, vocab_size)
        labels = np.random.randint(0, vocab_size, size=(batch_size, time_steps))

        eval_pred = EvalPrediction(predictions=predictions, label_ids=labels)

        result = computer(eval_pred)

        # Should call argmax for CTC
        assert "eval_wer" in result
        assert "eval_cer" in result

    def test_tuple_predictions_handling(self):
        """Test handling of tuple predictions"""
        processor = Mock()
        processor.tokenizer = Mock()
        processor.tokenizer.pad_token_id = 0
        processor.batch_decode = Mock(return_value=["привет"])

        computer = HuggingFaceMetricsComputer(
            processor=processor,
            model_type="seq2seq",
            language="ru"
        )

        # Some models return (logits, additional_outputs)
        predictions = (np.array([[1, 2, 3]]),)
        labels = np.array([[1, 2, 3]])

        eval_pred = EvalPrediction(predictions=predictions, label_ids=labels)

        # Should handle tuple without error
        result = computer(eval_pred)
        assert "eval_wer" in result

    def test_shape_validation_warning(self):
        """Test shape validation warning for unexpected shapes"""
        processor = Mock()
        processor.tokenizer = Mock()
        processor.tokenizer.pad_token_id = 0
        processor.batch_decode = Mock(return_value=["test"])

        computer = HuggingFaceMetricsComputer(
            processor=processor,
            model_type="seq2seq",
            language="ru"
        )

        # 1D predictions (unexpected)
        predictions = np.array([1, 2, 3])
        labels = np.array([[1, 2, 3]])

        eval_pred = EvalPrediction(predictions=predictions, label_ids=labels)

        # Should log warning but not crash
        result = computer(eval_pred)
        assert result is not None


# ============================================================
# Test PerformanceAnalyzer
# ============================================================
class TestPerformanceAnalyzer:
    """Test performance analysis functionality"""

    def test_analyze_errors_basic(self):
        """Test basic error analysis"""
        analyzer = PerformanceAnalyzer(language="ru")

        predictions = ["привет мир", "как дела", "всё хорошо"]
        references = ["привет мир", "как дела", "всё хорошо"]

        analysis = analyzer.analyze_errors(predictions, references)

        assert analysis["total_samples"] == 3
        assert analysis["perfect_matches"] == 3
        assert analysis["empty_predictions"] == 0
        assert analysis["overall_wer"] == 0.0
        assert analysis["perfect_match_rate"] == 1.0

    def test_analyze_errors_with_empty_predictions(self):
        """Test error analysis with empty predictions"""
        analyzer = PerformanceAnalyzer(language="ru")

        predictions = ["привет мир", "", "всё хорошо"]
        references = ["привет мир", "пустой", "всё хорошо"]

        analysis = analyzer.analyze_errors(predictions, references)

        assert analysis["total_samples"] == 3
        assert analysis["empty_predictions"] == 1
        assert analysis["perfect_matches"] == 2

    def test_analyze_errors_with_duration_bins(self):
        """Test error analysis with audio duration binning"""
        analyzer = PerformanceAnalyzer(language="ru")

        predictions = ["привет", "мир", "дела"]
        references = ["привет", "мир", "дела"]
        durations = [3.5, 12.0, 25.0]

        analysis = analyzer.analyze_errors(
            predictions,
            references,
            audio_durations=durations
        )

        assert analysis["wer_by_duration"] is not None
        assert "0-5s" in analysis["wer_by_duration"]
        assert "5-15s" in analysis["wer_by_duration"]
        assert "15-30s" in analysis["wer_by_duration"]

    def test_analyze_errors_by_text_length(self):
        """Test error analysis by text length bins"""
        analyzer = PerformanceAnalyzer(language="ru")

        predictions = [
            "привет",  # Short (1 word) → bin "1-4" (1-3 words)
            "привет мир как дела",  # Medium (4 words) → bin "4-7" (4-6 words)
            "это очень длинный текст с большим количеством слов для тестирования"  # Long (10 words) → bin "10-13" (10-12 words)
        ]
        references = predictions.copy()

        analysis = analyzer.analyze_errors(predictions, references)

        assert "wer_by_length" in analysis
        # Check for new bins (intervals: [start, end))
        assert "1-4" in analysis["wer_by_length"]
        assert "4-7" in analysis["wer_by_length"]
        assert "10-13" in analysis["wer_by_length"]

        # Verify bin structure (corpus-level WER)
        assert "wer" in analysis["wer_by_length"]["1-4"]
        assert "count" in analysis["wer_by_length"]["1-4"]

        # Perfect matches should have WER=0.0
        assert analysis["wer_by_length"]["1-4"]["wer"] == 0.0
        assert analysis["wer_by_length"]["4-7"]["wer"] == 0.0
        assert analysis["wer_by_length"]["10-13"]["wer"] == 0.0

        # Check counts
        assert analysis["wer_by_length"]["1-4"]["count"] == 1
        assert analysis["wer_by_length"]["4-7"]["count"] == 1
        assert analysis["wer_by_length"]["10-13"]["count"] == 1

    def test_compare_models(self):
        """Test model comparison functionality"""
        analyzer = PerformanceAnalyzer(language="ru")

        predictions_dict = {
            "model_a": ["привет мир", "как дела"],
            "model_b": ["првет мир", "как дела"]  # One error
        }
        references = ["привет мир", "как дела"]

        comparison = analyzer.compare_models(predictions_dict, references)

        assert "models" in comparison
        assert "metrics" in comparison
        assert "win_matrix" in comparison

        assert "model_a" in comparison["metrics"]
        assert "model_b" in comparison["metrics"]

        # Model A should have better WER (0.0 vs >0.0)
        assert comparison["metrics"]["model_a"]["wer"] < comparison["metrics"]["model_b"]["wer"]

    def test_error_examples_collection(self):
        """Test that error examples are collected"""
        analyzer = PerformanceAnalyzer(language="ru")

        predictions = ["првет", "мр", "дла"]
        references = ["привет", "мир", "дела"]

        analysis = analyzer.analyze_errors(predictions, references)

        assert "error_examples" in analysis
        assert len(analysis["error_examples"]) > 0

        # Check first error example structure
        example = analysis["error_examples"][0]
        assert "reference" in example
        assert "prediction" in example
        assert "wer" in example


# ============================================================
# Test create_metrics_summary
# ============================================================
class TestCreateMetricsSummary:
    """Test metrics summary creation"""

    def test_create_summary_single_model(self):
        """Test summary creation for single model"""
        result = MetricResult(
            wer=0.25,
            cer=0.15,
            bleu=0.65,
            mer=0.20,
            wil=0.30,
            insertions=5,
            deletions=3,
            substitutions=7,
            hits=85,
            total_predictions=100,
            empty_predictions=2
        )

        summary = create_metrics_summary([result], ["Whisper-Small"])

        assert "Whisper-Small" in summary
        assert "WER: 0.2500" in summary
        assert "CER: 0.1500" in summary
        assert "BLEU: 0.6500" in summary
        assert "MER: 0.2000" in summary
        assert "WIL: 0.3000" in summary
        assert "Substitutions: 7" in summary
        assert "Deletions: 3" in summary
        assert "Insertions: 5" in summary
        assert "Correct (Hits): 85" in summary
        assert "Empty Predictions: 2" in summary

    def test_create_summary_multiple_models(self):
        """Test summary creation for multiple models"""
        result1 = MetricResult(wer=0.25, cer=0.15, total_predictions=100)
        result2 = MetricResult(wer=0.30, cer=0.20, total_predictions=100)

        summary = create_metrics_summary(
            [result1, result2],
            ["Model A", "Model B"]
        )

        assert "Model A" in summary
        assert "Model B" in summary

    def test_create_summary_length_mismatch(self):
        """Test that length mismatch raises ValueError"""
        result = MetricResult(wer=0.25, cer=0.15)

        with pytest.raises(ValueError, match="must have the same length"):
            create_metrics_summary([result], ["Model A", "Model B"])


# ============================================================
# Integration Tests
# ============================================================
class TestMetricsIntegration:
    """Integration tests for complete metrics workflow"""

    def test_end_to_end_metrics_computation(self):
        """Test end-to-end metrics computation workflow"""
        # Create metrics computer
        metrics = STTMetrics(language="ru", use_bleu=False)

        # Sample predictions and references
        predictions = [
            "привет мир",
            "как дела",
            "всё хорошо",
            "",  # Empty prediction
            "ошибка в тексте"
        ]
        references = [
            "привет мир",
            "как дела",
            "всё хорошо",
            "пустой текст",
            "правильный текст"
        ]

        # Compute all metrics
        result = metrics.compute_all_metrics(predictions, references)

        # Verify result structure
        assert isinstance(result, MetricResult)
        assert result.total_predictions == 5
        assert result.empty_predictions == 1
        assert result.wer >= 0.0
        assert result.cer >= 0.0

        # Create summary
        summary = create_metrics_summary([result], ["Test Model"])
        assert len(summary) > 0

    def test_analyzer_with_metrics(self):
        """Test analyzer integration with metrics"""
        analyzer = PerformanceAnalyzer(language="ru")

        predictions = ["привет мир", "как дела"]
        references = ["привет друг", "как дела"]
        durations = [2.5, 3.0]

        analysis = analyzer.analyze_errors(
            predictions,
            references,
            audio_durations=durations
        )

        # Should have comprehensive analysis
        assert "overall_wer" in analysis
        assert "wer_by_length" in analysis
        assert "wer_by_duration" in analysis
        assert analysis["total_samples"] == 2


# ============================================================
# Edge Cases and Error Handling
# ============================================================
class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_all_empty_predictions(self):
        """Test handling when all predictions are empty"""
        metrics = STTMetrics(language="ru")

        predictions = ["", "", ""]
        references = ["test1", "test2", "test3"]

        result = metrics.compute_all_metrics(predictions, references)

        assert result.empty_predictions == 3
        assert result.total_predictions == 3
        assert result.wer == 1.0

    def test_unicode_text_handling(self):
        """Test handling of various Unicode characters"""
        normalizer = TextNormalizer(language="ru")

        text_with_emoji = "Привет 👋 мир 🌍"
        normalized = normalizer.normalize(text_with_emoji)

        # Should remove emojis
        assert "👋" not in normalized
        assert "🌍" not in normalized

    def test_very_long_text(self):
        """Test handling of very long texts"""
        metrics = STTMetrics(language="ru")

        long_text = " ".join(["слово"] * 1000)
        predictions = [long_text]
        references = [long_text]

        # Should not crash
        result = metrics.compute_all_metrics(predictions, references)
        assert result.wer == 0.0

    def test_single_character_texts(self):
        """Test handling of single character texts"""
        metrics = STTMetrics(language="ru")

        predictions = ["а"]
        references = ["б"]

        result = metrics.compute_all_metrics(predictions, references)
        assert result.wer > 0.0
        assert result.cer > 0.0

    def test_special_characters_normalization(self):
        """Test normalization of special characters"""
        normalizer = TextNormalizer(language="ru")

        text = "Тест №1: (проверка) [скобок] {разных} типов!"
        normalized = normalizer.normalize(text)

        # Should remove all special punctuation
        assert "(" not in normalized
        assert ")" not in normalized
        assert "[" not in normalized
        assert "]" not in normalized
        assert "{" not in normalized
        assert "}" not in normalized
        assert "№" not in normalized


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])