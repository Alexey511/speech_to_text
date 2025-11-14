"""
Evaluation metrics for speech-to-text models.
Implements WER, CER, BLEU and other relevant metrics.
"""

import re
import logging
from typing import List, Dict, Tuple, Any, Optional, Union
import numpy as np
import numpy.typing as npt
import torch
from jiwer import wer, cer, mer, wil, process_words
from transformers import EvalPrediction
from dataclasses import dataclass
import evaluate

logger = logging.getLogger(__name__)


@dataclass
class MetricResult:
    """Container for metric results"""
    wer: float = 0.0
    cer: float = 0.0
    bleu: float = 0.0
    mer: float = 0.0  # Match Error Rate
    wil: float = 0.0  # Word Information Lost
    insertions: int = 0
    deletions: int = 0
    substitutions: int = 0
    hits: int = 0
    total_predictions: int = 0
    empty_predictions: int = 0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            "wer": self.wer,
            "cer": self.cer,
            "bleu": self.bleu,
            "mer": self.mer,
            "wil": self.wil,
            "insertions": self.insertions,
            "deletions": self.deletions,
            "substitutions": self.substitutions,
            "hits": self.hits,
            "total_predictions": self.total_predictions,
            "empty_predictions": self.empty_predictions,
            "empty_prediction_rate": self.empty_predictions / max(self.total_predictions, 1)
        }


class TextNormalizer:
    """Text normalization for consistent evaluation"""
    
    def __init__(self, language: str = "ru"):
        self.language = language.lower()
        
        # Russian-specific normalization patterns
        if self.language == "ru":
            self.patterns = [
                # Remove punctuation except apostrophes
                (r"[^\w\s']", " "),
                # Normalize multiple spaces
                (r"\s+", " "),
                # Convert to lowercase
            ]
        else:
            self.patterns = [
                (r"[^\w\s']", " "),
                (r"\s+", " "),
            ]
    
    def normalize(self, text: str) -> str:
        """Normalize text for evaluation"""
        if not text:
            return ""
        
        # Apply patterns
        normalized = text
        for pattern, replacement in self.patterns:
            normalized = re.sub(pattern, replacement, normalized)
        
        # Convert to lowercase and strip
        normalized = normalized.lower().strip()
        
        return normalized
    
    def normalize_batch(self, texts: List[str]) -> List[str]:
        """Normalize a batch of texts"""
        return [self.normalize(text) for text in texts]


class STTMetrics:
    """Speech-to-text evaluation metrics"""
    
    def __init__(self, language: str = "ru", use_bleu: bool = False):
        self.language = language
        self.use_bleu = use_bleu
        self.normalizer = TextNormalizer(language)

        # Load evaluation metrics
        if self.use_bleu:
            try:
                # Load BLEU metric with keep_in_memory=True to avoid Windows cache file issues
                self.bleu_metric = evaluate.load("bleu", keep_in_memory=True)
            except Exception as e:
                logger.warning(f"Could not load BLEU metric: {e}")
                self.bleu_metric = None
                self.use_bleu = False
    
    def compute_wer(self, predictions: List[str], references: List[str]) -> float:
        """Compute Word Error Rate"""
        if not predictions or not references:
            return 1.0
        
        # Normalize texts
        pred_normalized = self.normalizer.normalize_batch(predictions)
        ref_normalized = self.normalizer.normalize_batch(references)
        
        # Filter out empty predictions/references
        valid_pairs = [
            (p, r) for p, r in zip(pred_normalized, ref_normalized) 
            if p.strip() and r.strip()
        ]
        
        if not valid_pairs:
            return 1.0

        pred_valid, ref_valid = zip(*valid_pairs)

        try:
            # Convert tuples to lists for jiwer
            wer_score = wer(list(ref_valid), list(pred_valid))
            return float(wer_score)
        except Exception as e:
            logger.error(f"Error computing WER: {e}")
            return 1.0
    
    def compute_cer(self, predictions: List[str], references: List[str]) -> float:
        """Compute Character Error Rate"""
        if not predictions or not references:
            return 1.0
        
        # Normalize texts
        pred_normalized = self.normalizer.normalize_batch(predictions)
        ref_normalized = self.normalizer.normalize_batch(references)
        
        # Filter out empty predictions/references
        valid_pairs = [
            (p, r) for p, r in zip(pred_normalized, ref_normalized) 
            if p.strip() and r.strip()
        ]
        
        if not valid_pairs:
            return 1.0

        pred_valid, ref_valid = zip(*valid_pairs)

        try:
            # Convert tuples to lists for jiwer
            cer_score = cer(list(ref_valid), list(pred_valid))
            # cer() can return float or dict depending on parameters
            if isinstance(cer_score, dict):
                return float(cer_score.get('cer', 1.0))
            return float(cer_score)
        except Exception as e:
            logger.error(f"Error computing CER: {e}")
            return 1.0
    
    def compute_bleu(self, predictions: List[str], references: List[str]) -> float:
        """Compute BLEU score"""
        if not self.use_bleu:
            logger.debug("BLEU computation skipped: use_bleu=False")
            return 0.0

        if self.bleu_metric is None:
            logger.warning("BLEU computation skipped: bleu_metric is None")
            return 0.0

        if not predictions or not references:
            logger.warning("BLEU computation skipped: empty predictions or references")
            return 0.0

        # Normalize texts
        pred_normalized = self.normalizer.normalize_batch(predictions)
        ref_normalized = self.normalizer.normalize_batch(references)

        # Filter empty predictions/references
        # Note: HuggingFace evaluate BLEU expects strings, not tokenized lists
        valid_pairs = [
            (p, [r]) for p, r in zip(pred_normalized, ref_normalized)
            if p.strip() and r.strip()
        ]

        if not valid_pairs:
            logger.warning(f"BLEU computation skipped: no valid pairs after filtering (total: {len(predictions)})")
            return 0.0

        predictions_str, references_str = zip(*valid_pairs)

        logger.debug(f"Computing BLEU for {len(valid_pairs)} valid pairs")

        try:
            result = self.bleu_metric.compute(
                predictions=list(predictions_str),
                references=list(references_str)
            )
            # Ensure result is not None and has 'bleu' key
            if result is None or "bleu" not in result:
                logger.warning(f"BLEU metric returned invalid result: {result}")
                return 0.0

            return float(result["bleu"])
        except Exception as e:
            logger.error(f"Error computing BLEU: {e}")
            return 0.0
    
    def compute_detailed_measures(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, Any]:
        """Compute detailed error measures including WER, CER, MER, WIL"""
        total_predictions = len(predictions)

        if not predictions or not references:
            logger.warning("Empty predictions or references provided to compute_detailed_measures")
            return {
                "wer": 1.0, "cer": 1.0, "mer": 1.0, "wil": 1.0,
                "substitutions": 0, "deletions": 0, "insertions": 0, "hits": 0,
                "total_predictions": total_predictions, "empty_predictions": total_predictions
            }

        # Normalize texts
        pred_normalized = self.normalizer.normalize_batch(predictions)
        ref_normalized = self.normalizer.normalize_batch(references)

        # Count empty predictions
        empty_predictions = sum(1 for p in pred_normalized if not p.strip())
        if empty_predictions > 0:
            logger.warning(f"Found {empty_predictions}/{total_predictions} empty predictions after normalization")

        # Filter out empty predictions/references
        valid_pairs = [
            (p, r) for p, r in zip(pred_normalized, ref_normalized)
            if p.strip() and r.strip()
        ]

        if not valid_pairs:
            logger.warning("No valid prediction-reference pairs after filtering empty strings")
            return {
                "wer": 1.0, "cer": 1.0, "mer": 1.0, "wil": 1.0,
                "substitutions": 0, "deletions": 0, "insertions": 0, "hits": 0,
                "total_predictions": total_predictions, "empty_predictions": empty_predictions
            }

        pred_valid, ref_valid = zip(*valid_pairs)

        try:
            # Convert tuples to lists for jiwer
            ref_list = list(ref_valid)
            pred_list = list(pred_valid)

            # Use process_words to get all metrics at once
            result = process_words(ref_list, pred_list)
            cer_score = cer(ref_list, pred_list)

            # Handle cer() return type (can be float or dict)
            if isinstance(cer_score, dict):
                cer_value = float(cer_score.get('cer', 1.0))
            else:
                cer_value = float(cer_score)

            return {
                "wer": float(result.wer),
                "cer": cer_value,
                "mer": float(result.mer),
                "wil": float(result.wil),
                "substitutions": int(result.substitutions),
                "deletions": int(result.deletions),
                "insertions": int(result.insertions),
                "hits": int(result.hits),
                "total_predictions": total_predictions,
                "empty_predictions": empty_predictions
            }
        except Exception as e:
            logger.error(f"Error computing detailed measures: {e}")
            return {
                "wer": 1.0, "cer": 1.0, "mer": 1.0, "wil": 1.0,
                "substitutions": 0, "deletions": 0, "insertions": 0, "hits": 0,
                "total_predictions": total_predictions, "empty_predictions": empty_predictions
            }
    
    def compute_all_metrics(
        self,
        predictions: List[str],
        references: List[str]
    ) -> MetricResult:
        """Compute all available metrics"""
        detailed = self.compute_detailed_measures(predictions, references)
        bleu_score = self.compute_bleu(predictions, references) if self.use_bleu else 0.0

        return MetricResult(
            wer=detailed["wer"],
            cer=detailed["cer"],
            bleu=bleu_score,
            mer=detailed["mer"],
            wil=detailed["wil"],
            substitutions=detailed["substitutions"],
            deletions=detailed["deletions"],
            insertions=detailed["insertions"],
            hits=detailed["hits"],
            total_predictions=detailed["total_predictions"],
            empty_predictions=detailed["empty_predictions"]
        )


class HuggingFaceMetricsComputer:
    """
    Metrics computer for HuggingFace Trainer.
    Supports both seq2seq (Whisper, Speech2Text) and CTC (Custom) models.
    """

    def __init__(
        self,
        processor,
        model_type: str = "seq2seq",
        language: str = "ru",
        use_bleu: bool = False
    ):
        """
        Initialize metrics computer.

        Args:
            processor: Model processor (WhisperProcessor, Speech2TextProcessor, etc.)
            model_type: Type of model ("seq2seq" for Whisper/Speech2Text, "ctc" for Custom)
            language: Language code for text normalization
            use_bleu: Whether to compute BLEU score
        """
        self.processor = processor
        self.model_type = model_type.lower()
        self.metrics = STTMetrics(language=language, use_bleu=use_bleu)

        # Validate model_type
        if self.model_type not in ["seq2seq", "ctc"]:
            raise ValueError(f"model_type must be 'seq2seq' or 'ctc', got '{model_type}'")

        logger.info(f"HuggingFaceMetricsComputer initialized with model_type='{self.model_type}'")

    def __call__(self, eval_pred: EvalPrediction) -> Dict[str, float]:
        """
        Compute metrics for HuggingFace Trainer.

        Args:
            eval_pred: EvalPrediction object with predictions and labels

        Returns:
            Dictionary with metric values
        """
        predictions, labels = eval_pred

        # Handle different model types (some return tuples)
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        # Ensure predictions is numpy array
        if not isinstance(predictions, np.ndarray):
            predictions = np.array(predictions)
        if not isinstance(labels, np.ndarray):
            labels = np.array(labels)

        # Log shapes for debugging
        logger.debug(f"Predictions shape: {predictions.shape}, Labels shape: {labels.shape}")

        # Validate predictions shape
        if predictions.ndim not in [2, 3]:
            logger.warning(
                f"Unexpected predictions shape: {predictions.shape}. "
                f"Expected 2D (batch, sequence) for seq2seq or 3D (batch, time, vocab) for CTC."
            )

        # Different decoding logic for seq2seq vs CTC models
        if self.model_type == "ctc":
            # CTC models (Custom): predictions are logits, need argmax
            logger.debug("Decoding CTC predictions using argmax")
            pred_ids: npt.NDArray[np.int64] = np.argmax(predictions, axis=-1)  # (batch, time)
            pred_str = self.processor.batch_decode(pred_ids)

            # For labels, replace -100 with pad token (0)
            label_ids: npt.NDArray[np.int64] = np.where(labels != -100, labels, 0)
            label_str = self.processor.batch_decode(label_ids)

        else:
            # Seq2seq models (Whisper, Speech2Text): predictions are token IDs from generate()
            logger.debug("Decoding seq2seq predictions")

            # Get pad_token_id from processor
            if hasattr(self.processor, 'tokenizer'):
                pad_token_id = self.processor.tokenizer.pad_token_id
            else:
                logger.warning("Processor has no tokenizer attribute, using pad_token_id=0")
                pad_token_id = 0

            # Replace -100 with pad_token_id for decoding
            pred_ids = np.where(predictions != -100, predictions, pad_token_id)
            pred_str = self.processor.batch_decode(pred_ids, skip_special_tokens=True)

            # Decode labels
            label_ids = np.where(labels != -100, labels, pad_token_id)
            label_str = self.processor.batch_decode(label_ids, skip_special_tokens=True)

        # Compute metrics
        result = self.metrics.compute_all_metrics(pred_str, label_str)

        return {
            "eval_wer": result.wer,
            "eval_cer": result.cer,
            "eval_bleu": result.bleu,
            "eval_mer": result.mer,
            "eval_wil": result.wil,
            "eval_empty_prediction_rate": result.empty_predictions / max(result.total_predictions, 1)
        }


class PerformanceAnalyzer:
    """Analyze model performance in detail"""
    
    def __init__(self, language: str = "ru"):
        self.language = language
        self.normalizer = TextNormalizer(language)
    
    def analyze_errors(
        self, 
        predictions: List[str], 
        references: List[str],
        audio_durations: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """Analyze prediction errors in detail"""
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have the same length")
        
        analysis = {
            "total_samples": len(predictions),
            "empty_predictions": 0,
            "perfect_matches": 0,
            "error_examples": [],
            "wer_by_length": {},
            "wer_by_duration": {} if audio_durations else None
        }
        
        # Normalize texts
        pred_normalized = self.normalizer.normalize_batch(predictions)
        ref_normalized = self.normalizer.normalize_batch(references)

        # Length bins for text analysis
        # Bins use [start, end) interval: start <= length < end
        length_bins = [
            (1, 4),   # 1-3 words
            (4, 7),   # 4-6 words
            (7, 10),  # 7-9 words
            (10, 13), # 10-12 words
            (13, 16), # 13-15 words
            (16, float('inf'))  # 16+ words
        ]
        duration_bins = [(0, 5), (5, 15), (15, 30), (30, float('inf'))] if audio_durations else []

        # Initialize bins with lists to collect predictions/references for corpus-level WER
        for start, end in length_bins:
            bin_name = f"{start}-{end if end != float('inf') else 'inf'}"
            analysis["wer_by_length"][bin_name] = {"predictions": [], "references": []}

        if audio_durations:
            for start, end in duration_bins:
                analysis["wer_by_duration"][f"{start}-{end if end != float('inf') else 'inf'}s"] = {"predictions": [], "references": []}

        # Collect all predictions/references for overall corpus-level WER
        all_valid_predictions = []
        all_valid_references = []

        for i, (pred, ref) in enumerate(zip(pred_normalized, ref_normalized)):
            # Check for empty predictions
            if not pred.strip():
                analysis["empty_predictions"] += 1
                continue

            # Check for perfect matches
            if pred.strip() == ref.strip():
                analysis["perfect_matches"] += 1

            # Collect for corpus-level WER
            all_valid_predictions.append(pred)
            all_valid_references.append(ref)

            # Store error examples (first 10 non-perfect matches)
            if pred.strip() != ref.strip() and len(analysis["error_examples"]) < 10:
                try:
                    individual_wer = wer([ref], [pred])
                    analysis["error_examples"].append({
                        "reference": references[i],  # Original, non-normalized
                        "prediction": predictions[i],  # Original, non-normalized
                        "wer": individual_wer
                    })
                except:
                    pass

            # Bin by text length (collect predictions/references for corpus-level WER)
            ref_length = len(ref.split())
            for start, end in length_bins:
                if start <= ref_length < end:
                    bin_name = f"{start}-{end if end != float('inf') else 'inf'}"
                    analysis["wer_by_length"][bin_name]["predictions"].append(pred)
                    analysis["wer_by_length"][bin_name]["references"].append(ref)
                    break

            # Bin by audio duration
            if audio_durations and i < len(audio_durations):
                duration = audio_durations[i]
                for start, end in duration_bins:
                    if start <= duration < end:
                        bin_name = f"{start}-{end if end != float('inf') else 'inf'}s"
                        analysis["wer_by_duration"][bin_name]["predictions"].append(pred)
                        analysis["wer_by_duration"][bin_name]["references"].append(ref)
                        break

        # Calculate corpus-level WER for each length bin
        for bin_name, bin_data in analysis["wer_by_length"].items():
            if bin_data["predictions"] and bin_data["references"]:
                try:
                    bin_wer = wer(bin_data["references"], bin_data["predictions"])
                except:
                    bin_wer = 1.0
            else:
                bin_wer = 0.0

            analysis["wer_by_length"][bin_name] = {
                "wer": bin_wer,
                "count": len(bin_data["predictions"])
            }

        # Calculate corpus-level WER for each duration bin
        if analysis["wer_by_duration"]:
            for bin_name, bin_data in analysis["wer_by_duration"].items():
                if bin_data["predictions"] and bin_data["references"]:
                    try:
                        bin_wer = wer(bin_data["references"], bin_data["predictions"])
                    except:
                        bin_wer = 1.0
                else:
                    bin_wer = 0.0

                analysis["wer_by_duration"][bin_name] = {
                    "wer": bin_wer,
                    "count": len(bin_data["predictions"])
                }

        # Overall corpus-level WER
        if all_valid_predictions and all_valid_references:
            try:
                analysis["overall_wer"] = wer(all_valid_references, all_valid_predictions)
            except:
                analysis["overall_wer"] = 1.0
        else:
            analysis["overall_wer"] = 1.0
        analysis["perfect_match_rate"] = analysis["perfect_matches"] / analysis["total_samples"]
        analysis["empty_prediction_rate"] = analysis["empty_predictions"] / analysis["total_samples"]
        
        return analysis
    
    def compare_models(
        self, 
        predictions_dict: Dict[str, List[str]], 
        references: List[str]
    ) -> Dict[str, Any]:
        """Compare multiple models' performance"""
        comparison = {
            "models": list(predictions_dict.keys()),
            "metrics": {},
            "win_matrix": {}
        }
        
        metrics = STTMetrics(language=self.language)
        
        # Compute metrics for each model
        for model_name, predictions in predictions_dict.items():
            result = metrics.compute_all_metrics(predictions, references)
            comparison["metrics"][model_name] = result.to_dict()
        
        # Create pairwise comparison matrix
        model_names = list(predictions_dict.keys())
        for model1 in model_names:
            comparison["win_matrix"][model1] = {}
            for model2 in model_names:
                if model1 == model2:
                    comparison["win_matrix"][model1][model2] = 0.5  # Tie
                else:
                    wer1 = comparison["metrics"][model1]["wer"]
                    wer2 = comparison["metrics"][model2]["wer"]
                    comparison["win_matrix"][model1][model2] = 1.0 if wer1 < wer2 else 0.0
        
        return comparison


def create_metrics_summary(results: List[MetricResult], model_names: List[str]) -> str:
    """Create a formatted summary of metrics results"""
    if len(results) != len(model_names):
        raise ValueError("Results and model names must have the same length")

    summary = "Model Performance Summary\n"
    summary += "=" * 80 + "\n\n"

    for result, name in zip(results, model_names):
        summary += f"{name}:\n"
        summary += f"  Primary Metrics:\n"
        summary += f"    WER: {result.wer:.4f} ({result.wer:.2%})\n"
        summary += f"    CER: {result.cer:.4f} ({result.cer:.2%})\n"
        if result.bleu > 0:
            summary += f"    BLEU: {result.bleu:.4f}\n"
        if result.mer > 0:
            summary += f"    MER: {result.mer:.4f} ({result.mer:.2%})\n"
        if result.wil > 0:
            summary += f"    WIL: {result.wil:.4f} ({result.wil:.2%})\n"

        total_errors = result.substitutions + result.deletions + result.insertions
        summary += f"\n  Error Breakdown:\n"
        summary += f"    Total Errors: {total_errors}\n"
        summary += f"      - Substitutions: {result.substitutions}\n"
        summary += f"      - Deletions: {result.deletions}\n"
        summary += f"      - Insertions: {result.insertions}\n"
        summary += f"    Correct (Hits): {result.hits}\n"

        if result.total_predictions > 0:
            empty_rate = result.empty_predictions / result.total_predictions
            summary += f"\n  Prediction Stats:\n"
            summary += f"    Total Predictions: {result.total_predictions}\n"
            summary += f"    Empty Predictions: {result.empty_predictions} ({empty_rate:.2%})\n"

        summary += "\n" + "-" * 80 + "\n\n"

    return summary
