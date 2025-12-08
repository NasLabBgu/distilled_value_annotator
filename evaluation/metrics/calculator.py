# Metrics calculator for model evaluation

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import Counter

from evaluation.models import (
    VideoAnnotation,
    PredictionResult,
    PredictionSet,
    GroundTruthDataset,
    MetricScores,
)
from evaluation.video_id_utils import normalize_video_id


# Define the 19 annotation categories
ANNOTATION_CATEGORIES = [
    "Self_Direction_Thought",
    "Self_Direction_Action",
    "Stimulation",
    "Hedonism",
    "Achievement",
    "Power_Resources",
    "Power_Dominance",
    "Face",
    "Security_Personal",
    "Security_Social",
    "Conformity_Rules",
    "Conformity_Interpersonal",
    "Tradition",
    "Humility",
    "Benevolence_Dependability",
    "Benevolence_Care",
    "Universalism_Concern",
    "Universalism_Nature",
    "Universalism_Tolerance",
]


@dataclass
class AlignmentResult:
    """Result of aligning predictions with ground truth."""
    
    aligned_pairs: List[Tuple[PredictionResult, VideoAnnotation]]
    """List of (prediction, ground_truth) pairs that were matched."""
    
    matched_count: int = 0
    """Number of predictions matched with ground truth."""
    
    unmatched_predictions: List[str] = field(default_factory=list)
    """Video IDs from predictions not found in ground truth."""
    
    missing_predictions: List[str] = field(default_factory=list)
    """Video IDs in ground truth without predictions."""


@dataclass
class CategoryResult:
    """Metrics for a single category."""
    category: str
    precision: float
    recall: float
    f1: float
    support: int
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0


@dataclass
class AggregateResult:
    """Aggregate metrics across categories."""
    macro_f1: float
    weighted_f1: float
    macro_precision: float
    macro_recall: float
    categories_evaluated: int


@dataclass 
class ModelEvaluationResult:
    """Complete evaluation result for a model."""
    model_name: str
    total_predictions: int
    successful_predictions: int
    failed_predictions: int
    matched_with_ground_truth: int
    unmatched_count: int
    missing_count: int
    
    # Aggregate metrics by value type
    endorsed_aggregate: AggregateResult
    conflict_aggregate: AggregateResult
    combined_aggregate: AggregateResult
    
    # Per-category metrics by value type
    per_category_endorsed: Dict[str, CategoryResult]
    per_category_conflict: Dict[str, CategoryResult]
    per_category_combined: Dict[str, CategoryResult]


class MetricsCalculator:
    """
    Calculates evaluation metrics by comparing model predictions to ground truth.
    
    Supports:
    - Per-category precision, recall, F1
    - Aggregate macro and weighted F1
    - Separate metrics for endorsed (1,2), conflict (-1), and combined
    - Rare category filtering by frequency threshold
    """
    
    def __init__(
        self,
        ground_truth: GroundTruthDataset,
        min_support: int = 0,
        min_frequency_threshold: float = 0.05,
        treat_dominant_as_present: bool = True,
    ):
        """
        Initialize the metrics calculator.
        
        Args:
            ground_truth: The ground truth dataset to compare against
            min_support: Minimum support (count) for a category to be included in metrics
            min_frequency_threshold: Minimum frequency (0.0-1.0) for a category to be included.
                                    Categories with frequency below this threshold are excluded.
            treat_dominant_as_present: If True, treat dominant (2) same as present (1) for multiclass metrics.
                                      Default is True (1 and 2 are considered equivalent).
        """
        self._ground_truth = ground_truth
        self._min_support = min_support
        self._min_frequency_threshold = min_frequency_threshold
        self._treat_dominant_as_present = treat_dominant_as_present
        
        # Calculate category frequencies in ground truth
        self._category_frequencies = self._calculate_category_frequencies()
        
        # Build lookup index for ground truth by video_id
        self._gt_by_video_id: Dict[str, VideoAnnotation] = {
            video.video_id: video for video in ground_truth.videos
        }
        
        # Build lookup by normalized video_id (username_videoid format)
        self._gt_by_normalized_id: Dict[str, VideoAnnotation] = {}
        for video in ground_truth.videos:
            normalized = normalize_video_id(video.video_id)
            self._gt_by_normalized_id[normalized] = video
            
            # Also index by numeric ID only for fallback matching
            parts = normalized.rsplit('_', 1)
            if len(parts) == 2 and parts[1].isdigit():
                numeric_id = parts[1]
                # Only add if not conflicting with another video
                if numeric_id not in self._gt_by_normalized_id:
                    self._gt_by_normalized_id[numeric_id] = video
        
        logging.info(
            f"MetricsCalculator initialized with {len(ground_truth.videos)} ground truth videos"
        )
    
    def _calculate_category_frequencies(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate frequency of each value type for each category in ground truth.
        
        Returns:
            Dictionary mapping category -> value_type -> frequency (0.0 to 1.0)
        """
        frequencies = {}
        total_videos = len(self._ground_truth.videos)
        
        if total_videos == 0:
            return {cat: {"endorsed": 0.0, "conflict": 0.0} for cat in ANNOTATION_CATEGORIES}
        
        for category in ANNOTATION_CATEGORIES:
            endorsed_count = 0
            conflict_count = 0
            
            for video in self._ground_truth.videos:
                value = video.annotations.get(category)
                if value in {1, 2}:
                    endorsed_count += 1
                elif value == -1:
                    conflict_count += 1
            
            frequencies[category] = {
                "endorsed": endorsed_count / total_videos,
                "conflict": conflict_count / total_videos,
            }
        
        return frequencies
    
    def get_excluded_categories(self, value_type: str = "endorsed") -> List[str]:
        """
        Get list of categories excluded due to low frequency.
        
        Args:
            value_type: "endorsed" or "conflict"
            
        Returns:
            List of category names below the frequency threshold
        """
        excluded = []
        for category, freqs in self._category_frequencies.items():
            if freqs.get(value_type, 0.0) < self._min_frequency_threshold:
                excluded.append(category)
        return excluded
    
    def align_predictions(
        self,
        predictions: PredictionSet,
    ) -> AlignmentResult:
        """
        Align predictions with ground truth by video_id.
        
        Matches by username_videoid format for better accuracy.
        
        Args:
            predictions: The prediction set to align
            
        Returns:
            AlignmentResult with matched pairs and unmatched/missing counts
        """
        aligned_pairs = []
        unmatched_predictions = []
        matched_gt_ids = set()
        
        for pred in predictions.predictions:
            if not pred.success:
                # Skip failed predictions
                continue
            
            # Try normalized match (username_videoid format)
            normalized_pred_id = normalize_video_id(pred.video_id)
            gt_video = self._gt_by_normalized_id.get(normalized_pred_id)
            
            # Try exact match next
            if gt_video is None:
                gt_video = self._gt_by_video_id.get(pred.video_id)
            
            if gt_video is not None:
                aligned_pairs.append((pred, gt_video))
                matched_gt_ids.add(gt_video.video_id)
            else:
                unmatched_predictions.append(pred.video_id)
        
        # Find ground truth videos without predictions
        missing_predictions = [
            video.video_id
            for video in self._ground_truth.videos
            if video.video_id not in matched_gt_ids
        ]
        
        result = AlignmentResult(
            aligned_pairs=aligned_pairs,
            matched_count=len(aligned_pairs),
            unmatched_predictions=unmatched_predictions,
            missing_predictions=missing_predictions,
        )
        
        logging.info(
            f"Alignment complete: {result.matched_count} matched, "
            f"{len(unmatched_predictions)} unmatched predictions, "
            f"{len(missing_predictions)} missing predictions"
        )
        
        return result
    
    def calculate_category_metrics(
        self,
        alignment: AlignmentResult,
        value_type: str = "endorsed",
    ) -> Dict[str, CategoryResult]:
        """
        Calculate per-category metrics.
        
        Args:
            alignment: The alignment result from align_predictions()
            value_type: One of "endorsed", "conflict", or "combined"
            
        Returns:
            Dictionary mapping category name to CategoryResult
        """
        if value_type not in {"endorsed", "conflict", "combined"}:
            raise ValueError(f"Invalid value_type: {value_type}")
        
        category_metrics = {}
        
        for category in ANNOTATION_CATEGORIES:
            # Collect predictions and ground truth for this category
            y_true = []
            y_pred = []
            
            for pred, gt in alignment.aligned_pairs:
                gt_value = gt.annotations.get(category)
                pred_value = pred.predictions.get(category)
                
                if gt_value is None or pred_value is None:
                    continue
                
                # Convert to binary based on value_type
                if value_type == "endorsed":
                    # Positive class: 1 or 2
                    y_true.append(1 if gt_value in {1, 2} else 0)
                    y_pred.append(1 if pred_value in {1, 2} else 0)
                elif value_type == "conflict":
                    # Positive class: -1
                    y_true.append(1 if gt_value == -1 else 0)
                    y_pred.append(1 if pred_value == -1 else 0)
                else:  # combined
                    # Multi-class: optionally treat dominant (2) as present (1)
                    if self._treat_dominant_as_present:
                        # Normalize: 2 -> 1, keep others as-is
                        gt_normalized = 1 if gt_value == 2 else gt_value
                        pred_normalized = 1 if pred_value == 2 else pred_value
                        y_true.append(gt_normalized)
                        y_pred.append(pred_normalized)
                    else:
                        # Keep original values
                        y_true.append(gt_value)
                        y_pred.append(pred_value)
            
            # Calculate metrics
            if value_type in {"endorsed", "conflict"}:
                metrics = self._calculate_binary_metrics(y_true, y_pred)
            else:
                metrics = self._calculate_multiclass_metrics(y_true, y_pred)
            
            # Calculate support (count of positive class in ground truth)
            if value_type == "endorsed":
                support = sum(1 for v in y_true if v == 1)
            elif value_type == "conflict":
                support = sum(1 for v in y_true if v == 1)
            else:
                # For combined: support is count of non-zero (endorsed OR conflict) in ground truth
                support = sum(1 for v in y_true if v != 0)
            
            category_metrics[category] = CategoryResult(
                category=category,
                precision=metrics["precision"],
                recall=metrics["recall"],
                f1=metrics["f1"],
                support=support,
                true_positives=metrics.get("tp", 0),
                false_positives=metrics.get("fp", 0),
                false_negatives=metrics.get("fn", 0),
            )
        
        return category_metrics
    
    def _calculate_binary_metrics(
        self,
        y_true: List[int],
        y_pred: List[int],
    ) -> Dict[str, float]:
        """Calculate precision, recall, F1 for binary classification."""
        if not y_true:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": 0}
        
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
    
    def _calculate_multiclass_metrics(
        self,
        y_true: List[int],
        y_pred: List[int],
    ) -> Dict[str, float]:
        """Calculate macro-averaged metrics for multi-class classification."""
        if not y_true:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        # Get all unique classes
        classes = sorted(set(y_true) | set(y_pred))
        
        # Calculate per-class metrics and average
        precisions = []
        recalls = []
        f1s = []
        
        for cls in classes:
            # Convert to binary: this class vs rest
            y_true_bin = [1 if v == cls else 0 for v in y_true]
            y_pred_bin = [1 if v == cls else 0 for v in y_pred]
            
            metrics = self._calculate_binary_metrics(y_true_bin, y_pred_bin)
            
            # Only include classes that appear in ground truth
            if sum(y_true_bin) > 0:
                precisions.append(metrics["precision"])
                recalls.append(metrics["recall"])
                f1s.append(metrics["f1"])
        
        return {
            "precision": sum(precisions) / len(precisions) if precisions else 0.0,
            "recall": sum(recalls) / len(recalls) if recalls else 0.0,
            "f1": sum(f1s) / len(f1s) if f1s else 0.0,
        }
    
    def calculate_aggregate_metrics(
        self,
        category_metrics: Dict[str, CategoryResult],
        value_type: str = "endorsed",
    ) -> AggregateResult:
        """
        Calculate aggregate metrics across all categories.
        
        Args:
            category_metrics: Per-category metrics from calculate_category_metrics()
            value_type: "endorsed", "conflict", or "combined" - used for frequency filtering
            
        Returns:
            AggregateMetrics with macro and weighted F1 scores
        """
        # Filter categories with minimum support
        valid_categories = {
            cat: metrics
            for cat, metrics in category_metrics.items()
            if metrics.support >= self._min_support
        }
        
        # Track excluded categories for logging
        excluded_by_support = [
            cat for cat in category_metrics
            if category_metrics[cat].support < self._min_support
        ]
        excluded_by_frequency = []
        
        # Also filter by frequency threshold if set
        if self._min_frequency_threshold > 0.0 and value_type in {"endorsed", "conflict"}:
            for cat in list(valid_categories.keys()):
                freq = self._category_frequencies.get(cat, {}).get(value_type, 0.0)
                if freq < self._min_frequency_threshold:
                    excluded_by_frequency.append((cat, freq))
                    del valid_categories[cat]
        
        # Log excluded categories
        if excluded_by_support:
            logging.debug(
                f"[{value_type}] Excluded {len(excluded_by_support)} categories by min_support={self._min_support}: "
                f"{excluded_by_support}"
            )
        if excluded_by_frequency:
            logging.info(
                f"[{value_type}] Excluded {len(excluded_by_frequency)} categories by frequency < {self._min_frequency_threshold:.1%}: "
                f"{[(cat, f'{freq:.2%}') for cat, freq in excluded_by_frequency]}"
            )
        
        if not valid_categories:
            return AggregateResult(
                macro_f1=0.0,
                weighted_f1=0.0,
                macro_precision=0.0,
                macro_recall=0.0,
                categories_evaluated=0,
            )
        
        # Macro F1: unweighted mean of category F1 scores
        f1_scores = [m.f1 for m in valid_categories.values()]
        precision_scores = [m.precision for m in valid_categories.values()]
        recall_scores = [m.recall for m in valid_categories.values()]
        
        macro_f1 = sum(f1_scores) / len(f1_scores)
        macro_precision = sum(precision_scores) / len(precision_scores)
        macro_recall = sum(recall_scores) / len(recall_scores)
        
        # Weighted F1: weighted by support
        total_support = sum(m.support for m in valid_categories.values())
        if total_support > 0:
            weighted_f1 = sum(
                m.f1 * m.support for m in valid_categories.values()
            ) / total_support
        else:
            weighted_f1 = 0.0
        
        return AggregateResult(
            macro_f1=macro_f1,
            weighted_f1=weighted_f1,
            macro_precision=macro_precision,
            macro_recall=macro_recall,
            categories_evaluated=len(valid_categories),
        )
    
    def calculate_model_metrics(
        self,
        predictions: PredictionSet,
    ) -> ModelEvaluationResult:
        """
        Calculate all metrics for a model.
        
        Args:
            predictions: The prediction set to evaluate
            
        Returns:
            ModelMetrics with per-category and aggregate metrics
        """
        # Align predictions with ground truth
        alignment = self.align_predictions(predictions)
        
        # Calculate per-category metrics for each value type
        endorsed_category = self.calculate_category_metrics(alignment, "endorsed")
        conflict_category = self.calculate_category_metrics(alignment, "conflict")
        combined_category = self.calculate_category_metrics(alignment, "combined")
        
        # Calculate aggregate metrics with value_type for frequency filtering
        endorsed_aggregate = self.calculate_aggregate_metrics(endorsed_category, "endorsed")
        conflict_aggregate = self.calculate_aggregate_metrics(conflict_category, "conflict")
        combined_aggregate = self.calculate_aggregate_metrics(combined_category, "combined")
        
        return ModelEvaluationResult(
            model_name=predictions.model_name,
            total_predictions=predictions.total_count,
            successful_predictions=predictions.success_count,
            failed_predictions=predictions.failure_count,
            matched_with_ground_truth=alignment.matched_count,
            unmatched_count=len(alignment.unmatched_predictions),
            missing_count=len(alignment.missing_predictions),
            endorsed_aggregate=endorsed_aggregate,
            conflict_aggregate=conflict_aggregate,
            combined_aggregate=combined_aggregate,
            per_category_endorsed=endorsed_category,
            per_category_conflict=conflict_category,
            per_category_combined=combined_category,
        )