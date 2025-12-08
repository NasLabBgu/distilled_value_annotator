# Ground truth dataset loader and validator

import csv
import json
import os
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import random

from .models import VideoAnnotation, GroundTruthDataset
from .video_id_utils import normalize_video_id

logger = logging.getLogger(__name__)


# The 19 annotation categories based on Schwartz's value framework
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


class ValidationResult:
    """Result of ground truth validation."""
    
    def __init__(self):
        self.errors: List[str] = []
        self.valid_count: int = 0
        self.invalid_count: int = 0
    
    def add_error(self, error: str):
        """Add a validation error."""
        self.errors.append(error)
        self.invalid_count += 1
    
    def add_valid(self):
        """Increment valid count."""
        self.valid_count += 1
    
    @property
    def is_valid(self) -> bool:
        """Check if validation passed."""
        return len(self.errors) == 0


class GroundTruthLoader:
    """
    Loads and validates ground truth dataset from CSV or JSON files.
    
    Supports loading from local filesystem or GCS paths.
    Validates completeness and value ranges.
    Supports sampling for quick testing.
    """
    
    # Mapping from text values to numeric values
    VALUE_MAPPING = {
        '': 0,              # Empty/absent
        'absent': 0,        # Explicit absent
        'conflict': -1,     # Conflict value
        'present': 1,       # Present/endorsed
        'dominant': 2,      # Dominant/strongly endorsed
        '-1': -1,
        '0': 0,
        '1': 1,
        '2': 2,
        'None': 0,
        'none': 0,
        'null': 0,
        'NULL': 0,
        "{'present': 1}": 1,
        "{'absent': 0}": 0,
        "{'conflict': -1}": -1,
        "{'dominant': 2}": 2,
    }
    
    def __init__(
        self,
        dataset_path: str,
        sample_size: Optional[int] = None,
        random_seed: Optional[int] = None
    ):
        """
        Initialize the ground truth loader.
        
        Args:
            dataset_path: Path to the ground truth dataset file (CSV or JSON)
            sample_size: Optional number of videos to sample (None = use all)
            random_seed: Random seed for reproducible sampling
        """
        self.dataset_path = dataset_path
        self.sample_size = sample_size
        self.random_seed = random_seed
        
        if random_seed is not None:
            random.seed(random_seed)
    
    @staticmethod
    def _convert_value(value_str: str) -> Optional[int]:
        """
        Convert text annotation value to numeric value.
        
        Args:
            value_str: Text value (e.g., 'present', 'conflict', 'dominant', '')
            
        Returns:
            Numeric value (-1, 0, 1, 2) or None if invalid
        """
        # Normalize the value
        normalized = value_str.strip().lower()
        
        # Try direct mapping
        if normalized in GroundTruthLoader.VALUE_MAPPING:
            return GroundTruthLoader.VALUE_MAPPING[normalized]
        
        # Try parsing as integer (for files that already have numeric values)
        try:
            int_value = int(normalized)
            if int_value in {-1, 0, 1, 2}:
                return int_value
        except ValueError:
            pass
        
        # Invalid value
        return None
    
    def load(self) -> GroundTruthDataset:
        """
        Load the ground truth dataset from file.
        
        Returns:
            GroundTruthDataset object with loaded videos
            
        Raises:
            FileNotFoundError: If the dataset file doesn't exist
            ValueError: If the file format is not supported
        """
        # Determine file format from extension
        path = Path(self.dataset_path)
        
        # Handle GCS paths
        if self.dataset_path.startswith("gs://"):
            return self._load_from_gcs()
        
        # Check if file exists
        if not path.exists():
            raise FileNotFoundError(f"Ground truth file not found: {self.dataset_path}")
        
        # Load based on file extension
        if path.suffix.lower() == '.csv':
            videos = self._load_csv(path)
        elif path.suffix.lower() == '.json':
            videos = self._load_json(path)
        else:
            raise ValueError(
                f"Unsupported file format: {path.suffix}. "
                "Supported formats: .csv, .json"
            )
        
        # Validate the dataset
        validation_result = self.validate(videos)
        
        # Filter out invalid videos
        valid_videos = [
            video for video in videos
            if self._is_video_valid(video)
        ]
        
        # Apply sampling if requested
        if self.sample_size is not None and self.sample_size < len(valid_videos):
            valid_videos = self._sample_dataset(valid_videos)
        
        # Create and return dataset
        dataset = GroundTruthDataset(
            videos=valid_videos,
            total_count=len(videos),
            valid_count=len(valid_videos),
            validation_errors=validation_result.errors
        )
        
        return dataset
    
    def _load_csv(self, path: Path) -> List[VideoAnnotation]:
        """
        Load ground truth from CSV file.
        
        Expected CSV format:
        - video_id or 1_Link1: Video URI/identifier
        - Columns for each annotation category with text or numeric values
        
        Supports both formats:
        1. Standard format with video_id, video_uri, script_uri, has_sound, and category columns
        2. TikTok format with 1_Link1 and 1_Value1_<Category>_values columns
        
        Args:
            path: Path to CSV file
            
        Returns:
            List of VideoAnnotation objects
        """
        videos = []
        
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            # Detect format by checking column names
            first_row = next(reader, None)
            if first_row is None:
                return videos
            
            # Build case-insensitive column mapping for TikTok format
            # Maps lowercase category name -> actual column name in CSV
            column_mapping: Dict[str, str] = {}
            for col in first_row.keys():
                # Check for TikTok value columns pattern: 1_Value1_{Category}_values
                if col.startswith('1_Value1_') and col.endswith('_values'):
                    # Extract category part
                    category_part = col[9:-7]  # Remove '1_Value1_' prefix and '_values' suffix
                    column_mapping[category_part.lower()] = col
            
            # Log detected columns
            logger.info(f"Detected {len(column_mapping)} value columns in CSV")
            
            # Check for category name mismatches
            for category in ANNOTATION_CATEGORIES:
                expected_col = f'1_Value1_{category}_values'
                if expected_col not in first_row:
                    # Try case-insensitive match
                    lower_category = category.lower()
                    if lower_category in column_mapping:
                        actual_col = column_mapping[lower_category]
                        if actual_col != expected_col:
                            logger.warning(
                                f"Column name case mismatch: expected '{expected_col}', "
                                f"found '{actual_col}'"
                            )
                    else:
                        logger.warning(f"Column not found for category '{category}'")
            
            # Reset to beginning
            f.seek(0)
            reader = csv.DictReader(f)
            
            # Determine format
            has_standard_format = 'video_id' in first_row or 'video_uri' in first_row
            has_tiktok_format = '1_Link1' in first_row
            
            for row_num, row in enumerate(reader, start=2):  # Start at 2 (header is row 1)
                try:
                    if has_tiktok_format:
                        # TikTok format
                        video_uri = row.get('1_Link1', '').strip()
                        
                        # Use normalize_video_id for consistent ID extraction
                        video_id = normalize_video_id(video_uri)
                        if not video_id:
                            video_id = f"video_{row_num}"
                        
                        # Script URI - construct from video URI
                        script_uri = video_uri.replace('.mp4', '.txt').replace('/video/', '/scripts/')
                        
                        # Assume has sound for TikTok videos
                        has_sound = True
                        
                        # Extract annotations with case-insensitive matching
                        annotations = {}
                        for category in ANNOTATION_CATEGORIES:
                            # Try exact column name first
                            col_name = f'1_Value1_{category}_values'
                            value_str = row.get(col_name, '').strip()
                            
                            # If not found, try case-insensitive match
                            if not value_str and col_name not in row:
                                lower_category = category.lower()
                                if lower_category in column_mapping:
                                    actual_col = column_mapping[lower_category]
                                    value_str = row.get(actual_col, '').strip()
                            
                            # Convert text value to numeric
                            numeric_value = self._convert_value(value_str)
                            
                            # Log unexpected values (first occurrence only per category)
                            if numeric_value is None and value_str:
                                if not hasattr(self, '_logged_unexpected_values'):
                                    self._logged_unexpected_values = set()
                                key = (category, value_str)
                                if key not in self._logged_unexpected_values:
                                    logger.warning(
                                        f"Unexpected value '{value_str}' for category '{category}' "
                                        f"in row {row_num}. Expected: '', 'absent', 'conflict', "
                                        f"'present', 'dominant', or numeric -1, 0, 1, 2"
                                    )
                                    self._logged_unexpected_values.add(key)
                            
                            annotations[category] = numeric_value
                    
                    else:
                        # Standard format
                        video_id = row.get('video_id', '').strip()
                        video_uri = row.get('video_uri', '').strip()
                        script_uri = row.get('script_uri', '').strip()
                        has_sound_str = row.get('has_sound', '').strip().lower()
                        
                        # Parse has_sound
                        has_sound = has_sound_str in ('true', '1', 'yes')
                        
                        # Extract annotations for all 19 categories
                        annotations = {}
                        for category in ANNOTATION_CATEGORIES:
                            value_str = row.get(category, '').strip()
                            
                            # Convert value (handles both text and numeric)
                            numeric_value = self._convert_value(value_str)
                            annotations[category] = numeric_value
                    
                    # Create VideoAnnotation (validation happens in __post_init__)
                    # We'll catch validation errors and continue
                    try:
                        video = VideoAnnotation(
                            video_id=video_id,
                            video_uri=video_uri,
                            script_uri=script_uri,
                            annotations=annotations,
                            has_sound=has_sound
                        )
                        videos.append(video)
                    except (ValueError, TypeError) as e:
                        # Create a video with the error for validation reporting
                        videos.append(self._create_invalid_video(
                            video_id, video_uri, script_uri, annotations, has_sound
                        ))
                
                except Exception as e:
                    # Skip malformed rows but log them
                    print(f"Warning: Skipping malformed row {row_num}: {e}")
                    continue
        
        return videos
    
    def _load_json(self, path: Path) -> List[VideoAnnotation]:
        """
        Load ground truth from JSON file.
        
        Expected JSON format:
        [
            {
                "video_id": "...",
                "video_uri": "...",
                "script_uri": "...",
                "has_sound": true/false,
                "annotations": {
                    "category_name": value,
                    ...
                }
            },
            ...
        ]
        
        Args:
            path: Path to JSON file
            
        Returns:
            List of VideoAnnotation objects
        """
        videos = []
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise ValueError("JSON file must contain a list of video objects")
        
        for item in data:
            try:
                video_id = item.get('video_id', '')
                video_uri = item.get('video_uri', '')
                script_uri = item.get('script_uri', '')
                has_sound = item.get('has_sound', False)
                annotations = item.get('annotations', {})
                
                # Create VideoAnnotation
                try:
                    video = VideoAnnotation(
                        video_id=video_id,
                        video_uri=video_uri,
                        script_uri=script_uri,
                        annotations=annotations,
                        has_sound=has_sound
                    )
                    videos.append(video)
                except (ValueError, TypeError) as e:
                    # Create invalid video for validation reporting
                    videos.append(self._create_invalid_video(
                        video_id, video_uri, script_uri, annotations, has_sound
                    ))
            
            except Exception as e:
                print(f"Warning: Skipping malformed video entry: {e}")
                continue
        
        return videos
    
    def _load_from_gcs(self) -> GroundTruthDataset:
        """
        Load ground truth from Google Cloud Storage.
        
        Returns:
            GroundTruthDataset object
            
        Raises:
            NotImplementedError: GCS support not yet implemented
        """
        # TODO: Implement GCS loading using google-cloud-storage library
        raise NotImplementedError(
            "GCS loading not yet implemented. "
            "Please download the file locally first."
        )
    
    def _create_invalid_video(
        self,
        video_id: str,
        video_uri: str,
        script_uri: str,
        annotations: dict,
        has_sound: bool
    ) -> VideoAnnotation:
        """
        Create a VideoAnnotation object that bypasses validation.
        Used for tracking invalid videos during validation.
        """
        # Create object without calling __post_init__
        video = object.__new__(VideoAnnotation)
        video.video_id = video_id
        video.video_uri = video_uri
        video.script_uri = script_uri
        video.annotations = annotations
        video.has_sound = has_sound
        video.script_text = None
        return video
    
    def validate(self, videos: List[VideoAnnotation]) -> ValidationResult:
        """
        Validate the ground truth dataset.
        
        Checks:
        - All videos have annotations for all 19 categories
        - All annotation values are in {-1, 0, 1, 2}
        
        Args:
            videos: List of VideoAnnotation objects to validate
            
        Returns:
            ValidationResult with errors and counts
        """
        result = ValidationResult()
        
        for video in videos:
            video_errors = []
            
            # Check completeness: all 19 categories present
            missing_categories = set(ANNOTATION_CATEGORIES) - set(video.annotations.keys())
            if missing_categories:
                video_errors.append(
                    f"Video {video.video_id}: Missing annotations for categories: "
                    f"{', '.join(sorted(missing_categories))}"
                )
            
            # Check for extra categories
            extra_categories = set(video.annotations.keys()) - set(ANNOTATION_CATEGORIES)
            if extra_categories:
                video_errors.append(
                    f"Video {video.video_id}: Unexpected annotation categories: "
                    f"{', '.join(sorted(extra_categories))}"
                )
            
            # Check value ranges
            for category, value in video.annotations.items():
                if value is None:
                    video_errors.append(
                        f"Video {video.video_id}: Missing value for category {category}"
                    )
                elif value not in {-1, 0, 1, 2}:
                    video_errors.append(
                        f"Video {video.video_id}: Invalid value {value} for category {category}. "
                        f"Must be one of {{-1, 0, 1, 2}}"
                    )
            
            # Check required fields
            if not video.video_id:
                video_errors.append("Video has empty video_id")
            if not video.video_uri:
                video_errors.append(f"Video {video.video_id}: Missing video_uri")
            if not video.script_uri:
                video_errors.append(f"Video {video.video_id}: Missing script_uri")
            
            # Record results
            if video_errors:
                for error in video_errors:
                    result.add_error(error)
            else:
                result.add_valid()
        
        return result
    
    def _is_video_valid(self, video: VideoAnnotation) -> bool:
        """
        Check if a single video is valid.
        
        Args:
            video: VideoAnnotation to check
            
        Returns:
            True if valid, False otherwise
        """
        # Check required fields
        if not video.video_id or not video.video_uri or not video.script_uri:
            return False
        
        # Check completeness
        if set(video.annotations.keys()) != set(ANNOTATION_CATEGORIES):
            return False
        
        # Check value ranges
        for value in video.annotations.values():
            if value is None or value not in {-1, 0, 1, 2}:
                return False
        
        return True
    
    def _sample_dataset(self, videos: List[VideoAnnotation]) -> List[VideoAnnotation]:
        """
        Sample a subset of videos using stratified sampling.
        
        Maintains the distribution of endorsed/conflict/absent values across categories.
        
        Args:
            videos: List of all valid videos
            
        Returns:
            Sampled list of videos
        """
        if self.sample_size is None or self.sample_size >= len(videos):
            return videos
        
        # Calculate value type distribution for stratification
        # We'll stratify based on the proportion of endorsed values per video
        
        # Compute endorsed proportion for each video
        video_scores = []
        for video in videos:
            endorsed_count = sum(
                1 for value in video.annotations.values()
                if value in {1, 2}
            )
            endorsed_proportion = endorsed_count / len(ANNOTATION_CATEGORIES)
            video_scores.append((video, endorsed_proportion))
        
        # Sort by endorsed proportion
        video_scores.sort(key=lambda x: x[1])
        
        # Divide into strata (quintiles)
        num_strata = min(5, len(video_scores))  # Use fewer strata if we have few videos
        strata_size = len(video_scores) // num_strata
        strata = []
        
        for i in range(num_strata):
            start_idx = i * strata_size
            if i == num_strata - 1:
                # Last stratum gets remaining videos
                end_idx = len(video_scores)
            else:
                end_idx = (i + 1) * strata_size
            strata.append([v for v, _ in video_scores[start_idx:end_idx]])
        
        # Sample proportionally from each stratum
        sampled_videos = []
        samples_per_stratum = self.sample_size // num_strata
        remaining_samples = self.sample_size % num_strata
        
        for i, stratum in enumerate(strata):
            # Calculate samples for this stratum
            n_samples = samples_per_stratum
            if i < remaining_samples:
                n_samples += 1
            
            # Ensure we don't try to sample more than available
            n_samples = min(n_samples, len(stratum))
            
            # Sample from stratum
            if n_samples >= len(stratum):
                sampled_videos.extend(stratum)
            else:
                sampled_videos.extend(random.sample(stratum, n_samples))
        
        # If we still don't have enough samples (due to rounding), add more from remaining videos
        if len(sampled_videos) < self.sample_size:
            # Get videos not yet sampled
            sampled_ids = {v.video_id for v in sampled_videos}
            remaining = [v for v in videos if v.video_id not in sampled_ids]
            needed = self.sample_size - len(sampled_videos)
            if remaining and needed > 0:
                additional = random.sample(remaining, min(needed, len(remaining)))
                sampled_videos.extend(additional)
        
        return sampled_videos
