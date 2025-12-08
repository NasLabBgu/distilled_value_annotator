# Prediction loader for CSV files with model predictions

import csv
import re
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict

from evaluation.models import PredictionResult, PredictionSet
from evaluation.ground_truth_loader import ANNOTATION_CATEGORIES, GroundTruthLoader
from evaluation.video_id_utils import normalize_video_id


class PredictionLoader:
    """
    Loads model predictions from CSV files.
    
    Supports the same CSV formats as GroundTruthLoader:
    1. TikTok format with 'filename' or '1_Link1' column and 1_Value1_<Category>_values columns
    2. Standard format with video_id and category columns
    
    Reuses value conversion logic from GroundTruthLoader.
    """
    
    # Column name mapping from TikTok CSV format to standard category names
    COLUMN_MAPPING = {
        '1_Value1_Self_Direction_Thought_values': 'Self_Direction_Thought',
        '1_Value1_Self_Direction_Action_values': 'Self_Direction_Action',
        '1_Value1_Stimulation_values': 'Stimulation',
        '1_Value1_Hedonism_values': 'Hedonism',
        '1_Value1_Achievement_values': 'Achievement',
        '1_Value1_Power_Resources_values': 'Power_Resources',
        '1_Value1_Power_dominance_values': 'Power_Dominance',  # Note: lowercase 'd'
        '1_Value1_Face_values': 'Face',
        '1_Value1_Security_Personal_values': 'Security_Personal',
        '1_Value1_Security_Social_values': 'Security_Social',
        '1_Value1_Conformity_Rules_values': 'Conformity_Rules',
        '1_Value1_Conformity_Interpersonal_values': 'Conformity_Interpersonal',
        '1_Value1_Tradition_values': 'Tradition',
        '1_Value1_Humility_values': 'Humility',
        '1_Value1_Benevolence_Dependability_values': 'Benevolence_Dependability',
        '1_Value1_Benevolence_Care_values': 'Benevolence_Care',
        '1_Value1_Universalism_Concern_values': 'Universalism_Concern',
        '1_Value1_Universalism_Nature_values': 'Universalism_Nature',
        '1_Value1_Universalism_Tolerance_values': 'Universalism_Tolerance',
    }
    
    def __init__(self, model_name: str = "model"):
        """
        Initialize the prediction loader.
        
        Args:
            model_name: Name to assign to the loaded predictions
        """
        self.model_name = model_name
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def load(self, csv_path: str) -> PredictionSet:
        """
        Load predictions from a CSV file.
        
        Args:
            csv_path: Path to the predictions CSV file
            
        Returns:
            PredictionSet containing all loaded predictions
            
        Raises:
            FileNotFoundError: If the CSV file doesn't exist
        """
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Predictions file not found: {csv_path}")
        
        predictions = []
        parse_errors = []
        self._logged_unexpected_values = set()  # Track logged unexpected values
        
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            columns = reader.fieldnames or []
            
            # Detect format
            is_tiktok_format = any(col in columns for col in ['filename', '1_Link1'])
            has_value_columns = any('_values' in col for col in columns)
            
            self.logger.info(f"Detected format: TikTok={is_tiktok_format}, ValueColumns={has_value_columns}")
            
            # Build case-insensitive column mapping
            column_mapping = self._build_column_mapping(columns)
            self.logger.info(f"Built column mapping for {len(column_mapping)} categories")
            
            for row_num, row in enumerate(reader, start=2):
                try:
                    pred = self._parse_row(row, is_tiktok_format, has_value_columns, column_mapping, row_num)
                    if pred:
                        predictions.append(pred)
                except Exception as e:
                    parse_errors.append(f"Row {row_num}: {e}")
        
        if parse_errors:
            self.logger.warning(f"Parse errors: {len(parse_errors)}")
            for error in parse_errors[:5]:
                self.logger.warning(f"  {error}")
        
        self.logger.info(f"Loaded {len(predictions)} predictions from {csv_path}")
        
        return PredictionSet(
            model_name=self.model_name,
            predictions=predictions,
            total_count=len(predictions),
            success_count=len(predictions),
            failure_count=0,
            failed_video_ids=[],
        )
    
    def _build_column_mapping(self, columns: List[str]) -> Dict[str, str]:
        """
        Build case-insensitive mapping from standard category names to actual column names.
        
        Args:
            columns: List of column names from CSV
            
        Returns:
            Dict mapping lowercase category name -> actual column name
        """
        column_mapping = {}
        
        for col in columns:
            # Check for TikTok value columns pattern: 1_Value1_{Category}_values
            if col.startswith('1_Value1_') and col.endswith('_values'):
                # Extract category part
                category_part = col[9:-7]  # Remove '1_Value1_' prefix and '_values' suffix
                column_mapping[category_part.lower()] = col
        
        # Log any case mismatches
        for category in ANNOTATION_CATEGORIES:
            expected_col = f'1_Value1_{category}_values'
            if expected_col not in columns:
                lower_category = category.lower()
                if lower_category in column_mapping:
                    actual_col = column_mapping[lower_category]
                    self.logger.warning(
                        f"Column case mismatch for '{category}': expected '{expected_col}', "
                        f"found '{actual_col}'"
                    )
                else:
                    self.logger.warning(f"No column found for category '{category}'")
        
        return column_mapping
    
    def _parse_row(
        self,
        row: Dict[str, str],
        is_tiktok_format: bool,
        has_value_columns: bool,
        column_mapping: Dict[str, str],
        row_num: int,
    ) -> Optional[PredictionResult]:
        """
        Parse a single CSV row into a PredictionResult.
        
        Args:
            row: CSV row as dictionary
            is_tiktok_format: Whether this is TikTok format
            has_value_columns: Whether columns have _values suffix
            column_mapping: Mapping from lowercase category to actual column name
            row_num: Row number for logging
            
        Returns:
            PredictionResult or None if row is invalid
        """
        # Extract video ID
        if 'filename' in row:
            video_id = self._extract_video_id(row['filename'])
        elif '1_Link1' in row:
            video_id = normalize_video_id(row['1_Link1'])
        elif 'video_id' in row:
            video_id = row['video_id'].strip()
        else:
            return None
        
        if not video_id:
            return None
        
        # Extract annotations
        annotations = {}
        
        if has_value_columns:
            # TikTok format with 1_Value1_*_values columns
            for category in ANNOTATION_CATEGORIES:
                # Try exact column name first (from COLUMN_MAPPING)
                csv_col = None
                for col, cat in self.COLUMN_MAPPING.items():
                    if cat == category:
                        csv_col = col
                        break
                
                value_str = row.get(csv_col, '').strip() if csv_col else ''
                
                # If not found, try case-insensitive match
                if not value_str and csv_col not in row:
                    lower_category = category.lower()
                    if lower_category in column_mapping:
                        actual_col = column_mapping[lower_category]
                        value_str = row.get(actual_col, '').strip()
                
                numeric_value = self._convert_value(value_str, category, row_num)
                annotations[category] = numeric_value
        else:
            # Standard format with category names as columns
            for category in ANNOTATION_CATEGORIES:
                value_str = row.get(category, '').strip()
                numeric_value = self._convert_value(value_str, category, row_num)
                annotations[category] = numeric_value
        
        return PredictionResult(
            video_id=video_id,
            predictions=annotations,
            success=True,
            error_message=None,
            inference_time=0.0,
        )
    
    def _extract_video_id(self, filename: str) -> str:
        """
        Extract video ID from filename.
        
        Uses normalize_video_id to preserve username_videoid format.
        
        Examples:
            @alexkay_video_6783398367490854150 -> alexkay_6783398367490854150
            username_videoid.txt -> username_videoid
        """
        # Use the proper normalization function from video_id_utils
        return normalize_video_id(filename) or filename
    
    def _convert_value(self, value_str: str, category: str = "", row_num: int = 0) -> int:
        """
        Convert text annotation value to numeric value.
        
        Reuses the same mapping as GroundTruthLoader.
        
        Args:
            value_str: Text value ('present', 'conflict', 'dominant', '', etc.)
            category: Category name for logging
            row_num: Row number for logging
            
        Returns:
            Numeric value (-1, 0, 1, 2)
        """
        if value_str is None:
            return 0
        
        # Use GroundTruthLoader's conversion logic
        result = GroundTruthLoader._convert_value(value_str)
        
        # Log unexpected values (first occurrence only per category+value)
        if result is None and value_str:
            key = (category, value_str)
            if key not in self._logged_unexpected_values:
                self.logger.warning(
                    f"Unexpected value '{value_str}' for category '{category}' "
                    f"in row {row_num}. Expected: '', 'absent', 'conflict', "
                    f"'present', 'dominant', or numeric -1, 0, 1, 2"
                )
                self._logged_unexpected_values.add(key)
        
        # If conversion returned None, default to 0
        return result if result is not None else 0
