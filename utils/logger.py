"""
Pipeline logging module for structured error tracking and reporting.
"""

import logging
from typing import Dict, List
from datetime import datetime


class PipelineLogger:
    """
    Centralized logger for pipeline execution with structured error tracking.
    Stores errors by stage and item for comprehensive failure reporting.
    """
    
    def __init__(self, name: str = "PipelineLogger"):
        """
        Initialize the pipeline logger.
        
        Args:
            name: Name for the logger instance
        """
        self.logger = logging.getLogger(name)
        self.logger.propagate = False  # Prevent duplicate log messages
        self.errors: Dict[str, List[Dict[str, str]]] = {}
        self.warnings: List[str] = []
        self.info_messages: List[str] = []
    
    def log_error(self, stage: str, item: str, error: str) -> None:
        """
        Log an error with context about the stage and item.
        
        Args:
            stage: Pipeline stage where error occurred (e.g., 'video_to_script', 'script_to_annotation')
            item: Identifier for the item that failed (e.g., video URI, script path)
            error: Error message or exception details
        """
        timestamp = datetime.now().isoformat()
        
        # Store error in structured format
        if stage not in self.errors:
            self.errors[stage] = []
        
        self.errors[stage].append({
            'item': item,
            'error': error,
            'timestamp': timestamp
        })
        
        # Also log to standard logger
        self.logger.error(f"[{stage}] Error processing {item}: {error}")
    
    def log_warning(self, message: str) -> None:
        """
        Log a warning message.
        
        Args:
            message: Warning message to log
        """
        timestamp = datetime.now().isoformat()
        self.warnings.append(f"[{timestamp}] {message}")
        self.logger.warning(message)
    
    def log_info(self, message: str) -> None:
        """
        Log an informational message.
        
        Args:
            message: Info message to log
        """
        timestamp = datetime.now().isoformat()
        self.info_messages.append(f"[{timestamp}] {message}")
        self.logger.info(message)
    
    def get_failure_summary(self) -> Dict[str, List[Dict[str, str]]]:
        """
        Get a structured summary of all failures organized by stage.
        
        Returns:
            Dictionary mapping stage names to lists of failure details.
            Each failure detail contains 'item', 'error', and 'timestamp'.
            
        Example:
            {
                'video_to_script': [
                    {
                        'item': 'gs://bucket/video1.mp4',
                        'error': 'API rate limit exceeded',
                        'timestamp': '2024-01-01T12:00:00'
                    }
                ],
                'script_to_annotation': [
                    {
                        'item': 'gs://bucket/script1.txt',
                        'error': 'Invalid JSON response',
                        'timestamp': '2024-01-01T12:05:00'
                    }
                ]
            }
        """
        return self.errors.copy()
    
    def get_error_count(self) -> int:
        """
        Get the total number of errors logged.
        
        Returns:
            Total count of errors across all stages
        """
        return sum(len(errors) for errors in self.errors.values())
    
    def get_errors_by_stage(self, stage: str) -> List[Dict[str, str]]:
        """
        Get all errors for a specific stage.
        
        Args:
            stage: Pipeline stage name
        
        Returns:
            List of error details for the specified stage
        """
        return self.errors.get(stage, [])
    
    def has_errors(self) -> bool:
        """
        Check if any errors have been logged.
        
        Returns:
            True if errors exist, False otherwise
        """
        return len(self.errors) > 0
    
    def clear_errors(self) -> None:
        """Clear all logged errors."""
        self.errors.clear()
    
    def print_summary(self) -> None:
        """Print a formatted summary of all logged errors."""
        if not self.has_errors():
            self.logger.info("No errors logged")
            return
        
        self.logger.info("=" * 60)
        self.logger.info("ERROR SUMMARY")
        self.logger.info("=" * 60)
        
        for stage, errors in self.errors.items():
            self.logger.info(f"\n{stage}: {len(errors)} error(s)")
            for idx, error_detail in enumerate(errors, 1):
                self.logger.info(f"  {idx}. {error_detail['item']}")
                self.logger.info(f"     Error: {error_detail['error']}")
                self.logger.info(f"     Time: {error_detail['timestamp']}")
        
        self.logger.info("=" * 60)
