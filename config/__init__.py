# Configuration module for pipeline settings

import yaml
from typing import Optional, Dict, Any
from pathlib import Path


class PipelineConfig:
    """Configuration manager for the video annotation pipeline."""
    
    def __init__(self, config_path: str):
        """
        Initialize configuration from a YAML file.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid YAML
        """
        self._config_path = Path(config_path)
        if not self._config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(self._config_path, 'r') as f:
            self._config = yaml.safe_load(f)
        
        if self._config is None:
            raise ValueError("Configuration file is empty")
        
        # Validate configuration on initialization
        self.validate()
    
    # GCS Configuration Properties
    @property
    def gcs_bucket_name(self) -> str:
        """Get the GCS bucket name."""
        return self._config.get('gcs', {}).get('bucket_name', '')
    
    @property
    def video_source_path(self) -> str:
        """Get the GCS path for video sources."""
        return self._config.get('gcs', {}).get('video_source_path', '')
    
    @property
    def script_output_path(self) -> Optional[str]:
        """Get the GCS path for script outputs (optional)."""
        return self._config.get('gcs', {}).get('script_output_path')
    
    @property
    def csv_output_path(self) -> str:
        """Get the GCS path for CSV output."""
        return self._config.get('gcs', {}).get('csv_output_path', '')
    
    # Model Configuration Properties
    @property
    def model_name(self) -> str:
        """Get the LLM model name."""
        return self._config.get('model', {}).get('name', 'gemini-1.5-pro-002')
    
    @property
    def max_retries(self) -> int:
        """Get the maximum number of retries for API calls."""
        return self._config.get('model', {}).get('max_retries', 4)
    
    @property
    def retry_delay(self) -> int:
        """Get the base retry delay in seconds."""
        return self._config.get('model', {}).get('retry_delay', 40)
    
    @property
    def request_delay(self) -> int:
        """Get the delay between requests in seconds."""
        return self._config.get('model', {}).get('request_delay', 3)
    
    # Pipeline Configuration Properties
    @property
    def stage_to_run(self) -> str:
        """Get the pipeline stage to run ('both', 'video_to_script', 'script_to_annotation')."""
        return self._config.get('pipeline', {}).get('stage', 'both')
    
    @property
    def pipeline_mode(self) -> str:
        """Get the pipeline mode ('one_step' or 'two_step'). Default is 'two_step'."""
        return self._config.get('pipeline', {}).get('mode', 'two_step')
    
    @property
    def save_scripts(self) -> bool:
        """Get whether to save intermediate scripts."""
        return self._config.get('pipeline', {}).get('save_scripts', False)
    
    # Safety Settings Properties
    @property
    def safety_settings(self) -> Dict[str, str]:
        """Get the safety settings for LLM."""
        return self._config.get('safety_settings', {
            'harassment': 'BLOCK_NONE',
            'hate_speech': 'BLOCK_NONE',
            'sexually_explicit': 'BLOCK_NONE',
            'dangerous_content': 'BLOCK_NONE'
        })
    
    def validate(self) -> bool:
        """
        Validate the configuration file.
        
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid with specific error message
        """
        errors = []
        
        # Validate GCS configuration
        if not self.gcs_bucket_name:
            errors.append("GCS bucket_name is required")
        
        if not self.video_source_path:
            errors.append("GCS video_source_path is required")
        
        if not self.csv_output_path:
            errors.append("GCS csv_output_path is required")
        
        # Validate that if save_scripts is True, script_output_path must be provided
        if self.save_scripts and not self.script_output_path:
            errors.append("GCS script_output_path is required when save_scripts is True")
        
        # Validate model configuration
        if not self.model_name:
            errors.append("Model name is required")
        
        if not isinstance(self.max_retries, int) or self.max_retries < 0:
            errors.append("Model max_retries must be a non-negative integer")
        
        if not isinstance(self.retry_delay, int) or self.retry_delay < 0:
            errors.append("Model retry_delay must be a non-negative integer")
        
        if not isinstance(self.request_delay, int) or self.request_delay < 0:
            errors.append("Model request_delay must be a non-negative integer")
        
        # Validate pipeline configuration
        valid_stages = ['both', 'video_to_script', 'script_to_annotation']
        if self.stage_to_run not in valid_stages:
            errors.append(f"Pipeline stage must be one of {valid_stages}, got '{self.stage_to_run}'")
        
        valid_modes = ['one_step', 'two_step']
        if self.pipeline_mode not in valid_modes:
            errors.append(f"Pipeline mode must be one of {valid_modes}, got '{self.pipeline_mode}'")
        
        if not isinstance(self.save_scripts, bool):
            errors.append("Pipeline save_scripts must be a boolean")
        
        # Validate safety settings
        valid_safety_values = ['BLOCK_NONE', 'BLOCK_ONLY_HIGH', 'BLOCK_MEDIUM_AND_ABOVE', 'BLOCK_LOW_AND_ABOVE']
        safety_settings = self.safety_settings
        
        for key, value in safety_settings.items():
            if value not in valid_safety_values:
                errors.append(f"Safety setting '{key}' has invalid value '{value}'. Must be one of {valid_safety_values}")
        
        # Raise error if any validation failed
        if errors:
            error_message = "Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors)
            raise ValueError(error_message)
        
        return True
