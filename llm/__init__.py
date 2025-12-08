# LLM client module for Vertex AI interactions

import time
import random
import vertexai
from typing import Callable, Any, Dict, Optional
from vertexai.generative_models import (
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
)

# Initialize Vertex AI (uses Application Default Credentials)
# This will automatically detect project and location from environment
vertexai.init()


class BaseLLMClient:
    """Base class for LLM clients with retry logic and safety settings."""
    
    def __init__(
        self,
        model_name: str,
        system_instructions: str,
        safety_settings: Dict[str, str],
        max_retries: int,
        retry_delay: int
    ):
        """
        Initialize the base LLM client.
        
        Args:
            model_name: Name of the Vertex AI model to use
            system_instructions: System instructions for the model
            safety_settings: Dictionary of safety settings (harassment, hate_speech, etc.)
            max_retries: Maximum number of retry attempts
            retry_delay: Base delay in seconds for exponential backoff
        """
        self.model_name = model_name
        self.system_instructions = system_instructions
        self.safety_settings = self._convert_safety_settings(safety_settings)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.model = self._create_model()
    
    def _convert_safety_settings(self, config_settings: Dict[str, str]) -> Dict:
        """
        Convert configuration safety settings to Vertex AI format.
        
        Args:
            config_settings: Dictionary with keys like 'harassment', 'hate_speech', etc.
                           and values like 'BLOCK_NONE', 'BLOCK_ONLY_HIGH', etc.
        
        Returns:
            Dictionary mapping HarmCategory enums to HarmBlockThreshold enums
        """
        # Map config keys to HarmCategory enums
        category_mapping = {
            'harassment': HarmCategory.HARM_CATEGORY_HARASSMENT,
            'hate_speech': HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            'sexually_explicit': HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            'dangerous_content': HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        }
        
        # Map config values to HarmBlockThreshold enums
        threshold_mapping = {
            'BLOCK_NONE': HarmBlockThreshold.BLOCK_NONE,
            'BLOCK_ONLY_HIGH': HarmBlockThreshold.BLOCK_ONLY_HIGH,
            'BLOCK_MEDIUM_AND_ABOVE': HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            'BLOCK_LOW_AND_ABOVE': HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        }
        
        # Convert settings
        vertex_settings = {}
        for key, value in config_settings.items():
            if key in category_mapping and value in threshold_mapping:
                vertex_settings[category_mapping[key]] = threshold_mapping[value]
        
        return vertex_settings
    
    def _create_model(self) -> GenerativeModel:
        """
        Create and return a Vertex AI GenerativeModel instance.
        
        Returns:
            Configured GenerativeModel instance
        """
        return GenerativeModel(
            model_name=self.model_name,
            system_instruction=[self.system_instructions],
        )
    
    def _retry_with_backoff(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function with exponential backoff retry logic.
        
        Args:
            func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
        
        Returns:
            Result from the function if successful
            Error message string if all retries fail
        """
        for attempt in range(self.max_retries):
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                error_msg = str(e)
                print(f"Error on attempt {attempt + 1}/{self.max_retries}: {error_msg}")
                
                if attempt < self.max_retries - 1:
                    # Calculate exponential backoff delay with jitter
                    delay = (self.retry_delay * (2 ** attempt)) + random.uniform(0, 1)
                    print(f"Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
                else:
                    return f"Error: Failed after {self.max_retries} retries. Last error: {error_msg}"
        
        return f"Error: Failed after {self.max_retries} retries."



class VideoScriptLLMClient(BaseLLMClient):
    """LLM client for converting videos to movie scripts."""
    
    def __init__(
        self,
        model_name: str,
        safety_settings: Dict[str, str],
        max_retries: int,
        retry_delay: int
    ):
        """
        Initialize the video-to-script LLM client.
        
        Args:
            model_name: Name of the Vertex AI model to use
            safety_settings: Dictionary of safety settings
            max_retries: Maximum number of retry attempts
            retry_delay: Base delay in seconds for exponential backoff
        """
        # Load system instructions from file
        import os
        instructions_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'prompts',
            'video_to_script_instructions.txt'
        )
        
        with open(instructions_path, 'r', encoding='utf-8') as f:
            system_instructions = f.read()
        
        super().__init__(
            model_name=model_name,
            system_instructions=system_instructions,
            safety_settings=safety_settings,
            max_retries=max_retries,
            retry_delay=retry_delay
        )
    
    def generate_script_from_video(self, video_uri: str) -> str:
        """
        Generate a movie script from a video URI.
        
        Args:
            video_uri: GCS URI of the video file (e.g., gs://bucket/path/video.mp4)
        
        Returns:
            Generated movie script as a string, or error message if failed
        """
        from vertexai.generative_models import Part
        
        def _generate():
            # Create video part from URI
            contents = Part.from_uri(uri=video_uri, mime_type="video/mp4")
            prompt = [
                Part.from_text("Video: "),
                contents,
            ]
            
            # Generate content with safety settings
            response = self.model.generate_content(
                prompt,
                safety_settings=self.safety_settings
            )
            
            # Check if response has text
            if response.text:
                return response.text
            else:
                # Handle cases where content was blocked
                return f"Error: Could not generate script for {video_uri}, {response.prompt_feedback}"
        
        # Use retry logic
        return self._retry_with_backoff(_generate)



class AnnotationLLMClient(BaseLLMClient):
    """LLM client for extracting value annotations from movie scripts."""
    
    def __init__(
        self,
        model_name: str,
        safety_settings: Dict[str, str],
        max_retries: int,
        retry_delay: int
    ):
        """
        Initialize the script-to-annotation LLM client.
        
        Args:
            model_name: Name of the Vertex AI model to use
            safety_settings: Dictionary of safety settings
            max_retries: Maximum number of retry attempts
            retry_delay: Base delay in seconds for exponential backoff
        """
        # Load system instructions from file
        import os
        instructions_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'prompts',
            'script_to_annotation_instructions.txt'
        )
        
        with open(instructions_path, 'r', encoding='utf-8') as f:
            system_instructions = f.read()
        
        super().__init__(
            model_name=model_name,
            system_instructions=system_instructions,
            safety_settings=safety_settings,
            max_retries=max_retries,
            retry_delay=retry_delay
        )
    
    def generate_annotations_from_script(self, script_text: str) -> str:
        """
        Generate value annotations from a movie script.
        
        Args:
            script_text: The movie script text to analyze
        
        Returns:
            Generated annotations as a string (JSON format), or error message if failed
        """
        from vertexai.generative_models import Part
        
        def _generate():
            # Create prompt with script text
            prompt = [Part.from_text(script_text)]
            
            # Generate content with safety settings
            response = self.model.generate_content(
                prompt,
                safety_settings=self.safety_settings
            )
            
            # Check if response has text
            if response.text:
                return response.text
            else:
                # Handle cases where content was blocked
                return f"Error: Could not generate annotations, {response.prompt_feedback}"
        
        # Use retry logic
        return self._retry_with_backoff(_generate)


class OneStepAnnotationLLMClient(BaseLLMClient):
    """LLM client for generating value annotations directly from videos (one-step mode)."""
    
    def __init__(
        self,
        model_name: str,
        safety_settings: Dict[str, str],
        max_retries: int,
        retry_delay: int
    ):
        """
        Initialize the one-step video-to-annotation LLM client.
        
        Args:
            model_name: Name of the Vertex AI model to use
            safety_settings: Dictionary of safety settings
            max_retries: Maximum number of retry attempts
            retry_delay: Base delay in seconds for exponential backoff
        """
        # Load system instructions from file
        import os
        instructions_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'prompts',
            'videos_to_annotations_one-step.txt'
        )
        
        with open(instructions_path, 'r', encoding='utf-8') as f:
            system_instructions = f.read()
        
        super().__init__(
            model_name=model_name,
            system_instructions=system_instructions,
            safety_settings=safety_settings,
            max_retries=max_retries,
            retry_delay=retry_delay
        )
    
    def generate_annotations_from_video(self, video_uri: str) -> str:
        """
        Generate value annotations directly from a video URI (one-step mode).
        
        Args:
            video_uri: GCS URI of the video file (e.g., gs://bucket/path/video.mp4)
        
        Returns:
            Generated annotations as a string (JSON format), or error message if failed
        """
        from vertexai.generative_models import Part
        
        def _generate():
            # Create video part from URI (same pattern as VideoScriptLLMClient)
            contents = Part.from_uri(uri=video_uri, mime_type="video/mp4")
            prompt = [
                Part.from_text("Video: "),
                contents,
            ]
            
            # Generate content with safety settings
            response = self.model.generate_content(
                prompt,
                safety_settings=self.safety_settings
            )
            
            # Check if response has text
            if response.text:
                return response.text
            else:
                # Handle cases where content was blocked
                return f"Error: Could not generate annotations for {video_uri}, {response.prompt_feedback}"
        
        # Use retry logic
        return self._retry_with_backoff(_generate)
