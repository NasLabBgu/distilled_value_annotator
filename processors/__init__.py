# Processors module for video and script processing

import time
import logging
import json
import re
from typing import List, Tuple, Optional, Dict
from llm import VideoScriptLLMClient, AnnotationLLMClient, OneStepAnnotationLLMClient
from gcs import GCSInterface
from utils.logger import PipelineLogger


class VideoToScriptProcessor:
    """
    Processor for converting videos to movie scripts using LLM.
    Handles batch processing with retry logic and optional script saving.
    """
    
    def __init__(
        self,
        llm_client: VideoScriptLLMClient,
        gcs_interface: GCSInterface,
        request_delay: int,
        save_scripts: bool,
        script_output_path: Optional[str],
        pipeline_logger: Optional[PipelineLogger] = None
    ):
        """
        Initialize the video-to-script processor.
        
        Args:
            llm_client: VideoScriptLLMClient instance for generating scripts
            gcs_interface: GCSInterface instance for GCS operations
            request_delay: Delay in seconds between API requests
            save_scripts: Whether to save generated scripts to GCS
            script_output_path: GCS path for saving scripts (required if save_scripts is True)
            pipeline_logger: Optional PipelineLogger instance for structured error tracking
        """
        self.llm_client = llm_client
        self.gcs_interface = gcs_interface
        self.request_delay = request_delay
        self.save_scripts = save_scripts
        self.script_output_path = script_output_path
        self.pipeline_logger = pipeline_logger or PipelineLogger("VideoToScriptProcessor")
        
        # Validate configuration
        if self.save_scripts and not self.script_output_path:
            raise ValueError("script_output_path is required when save_scripts is True")
        
        logging.info("VideoToScriptProcessor initialized")
        self.pipeline_logger.log_info("VideoToScriptProcessor initialized")
    
    def _process_single_video(self, video_uri: str) -> Optional[str]:
        """
        Process a single video to generate a movie script.
        
        Args:
            video_uri: GCS URI of the video file
        
        Returns:
            Generated script text if successful, None if failed
        """
        try:
            logging.info(f"Processing video: {video_uri}")
            self.pipeline_logger.log_info(f"Processing video: {video_uri}")
            
            # Call LLM client to generate script
            script = self.llm_client.generate_script_from_video(video_uri)
            
            # Check if the result is an error message
            if isinstance(script, str) and script.startswith("Error:"):
                error_msg = f"Failed to generate script: {script}"
                logging.error(f"Failed to generate script for {video_uri}: {script}")
                self.pipeline_logger.log_error("video_to_script", video_uri, error_msg)
                return None
            
            logging.info(f"Successfully generated script for {video_uri}")
            self.pipeline_logger.log_info(f"Successfully generated script for {video_uri}")
            return script
        
        except Exception as e:
            error_msg = str(e)
            logging.error(f"Error processing video {video_uri}: {error_msg}")
            self.pipeline_logger.log_error("video_to_script", video_uri, error_msg)
            return None
    
    def process_videos(self, video_uris: List[str]) -> Tuple[List[str], List[str]]:
        """
        Process multiple videos to generate movie scripts.
        
        Args:
            video_uris: List of GCS URIs for video files
        
        Returns:
            Tuple of (scripts_or_script_uris, failed_uris)
            - If save_scripts is True: returns list of GCS URIs where scripts were saved
            - If save_scripts is False: returns list of script texts (in-memory)
            - failed_uris: list of video URIs that failed to process
        """
        scripts_or_uris = []
        failed_uris = []
        total_videos = len(video_uris)
        
        logging.info(f"Starting batch processing of {total_videos} videos")
        self.pipeline_logger.log_info(f"Starting batch processing of {total_videos} videos")
        
        for idx, video_uri in enumerate(video_uris, 1):
            progress_msg = f"Progress: {idx}/{total_videos} - Processing {video_uri}"
            logging.info(progress_msg)
            self.pipeline_logger.log_info(progress_msg)
            
            # Process single video
            script = self._process_single_video(video_uri)
            
            if script is None:
                # Track failure
                failed_uris.append(video_uri)
                warning_msg = f"Failed to process video {idx}/{total_videos}: {video_uri}"
                logging.warning(warning_msg)
                self.pipeline_logger.log_warning(warning_msg)
            else:
                # Handle successful script generation
                if self.save_scripts:
                    # Extract video filename (e.g., @username_video_12345.mp4) and create script path
                    # Preserve the full filename format for video identification in CSV
                    video_filename = video_uri.split('/')[-1]  # e.g., @username_video_12345.mp4
                    script_filename = video_filename.rsplit('.', 1)[0] + '.txt'  # e.g., @username_video_12345.txt
                    script_path = f"{self.script_output_path.rstrip('/')}/{script_filename}"
                    
                    # Save script to GCS
                    success = self.gcs_interface.save_script(script, script_path)
                    
                    if success:
                        script_uri = f"gs://{self.gcs_interface.bucket_name}/{script_path}"
                        scripts_or_uris.append(script_uri)
                        logging.info(f"Saved script to {script_uri}")
                        self.pipeline_logger.log_info(f"Saved script to {script_uri}")
                    else:
                        # If save failed, track as failure
                        failed_uris.append(video_uri)
                        error_msg = "Failed to save script to GCS"
                        logging.error(f"Failed to save script for {video_uri}")
                        self.pipeline_logger.log_error("video_to_script", video_uri, error_msg)
                else:
                    # Keep script in memory
                    scripts_or_uris.append(script)
                    logging.info(f"Script generated and kept in memory for {video_uri}")
                    self.pipeline_logger.log_info(f"Script generated and kept in memory for {video_uri}")
            
            # Apply delay between requests (except after the last video)
            if idx < total_videos:
                logging.debug(f"Waiting {self.request_delay} seconds before next request...")
                time.sleep(self.request_delay)
        
        # Log summary
        success_count = len(scripts_or_uris)
        failure_count = len(failed_uris)
        summary_msg = f"Batch processing complete: {success_count} successful, {failure_count} failed"
        logging.info(summary_msg)
        self.pipeline_logger.log_info(summary_msg)
        
        return scripts_or_uris, failed_uris



class ScriptToAnnotationProcessor:
    """
    Processor for extracting value annotations from movie scripts using LLM.
    Handles batch processing with retry logic and JSON parsing.
    """
    
    def __init__(
        self,
        llm_client: AnnotationLLMClient,
        gcs_interface: GCSInterface,
        request_delay: int,
        pipeline_logger: Optional[PipelineLogger] = None
    ):
        """
        Initialize the script-to-annotation processor.
        
        Args:
            llm_client: AnnotationLLMClient instance for generating annotations
            gcs_interface: GCSInterface instance for GCS operations
            request_delay: Delay in seconds between API requests
            pipeline_logger: Optional PipelineLogger instance for structured error tracking
        """
        self.llm_client = llm_client
        self.gcs_interface = gcs_interface
        self.request_delay = request_delay
        self.pipeline_logger = pipeline_logger or PipelineLogger("ScriptToAnnotationProcessor")
        
        logging.info("ScriptToAnnotationProcessor initialized")
        self.pipeline_logger.log_info("ScriptToAnnotationProcessor initialized")
    
    def _extract_json_and_text(self, response: str) -> Dict:
        """
        Extract JSON and text notes from LLM response.
        Handles responses with JSON in markdown code blocks or plain JSON.
        
        Args:
            response: Raw response string from LLM
        
        Returns:
            Dictionary containing parsed JSON data and optional notes
            Returns empty dict if parsing fails
        """
        try:
            # Try to extract JSON from markdown code blocks first
            # Pattern matches ```json ... ``` or ``` ... ```
            json_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
            matches = re.findall(json_pattern, response, re.DOTALL)
            
            json_data = None
            
            if matches:
                # Try to parse the first JSON block found
                for match in matches:
                    try:
                        json_data = json.loads(match.strip())
                        break
                    except json.JSONDecodeError:
                        continue
            
            # If no JSON found in code blocks, try parsing the entire response
            if json_data is None:
                try:
                    json_data = json.loads(response.strip())
                except json.JSONDecodeError:
                    # Try to find JSON object in the response
                    json_obj_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
                    json_matches = re.findall(json_obj_pattern, response, re.DOTALL)
                    
                    for json_match in json_matches:
                        try:
                            json_data = json.loads(json_match)
                            break
                        except json.JSONDecodeError:
                            continue
            
            if json_data is None:
                error_msg = "Failed to extract JSON from response"
                logging.error(error_msg)
                self.pipeline_logger.log_error("script_to_annotation", "JSON parsing", error_msg)
                return {}
            
            # Extract text notes if present (text outside code blocks)
            notes = ""
            if matches:
                # Remove code blocks from response to get remaining text
                text_without_code = re.sub(json_pattern, '', response, flags=re.DOTALL)
                notes = text_without_code.strip()
            
            # Add notes to the JSON data if present
            if notes and 'notes' not in json_data:
                json_data['notes'] = notes
            
            return json_data
        
        except Exception as e:
            error_msg = f"Error extracting JSON and text: {str(e)}"
            logging.error(error_msg)
            self.pipeline_logger.log_error("script_to_annotation", "JSON parsing", error_msg)
            return {}
    
    def _process_single_script(self, script_source: str) -> Optional[Dict]:
        """
        Process a single script to generate value annotations.
        
        Args:
            script_source: Either a GCS URI (gs://...) or in-memory script text
        
        Returns:
            Dictionary containing video_id and annotation values if successful, None if failed
        """
        try:
            # Determine if script_source is a URI or in-memory text
            if script_source.startswith("gs://"):
                # Read script from GCS
                logging.info(f"Processing script from GCS: {script_source}")
                self.pipeline_logger.log_info(f"Processing script from GCS: {script_source}")
                script_text = self.gcs_interface.read_script(script_source)
                
                # Extract video ID from filename
                # Format: gs://bucket/path/@username_video_12345.txt -> @username_video_12345
                filename = script_source.split('/')[-1]
                video_id = filename.rsplit('.', 1)[0]  # Remove extension
            else:
                # Use in-memory script
                logging.info("Processing in-memory script")
                self.pipeline_logger.log_info("Processing in-memory script")
                script_text = script_source
                
                # For in-memory scripts, we need to extract video_id from the script or use a placeholder
                # This will be handled by the caller passing the video URI alongside
                video_id = "unknown"
            
            # Call LLM client to generate annotations
            response = self.llm_client.generate_annotations_from_script(script_text)
            
            # Check if the result is an error message
            if isinstance(response, str) and response.startswith("Error:"):
                error_msg = f"Failed to generate annotations: {response}"
                logging.error(error_msg)
                self.pipeline_logger.log_error("script_to_annotation", script_source, error_msg)
                return None
            
            # Parse response using _extract_json_and_text
            annotation_data = self._extract_json_and_text(response)
            
            if not annotation_data:
                error_msg = "Failed to parse annotation response"
                logging.error(error_msg)
                self.pipeline_logger.log_error("script_to_annotation", script_source, error_msg)
                return None
            
            # Add video_id to the annotation data
            annotation_data['video_id'] = video_id
            
            logging.info(f"Successfully generated annotations for {video_id}")
            self.pipeline_logger.log_info(f"Successfully generated annotations for {video_id}")
            return annotation_data
        
        except Exception as e:
            error_msg = str(e)
            logging.error(f"Error processing script: {error_msg}")
            self.pipeline_logger.log_error("script_to_annotation", script_source, error_msg)
            return None
    
    def process_scripts(self, script_sources: List[str]) -> Tuple[List[Dict], List[str]]:
        """
        Process multiple scripts to generate value annotations.
        
        Args:
            script_sources: List of GCS URIs or in-memory script texts
        
        Returns:
            Tuple of (annotations, failed_sources)
            - annotations: list of annotation dictionaries
            - failed_sources: list of script sources that failed to process
        """
        annotations = []
        failed_sources = []
        total_scripts = len(script_sources)
        
        logging.info(f"Starting batch processing of {total_scripts} scripts")
        self.pipeline_logger.log_info(f"Starting batch processing of {total_scripts} scripts")
        
        for idx, script_source in enumerate(script_sources, 1):
            # Create a display identifier for logging
            if script_source.startswith("gs://"):
                display_id = script_source.split('/')[-1]
            else:
                display_id = f"in-memory script {idx}"
            
            progress_msg = f"Progress: {idx}/{total_scripts} - Processing {display_id}"
            logging.info(progress_msg)
            self.pipeline_logger.log_info(progress_msg)
            
            # Process single script
            annotation = self._process_single_script(script_source)
            
            if annotation is None:
                # Track failure
                failed_sources.append(script_source)
                warning_msg = f"Failed to process script {idx}/{total_scripts}: {display_id}"
                logging.warning(warning_msg)
                self.pipeline_logger.log_warning(warning_msg)
            else:
                # Add successful annotation
                annotations.append(annotation)
                success_msg = f"Successfully processed script {idx}/{total_scripts}"
                logging.info(success_msg)
                self.pipeline_logger.log_info(success_msg)
            
            # Apply delay between requests (except after the last script)
            if idx < total_scripts:
                logging.debug(f"Waiting {self.request_delay} seconds before next request...")
                time.sleep(self.request_delay)
        
        # Log summary
        success_count = len(annotations)
        failure_count = len(failed_sources)
        summary_msg = f"Batch processing complete: {success_count} successful, {failure_count} failed"
        logging.info(summary_msg)
        self.pipeline_logger.log_info(summary_msg)
        
        return annotations, failed_sources


class VideoToAnnotationProcessor:
    """
    Processor for converting videos directly to value annotations using LLM (one-step mode).
    Handles batch processing with retry logic and JSON parsing.
    """
    
    def __init__(
        self,
        llm_client: OneStepAnnotationLLMClient,
        gcs_interface: GCSInterface,
        request_delay: int,
        pipeline_logger: Optional[PipelineLogger] = None
    ):
        """
        Initialize the video-to-annotation processor (one-step mode).
        
        Args:
            llm_client: OneStepAnnotationLLMClient instance for generating annotations
            gcs_interface: GCSInterface instance for GCS operations
            request_delay: Delay in seconds between API requests
            pipeline_logger: Optional PipelineLogger instance for structured error tracking
        """
        self.llm_client = llm_client
        self.gcs_interface = gcs_interface
        self.request_delay = request_delay
        self.pipeline_logger = pipeline_logger or PipelineLogger("VideoToAnnotationProcessor")
        
        logging.info("VideoToAnnotationProcessor initialized")
        self.pipeline_logger.log_info("VideoToAnnotationProcessor initialized")
    
    def _extract_json_and_text(self, response: str) -> Dict:
        """
        Extract JSON and text notes from LLM response.
        Handles responses with JSON in markdown code blocks or plain JSON.
        
        Args:
            response: Raw response string from LLM
        
        Returns:
            Dictionary containing parsed JSON data and optional notes
            Returns empty dict if parsing fails
        """
        try:
            # Try to extract JSON from markdown code blocks first
            # Pattern matches ```json ... ``` or ``` ... ```
            json_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
            matches = re.findall(json_pattern, response, re.DOTALL)
            
            json_data = None
            
            if matches:
                # Try to parse the first JSON block found
                for match in matches:
                    try:
                        json_data = json.loads(match.strip())
                        break
                    except json.JSONDecodeError:
                        continue
            
            # If no JSON found in code blocks, try parsing the entire response
            if json_data is None:
                try:
                    json_data = json.loads(response.strip())
                except json.JSONDecodeError:
                    # Try to find JSON object in the response
                    json_obj_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
                    json_matches = re.findall(json_obj_pattern, response, re.DOTALL)
                    
                    for json_match in json_matches:
                        try:
                            json_data = json.loads(json_match)
                            break
                        except json.JSONDecodeError:
                            continue
            
            if json_data is None:
                error_msg = "Failed to extract JSON from response"
                logging.error(error_msg)
                self.pipeline_logger.log_error("video_to_annotation", "JSON parsing", error_msg)
                return {}
            
            # Extract text notes if present (text outside code blocks)
            notes = ""
            if matches:
                # Remove code blocks from response to get remaining text
                text_without_code = re.sub(json_pattern, '', response, flags=re.DOTALL)
                notes = text_without_code.strip()
            
            # Add notes to the JSON data if present
            if notes and 'notes' not in json_data:
                json_data['notes'] = notes
            
            return json_data
        
        except Exception as e:
            error_msg = f"Error extracting JSON and text: {str(e)}"
            logging.error(error_msg)
            self.pipeline_logger.log_error("video_to_annotation", "JSON parsing", error_msg)
            return {}
    
    def _process_single_video(self, video_uri: str) -> Optional[Dict]:
        """
        Process a single video directly to generate value annotations (one-step mode).
        
        Args:
            video_uri: GCS URI of the video file (e.g., gs://bucket/path/video.mp4)
        
        Returns:
            Dictionary containing video_id and annotation values if successful, None if failed
        """
        try:
            logging.info(f"Processing video (one-step): {video_uri}")
            self.pipeline_logger.log_info(f"Processing video (one-step): {video_uri}")
            
            # Extract video ID from filename
            # Format: gs://bucket/path/@username_video_12345.mp4 -> @username_video_12345
            filename = video_uri.split('/')[-1]
            video_id = filename.rsplit('.', 1)[0]  # Remove extension
            
            # Call LLM client to generate annotations directly from video
            response = self.llm_client.generate_annotations_from_video(video_uri)
            
            # Check if the result is an error message
            if isinstance(response, str) and response.startswith("Error:"):
                error_msg = f"Failed to generate annotations: {response}"
                logging.error(error_msg)
                self.pipeline_logger.log_error("video_to_annotation", video_uri, error_msg)
                return None
            
            # Parse response using _extract_json_and_text
            annotation_data = self._extract_json_and_text(response)
            
            if not annotation_data:
                error_msg = "Failed to parse annotation response"
                logging.error(error_msg)
                self.pipeline_logger.log_error("video_to_annotation", video_uri, error_msg)
                return None
            
            # Add video_id to the annotation data
            annotation_data['video_id'] = video_id
            
            logging.info(f"Successfully generated annotations for {video_id}")
            self.pipeline_logger.log_info(f"Successfully generated annotations for {video_id}")
            return annotation_data
        
        except Exception as e:
            error_msg = str(e)
            logging.error(f"Error processing video {video_uri}: {error_msg}")
            self.pipeline_logger.log_error("video_to_annotation", video_uri, error_msg)
            return None
    
    def process_videos(self, video_uris: List[str]) -> Tuple[List[Dict], List[str]]:
        """
        Process multiple videos directly to generate value annotations (one-step mode).
        
        Args:
            video_uris: List of GCS URIs for video files
        
        Returns:
            Tuple of (annotations, failed_uris)
            - annotations: list of annotation dictionaries
            - failed_uris: list of video URIs that failed to process
        """
        annotations = []
        failed_uris = []
        total_videos = len(video_uris)
        
        logging.info(f"Starting batch processing of {total_videos} videos (one-step mode)")
        self.pipeline_logger.log_info(f"Starting batch processing of {total_videos} videos (one-step mode)")
        
        for idx, video_uri in enumerate(video_uris, 1):
            progress_msg = f"Progress: {idx}/{total_videos} - Processing {video_uri}"
            logging.info(progress_msg)
            self.pipeline_logger.log_info(progress_msg)
            
            # Process single video
            annotation = self._process_single_video(video_uri)
            
            if annotation is None:
                # Track failure
                failed_uris.append(video_uri)
                warning_msg = f"Failed to process video {idx}/{total_videos}: {video_uri}"
                logging.warning(warning_msg)
                self.pipeline_logger.log_warning(warning_msg)
            else:
                # Add successful annotation
                annotations.append(annotation)
                success_msg = f"Successfully processed video {idx}/{total_videos}"
                logging.info(success_msg)
                self.pipeline_logger.log_info(success_msg)
            
            # Apply delay between requests (except after the last video)
            if idx < total_videos:
                logging.debug(f"Waiting {self.request_delay} seconds before next request...")
                time.sleep(self.request_delay)
        
        # Log summary
        success_count = len(annotations)
        failure_count = len(failed_uris)
        summary_msg = f"Batch processing complete: {success_count} successful, {failure_count} failed"
        logging.info(summary_msg)
        self.pipeline_logger.log_info(summary_msg)
        
        return annotations, failed_uris
