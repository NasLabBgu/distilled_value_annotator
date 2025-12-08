# Pipeline orchestrator module for coordinating pipeline execution

import logging
from typing import Dict, Any, List, Tuple
from config import PipelineConfig
from gcs import GCSInterface
from llm import VideoScriptLLMClient, AnnotationLLMClient, OneStepAnnotationLLMClient
from processors import VideoToScriptProcessor, ScriptToAnnotationProcessor, VideoToAnnotationProcessor
from utils import CSVGenerator, PipelineLogger


class PipelineOrchestrator:
    """
    Orchestrates the complete video annotation pipeline.
    Coordinates execution of video-to-script and script-to-annotation stages.
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize the pipeline orchestrator with configuration.
        
        Args:
            config: PipelineConfig instance with all pipeline settings
        """
        self.config = config
        
        # Initialize pipeline logger
        self.pipeline_logger = PipelineLogger("PipelineOrchestrator")
        
        # Initialize GCS interface
        self.gcs_interface = GCSInterface(bucket_name=config.gcs_bucket_name)
        
        # Initialize CSV generator
        self.csv_generator = CSVGenerator(gcs_interface=self.gcs_interface)
        
        # Initialize components based on pipeline mode
        if config.pipeline_mode == 'one_step':
            # One-step mode: video directly to annotations
            self.one_step_client = OneStepAnnotationLLMClient(
                model_name=config.model_name,
                safety_settings=config.safety_settings,
                max_retries=config.max_retries,
                retry_delay=config.retry_delay
            )
            
            self.one_step_processor = VideoToAnnotationProcessor(
                llm_client=self.one_step_client,
                gcs_interface=self.gcs_interface,
                request_delay=config.request_delay,
                pipeline_logger=self.pipeline_logger
            )
            
            # Set two-step components to None for clarity
            self.video_script_client = None
            self.annotation_client = None
            self.video_processor = None
            self.script_processor = None
            
            logging.info("PipelineOrchestrator initialized in ONE-STEP mode")
            self.pipeline_logger.log_info("PipelineOrchestrator initialized in ONE-STEP mode")
        else:
            # Two-step mode (default): video to script, then script to annotation
            self.video_script_client = VideoScriptLLMClient(
                model_name=config.model_name,
                safety_settings=config.safety_settings,
                max_retries=config.max_retries,
                retry_delay=config.retry_delay
            )
            
            self.annotation_client = AnnotationLLMClient(
                model_name=config.model_name,
                safety_settings=config.safety_settings,
                max_retries=config.max_retries,
                retry_delay=config.retry_delay
            )
            
            self.video_processor = VideoToScriptProcessor(
                llm_client=self.video_script_client,
                gcs_interface=self.gcs_interface,
                request_delay=config.request_delay,
                save_scripts=config.save_scripts,
                script_output_path=config.script_output_path,
                pipeline_logger=self.pipeline_logger
            )
            
            self.script_processor = ScriptToAnnotationProcessor(
                llm_client=self.annotation_client,
                gcs_interface=self.gcs_interface,
                request_delay=config.request_delay,
                pipeline_logger=self.pipeline_logger
            )
            
            # Set one-step components to None for clarity
            self.one_step_client = None
            self.one_step_processor = None
            
            logging.info("PipelineOrchestrator initialized in TWO-STEP mode")
            self.pipeline_logger.log_info("PipelineOrchestrator initialized in TWO-STEP mode")

    def _run_video_to_script_stage(self) -> Tuple[List[str], List[str]]:
        """
        Execute the video-to-script stage of the pipeline.
        Lists videos from GCS and processes them to generate scripts.
        
        Returns:
            Tuple of (scripts_or_script_uris, failed_video_uris)
            - If save_scripts is True: returns list of GCS URIs where scripts were saved
            - If save_scripts is False: returns list of script texts (in-memory)
            - failed_video_uris: list of video URIs that failed to process
        """
        logging.info("=" * 60)
        logging.info("Starting video-to-script stage")
        logging.info("=" * 60)
        self.pipeline_logger.log_info("Starting video-to-script stage")
        
        try:
            # List videos from GCS
            video_uris = self.gcs_interface.list_videos(prefix=self.config.video_source_path)
            
            if not video_uris:
                warning_msg = f"No videos found in {self.config.video_source_path}"
                logging.warning(warning_msg)
                self.pipeline_logger.log_warning(warning_msg)
                return [], []
            
            info_msg = f"Found {len(video_uris)} videos to process"
            logging.info(info_msg)
            self.pipeline_logger.log_info(info_msg)
            
            # Process videos using VideoToScriptProcessor
            scripts_or_uris, failed_uris = self.video_processor.process_videos(video_uris)
            
            # Log summary
            success_count = len(scripts_or_uris)
            failure_count = len(failed_uris)
            summary_msg = f"Video-to-script stage complete: {success_count} successful, {failure_count} failed"
            logging.info("=" * 60)
            logging.info(summary_msg)
            logging.info("=" * 60)
            self.pipeline_logger.log_info(summary_msg)
            
            return scripts_or_uris, failed_uris
        
        except Exception as e:
            error_msg = f"Error in video-to-script stage: {str(e)}"
            logging.error(error_msg)
            self.pipeline_logger.log_error("video_to_script", "stage_execution", error_msg)
            raise

    def _run_script_to_annotation_stage(self, script_sources: List[str]) -> Tuple[List[Dict], List[str]]:
        """
        Execute the script-to-annotation stage of the pipeline.
        Processes scripts (from GCS URIs or in-memory) to generate value annotations.
        
        Args:
            script_sources: List of GCS URIs or in-memory script texts
        
        Returns:
            Tuple of (annotations, failed_sources)
            - annotations: list of annotation dictionaries
            - failed_sources: list of script sources that failed to process
        """
        logging.info("=" * 60)
        logging.info("Starting script-to-annotation stage")
        logging.info("=" * 60)
        self.pipeline_logger.log_info("Starting script-to-annotation stage")
        
        try:
            if not script_sources:
                warning_msg = "No scripts provided for annotation"
                logging.warning(warning_msg)
                self.pipeline_logger.log_warning(warning_msg)
                return [], []
            
            info_msg = f"Processing {len(script_sources)} scripts for annotation"
            logging.info(info_msg)
            self.pipeline_logger.log_info(info_msg)
            
            # Process scripts using ScriptToAnnotationProcessor
            annotations, failed_sources = self.script_processor.process_scripts(script_sources)
            
            # Log summary
            success_count = len(annotations)
            failure_count = len(failed_sources)
            summary_msg = f"Script-to-annotation stage complete: {success_count} successful, {failure_count} failed"
            logging.info("=" * 60)
            logging.info(summary_msg)
            logging.info("=" * 60)
            self.pipeline_logger.log_info(summary_msg)
            
            return annotations, failed_sources
        
        except Exception as e:
            error_msg = f"Error in script-to-annotation stage: {str(e)}"
            logging.error(error_msg)
            self.pipeline_logger.log_error("script_to_annotation", "stage_execution", error_msg)
            raise

    def _run_complete_pipeline(self) -> Dict[str, Any]:
        """
        Execute the complete pipeline: video-to-script followed by script-to-annotation.
        Generates and saves CSV output with all annotations.
        
        Returns:
            Dictionary containing execution summary with:
            - stage: 'complete'
            - total_videos: number of videos processed
            - successful_videos: number of videos successfully converted to scripts
            - failed_videos: list of video URIs that failed
            - total_annotations: number of annotations generated
            - failed_scripts: list of script sources that failed annotation
            - csv_saved: boolean indicating if CSV was saved successfully
            - csv_path: GCS path where CSV was saved (if successful)
            - failure_summary: structured summary of all failures by stage
        """
        logging.info("=" * 60)
        logging.info("Starting complete pipeline execution")
        logging.info("=" * 60)
        self.pipeline_logger.log_info("Starting complete pipeline execution")
        
        # Stage 1: Video to Script
        scripts_or_uris, failed_videos = self._run_video_to_script_stage()
        
        # Stage 2: Script to Annotation
        # Pass the results from stage 1 (either URIs or in-memory scripts)
        annotations, failed_scripts = self._run_script_to_annotation_stage(scripts_or_uris)
        
        # Stage 3: Generate and save CSV
        logging.info("=" * 60)
        logging.info("Generating and saving CSV output")
        logging.info("=" * 60)
        self.pipeline_logger.log_info("Generating and saving CSV output")
        
        csv_saved = False
        csv_path = None
        
        if annotations:
            csv_saved = self.csv_generator.generate_and_save(
                annotations=annotations,
                output_path=self.config.csv_output_path
            )
            
            if csv_saved:
                csv_path = f"gs://{self.gcs_interface.bucket_name}/{self.config.csv_output_path}"
                success_msg = f"CSV successfully saved to {csv_path}"
                logging.info(success_msg)
                self.pipeline_logger.log_info(success_msg)
            else:
                error_msg = "Failed to save CSV output"
                logging.error(error_msg)
                self.pipeline_logger.log_error("csv_generation", "output", error_msg)
        else:
            warning_msg = "No annotations to save, skipping CSV generation"
            logging.warning(warning_msg)
            self.pipeline_logger.log_warning(warning_msg)
        
        # Get failure summary from logger
        failure_summary = self.pipeline_logger.get_failure_summary()
        
        # Create execution summary
        summary = {
            'stage': 'complete',
            'total_videos': len(scripts_or_uris) + len(failed_videos),
            'successful_videos': len(scripts_or_uris),
            'failed_videos': failed_videos,
            'total_annotations': len(annotations),
            'failed_scripts': failed_scripts,
            'csv_saved': csv_saved,
            'csv_path': csv_path,
            'failure_summary': failure_summary
        }
        
        logging.info("=" * 60)
        logging.info("Complete pipeline execution finished")
        logging.info(f"Videos processed: {summary['successful_videos']}/{summary['total_videos']}")
        logging.info(f"Annotations generated: {summary['total_annotations']}")
        logging.info(f"CSV saved: {csv_saved}")
        logging.info("=" * 60)
        self.pipeline_logger.log_info("Complete pipeline execution finished")
        
        # Print error summary if there are errors
        if self.pipeline_logger.has_errors():
            self.pipeline_logger.print_summary()
        
        return summary

    def _run_one_step_pipeline(self) -> Dict[str, Any]:
        """
        Execute the one-step pipeline: video directly to annotations (no intermediate scripts).
        Generates and saves CSV output with all annotations.
        
        Returns:
            Dictionary containing execution summary with:
            - stage: 'one_step'
            - total_videos: number of videos processed
            - successful_annotations: number of annotations generated
            - failed_videos: list of video URIs that failed
            - csv_saved: boolean indicating if CSV was saved successfully
            - csv_path: GCS path where CSV was saved (if successful)
            - failure_summary: structured summary of all failures
        """
        logging.info("=" * 60)
        logging.info("Starting ONE-STEP pipeline execution")
        logging.info("=" * 60)
        self.pipeline_logger.log_info("Starting ONE-STEP pipeline execution")
        
        try:
            # List videos from GCS
            video_uris = self.gcs_interface.list_videos(prefix=self.config.video_source_path)
            
            if not video_uris:
                warning_msg = f"No videos found in {self.config.video_source_path}"
                logging.warning(warning_msg)
                self.pipeline_logger.log_warning(warning_msg)
                return {
                    'stage': 'one_step',
                    'total_videos': 0,
                    'successful_annotations': 0,
                    'failed_videos': [],
                    'csv_saved': False,
                    'csv_path': None,
                    'failure_summary': {}
                }
            
            info_msg = f"Found {len(video_uris)} videos to process (one-step mode)"
            logging.info(info_msg)
            self.pipeline_logger.log_info(info_msg)
            
            # Process videos directly to annotations using one-step processor
            annotations, failed_videos = self.one_step_processor.process_videos(video_uris)
            
            # Generate and save CSV
            logging.info("=" * 60)
            logging.info("Generating and saving CSV output")
            logging.info("=" * 60)
            self.pipeline_logger.log_info("Generating and saving CSV output")
            
            csv_saved = False
            csv_path = None
            
            if annotations:
                csv_saved = self.csv_generator.generate_and_save(
                    annotations=annotations,
                    output_path=self.config.csv_output_path
                )
                
                if csv_saved:
                    csv_path = f"gs://{self.gcs_interface.bucket_name}/{self.config.csv_output_path}"
                    success_msg = f"CSV successfully saved to {csv_path}"
                    logging.info(success_msg)
                    self.pipeline_logger.log_info(success_msg)
                else:
                    error_msg = "Failed to save CSV output"
                    logging.error(error_msg)
                    self.pipeline_logger.log_error("csv_generation", "output", error_msg)
            else:
                warning_msg = "No annotations to save, skipping CSV generation"
                logging.warning(warning_msg)
                self.pipeline_logger.log_warning(warning_msg)
            
            # Get failure summary from logger
            failure_summary = self.pipeline_logger.get_failure_summary()
            
            # Create execution summary
            summary = {
                'stage': 'one_step',
                'total_videos': len(annotations) + len(failed_videos),
                'successful_annotations': len(annotations),
                'failed_videos': failed_videos,
                'csv_saved': csv_saved,
                'csv_path': csv_path,
                'failure_summary': failure_summary
            }
            
            logging.info("=" * 60)
            logging.info("ONE-STEP pipeline execution finished")
            logging.info(f"Videos processed: {summary['successful_annotations']}/{summary['total_videos']}")
            logging.info(f"CSV saved: {csv_saved}")
            logging.info("=" * 60)
            self.pipeline_logger.log_info("ONE-STEP pipeline execution finished")
            
            # Print error summary if there are errors
            if self.pipeline_logger.has_errors():
                self.pipeline_logger.print_summary()
            
            return summary
        
        except Exception as e:
            error_msg = f"Error in one-step pipeline: {str(e)}"
            logging.error(error_msg)
            self.pipeline_logger.log_error("one_step_pipeline", "execution", error_msg)
            raise

    def run(self) -> Dict[str, Any]:
        """
        Execute the pipeline based on the configured mode and stage settings.
        Routes to one-step pipeline if mode is 'one_step', otherwise uses stage_to_run.
        
        Returns:
            Dictionary containing execution summary with stage-specific results
            
        Raises:
            ValueError: If an invalid stage is specified in configuration
        """
        try:
            # Check pipeline mode first - one_step mode takes precedence
            if self.config.pipeline_mode == 'one_step':
                info_msg = "Pipeline orchestrator starting in ONE-STEP mode"
                logging.info(info_msg)
                self.pipeline_logger.log_info(info_msg)
                return self._run_one_step_pipeline()
            
            # Two-step mode: route based on stage setting
            stage = self.config.stage_to_run
            info_msg = f"Pipeline orchestrator starting with stage: {stage}"
            logging.info(info_msg)
            self.pipeline_logger.log_info(info_msg)
            
            if stage == 'both':
                # Run complete pipeline
                return self._run_complete_pipeline()
            
            elif stage == 'video_to_script':
                # Run only video-to-script stage
                scripts_or_uris, failed_videos = self._run_video_to_script_stage()
                
                # Get failure summary from logger
                failure_summary = self.pipeline_logger.get_failure_summary()
                
                summary = {
                    'stage': 'video_to_script',
                    'total_videos': len(scripts_or_uris) + len(failed_videos),
                    'successful_videos': len(scripts_or_uris),
                    'failed_videos': failed_videos,
                    'scripts_saved': self.config.save_scripts,
                    'script_output_path': self.config.script_output_path if self.config.save_scripts else None,
                    'failure_summary': failure_summary
                }
                
                logging.info("=" * 60)
                logging.info("Video-to-script stage execution finished")
                logging.info(f"Videos processed: {summary['successful_videos']}/{summary['total_videos']}")
                logging.info("=" * 60)
                self.pipeline_logger.log_info("Video-to-script stage execution finished")
                
                # Print error summary if there are errors
                if self.pipeline_logger.has_errors():
                    self.pipeline_logger.print_summary()
                
                return summary
            
            elif stage == 'script_to_annotation':
                # Run only script-to-annotation stage
                # List scripts from GCS
                script_uris = self.gcs_interface.list_scripts(prefix=self.config.script_output_path)
                
                if not script_uris:
                    warning_msg = f"No scripts found in {self.config.script_output_path}"
                    logging.warning(warning_msg)
                    self.pipeline_logger.log_warning(warning_msg)
                    return {
                        'stage': 'script_to_annotation',
                        'total_scripts': 0,
                        'successful_annotations': 0,
                        'failed_scripts': [],
                        'csv_saved': False,
                        'csv_path': None,
                        'failure_summary': {}
                    }
                
                annotations, failed_scripts = self._run_script_to_annotation_stage(script_uris)
                
                # Generate and save CSV
                logging.info("=" * 60)
                logging.info("Generating and saving CSV output")
                logging.info("=" * 60)
                self.pipeline_logger.log_info("Generating and saving CSV output")
                
                csv_saved = False
                csv_path = None
                
                if annotations:
                    csv_saved = self.csv_generator.generate_and_save(
                        annotations=annotations,
                        output_path=self.config.csv_output_path
                    )
                    
                    if csv_saved:
                        csv_path = f"gs://{self.gcs_interface.bucket_name}/{self.config.csv_output_path}"
                        success_msg = f"CSV successfully saved to {csv_path}"
                        logging.info(success_msg)
                        self.pipeline_logger.log_info(success_msg)
                    else:
                        error_msg = "Failed to save CSV output"
                        logging.error(error_msg)
                        self.pipeline_logger.log_error("csv_generation", "output", error_msg)
                else:
                    warning_msg = "No annotations to save, skipping CSV generation"
                    logging.warning(warning_msg)
                    self.pipeline_logger.log_warning(warning_msg)
                
                # Get failure summary from logger
                failure_summary = self.pipeline_logger.get_failure_summary()
                
                summary = {
                    'stage': 'script_to_annotation',
                    'total_scripts': len(annotations) + len(failed_scripts),
                    'successful_annotations': len(annotations),
                    'failed_scripts': failed_scripts,
                    'csv_saved': csv_saved,
                    'csv_path': csv_path,
                    'failure_summary': failure_summary
                }
                
                logging.info("=" * 60)
                logging.info("Script-to-annotation stage execution finished")
                logging.info(f"Scripts processed: {summary['successful_annotations']}/{summary['total_scripts']}")
                logging.info(f"CSV saved: {csv_saved}")
                logging.info("=" * 60)
                self.pipeline_logger.log_info("Script-to-annotation stage execution finished")
                
                # Print error summary if there are errors
                if self.pipeline_logger.has_errors():
                    self.pipeline_logger.print_summary()
                
                return summary
            
            else:
                error_msg = f"Invalid stage specified: {stage}. Must be 'both', 'video_to_script', or 'script_to_annotation'"
                logging.error(error_msg)
                self.pipeline_logger.log_error("orchestrator", "configuration", error_msg)
                raise ValueError(error_msg)
        
        except Exception as e:
            error_msg = f"Pipeline execution failed: {str(e)}"
            logging.error(error_msg)
            self.pipeline_logger.log_error("orchestrator", "execution", error_msg)
            raise
