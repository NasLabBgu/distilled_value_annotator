#!/usr/bin/env python3
"""
Video Annotation Pipeline - Main Entry Point

This script orchestrates the video annotation pipeline that processes TikTok videos
through two stages: video-to-script conversion and script-to-annotation.
"""

import sys
import argparse
import logging
from pathlib import Path
from config import PipelineConfig
from orchestrator import PipelineOrchestrator


def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description='Video Annotation Pipeline - Process videos and generate value annotations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --config config.yaml
  %(prog)s --config /path/to/config.yaml

Configuration File:
  The configuration file should be in YAML format and include:
  - GCS bucket and path settings
  - Model configuration (name, retries, delays)
  - Pipeline stage selection
  - Safety settings

For more information, see the README.md file.
        """
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to the configuration file (YAML format)'
    )
    return parser.parse_args()


def setup_logging():
    """Configure logging for the pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def print_execution_summary(summary: dict):
    """
    Print a formatted execution summary.
    
    Args:
        summary: Dictionary containing execution results
    """
    print("\n" + "=" * 60)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 60)
    
    stage = summary.get('stage', 'unknown')
    print(f"Stage: {stage}")
    print()
    
    if stage == 'complete':
        # Complete pipeline summary
        total_videos = summary.get('total_videos', 0)
        successful_videos = summary.get('successful_videos', 0)
        failed_videos = summary.get('failed_videos', [])
        total_annotations = summary.get('total_annotations', 0)
        failed_scripts = summary.get('failed_scripts', [])
        csv_saved = summary.get('csv_saved', False)
        csv_path = summary.get('csv_path', '')
        
        print(f"Videos Processed: {successful_videos}/{total_videos}")
        if failed_videos:
            print(f"Failed Videos: {len(failed_videos)}")
        
        print(f"Annotations Generated: {total_annotations}")
        if failed_scripts:
            print(f"Failed Scripts: {len(failed_scripts)}")
        
        print(f"CSV Saved: {'Yes' if csv_saved else 'No'}")
        if csv_saved and csv_path:
            print(f"CSV Location: {csv_path}")
    
    elif stage == 'one_step':
        # One-step pipeline summary (video directly to annotations)
        total_videos = summary.get('total_videos', 0)
        successful_annotations = summary.get('successful_annotations', 0)
        failed_videos = summary.get('failed_videos', [])
        csv_saved = summary.get('csv_saved', False)
        csv_path = summary.get('csv_path', '')
        
        print(f"Videos Processed: {successful_annotations}/{total_videos}")
        if failed_videos:
            print(f"Failed Videos: {len(failed_videos)}")
        
        print(f"CSV Saved: {'Yes' if csv_saved else 'No'}")
        if csv_saved and csv_path:
            print(f"CSV Location: {csv_path}")
    
    elif stage == 'video_to_script':
        # Video-to-script stage summary
        total_videos = summary.get('total_videos', 0)
        successful_videos = summary.get('successful_videos', 0)
        failed_videos = summary.get('failed_videos', [])
        scripts_saved = summary.get('scripts_saved', False)
        script_output_path = summary.get('script_output_path', '')
        
        print(f"Videos Processed: {successful_videos}/{total_videos}")
        if failed_videos:
            print(f"Failed Videos: {len(failed_videos)}")
        
        print(f"Scripts Saved: {'Yes' if scripts_saved else 'No (in-memory only)'}")
        if scripts_saved and script_output_path:
            print(f"Script Location: {script_output_path}")
    
    elif stage == 'script_to_annotation':
        # Script-to-annotation stage summary
        total_scripts = summary.get('total_scripts', 0)
        successful_annotations = summary.get('successful_annotations', 0)
        failed_scripts = summary.get('failed_scripts', [])
        csv_saved = summary.get('csv_saved', False)
        csv_path = summary.get('csv_path', '')
        
        print(f"Scripts Processed: {successful_annotations}/{total_scripts}")
        if failed_scripts:
            print(f"Failed Scripts: {len(failed_scripts)}")
        
        print(f"CSV Saved: {'Yes' if csv_saved else 'No'}")
        if csv_saved and csv_path:
            print(f"CSV Location: {csv_path}")
    
    # Print failure details if any
    failure_summary = summary.get('failure_summary', {})
    if failure_summary:
        print()
        print("Failures by Stage:")
        for stage_name, failures in failure_summary.items():
            if failures:
                print(f"  {stage_name}: {len(failures)} failure(s)")
    
    print("=" * 60)


def main():
    """
    Main entry point for the pipeline.
    
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    try:
        # Setup logging
        setup_logging()
        
        # Parse command-line arguments
        args = parse_arguments()
        
        # Validate config file exists
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Error: Configuration file not found: {args.config}")
            print(f"Please provide a valid path to a YAML configuration file.")
            return 1
        
        print(f"Loading configuration from: {args.config}")
        
        # Load configuration
        try:
            config = PipelineConfig(str(config_path))
            print("Configuration loaded and validated successfully")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return 1
        except ValueError as e:
            print(f"Configuration validation error:\n{e}")
            return 1
        except Exception as e:
            print(f"Error loading configuration: {e}")
            return 1
        
        # Create and run pipeline orchestrator
        print("\nInitializing pipeline orchestrator...")
        try:
            orchestrator = PipelineOrchestrator(config)
            print("Pipeline orchestrator initialized successfully")
        except Exception as e:
            print(f"Error initializing pipeline orchestrator: {e}")
            logging.exception("Failed to initialize pipeline orchestrator")
            return 1
        
        # Run the pipeline
        print("\nStarting pipeline execution...")
        if config.pipeline_mode == 'one_step':
            print("Mode: one_step (direct video to annotations)")
        else:
            print(f"Stage to run: {config.stage_to_run}")
        print()
        
        try:
            summary = orchestrator.run()
        except Exception as e:
            print(f"\nError during pipeline execution: {e}")
            logging.exception("Pipeline execution failed")
            return 1
        
        # Print execution summary
        print_execution_summary(summary)
        
        # Determine exit code based on results
        if summary.get('stage') == 'complete':
            # For complete pipeline, check if we got any annotations
            if summary.get('total_annotations', 0) > 0:
                print("\nPipeline completed successfully!")
                return 0
            else:
                print("\nPipeline completed but no annotations were generated.")
                return 1
        elif summary.get('stage') == 'video_to_script':
            # For video-to-script stage, check if we got any scripts
            if summary.get('successful_videos', 0) > 0:
                print("\nVideo-to-script stage completed successfully!")
                return 0
            else:
                print("\nVideo-to-script stage completed but no scripts were generated.")
                return 1
        elif summary.get('stage') == 'script_to_annotation':
            # For script-to-annotation stage, check if we got any annotations
            if summary.get('successful_annotations', 0) > 0:
                print("\nScript-to-annotation stage completed successfully!")
                return 0
            else:
                print("\nScript-to-annotation stage completed but no annotations were generated.")
                return 1
        elif summary.get('stage') == 'one_step':
            # For one-step mode, check if we got any annotations
            if summary.get('successful_annotations', 0) > 0:
                print("\nOne-step pipeline completed successfully!")
                return 0
            else:
                print("\nOne-step pipeline completed but no annotations were generated.")
                return 1
        else:
            print("\nPipeline completed with unknown status.")
            return 1
        
    except KeyboardInterrupt:
        print("\n\nPipeline execution interrupted by user.")
        return 1
    except Exception as e:
        print(f"\nFatal error: {e}")
        logging.exception("Fatal error in main")
        return 1


if __name__ == "__main__":
    sys.exit(main())
