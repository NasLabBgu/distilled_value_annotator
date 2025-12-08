# Video Annotation Pipeline

A Python-based pipeline for processing TikTok videos to extract 19 human value annotations based on Schwartz's value framework. Supports two modes: a **two-step process** (video → script → annotations) or a faster **one-step mode** (video → annotations directly).

## Overview

The pipeline processes videos stored in Google Cloud Storage (GCS) and outputs a CSV file containing value annotations for each video. It supports flexible execution modes, configurable retry logic, and optional intermediate script storage.

### Key Features

- **Two processing modes**: Choose between two-step (via scripts) or one-step (direct) annotation
- **Flexible execution**: Run complete pipeline or individual stages
- **Robust error handling**: Exponential backoff retry logic with configurable delays
- **Cloud-native**: Built for Google Cloud Platform with GCS and Vertex AI
- **Configurable**: YAML-based configuration for all pipeline parameters
- **Optional script storage**: Save intermediate scripts or process in-memory (two-step mode)

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Pipeline Stages](#pipeline-stages)
- [Output Format](#output-format)
- [Troubleshooting](#troubleshooting)
- [Advanced Usage](#advanced-usage)

## Prerequisites

Before installing the pipeline, ensure you have:

1. **Python 3.9 or higher**
   ```bash
   python --version
   ```

2. **Google Cloud Platform account** with:
   - A GCS bucket containing your video files
   - Vertex AI API enabled
   - Appropriate IAM permissions

3. **Google Cloud SDK** installed and configured
   ```bash
   gcloud --version
   ```

4. **Authentication** set up using one of:
   - Application Default Credentials (ADC)
   - Service Account Key

### Setting Up Google Cloud Authentication

**Option 1: Application Default Credentials (Recommended for local development)**
```bash
gcloud auth application-default login
```

**Option 2: Service Account Key**
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
```

### Required GCP Permissions

Your account or service account needs:
- `storage.objects.get` - Read videos from GCS
- `storage.objects.create` - Write scripts and CSV to GCS
- `aiplatform.endpoints.predict` - Call Vertex AI models

## Installation

1. **Clone or download the repository**
   ```bash
   cd video-annotation-pipeline
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```bash
   python main.py --help
   ```

## Configuration

The pipeline uses a YAML configuration file to specify all parameters. A sample `config.yaml` is provided.

### Configuration File Structure

```yaml
# Google Cloud Storage Configuration
gcs:
  bucket_name: "your-bucket-name"
  video_source_path: "path/to/videos/"
  script_output_path: "path/to/scripts/"  # Optional
  csv_output_path: "path/to/output.csv"

# LLM Model Configuration
model:
  name: "gemini-1.5-pro-002"
  max_retries: 4
  retry_delay: 40
  request_delay: 3

# Pipeline Execution Configuration
pipeline:
  stage: "both"  # Options: "both", "video_to_script", "script_to_annotation"
  save_scripts: true

# Safety Settings
safety_settings:
  harassment: "BLOCK_NONE"
  hate_speech: "BLOCK_NONE"
  sexually_explicit: "BLOCK_NONE"
  dangerous_content: "BLOCK_NONE"
```

### Configuration Options

#### GCS Settings

| Option | Type | Required | Description |
|--------|------|----------|-------------|
| `bucket_name` | string | Yes | Name of your GCS bucket |
| `video_source_path` | string | Yes | Path prefix where videos are located (e.g., "videos/") |
| `script_output_path` | string | No | Path prefix for saving scripts (only if `save_scripts: true`) |
| `csv_output_path` | string | Yes | Full path including filename for CSV output |

#### Model Settings

| Option | Type | Required | Description |
|--------|------|----------|-------------|
| `name` | string | Yes | Vertex AI model name (e.g., "gemini-1.5-pro-002") |
| `max_retries` | integer | Yes | Maximum retry attempts for failed API calls (recommended: 3-5) |
| `retry_delay` | integer | Yes | Base delay in seconds for exponential backoff (recommended: 30-60) |
| `request_delay` | integer | Yes | Delay in seconds between consecutive requests (recommended: 2-5) |

#### Pipeline Settings

| Option | Type | Required | Description |
|--------|------|----------|-------------|
| `mode` | string | No | Pipeline mode: "one_step" (direct) or "two_step" (via scripts). Default: "two_step" |
| `stage` | string | Yes* | Which stage(s) to run: "both", "video_to_script", or "script_to_annotation" (*ignored in one_step mode) |
| `save_scripts` | boolean | Yes* | Whether to save intermediate scripts to GCS (*only applies to two_step mode) |

#### Safety Settings

| Option | Type | Required | Description |
|--------|------|----------|-------------|
| `harassment` | string | Yes | Filter level for harassment content |
| `hate_speech` | string | Yes | Filter level for hate speech |
| `sexually_explicit` | string | Yes | Filter level for sexually explicit content |
| `dangerous_content` | string | Yes | Filter level for dangerous content |

**Valid safety values**: `BLOCK_NONE`, `BLOCK_ONLY_HIGH`, `BLOCK_MEDIUM_AND_ABOVE`, `BLOCK_LOW_AND_ABOVE`

## Usage

### Basic Usage

Run the complete pipeline with default configuration:

```bash
python main.py --config config.yaml
```

### Running Specific Stages

**Video to Script Only**
```yaml
# In config.yaml
pipeline:
  stage: "video_to_script"
  save_scripts: true
```
```bash
python main.py --config config.yaml
```

**Script to Annotation Only**
```yaml
# In config.yaml
pipeline:
  stage: "script_to_annotation"
```
```bash
python main.py --config config.yaml
```

### Example Workflow

1. **First run**: Process videos and save scripts
   ```yaml
   pipeline:
     stage: "both"
     save_scripts: true
   ```

2. **Reprocess annotations**: Use existing scripts
   ```yaml
   pipeline:
     stage: "script_to_annotation"
   ```

### One-Step Mode (Direct Video to Annotations)

For faster processing without intermediate script generation, use one-step mode:

```yaml
# One-step mode configuration
gcs:
  bucket_name: "your-bucket-name"
  video_source_path: "path/to/videos/"
  csv_output_path: "path/to/output.csv"

model:
  name: "gemini-1.5-pro-002"
  max_retries: 4
  retry_delay: 40
  request_delay: 3

pipeline:
  mode: "one_step"  # Direct video to annotations (no scripts)

safety_settings:
  harassment: "BLOCK_NONE"
  hate_speech: "BLOCK_NONE"
  sexually_explicit: "BLOCK_NONE"
  dangerous_content: "BLOCK_NONE"
```

**Benefits of one-step mode:**
- **Faster processing**: Single LLM call per video instead of two
- **Lower costs**: Reduced API usage
- **Simpler workflow**: No intermediate artifacts

**When to use two-step mode instead:**
- You need to review/edit intermediate scripts
- You want to reprocess annotations without re-processing videos
- You need the detailed movie script output

## Pipeline Stages

### Stage 1: Video to Movie Script

Converts video files to structured movie scripts including:
- Audio transcription
- Visual descriptions
- On-screen text/captions
- Scene descriptions

**Input**: MP4 video files in GCS
**Output**: Text scripts (saved to GCS or kept in memory)

### Stage 2: Script to Value Annotations

Extracts 19 human value annotations from movie scripts based on Schwartz's value framework.

**Input**: Movie scripts (from GCS or memory)
**Output**: JSON annotations with values and metadata

### Stage 3: CSV Generation

Aggregates all annotations into a single CSV file.

**Input**: Annotation dictionaries
**Output**: CSV file in GCS

## Output Format

The pipeline generates a CSV file with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `video_id` | string | Video filename identifier |
| `Self_Direction_Thought` | integer | Value score (-1, 0, 1, 2) |
| `Self_Direction_Action` | integer | Value score (-1, 0, 1, 2) |
| `Stimulation` | integer | Value score (-1, 0, 1, 2) |
| `Hedonism` | integer | Value score (-1, 0, 1, 2) |
| `Achievement` | integer | Value score (-1, 0, 1, 2) |
| `Power_Resources` | integer | Value score (-1, 0, 1, 2) |
| `Power_Dominance` | integer | Value score (-1, 0, 1, 2) |
| `Face` | integer | Value score (-1, 0, 1, 2) |
| `Security_Personal` | integer | Value score (-1, 0, 1, 2) |
| `Security_Social` | integer | Value score (-1, 0, 1, 2) |
| `Conformity_Rules` | integer | Value score (-1, 0, 1, 2) |
| `Conformity_Interpersonal` | integer | Value score (-1, 0, 1, 2) |
| `Tradition` | integer | Value score (-1, 0, 1, 2) |
| `Humility` | integer | Value score (-1, 0, 1, 2) |
| `Benevolence_Dependability` | integer | Value score (-1, 0, 1, 2) |
| `Benevolence_Care` | integer | Value score (-1, 0, 1, 2) |
| `Universalism_Concern` | integer | Value score (-1, 0, 1, 2) |
| `Universalism_Nature` | integer | Value score (-1, 0, 1, 2) |
| `Universalism_Tolerance` | integer | Value score (-1, 0, 1, 2) |
| `Has_sound` | boolean | Whether video has audio |
| `notes` | string | Optional text notes from annotation |

### Value Score Interpretation

- `-1`: Value is contradicted or opposed
- `0`: Value is not present or neutral
- `1`: Value is present or supported
- `2`: Value is strongly emphasized

## Troubleshooting

### Common Issues

#### 1. Authentication Errors

**Error**: `google.auth.exceptions.DefaultCredentialsError`

**Solution**:
```bash
# Set up application default credentials
gcloud auth application-default login

# Or set service account key
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/key.json"
```

#### 2. Permission Denied Errors

**Error**: `403 Forbidden` or `Permission denied`

**Solution**:
- Verify your account has the required IAM roles
- Check bucket permissions: `gsutil iam get gs://your-bucket-name`
- Ensure Vertex AI API is enabled: `gcloud services enable aiplatform.googleapis.com`

#### 3. Rate Limiting / Quota Errors

**Error**: `429 Too Many Requests` or `ResourceExhausted`

**Solution**:
- Increase `request_delay` in config (e.g., from 3 to 5 seconds)
- Increase `retry_delay` for longer backoff (e.g., from 40 to 60 seconds)
- Reduce batch size by processing videos in smaller groups

#### 4. Configuration Validation Errors

**Error**: `Configuration validation error`

**Solution**:
- Check YAML syntax is valid
- Ensure all required fields are present
- Verify field types match expected values
- Check that paths don't have trailing spaces

#### 5. Video Processing Failures

**Error**: Individual videos fail to process

**Solution**:
- Check video format (must be MP4)
- Verify video file is not corrupted
- Check video size (very large files may timeout)
- Review safety settings - content may be blocked
- Check pipeline logs for specific error messages

#### 6. Model Not Found

**Error**: `Model not found` or `Invalid model name`

**Solution**:
- Verify model name is correct (e.g., "gemini-1.5-pro-002")
- Check model is available in your GCP region
- Ensure Vertex AI API is enabled

#### 7. Out of Memory Errors

**Error**: `MemoryError` or system slowdown

**Solution**:
- Set `save_scripts: false` to reduce memory usage
- Process videos in smaller batches
- Increase system memory or use a machine with more RAM

### Debugging Tips

1. **Enable verbose logging**:
   ```python
   # In main.py, change logging level
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **Test with a small dataset**:
   - Start with 2-3 videos to verify configuration
   - Check output format before processing full dataset

3. **Run stages independently**:
   - Test video-to-script stage first
   - Verify scripts look correct
   - Then run script-to-annotation stage

4. **Check GCS paths**:
   ```bash
   # List videos in your bucket
   gsutil ls gs://your-bucket-name/path/to/videos/
   
   # Verify bucket access
   gsutil ls gs://your-bucket-name/
   ```

5. **Monitor API quotas**:
   - Check Vertex AI quotas in GCP Console
   - Monitor API usage in Cloud Monitoring

### Getting Help

If you encounter issues not covered here:

1. Check the execution summary for specific error messages
2. Review the logs for detailed error traces
3. Verify your GCP project configuration
4. Ensure all dependencies are up to date: `pip install --upgrade -r requirements.txt`

## Advanced Usage

### Processing Large Datasets

For large video collections:

1. **Adjust delays to avoid rate limits**:
   ```yaml
   model:
     request_delay: 5  # Increase delay between requests
     retry_delay: 60   # Increase backoff delay
   ```

2. **Process in batches**:
   - Split videos into subdirectories
   - Process each batch separately
   - Combine CSV outputs afterward

3. **Use faster model for testing**:
   ```yaml
   model:
     name: "gemini-1.5-flash-002"  # Faster, lower cost
   ```

### Custom System Instructions

The pipeline uses instruction files in the `prompts/` directory:
- `prompts/video_to_script_instructions.txt` - Video to script conversion
- `prompts/script_to_annotation_instructions.txt` - Script to annotation extraction

You can modify these files to customize the LLM behavior.

### Monitoring Progress

The pipeline provides real-time progress updates:
```
Processing video 1/10: video_001.mp4
Processing video 2/10: video_002.mp4
...
```

Failed items are logged and summarized at the end.

### Cost Optimization

To reduce costs:

1. **Use Flash model**: `gemini-1.5-flash-002` (faster, cheaper)
2. **Don't save scripts**: Set `save_scripts: false`
3. **Process in-memory**: Run complete pipeline without intermediate storage
4. **Batch processing**: Process multiple videos in one session to amortize startup costs

## Project Structure

```
video-annotation-pipeline/
├── config/              # Configuration module
├── evaluation/          # Model Evaluation Module
│   ├── adapters/        # Model adapters (base, gemini, MLM, script)
│   ├── metrics/         # Metrics calculation
│   └── reports/         # Report generation
├── examples/            # Example configs and sample data
├── gcs/                 # GCS interface module
├── llm/                 # LLM client modules
├── orchestrator/        # Pipeline orchestrator
├── processors/          # Video and script processors
├── prompts/             # System instruction files
├── tests/               # Test suite
├── utils/               # Utility modules (logging)
├── config.yaml          # Configuration file
├── main.py              # Main entry point
├── run_evaluation.py    # Model evaluation CLI
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

---

## Model Evaluation Module

The Model Evaluation Module provides a framework for evaluating and comparing different model predictions against ground truth annotations. It calculates comprehensive metrics and generates detailed reports.

### Features

- **Multiple Adapter Support**: Evaluate different models through a unified interface
- **Comprehensive Metrics**: F1 scores (macro, weighted), precision, recall for each category
- **Endorsed/Conflict Analysis**: Separate metrics for endorsed values (1,2) and conflict values (-1)
- **Flexible Configuration**: YAML/JSON configuration files
- **Report Generation**: CSV and JSON reports with model comparisons
- **Sampling Support**: Evaluate on subsets with reproducible random sampling

### Quick Start

1. **Create a configuration file** (`evaluation_config.yaml`):

   ```yaml
   ground_truth_path: "path/to/ground_truth.csv"
   scripts_path: "path/to/scripts/"
   output_dir: "evaluation_output/"
   
   models:
     - model_type: gemini
       model_name: gemini-1.5-pro
       adapter_class: GeminiAdapter
       config:
         model_id: "gemini-1.5-pro-002"
         project_id: "your-project-id"
         location: "us-central1"
   ```

2. **Run the evaluation**:

   ```bash
   python run_evaluation.py --config evaluation_config.yaml
   ```

3. **View results** in the `evaluation_output/` directory.

### CLI Options

```bash
python run_evaluation.py --config CONFIG_FILE [OPTIONS]

Options:
  --config, -c PATH     Path to configuration file (required)
  --verbose, -v         Enable verbose output (DEBUG level)
  --quiet, -q           Suppress non-essential output (WARNING level)
  --dry-run             Validate configuration without running
  --output-dir PATH     Override output directory
  --skip-reports        Skip report generation
  --models MODEL        Filter to specific models (can repeat)
```

### Examples

**Dry run to validate configuration:**
```bash
python run_evaluation.py --config config.yaml --dry-run
```

**Verbose output for debugging:**
```bash
python run_evaluation.py --config config.yaml --verbose
```

**Evaluate only specific models:**
```bash
python run_evaluation.py --config config.yaml --models model_a --models model_b
```

### Configuration Reference

See `examples/evaluation_config.yaml` for a fully documented example.

#### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `ground_truth_path` | string | Path to CSV file with ground truth annotations |
| `scripts_path` | string | Directory containing script files |
| `output_dir` | string | Directory for output reports |
| `models` | list | List of model configurations |

#### Optional Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `sample_size` | integer | null | Number of videos to sample (null = all) |
| `random_seed` | integer | null | Seed for reproducible sampling |
| `min_frequency_threshold` | float | 0.05 | Min category frequency for aggregate metrics |
| `parallel_execution` | boolean | true | Enable parallel prediction |
| `max_workers` | integer | 4 | Maximum parallel workers |

#### Model Configuration

Each model in the `models` list requires:

| Field | Type | Description |
|-------|------|-------------|
| `model_type` | string | Type identifier (e.g., "gemini", "custom") |
| `model_name` | string | Unique name for this model |
| `adapter_class` | string | Adapter class name to use |
| `config` | object | Model-specific configuration |

### Ground Truth Format

The ground truth CSV should have columns:
- `video_id`: Unique video identifier
- `video_uri`: Video file path/URI
- `script_uri`: Script file path
- Category columns (19 value categories): `Achievement`, `Benevolence`, `Conformity`, etc.

Values should be:
- `-1`: Value is contradicted
- `0`: Value not present
- `1`: Value is present
- `2`: Value is strongly emphasized

See `examples/sample_ground_truth.csv` for a sample file.

### Metrics Explained

The evaluation calculates these metrics for each category:

| Metric | Description |
|--------|-------------|
| `precision` | True positives / (True positives + False positives) |
| `recall` | True positives / (True positives + False negatives) |
| `f1_score` | Harmonic mean of precision and recall |
| `support` | Number of ground truth instances |

Aggregate metrics are provided for:
- **Endorsed values**: Categories with values 1 or 2 (collapsed to binary)
- **Conflict values**: Categories with value -1
- **Combined**: All predictions together

### Output Reports

After evaluation, the following files are generated:

1. **`{model_name}_category_metrics.csv`**: Per-category metrics
2. **`{model_name}_aggregate_metrics.csv`**: Summary metrics
3. **`{model_name}_report.json`**: Complete JSON report
4. **`model_comparison.csv`**: Side-by-side model comparison (if multiple models)

### Creating Custom Adapters

To evaluate a custom model, create an adapter class:

```python
from evaluation.adapters import ModelAdapter
from evaluation.models import VideoAnnotation, PredictionResult

class MyModelAdapter(ModelAdapter):
    def __init__(self, model_name: str, **config):
        super().__init__(model_name=model_name, **config)
        # Initialize your model
    
    def initialize(self) -> bool:
        # Return True if initialization succeeds
        return True
    
    def predict(self, video: VideoAnnotation) -> PredictionResult:
        # Run prediction and return result
        predictions = {"Achievement": 1, "Benevolence": 0, ...}
        return PredictionResult(
            video_id=video.video_id,
            predictions=predictions,
            success=True
        )
    
    def get_model_type(self) -> str:
        return "my_model"
    
    def get_model_name(self) -> str:
        return self._model_name
```

Register and use your adapter:

```python
from evaluation import EvaluationOrchestrator

EvaluationOrchestrator.register_adapter("MyModelAdapter", MyModelAdapter)
```

### Sample Data

The `examples/` directory contains:
- `evaluation_config.yaml`: Documented configuration template
- `sample_ground_truth.csv`: Sample dataset with 10 videos
- `sample_scripts/`: Sample script files

---

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]

## Support

For questions or issues, please [add contact information or issue tracker link].
