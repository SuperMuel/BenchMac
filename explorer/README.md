# BenchMAC Results Explorer

A Streamlit application for exploring and analyzing BenchMAC evaluation results stored in JSONL format.

## Features

- 📊 **Interactive Dashboard**: Summary statistics and overall success rates
- 📈 **Metrics Visualization**: Bar charts showing success rates per evaluation metric
- 🔍 **Detailed Results**: Drill down into individual evaluation results
- 🔬 **Execution Steps**: Command-by-command execution details for each run
- 🚨 **Harness Failures**: Failures section shown only when failures are present
- 🟢🟠❌ **Status Emoji**: Per-instance status in expanders
  - Green (✅): all available metrics passed
  - Orange (🟠): some metrics passed
  - Red (❌): no metrics passed (or no metrics available)

## Installation

1. Install the required dependencies (from this `explorer/` directory):
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser and navigate to the displayed URL (usually `http://localhost:8501`).

3. The app automatically aggregates all results from the configured directories:
   - `settings.evaluations_dir`: scans recursively for `*.jsonl` outcome files
   - `settings.experiments_dir/submissions/*.jsonl`: loads submissions metadata

4. Explore your evaluation results through the interactive interface.

## Understanding Results

The BenchMAC evaluation results contain:

- **Status**: Either "success" or "failure" (harness failure)
- **Instance ID**: Unique identifier for the benchmark instance
- **Metrics**: Tri-state booleans indicating success for various criteria:
  - `patch_application_success`: Whether the patch applied cleanly
  - `target_version_achieved`: Whether target Angular version was reached
  - `build_success`: Whether the build command succeeded
  - `install_success`: Whether dependencies installed successfully
- **Execution Steps**: Detailed command execution results including:
  - Command executed
  - Exit code
  - Standard output and error
  - Execution timing

## File Format

Results are stored in JSONL (JSON Lines) format where each line represents one evaluation result:

```json
{"status":"success","result":{"instance_id":"example","execution":{"steps":[...]},"metrics":{...}}}
{"status":"failure","instance_id":"example","error":"Error message"}
```

## Contributing

This explorer app is designed to work with BenchMAC evaluation results. For more information about BenchMAC, see the main project documentation.
