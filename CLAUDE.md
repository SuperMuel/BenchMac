# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Dependencies and Environment
- **Install dependencies**: `uv sync`
- **Install pre-commit hooks**: `uv run pre-commit install`

### Testing
- **Run unit tests only**: `uv run pytest` (default configuration excludes integration/e2e/slow tests)
- **Run all tests**: `uv run pytest -m ""`
- **Run integration tests**: `uv run pytest -m integration`
- **Run e2e tests**: `uv run pytest -m e2e`
- **Run with coverage**: `uv run pytest --cov=src --cov-report=html`
- **Run specific test file**: `uv run pytest tests/unit/test_models.py`

### Data Validation
These tests are very slow and validate the correctness of the benchmark instances.
They should be run manually after adding or changing instances in `data/`.
**Run baseline data validation**: `uv run pytest -m instance_validation`



### Code Quality
- **Format and lint code**: `uv run ruff format` and `uv run ruff check`
- **Fix linting issues**: `uv run ruff check --fix`
- **Type checking**: Use pyrightconfig.json for type checking configuration

### Running the Application
- **Main CLI entry point**: `uv run benchmac` or `uv run python -m bench_mac.cli`
- **Evaluate submissions**: `uv run benchmac evaluate path/to/submissions.jsonl`
- **Filter by instance ID**: `uv run benchmac evaluate submissions.jsonl --instance-id angular2-hn_v10_to_v11`

## Architecture Overview

BenchMAC is a benchmark for evaluating AI agents on Angular version migration tasks. The system uses Docker containers to provide isolated, reproducible evaluation environments.

### Core Components

1. **CLI Layer** (`cli.py`): Entry point using `cyclopts` for command-line interface
2. **Models** (`models.py`): Pydantic models defining the data structures:
   - `BenchmarkInstance`: Defines a migration task (repo, versions, commands)
   - `Submission`: Contains the patch submitted by an AI agent
   - `EvaluationTask`: Pairs an instance with a submission for evaluation
   - `ExecutionTrace`: Records all command outputs during evaluation
   - `MetricsReport`: Calculated performance metrics
   - `EvaluationResult`: Success/failure result with details

3. **Executor** (`executor.py`): Core evaluation logic that:
   - Prepares Docker environment for each instance
   - Applies submitted patches to Angular projects
   - Runs install, build, lint, and test commands
   - Captures detailed execution traces

4. **Runner** (`runner.py`): Orchestrates parallel evaluation using `ProcessPoolExecutor`
   - Manages worker processes for concurrent evaluations
   - Handles progress tracking and result callbacks

5. **Docker System** (`docker/`):
   - `manager.py`: High-level Docker API wrapper
   - `builder.py`: Creates specialized Docker images for each benchmark instance

6. **Metrics** (`metrics.py`): Analyzes execution traces to compute performance metrics
7. **Configuration** (`config.py`): Centralized settings using `pydantic-settings`

### Key Data Flow

1. **Input**: JSONL files containing benchmark instances and submissions
2. **Matching**: Submissions are matched to corresponding benchmark instances
3. **Execution**: Each job runs in isolated Docker container:
   - Clone repository at specific commit
   - Apply submitted patch
   - Run Angular commands (install/build/lint/test)
4. **Analysis**: Command outputs are analyzed to compute success metrics
5. **Output**: Results saved as JSONL with detailed execution traces and metrics

### File Structure
- `src/bench_mac/`: Main package code
- `data/`: Benchmark instances and reference patches
- `tests/`: Unit, integration, and e2e tests
- `scripts/`: Utility scripts for data generation and validation

### Docker Requirements
- Docker daemon must be running for evaluations
- Base images are built dynamically for each Angular version/Node.js combination
- Containers are automatically cleaned up after evaluation

### Configuration
Settings can be configured via:
- Environment variables with `BENCHMAC_` prefix
- `.env` file in project root
- Key settings: log levels, Docker host, cache directories