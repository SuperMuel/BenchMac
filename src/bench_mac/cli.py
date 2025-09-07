import json
import os
import sys
from collections.abc import Generator, Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import cyclopts
from cyclopts import Parameter
from loguru import logger
from rich.console import Console, Group
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

from bench_mac.config import settings
from bench_mac.logging_config import setup_main_process_logging
from bench_mac.models import (
    EvaluationResult,
    EvaluationResultAdapter,
    EvaluationTask,
    utc_now,
)
from bench_mac.runner import BenchmarkRunner
from bench_mac.utils import load_instances
from bench_mac.version import harness_version
from experiments.models import (
    CompletedExperiment,
    ExperimentResult,
    FailedExperiment,
)

app = cyclopts.App(
    help="BenchMAC: A benchmark for evaluating AI on Angular Codebase Migrations.",
    version=harness_version(),
)


def _iter_lines_from_jsonl_files(
    files: Iterable[Path],
) -> Iterable[tuple[Path, int, str]]:
    """Iterate over lines from JSONL files, yielding (file_path, line_num, line)."""
    for file_path in files:
        try:
            with file_path.open("r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    yield (file_path, line_num, line)
        except Exception as e:
            logger.warning(f"Unable to read {file_path}: {e}")


def _load_experiment_results(
    results_dir: Path,
) -> Generator[tuple[ExperimentResult, Path, int], None, None]:
    """Load experiment results from JSONL files in the results directory.

    Yields tuples of (experiment_result, file_path, line_num) for each valid
    experiment result.
    """
    if not results_dir.exists():
        logger.warning(f"Results directory not found: {results_dir}")
        return

    jsonl_files = list(results_dir.rglob("*.jsonl"))
    if not jsonl_files:
        logger.warning(f"No JSONL files found in results directory: {results_dir}")
        return

    for file_path, line_num, line in _iter_lines_from_jsonl_files(jsonl_files):
        try:
            data = json.loads(line)
            experiment_result = ExperimentResult.model_validate(data)
            yield (experiment_result, file_path, line_num)
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(
                f"⚠️ Warning: Skipping invalid experiment result in "
                f"{file_path.name} line {line_num}: {e}"
            )


def _prepare_tasks(
    *,
    experiments_results_dir: Path,
    evaluations_dir: Path | None = None,
    instances_path: Path,
    filter_ids: list[str] | None = None,
) -> Generator[EvaluationTask, None, None]:
    """Matches experiment results to instances and yields tasks to be run.

    Only processes completed experiments and shows warnings for failed ones.
    Skips submissions that have already been evaluated successfully.
    """
    logger.info("Loading benchmark instances...")
    instances_map = load_instances(instances_path)
    logger.info(f"Loaded {len(instances_map)} instances.")

    # Load previous evaluation results to avoid re-running completed submissions
    if evaluations_dir is None:
        evaluations_dir = settings.evaluations_dir

    logger.info(f"Loading previous evaluation results from: {evaluations_dir}")
    completed_submission_ids = _load_previous_evaluation_results(evaluations_dir)
    if completed_submission_ids:
        logger.info(
            f"Found {len(completed_submission_ids)} previously completed evaluations"
        )

    logger.debug("Loading and matching experiment results...")
    for experiment_result, file_path, line_num in _load_experiment_results(
        experiments_results_dir
    ):
        if experiment_result.is_failed:
            failed_experiment = experiment_result.root
            assert isinstance(failed_experiment, FailedExperiment)
            logger.warning(
                f"⚠️ Warning: Skipping failed experiment in {file_path.name} "
                f"line {line_num}: {failed_experiment.error}"
            )
            continue

        # Only process completed experiments
        completed_experiment = experiment_result.root
        assert isinstance(completed_experiment, CompletedExperiment)
        submission = completed_experiment.submission

        # Check if this submission has already been evaluated successfully
        if submission.submission_id in completed_submission_ids:
            logger.debug(
                f"⏭️ Skipping already evaluated submission {submission.submission_id} "
                f"for instance {submission.instance_id} "
                f"(from {file_path.name} line {line_num})"
            )
            continue

        if filter_ids and submission.instance_id not in filter_ids:
            continue

        if submission.instance_id in instances_map:
            yield EvaluationTask(
                instance=instances_map[submission.instance_id],
                submission=submission,
            )
        else:
            logger.warning(
                f"⚠️ Warning: Submission for '{submission.instance_id}' found in "
                f"{file_path.name} line {line_num}, but no matching "
                "instance exists. Skipping."
            )


def _run_interactive(
    runner: BenchmarkRunner,
    tasks: Sequence[EvaluationTask],
    output_file: Path,
    logs_dir: Path,
    run_id: str,
) -> list[EvaluationResult]:
    """Handles the evaluation run with a rich progress bar for interactive terminals."""
    total_tasks = len(tasks)
    success_count = 0
    failure_count = 0
    outcomes: list[EvaluationResult] = []

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[bold green]{task.fields[successes]}✓[/]"),
        TextColumn("[bold red]{task.fields[failures]}✗[/]"),
        TextColumn("({task.completed}/{task.total})"),
        TimeRemainingColumn(),
        transient=True,  # The bar will disappear on completion
    )

    with progress, output_file.open("a", encoding="utf-8") as f:
        # Fix: Pass the total number of tasks at creation to avoid "(0 of None)"
        task_id = progress.add_task(
            "Evaluating...", total=total_tasks, successes=0, failures=0
        )

        def on_result(outcome: EvaluationResult):
            nonlocal success_count, failure_count

            # Collect outcome for summary
            outcomes.append(outcome)

            # Update counters based on outcome
            if outcome.status == "completed":
                success_count += 1
                logger.info(f"✅ Completed: {outcome.result.instance_id}")
            else:
                failure_count += 1
                logger.error(f"❌ FAILURE: {outcome.instance_id} - {outcome.error}")

            # Write result to file
            f.write(outcome.model_dump_json() + "\n")

            # Update the progress bar with new counts and advance by 1
            progress.update(
                task_id,
                advance=1,
                successes=success_count,
                failures=failure_count,
            )

        runner.run(
            tasks=tasks,
            log_dir=logs_dir,
            run_id=run_id,
            on_result=on_result,
        )

    return outcomes


def _run_non_interactive(
    runner: BenchmarkRunner,
    tasks: Sequence[EvaluationTask],
    output_file: Path,
    logs_dir: Path,
    run_id: str,
) -> list[EvaluationResult]:
    """Handles the evaluation run with simple line-by-line logging for CI/CD."""
    task_list = list(tasks)
    total_tasks = len(task_list)
    outcomes: list[EvaluationResult] = []
    logger.info(f"Running in non-interactive mode. Evaluating {total_tasks} tasks.")

    with output_file.open("a", encoding="utf-8") as f:

        def on_result(outcome: EvaluationResult):
            # Collect outcome for summary
            outcomes.append(outcome)

            # Write to file
            f.write(outcome.model_dump_json() + "\n")

            # Also print a summary to the console log
            if outcome.status == "completed":
                logger.info(f"✅ Completed: {outcome.result.instance_id}")
            elif outcome.status == "failed":
                logger.error(f"❌ Failed: {outcome.instance_id} - {outcome.error}")

        runner.run(
            tasks=task_list,
            log_dir=logs_dir,
            run_id=run_id,
            on_result=on_result,
        )

    return outcomes


def _load_eval_results_from_file(results_file: Path) -> list[EvaluationResult]:
    """Load EvaluationResult objects from a JSONL results file."""
    outcomes: list[EvaluationResult] = []
    if not results_file.exists():
        logger.error(f"❌ Results file not found: {results_file}")
        return outcomes

    with results_file.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                outcome = EvaluationResultAdapter.validate_json(line)
                outcomes.append(outcome)
            except Exception as e:
                logger.warning(
                    f"⚠️ Warning: Skipping invalid result on line {line_num}: {e}"
                )

    return outcomes


def _load_previous_evaluation_results(
    evaluations_dir: Path,
) -> set[str]:
    """
    Load all previous evaluation results from JSONL files in the evaluations directory
    and return the set of submission IDs that have completed evaluations.

    Only considers completed evaluations (failed ones should be re-run).
    """
    completed_submission_ids: set[str] = set()

    if not evaluations_dir.exists():
        logger.debug(f"Evaluations directory not found: {evaluations_dir}")
        return completed_submission_ids

    # Find all JSONL files recursively
    jsonl_files = list(evaluations_dir.rglob("*.jsonl"))
    if not jsonl_files:
        logger.debug(
            f"No JSONL files found in evaluations directory: {evaluations_dir}"
        )
        return completed_submission_ids

    logger.debug(
        f"Loading previous evaluation results from {len(jsonl_files)} files..."
    )

    for results_file in jsonl_files:
        try:
            results = _load_eval_results_from_file(results_file)
            for result in results:
                if result.status == "completed":
                    # Extract submission_id from the completed evaluation
                    submission_id = result.result.submission_id
                    completed_submission_ids.add(submission_id)
                # Skip failed evaluations - they should be re-run
        except Exception as e:
            logger.warning(
                f"⚠️ Warning: Failed to load results from {results_file}: {e}"
            )

    logger.debug(
        f"Found {len(completed_submission_ids)} previously completed evaluations"
    )
    return completed_submission_ids


def _collect_network_error_details(
    outcomes: list[EvaluationResult],
) -> dict[str, list[str]]:
    """Return mapping of instance_id -> list of step commands impacted
     by network-like errors.

    Includes harness-level failures (no steps) under a synthetic "harness" entry.
    """
    network_error_keywords = [
        "socket timeout",
        "econnreset",
        "etimedout",
        "network is unreachable",
        "failed to fetch",
        "proxy",
    ]

    details: dict[str, list[str]] = {}

    for outcome in outcomes:
        if outcome.status == "completed":
            instance_id = outcome.result.instance_id
            for step in outcome.result.execution.steps:
                if not step.success:
                    stderr_lower = step.stderr.lower()
                    if any(
                        keyword in stderr_lower for keyword in network_error_keywords
                    ):
                        details.setdefault(instance_id, [])
                        if step.command not in details[instance_id]:
                            details[instance_id].append(step.command)
        else:  # failure at harness level
            instance_id = outcome.instance_id
            error_lower = outcome.error.lower()
            if any(keyword in error_lower for keyword in network_error_keywords):
                details.setdefault(instance_id, [])
                if "harness" not in details[instance_id]:
                    details[instance_id].append("harness")

    return details


def _print_evaluation_summary(
    outcomes: list[EvaluationResult],
    run_id: str | None = None,
    results_file: Path | None = None,
    logs_dir: Path | None = None,
) -> None:
    """Print a comprehensive summary of the evaluation results using Rich."""
    console = Console()

    # Calculate metrics
    total_jobs = len(outcomes)
    successful_jobs = sum(1 for outcome in outcomes if outcome.status == "completed")
    failed_jobs = total_jobs - successful_jobs

    # Metrics breakdown - initialize counters for all metrics
    patch_success_count = 0
    patch_total_count = 0
    target_version_success_count = 0
    target_version_total_count = 0
    build_success_count = 0
    build_total_count = 0
    install_success_count = 0
    install_total_count = 0

    for outcome in outcomes:
        if outcome.status == "completed":
            metrics = outcome.result.metrics

            # Patch application metrics
            if metrics.patch_application_success is not None:
                patch_total_count += 1
                if metrics.patch_application_success:
                    patch_success_count += 1

            # Target version metrics
            if metrics.target_version_achieved is not None:
                target_version_total_count += 1
                if metrics.target_version_achieved:
                    target_version_success_count += 1

            # Build success metrics
            if metrics.build_success is not None:
                build_total_count += 1
                if metrics.build_success:
                    build_success_count += 1

            # Install success metrics
            if metrics.install_success is not None:
                install_total_count += 1
                if metrics.install_success:
                    install_success_count += 1

    # Create summary table
    summary_table = Table(title="🎯 BenchMAC Evaluation Summary", show_header=False)
    summary_table.add_column("Field", style="bold cyan", width=20)
    summary_table.add_column("Value", style="white")

    if run_id:
        summary_table.add_row("Run ID:", run_id)
    if results_file:
        summary_table.add_row("Results File:", str(results_file))
    if logs_dir:
        summary_table.add_row("Logs Directory:", str(logs_dir))

    results_table = None
    # Only show the failed instance IDs if there are unsuccessful jobs
    if failed_jobs > 0 and total_jobs > 0:
        failed_instance_ids = [
            outcome.instance_id for outcome in outcomes if outcome.status != "completed"
        ]
        results_table = Table(
            title="❌ Harness Failures",
            show_header=True,
            caption="Note: These are instances where the evaluation harness failed to "
            "process the AI agent's submission. "
            "This does not necessarily mean the AI agent failed; the error may be due "
            "to infrastructure or evaluation issues.",
        )
        results_table.add_column("Instance ID", style="bold red")
        for instance_id in failed_instance_ids:
            results_table.add_row(instance_id)

    # Metrics breakdown table
    metrics_table = Table(title="📋 Metrics Breakdown", show_header=True)
    metrics_table.add_column("Metric", style="bold")
    metrics_table.add_column("Success", justify="right")
    metrics_table.add_column("Total", justify="right")
    metrics_table.add_column("Success Rate", justify="right")

    # Helper function to add metric rows
    def add_metric_row(name: str, success_count: int, total_count: int):
        if total_count > 0:
            success_rate = (success_count / total_count) * 100
            metrics_table.add_row(
                name,
                str(success_count),
                str(total_count),
                f"{success_rate:.1f}%",
            )
        else:
            metrics_table.add_row(name, "N/A", "N/A", "N/A")

    # Add all metric rows
    add_metric_row("Patch Application", patch_success_count, patch_total_count)
    add_metric_row(
        "Target Version Achieved",
        target_version_success_count,
        target_version_total_count,
    )
    add_metric_row("Install Success", install_success_count, install_total_count)
    add_metric_row("Build Success", build_success_count, build_total_count)

    # Print all tables
    console.print()
    console.print(Panel(summary_table, expand=False))
    console.print()
    if results_table:
        console.print(Panel(results_table, expand=False))
    console.print()
    console.print(Panel(metrics_table, expand=False))
    console.print()

    network_error_details = _collect_network_error_details(outcomes)
    if network_error_details:
        details_table = Table(title="Affected instances and steps", show_header=True)
        details_table.add_column("Instance", style="bold")
        details_table.add_column("Step(s)")
        for instance_id, commands in network_error_details.items():
            details_table.add_row(instance_id, "\n".join(commands))

        combined_panel = Panel(
            Group(
                Text(
                    "One or more steps failed due to a potential network issue "
                    "(e.g., 'Socket timeout'). These failures may be transient "
                    "and not reflective of the submission's quality. Consider "
                    "re-running the evaluation with a stable network connection.",
                    justify="center",
                ),
                "\n",
                details_table,
            ),
            title="[bold yellow]⚠️ Network Error Detected[/bold yellow]",
            border_style="yellow",
            expand=False,
        )
        console.print(combined_panel)


# --- Main CLI Command (The Dispatcher) ---


@app.command
def eval(
    experiments_dir: Path = settings.experiments_dir / "results",
    *,
    output_file: Path | None = None,
    instances_file: Path | None = None,
    instance_id: Annotated[
        list[str] | None,
        Parameter(
            help="Filter experiment results by instance ID. "
            "Can be used multiple times.",
            negative=(),
        ),
    ] = None,
    workers: int = os.cpu_count() or 1,
) -> None:
    """
    Run the BenchMAC evaluation on a set of experiment results.

    This command reads ExperimentResult objects from JSONL files in the results
    directory, filters out failed experiments (showing warnings), and evaluates
    the completed ones.
    """
    run_id = utc_now().strftime("%Y-%m-%d_%H%M%S")

    run_dir = settings.evaluations_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Configure logging for the main process
    setup_main_process_logging(run_id, logs_dir)

    if not experiments_dir.exists():
        logger.error(f"❌ Error: Results directory not found at '{experiments_dir}'")
        return

    logger.info(
        f"Starting evaluation run `{run_id}`."
        f" Results for individual instances will be saved to `{run_dir}`."
    )

    # 1. Prepare tasks and output path
    tasks = list(
        _prepare_tasks(
            experiments_results_dir=experiments_dir,
            instances_path=instances_file or settings.instances_file,
            filter_ids=instance_id,
        )
    )

    if not output_file:
        timestamp = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S")
        output_file = run_dir / Path(f"results_{timestamp}.jsonl")

    logger.info(f"Results will be saved to: {output_file}")

    runner = BenchmarkRunner(workers=workers)

    try:
        if sys.stdout.isatty():
            outcomes = _run_interactive(
                runner, tasks, output_file, logs_dir=logs_dir, run_id=run_id
            )
        else:
            outcomes = _run_non_interactive(
                runner, tasks, output_file, logs_dir=logs_dir, run_id=run_id
            )

        logger.info("✅ Evaluation complete.")

        # Print comprehensive summary
        _print_evaluation_summary(
            outcomes,
            run_id,
            output_file,
            logs_dir,
        )
    except Exception as e:
        logger.exception(
            f"❌ An unexpected error occurred during the run: {e.__class__.__name__}: {e}"  # noqa: E501
        )

    logger.complete()


@app.command
def summary(results_file: Path) -> None:
    """
    Display a summary of evaluation results from an existing results.jsonl file.
    """
    if not results_file.exists():
        logger.error(f"❌ Results file not found: {results_file}")
        return

    logger.info(f"Loading results from: {results_file}")

    # Load eval_results from file
    eval_results = _load_eval_results_from_file(results_file)

    if not eval_results:
        logger.error("❌ No valid results found in file")
        return

    logger.info(f"Loaded {len(eval_results)} results")

    # Extract run info from file path if possible
    run_id = None
    logs_dir = None
    if results_file.parent.name.startswith("2025-"):  # Run ID pattern
        run_id = results_file.parent.name
        logs_dir = results_file.parent / "logs"

    # Print summary
    _print_evaluation_summary(
        outcomes=eval_results,
        run_id=run_id,
        results_file=results_file,
        logs_dir=logs_dir,
    )


if __name__ == "__main__":
    app()
