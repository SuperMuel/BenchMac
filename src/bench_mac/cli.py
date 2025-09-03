import json
import os
import sys
from collections.abc import Generator, Sequence
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
    EvaluationCompleted,
    EvaluationFailed,
    EvaluationResult,
    EvaluationTask,
    Submission,
    utc_now,
)
from bench_mac.runner import BenchmarkRunner
from bench_mac.utils import load_instances
from bench_mac.version import harness_version

app = cyclopts.App(
    help="BenchMAC: A benchmark for evaluating AI on Angular Codebase Migrations.",
    version=harness_version(),
)


def _load_submissions(
    submissions_path: Path,
) -> Generator[Submission, None, None]:
    """Loads submissions from a JSONL file."""
    with submissions_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                yield Submission.model_validate(data)
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(
                    f"⚠️ Warning: Skipping invalid submission on line {i}: {e}"
                )


def _prepare_tasks(
    submissions_path: Path,
    instances_path: Path,
    filter_ids: list[str] | None = None,
) -> Generator[EvaluationTask, None, None]:
    """Matches submissions to instances and yields tasks to be run."""
    logger.info("Loading benchmark instances...")
    instances_map = load_instances(instances_path)
    logger.info(f"Loaded {len(instances_map)} instances.")

    logger.debug("Loading and matching submissions...")
    for sub in _load_submissions(submissions_path):
        if filter_ids and sub.instance_id not in filter_ids:
            continue

        if sub.instance_id in instances_map:
            yield EvaluationTask(
                instance=instances_map[sub.instance_id],
                submission=sub,
            )
        else:
            logger.warning(
                f"⚠️ Warning: Submission for '{sub.instance_id}' found, but no matching "
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
            if outcome.status == "success":
                success_count += 1
                logger.info(f"✅ SUCCESS: {outcome.result.instance_id}")
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
            if outcome.status == "success":
                logger.info(f"✅ SUCCESS: {outcome.result.instance_id}")
            elif outcome.status == "failure":
                logger.error(f"❌ FAILURE: {outcome.instance_id} - {outcome.error}")

        runner.run(
            tasks=task_list,
            log_dir=logs_dir,
            run_id=run_id,
            on_result=on_result,
        )

    return outcomes


def _load_outcomes_from_file(results_file: Path) -> list[EvaluationResult]:
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
                data = json.loads(line)
                # Handle union type by checking the status field
                if data.get("status") == "success":
                    outcome = EvaluationCompleted.model_validate(data)
                elif data.get("status") == "failure":
                    outcome = EvaluationFailed.model_validate(data)
                else:
                    raise ValueError(f"Unknown status: {data.get('status')}")
                outcomes.append(outcome)
            except Exception as e:
                logger.warning(
                    f"⚠️ Warning: Skipping invalid result on line {line_num}: {e}"
                )

    return outcomes


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
        if outcome.status == "success":
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
    successful_jobs = sum(1 for outcome in outcomes if outcome.status == "success")
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
        if outcome.status == "success":
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
            outcome.instance_id for outcome in outcomes if outcome.status != "success"
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
    add_metric_row("Build Success", build_success_count, build_total_count)
    add_metric_row("Install Success", install_success_count, install_total_count)

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
    submissions_file: Path,
    *,
    output_file: Path | None = None,
    instances_file: Path | None = None,
    instance_id: Annotated[
        list[str] | None,
        Parameter(
            help="Filter submissions by instance ID. Can be used multiple times.",
            negative=(),
        ),
    ] = None,
    workers: int = os.cpu_count() or 1,
) -> None:
    """
    Run the BenchMAC evaluation on a set of submissions.
    """
    run_id = utc_now().strftime("%Y-%m-%d_%H%M%S")

    run_dir = settings.evaluations_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Configure logging for the main process
    setup_main_process_logging(run_id, logs_dir)

    if not submissions_file.exists():
        logger.error(f"❌ Error: Submissions file not found at '{submissions_file}'")
        return

    logger.info(
        f"Starting evaluation run `{run_id}`."
        f" Results for individual instances will be saved to `{run_dir}`."
    )

    # 1. Prepare tasks and output path
    tasks = list(
        _prepare_tasks(
            submissions_path=submissions_file,
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

    # Load outcomes from file
    outcomes = _load_outcomes_from_file(results_file)

    if not outcomes:
        logger.error("❌ No valid results found in file")
        return

    logger.info(f"Loaded {len(outcomes)} results")

    # Extract run info from file path if possible
    run_id = None
    logs_dir = None
    if results_file.parent.name.startswith("2025-"):  # Run ID pattern
        run_id = results_file.parent.name
        logs_dir = results_file.parent / "logs"

    # Print summary
    _print_evaluation_summary(
        outcomes=outcomes,
        run_id=run_id,
        results_file=results_file,
        logs_dir=logs_dir,
    )


if __name__ == "__main__":
    app()
