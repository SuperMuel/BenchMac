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
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

from bench_mac.config import settings
from bench_mac.logging_config import setup_main_process_logging
from bench_mac.models import BenchmarkInstance, ExecutionJob, RunOutcome, Submission
from bench_mac.runner import BenchmarkRunner

app = cyclopts.App(
    help="BenchMAC: A benchmark for evaluating AI on Angular Codebase Migrations."
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


def _load_instances(instances_path: Path) -> dict[str, BenchmarkInstance]:
    """Loads benchmark instances into a dict for fast lookup."""
    instances: dict[str, BenchmarkInstance] = {}
    with instances_path.open("r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            instance = BenchmarkInstance.model_validate(data)
            instances[instance.instance_id] = instance
    return instances


def _prepare_tasks(
    submissions_path: Path,
    instances_path: Path,
    filter_ids: list[str] | None = None,
) -> Generator[ExecutionJob, None, None]:
    """Matches submissions to instances and yields tasks to be run."""
    logger.info("Loading benchmark instances...")
    instances_map = _load_instances(instances_path)
    logger.info(f"Loaded {len(instances_map)} instances.")

    logger.debug("Loading and matching submissions...")
    for sub in _load_submissions(submissions_path):
        if filter_ids and sub.instance_id not in filter_ids:
            continue

        if sub.instance_id in instances_map:
            yield ExecutionJob(
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
    tasks: Sequence[ExecutionJob],
    output_file: Path,
    logs_dir: Path,
    run_id: str,
) -> None:
    """Handles the evaluation run with a rich progress bar for interactive terminals."""
    total_tasks = len(tasks)
    success_count = 0
    failure_count = 0

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

        def on_result(outcome: RunOutcome):
            nonlocal success_count, failure_count

            # Update counters based on outcome
            if outcome.status == "success":
                success_count += 1
            else:
                failure_count += 1

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


def _run_non_interactive(
    runner: BenchmarkRunner,
    tasks: Sequence[ExecutionJob],
    output_file: Path,
    logs_dir: Path,
    run_id: str,
) -> None:
    """Handles the evaluation run with simple line-by-line logging for CI/CD."""
    task_list = list(tasks)
    total_tasks = len(task_list)
    logger.info(f"Running in non-interactive mode. Evaluating {total_tasks} tasks.")

    with output_file.open("a", encoding="utf-8") as f:

        def on_result(outcome: RunOutcome):
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


# --- Main CLI Command (The Dispatcher) ---


@app.command
def evaluate(
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
    run_id = datetime.now(UTC).strftime("%Y-%m-%d_%H%M%S")

    run_dir = settings.results_dir / run_id
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
            _run_interactive(
                runner, tasks, output_file, logs_dir=logs_dir, run_id=run_id
            )
        else:
            _run_non_interactive(
                runner, tasks, output_file, logs_dir=logs_dir, run_id=run_id
            )

        logger.info("✅ Evaluation complete.")
    except Exception as e:
        logger.exception(
            f"❌ An unexpected error occurred during the run: {e.__class__.__name__}: {e}"  # noqa: E501
        )

    logger.complete()


if __name__ == "__main__":
    app()
