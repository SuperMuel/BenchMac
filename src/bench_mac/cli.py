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
from bench_mac.models import BenchmarkInstance, EvaluationTask, RunOutcome, Submission
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
    logs_dir: Path,
    filter_ids: list[str] | None = None,
) -> Generator[EvaluationTask, None, None]:
    """Matches submissions to instances and yields tasks to be run."""
    logger.info("Loading benchmark instances...")
    instances_map = _load_instances(instances_path)
    logger.info(f"Loaded {len(instances_map)} instances.")

    logger.info("Loading and matching submissions...")
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
) -> None:
    """Handles the evaluation run with a rich progress bar for interactive terminals."""
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed} of {task.total})"),
        TimeRemainingColumn(),
        transient=True,  # The bar will disappear on completion
    )

    with progress, output_file.open("a", encoding="utf-8") as f:
        task_id = progress.add_task("Evaluating...", total=None)

        def on_progress(completed: int, total: int):
            progress.update(task_id, total=total, completed=completed)

        def on_result(outcome: RunOutcome):
            f.write(outcome.model_dump_json() + "\n")

        runner.run(
            tasks=tasks, log_dir=logs_dir, on_result=on_result, on_progress=on_progress
        )


def _run_non_interactive(
    runner: BenchmarkRunner,
    tasks: Sequence[EvaluationTask],
    output_file: Path,
    logs_dir: Path,
) -> None:
    """Handles the evaluation run with simple line-by-line logging for CI/CD."""
    task_list = list(tasks)
    total_tasks = len(task_list)
    logger.info(f"Running in non-interactive mode. Evaluating {total_tasks} tasks.")

    with output_file.open("a", encoding="utf-8") as f:

        def on_progress(completed: int, total: int):
            # Simple print statement for progress
            logger.info(f"Progress: {completed}/{total} instances evaluated.")

        def on_result(outcome: RunOutcome):
            # Write to file
            f.write(outcome.model_dump_json() + "\n")
            # Also print a summary to the console log
            if outcome.status == "success":
                logger.info(f"✅ SUCCESS: {outcome.result.instance_id}")
            else:
                logger.error(f"❌ FAILURE: {outcome.instance_id} - {outcome.error}")

        runner.run(
            tasks=task_list,
            log_dir=logs_dir,
            on_result=on_result,
            on_progress=on_progress,
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

    run_dir = (
        Path("results") / run_id
    )  # TODO: make this configurable via config or cli arg
    run_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Configure loguru global logger
    logger.remove()  # Remove the default handler

    # Sink 1: Console output (for humans)
    logger.add(
        sys.stderr,
        level=settings.cli_default_log_level,
        format="<level>{message}</level>",
    )

    # Sink 2: Central run log file (for machines/debugging)
    logger.add(
        logs_dir / "run.log",
        level="DEBUG",
        serialize=True,
        enqueue=True,  # CRITICAL for multiprocessing
        backtrace=True,  # Automatically log full stack traces on exceptions
        diagnose=True,  # Adds extra details to exception traces
    )

    # 3. Add the run_id to ALL subsequent log records globally
    logger.configure(extra={"run_id": run_id})

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
            logs_dir=run_dir / "logs",
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
            _run_interactive(runner, tasks, output_file, logs_dir=logs_dir)
        else:
            _run_non_interactive(runner, tasks, output_file, logs_dir=logs_dir)

        logger.info("✅ Evaluation complete.")
    except Exception as e:
        logger.error(f"❌ An unexpected error occurred during the run: {e}")

    logger.complete()


if __name__ == "__main__":
    app()
