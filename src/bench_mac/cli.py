import json
import os
import sys
from collections.abc import Generator, Sequence
from datetime import UTC, datetime
from pathlib import Path

import cyclopts
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

from bench_mac.config import settings
from bench_mac.models import BenchmarkInstance, RunOutcome, Submission
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
                print(f"⚠️ Warning: Skipping invalid submission on line {i}: {e}")


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
) -> Generator[tuple[BenchmarkInstance, Submission], None, None]:
    """Matches submissions to instances and yields tasks to be run."""
    print("Loading benchmark instances...")
    instances_map = _load_instances(instances_path)
    print(f"Loaded {len(instances_map)} instances.")

    print("Loading and matching submissions...")
    for sub in _load_submissions(submissions_path):
        if filter_ids and sub.instance_id not in filter_ids:
            continue

        if sub.instance_id in instances_map:
            yield (instances_map[sub.instance_id], sub)
        else:
            print(
                f"⚠️ Warning: Submission for '{sub.instance_id}' found, but no matching "
                "instance exists. Skipping."
            )


def _run_interactive(
    runner: BenchmarkRunner,
    tasks: Sequence[tuple[BenchmarkInstance, Submission]],
    output_file: Path,
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

        runner.run(tasks=tasks, on_result=on_result, on_progress=on_progress)


def _run_non_interactive(
    runner: BenchmarkRunner,
    tasks: Sequence[tuple[BenchmarkInstance, Submission]],
    output_file: Path,
) -> None:
    """Handles the evaluation run with simple line-by-line logging for CI/CD."""
    task_list = list(tasks)
    total_tasks = len(task_list)
    print(f"Running in non-interactive mode. Evaluating {total_tasks} tasks.")

    with output_file.open("a", encoding="utf-8") as f:

        def on_progress(completed: int, total: int):
            # Simple print statement for progress
            print(f"Progress: {completed}/{total} instances evaluated.")

        def on_result(outcome: RunOutcome):
            # Write to file
            f.write(outcome.model_dump_json() + "\n")
            # Also print a summary to the console log
            if outcome.status == "success":
                print(f"✅ SUCCESS: {outcome.result.instance_id}")
            else:
                print(f"❌ FAILURE: {outcome.instance_id} - {outcome.error}")

        runner.run(tasks=task_list, on_result=on_result, on_progress=on_progress)


# --- Main CLI Command (The Dispatcher) ---


@app.command
def evaluate(
    submissions_file: Path,
    *,
    output_file: Path | None = None,
    instances_file: Path | None = None,
    instance_id: list[str] | None = None,
    workers: int = os.cpu_count() or 1,
) -> None:
    """
    Run the BenchMAC evaluation on a set of submissions.
    ...
    """
    if not submissions_file.exists():
        print(f"❌ Error: Submissions file not found at '{submissions_file}'")
        return

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
        output_file = Path(f"results_{timestamp}.jsonl")

    print(f"Results will be saved to: {output_file}")

    # 2. Instantiate the runner
    runner = BenchmarkRunner(workers=workers)

    # 3. Dispatch to the correct runner function based on the environment
    try:
        if sys.stdout.isatty():
            _run_interactive(runner, tasks, output_file)
        else:
            _run_non_interactive(runner, tasks, output_file)

        print("✅ Evaluation complete.")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred during the run: {e}")


if __name__ == "__main__":
    app()
