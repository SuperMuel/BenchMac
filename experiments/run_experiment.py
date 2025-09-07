# ruff: noqa: E501
# this is temporary a Typer app, but we'll fusion this with the main CLI later

import json
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import typer
from dotenv import load_dotenv
from pydantic import ValidationError
from rich.console import Console
from rich.progress import Progress

from bench_mac.docker.manager import DockerManager
from bench_mac.models import (
    BenchmarkInstance,
    InstanceID,
    Submission,
    SubmissionID,
)
from experiments.agents.base import BaseAgent
from experiments.agents.mini_swe_agent.agent import MiniSweAgent
from experiments.models import (
    AgentConfig,
    CompletedExperiment,
    ExperimentResult,
    ExperimentTask,
    FailedExperiment,
)
from src.bench_mac.config import settings
from src.bench_mac.utils import load_instances

load_dotenv()

# --- Configuration ---
app = typer.Typer(add_completion=False)
console = Console()


def get_results_file_path(experiments_dir: Path, now: datetime | None = None) -> Path:
    if now is None:
        now = datetime.now(UTC)
    assert experiments_dir.is_dir() and experiments_dir.exists()
    results_dir = experiments_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    return results_dir / f"results_{now.strftime('%Y-%m-%d_%H-%M-%S')}.jsonl"


def collect_tasks(
    instances: dict[str, BenchmarkInstance], agent_configs: list[AgentConfig]
) -> list[ExperimentTask]:
    """
    Collect all patch generation tasks by combining instances with agent configurations.

    Args:
        instances: Dictionary of benchmark instances keyed by instance_id
        agent_configs: List of agent configurations to use for patch generation

    Returns:
        List of ExperimentTask objects, one for each instance-agent_config combination
    """
    return [
        ExperimentTask(instance_id=instance_id, agent_config=agent_config)
        for instance_id in instances
        for agent_config in agent_configs
    ]


def collect_old_results(experiments_dir: Path) -> list[ExperimentResult]:
    """
    Collect all old results from the experiments directory.

    Args:
        experiments_dir: Path to the experiments directory containing results

    Returns:
        List of all ExperimentResult objects found in results files
    """
    results: list[ExperimentResult] = []
    results_dir = experiments_dir / "results"

    if not results_dir.exists():
        return results

    result_files = list(results_dir.glob("*.jsonl"))

    for result_file in result_files:
        try:
            with result_file.open("r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        result = ExperimentResult.model_validate(data)
                        results.append(result)
        except (json.JSONDecodeError, ValidationError) as e:
            console.print(
                f"[yellow]Warning: Could not parse {result_file}: {e}[/yellow]"
            )
            continue

    return results


def filter_completed_tasks(
    tasks: list[ExperimentTask], old_results: list[ExperimentResult]
) -> list[ExperimentTask]:
    """
    Filter out tasks that have already been completed based on old submissions.

    A task is considered completed if there's already a completed result with the same
    instance_id and the same full AgentConfig.

    Args:
        tasks: List of tasks to potentially run
        old_results: List of existing results

    Returns:
        List of tasks that haven't been completed yet
    """
    # Create a set of completed (instance_id, agent_config_key) combinations
    completed_combinations = set()
    for r in old_results:
        if r.root.status != "completed":
            continue
        try:
            instance_id = r.root.task.instance_id
            key = r.root.task.agent_config.key
            completed_combinations.add((instance_id, key))
        except Exception:
            # Be defensive: if structure is unexpected, skip that result
            continue

    # Filter tasks that haven't been completed
    filtered_tasks: list[ExperimentTask] = []
    skipped_count = 0

    for task in tasks:
        task_key = (task.instance_id, task.agent_config.key)
        if task_key in completed_combinations:
            console.print(
                f"[blue]Skipping already completed task: {task.instance_id} ({task.agent_config.model_name})[/blue]"
            )
            skipped_count += 1
        else:
            filtered_tasks.append(task)

    if skipped_count > 0:
        console.print(f"[blue]Skipped {skipped_count} already completed tasks[/blue]")

    return filtered_tasks


def process_single_task(
    task: ExperimentTask,
    agent: BaseAgent,
    submission_id: str,
    dt_factory: Callable[[], datetime] | None = None,
) -> CompletedExperiment | FailedExperiment:
    """
    Process a single experiment task by running the agent and returning the result.

    Args:
        task: The experiment task to process
        instance: The benchmark instance to process
        submission_id: Unique identifier for the submission
        started_at: When the task processing started

    Returns:
        Either a CompletedExperiment or FailedExperiment result
    """
    dt_factory = dt_factory or (lambda: datetime.now(UTC))
    started_at = dt_factory()

    try:
        # Run the agent
        console.print("Running agent")
        model_patch = agent.run(submission_id=submission_id)
        submission = Submission(
            submission_id=SubmissionID(submission_id),
            instance_id=InstanceID(task.instance_id),
            model_patch=model_patch,
        )
        completed = CompletedExperiment(
            task=task,
            submission=submission,
            started_at=started_at,
            ended_at=dt_factory(),
        )
        return completed
    except Exception as e:
        console.print(
            f"[bold red]Error processing {task.instance_id} ({task.agent_config.model_name}): {e}[/bold red]"
        )
        failed = FailedExperiment(
            task=task,
            error=str(e),
            started_at=started_at,
            ended_at=dt_factory(),
        )
        return failed


def create_agent(instance: BenchmarkInstance, agent_config: AgentConfig) -> BaseAgent:
    return MiniSweAgent(instance, agent_config, DockerManager())


DEFAULT_MODEL_NAMES = ["mistral/devstral-medium-2507"]


@app.command()
def main(
    model_names: list[str] = typer.Option(  # noqa: B008
        DEFAULT_MODEL_NAMES,
        "--model-name",
        "-m",
        help="Name of the model(s) to use (e.g., 'mistral/devstral'). Can be used multiple times.",
    ),
    instances_file: Path = typer.Option(  # noqa: B008
        settings.instances_file, help="Path to the benchmark instances file."
    ),
    results_file: Path | None = typer.Option(  # noqa: B008
        None,
        "--results-file",
        "-r",
        help="Path to the results JSONL file.",
    ),
    instance_ids: list[str] | None = typer.Option(  # noqa: B008
        None,
        "--instance-id",
        help="Filter by specific instance ID(s). Can be used multiple times.",
    ),
) -> None:
    """
    Runs an LLM-powered agent on BenchMAC instances and generates a submission file.
    """
    settings.initialize_directories()
    console.print("[bold green]Starting BenchMAC Experiment Runner[/bold green]")

    # Create agent configurations from model names
    agent_configs = [AgentConfig(model_name=model_name) for model_name in model_names]
    model_names_str = ", ".join(
        f"[cyan]{config.model_name}[/cyan]" for config in agent_configs
    )
    console.print(f"📝 Model(s): {model_names_str}")

    # Load all instances first
    all_instances = load_instances(instances_file)

    # Filter by instance IDs if specified
    if instance_ids:
        instances = {
            instance_id: instance
            for instance_id, instance in all_instances.items()
            if instance_id in instance_ids
        }
        console.print(
            f"Filtering to {len(instances)} specified instance(s): {', '.join(instance_ids)}"
        )

        # Check if any specified instances were not found
        missing_instances = set(instance_ids) - set(instances.keys())
        if missing_instances:
            console.print(
                f"[yellow]Warning: The following instance IDs were not found: {', '.join(missing_instances)}[/yellow]"
            )
    else:
        instances = all_instances

    if results_file is None:
        results_file = get_results_file_path(settings.experiments_dir)

    # Collect old results to avoid re-running completed tasks
    old_results = collect_old_results(settings.experiments_dir)
    console.print(f"Found {len(old_results)} existing results")

    # Collect all tasks
    tasks = collect_tasks(instances, agent_configs)
    console.print(f"Generated {len(tasks)} potential tasks")

    # Filter out already completed tasks
    tasks = filter_completed_tasks(tasks, old_results)

    if not tasks:
        console.print("No tasks to process. Exiting...")
        return

    try:
        with Progress(console=console) as progress:
            task_progress = progress.add_task(
                "[cyan]Processing Tasks...", total=len(tasks)
            )

        for task in tasks:
            instance = instances[task.instance_id]
            submission_id = str(uuid.uuid4())
            progress.update(
                task_progress,
                description=f"[cyan]Processing: {task.instance_id} ({task.agent_config.model_name})[/cyan]",
            )

            agent = create_agent(instance, task.agent_config)

            result = process_single_task(
                task,
                agent,
                submission_id,
            )

            result_wrapper = ExperimentResult(root=result)
            with results_file.open("a") as f:
                f.write(result_wrapper.model_dump_json() + "\n")

            progress.advance(task_progress)

    except KeyboardInterrupt:
        console.print(
            "\n[bold yellow]Experiment interrupted by user (KeyboardInterrupt).[/bold yellow]"
        )
        console.print(f"Partial results saved to [cyan]{results_file}[/cyan]")
        return

    console.print("\n[bold green]✅ Experiment finished![/bold green]")
    console.print(f"Results saved to [cyan]{results_file}[/cyan]")


if __name__ == "__main__":
    app()
