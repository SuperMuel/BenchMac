# ruff: noqa: E501
# this is temporary a Typer app, but we'll fusion this with the main CLI later

import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from pathlib import Path

import typer
from dotenv import load_dotenv
from loguru import logger
from pydantic import ValidationError
from rich.console import Console
from rich.progress import Progress
from uuid6 import uuid7

from bench_mac.config import settings
from bench_mac.docker.manager import DockerManager
from bench_mac.models import (
    BenchmarkInstance,
    InstanceID,
    Submission,
    SubmissionID,
)
from bench_mac.utils import load_instances
from experiments.agents.angular_schematics import AngularSchematicsAgent
from experiments.agents.base import BaseAgent
from experiments.agents.mini_swe_agent.agent import MiniSweAgent
from experiments.logging_utils import bind_run_context, setup_experiment_logging
from experiments.models import (
    AgentConfig,
    AngularSchematicsConfig,
    CompletedExperiment,
    ExperimentArtifacts,
    ExperimentResult,
    ExperimentTask,
    FailedExperiment,
    MiniSweAgentConfig,
)

load_dotenv()

# --- Configuration ---
app = typer.Typer(add_completion=False)
console = Console()


SUPPORTED_SCAFFOLDS = {"swe-agent-mini", "angular-schematics"}
DEFAULT_SCAFFOLDS = ("swe-agent-mini",)
DEFAULT_MODEL_NAMES = ("mistral/devstral-medium-2507",)


def _resolve_minisweagent_version() -> str:
    try:
        return pkg_version("minisweagent")
    except PackageNotFoundError as exc:  # pragma: no cover - defensive guard
        msg = "minisweagent must be installed to run the swe-agent-mini scaffold"
        raise RuntimeError(msg) from exc


def build_agent_configs(
    scaffolds: list[str],
    model_names: list[str],
) -> list[AgentConfig]:
    configs: list[AgentConfig] = []
    mini_swe_agent_version: str | None = None
    for scaffold in scaffolds:
        scaffold_key = scaffold.strip().lower()

        if scaffold_key not in SUPPORTED_SCAFFOLDS:
            raise typer.BadParameter(
                f"Unsupported scaffold '{scaffold}'. Supported: {', '.join(sorted(SUPPORTED_SCAFFOLDS))}"
            )

        if scaffold_key == "swe-agent-mini":
            if not model_names:
                raise typer.BadParameter(
                    "At least one --model-name must be provided when using the swe-agent-mini scaffold."
                )
            if mini_swe_agent_version is None:
                mini_swe_agent_version = _resolve_minisweagent_version()
            for model_name in model_names:
                configs.append(
                    MiniSweAgentConfig(
                        model_name=model_name,
                        library_version=mini_swe_agent_version,
                    )
                )
        elif scaffold_key == "angular-schematics":
            configs.append(AngularSchematicsConfig())

    return configs


def get_results_file_path(experiments_dir: Path, now: datetime | None = None) -> Path:
    if now is None:
        now = datetime.now(UTC)
    assert experiments_dir.is_dir() and experiments_dir.exists()
    results_dir = experiments_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    return results_dir / f"results_{now.strftime('%Y-%m-%d_%H-%M-%S')}.jsonl"


def _format_timedelta(td: timedelta) -> str:
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


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
                f"[blue]Skipping already completed task: {task.instance_id} "
                f"({task.agent_config.display_name})[/blue]"
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
    task_logger = bind_run_context(
        instance=task.instance_id,
        model=task.agent_config.display_name,
        submission=submission_id,
    )

    try:
        # Run the agent
        console.print("Running agent")
        task_logger.info("Starting agent run")
        agent_result = agent.run(submission_id=submission_id)
        submission = Submission(
            submission_id=SubmissionID(submission_id),
            instance_id=InstanceID(task.instance_id),
            model_patch=agent_result.model_patch,
        )
        artifacts = None
        if agent_result.artifacts is not None:
            artifacts = ExperimentArtifacts(
                execution_trace=agent_result.artifacts.execution_trace,
                cost_usd=agent_result.artifacts.cost_usd,
                n_calls=agent_result.artifacts.n_calls,
            )
        completed = CompletedExperiment(
            task=task,
            submission=submission,
            started_at=started_at,
            ended_at=dt_factory(),
            artifacts=artifacts,
        )
        # Post-completion summary
        price_str = "unknown"
        steps_str = "unknown"
        duration_str = "unknown"
        if completed.artifacts is not None:
            if completed.artifacts.cost_usd is not None:
                price_str = f"${completed.artifacts.cost_usd:.4f}"
            if completed.artifacts.execution_trace is not None:
                try:
                    steps = len(completed.artifacts.execution_trace.steps)
                    steps_str = str(steps)
                    duration_td = completed.artifacts.execution_trace.total_duration
                    duration_str = _format_timedelta(duration_td)
                except Exception:
                    # Be tolerant of partial/malformed traces
                    pass
        console.print(
            "[green]✓ Completed:[/green] "
            f"{task.instance_id} ({task.agent_config.display_name}) — "
            f"price: {price_str}, steps: {steps_str}, total: {duration_str}"
        )
        task_logger.success("Task completed successfully")
        return completed
    except Exception as e:
        console.print(
            f"[bold red]Error processing {task.instance_id} "
            f"({task.agent_config.display_name}): {e}[/bold red]"
        )
        task_logger.opt(exception=True).error("Task failed during agent run")
        failed_artifacts = None
        agent_artifacts = agent.collect_artifacts()
        if agent_artifacts is not None:
            failed_artifacts = ExperimentArtifacts(
                execution_trace=agent_artifacts.execution_trace,
                cost_usd=agent_artifacts.cost_usd,
                n_calls=agent_artifacts.n_calls,
            )
        failed = FailedExperiment(
            task=task,
            error=str(e),
            started_at=started_at,
            ended_at=dt_factory(),
            artifacts=failed_artifacts,
        )
        return failed


def create_agent(instance: BenchmarkInstance, agent_config: AgentConfig) -> BaseAgent:
    docker_manager = DockerManager()

    match agent_config:
        case MiniSweAgentConfig():
            return MiniSweAgent(instance, agent_config, docker_manager)
        case AngularSchematicsConfig():
            return AngularSchematicsAgent(instance, agent_config, docker_manager)

    msg = f"Unsupported scaffold '{agent_config.scaffold}'"
    raise ValueError(msg)


@app.command()
def main(
    scaffolds: list[str] = typer.Option(  # noqa: B008
        list(DEFAULT_SCAFFOLDS),
        "--scaffold",
        help="Agent scaffold(s) to run (e.g., 'swe-agent-mini', 'angular-schematics'). Can be used multiple times.",
    ),
    model_names: list[str] = typer.Option(  # noqa: B008
        list(DEFAULT_MODEL_NAMES),
        "--model-name",
        "-m",
        help="Name of the model(s) to use when applicable for the scaffold (e.g., 'mistral/devstral'). Can be used multiple times.",
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

    selected_scaffolds = scaffolds or list(DEFAULT_SCAFFOLDS)
    agent_configs = build_agent_configs(selected_scaffolds, model_names)

    if not agent_configs:
        console.print("[yellow]No agent configurations specified. Exiting...[/yellow]")
        return

    agents_display = ", ".join(
        f"[cyan]{config.display_name}[/cyan]" for config in agent_configs
    )
    console.print(f"📝 Agent(s): {agents_display}")

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

    log_path = setup_experiment_logging(results_file.stem)
    console.print(f"🗒 Logging to [cyan]{log_path}[/cyan]")
    logger.info("Initialized experiment logging -> {log_path}", log_path=str(log_path))

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
            submission_id = str(uuid7())
            task_logger = bind_run_context(
                instance=task.instance_id,
                model=task.agent_config.display_name,
                submission=submission_id,
            )
            progress.update(
                task_progress,
                description=(
                    f"[cyan]Processing: {task.instance_id} "
                    f"({task.agent_config.display_name})[/cyan]"
                ),
            )

            agent = create_agent(instance, task.agent_config)
            task_logger.debug("Agent created for task")

            result = process_single_task(
                task,
                agent,
                submission_id,
            )

            result_wrapper = ExperimentResult(root=result)
            with results_file.open("a") as f:
                f.write(result_wrapper.model_dump_json() + "\n")

            task_logger.info(
                "Persisted result to {results_file}",
                results_file=str(results_file),
            )

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
