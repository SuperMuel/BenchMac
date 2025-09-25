# ruff: noqa: E501
# this is temporary a Typer app, but we'll fusion this with the main CLI later

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from pathlib import Path
from threading import Semaphore
from typing import Any, Literal

import litellm
import typer
import yaml
from dotenv import load_dotenv
from loguru import logger
from rich.console import Console
from rich.progress import Progress, TaskID
from uuid6 import uuid7

from bench_mac.core.config import settings
from bench_mac.core.models import (
    BenchmarkInstance,
    InstanceID,
    Submission,
    SubmissionID,
)
from bench_mac.core.utils import load_instances
from bench_mac.docker.manager import DockerManager
from experiments.agents.angular_schematics import AngularSchematicsAgent
from experiments.agents.base import BaseAgent
from experiments.agents.mini_swe_agent.agent import MiniSweAgent
from experiments.logging_utils import bind_run_context, setup_experiment_logging
from experiments.models import (
    AgentConfig,
    AngularSchematicsConfig,
    CompletedExperiment,
    ExperimentResult,
    ExperimentTask,
    FailedExperiment,
    MiniSweAgentConfig,
)
from experiments.storage import (
    RESULTS_DIR_NAME,
    create_results_run_dir,
    load_all_experiment_results,
    save_experiment_result,
)

load_dotenv()


litellm.register_model(  # pyright: ignore[reportPrivateImportUsage]
    {
        "mistral/magistral-medium-2509": {
            "max_tokens": 128000,
            "input_cost_per_token": 2e-06,
            "output_cost_per_token": 5e-06,
            "litellm_provider": "mistral",
            "mode": "chat",
        },
        "mistral/magistral-small-2509": {
            "max_tokens": 128000,
            "input_cost_per_token": 5e-07,
            "output_cost_per_token": 1.5e-06,
            "litellm_provider": "mistral",
            "mode": "chat",
        },
        "mistral/mistral-medium-2508": {
            "max_tokens": 128000,
            "input_cost_per_token": 4e-07,
            "output_cost_per_token": 2e-06,
            "litellm_provider": "mistral",
            "mode": "chat",
        },
        "mistral/mistral-small-2506": {
            "max_tokens": 128000,
            "input_cost_per_token": 1e-07,
            "output_cost_per_token": 3e-07,
            "litellm_provider": "mistral",
            "mode": "chat",
        },
    }
)
litellm.drop_params = True


# --- Configuration ---
app = typer.Typer(add_completion=False)
console = Console()


SUPPORTED_SCAFFOLDS = {"swe-agent-mini", "angular-schematics"}
DEFAULT_SCAFFOLDS = ("swe-agent-mini",)
DEFAULT_MODEL_NAMES = ("mistral/devstral-medium-2507",)


def _is_known_model(model_name: str) -> bool:
    """Return True if a model is known to litellm (built-in or registered).

    Accepts provider-prefixed names like ``openai/gpt-4o`` by also checking the
    suffix after the first ``/``.
    """
    candidates: list[str] = [model_name]
    if "/" in model_name:
        candidates.append(model_name.split("/", 1)[1])

    for candidate in candidates:
        # 1) Directly in the cost map (includes manual registrations)
        try:
            if candidate in litellm.model_cost:
                return True
        except Exception:
            pass

        # 2) Aggregated known model list
        try:
            model_list_set = getattr(litellm, "model_list_set", None)
            if model_list_set and candidate in model_list_set:
                return True
        except Exception:
            pass

        # 3) By provider groupings
        try:
            for models in getattr(litellm, "models_by_provider", {}).values():
                if isinstance(models, set | list | tuple) and candidate in models:
                    return True
        except Exception:
            pass

    return False


def _validate_model_names_or_exit(model_names: list[str]) -> None:
    """Validate that all provided model names are known to litellm.

    Raises a Typer error when one or more models are unknown.
    """
    unknown = [m for m in model_names if not _is_known_model(m)]
    if unknown:
        missing = ", ".join(unknown)
        msg = (
            "Unknown model(s): "
            f"{missing}. Register them via litellm.register_model(...) or choose valid built-in models."
        )
        raise typer.BadParameter(msg)


@dataclass(slots=True)
class TaskEvent:
    status: Literal["started", "completed", "failed"]
    message: str
    instance_id: str
    agent_display_name: str
    submission_id: str
    details: dict[str, Any] | None = None


def _resolve_minisweagent_version() -> str:
    try:
        return pkg_version("mini-swe-agent")
    except PackageNotFoundError as exc:  # pragma: no cover - defensive guard
        msg = "minisweagent must be installed to run the swe-agent-mini scaffold"
        raise RuntimeError(msg) from exc


def build_agent_configs(
    *,
    scaffolds: list[str],
    model_names: list[str],
    task_templates: str,
    agent_settings: dict[str, Any],
    step_limit: int | None = None,
    cost_limit_usd: float | None = None,
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
                        swe_agent_mini_version=mini_swe_agent_version,
                        task_template=task_templates,
                        agent_settings=agent_settings,
                        step_limit=step_limit,
                        cost_limit_usd=cost_limit_usd,
                    )
                )
        elif scaffold_key == "angular-schematics":
            configs.append(AngularSchematicsConfig())

    return configs


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
    results_root = experiments_dir / RESULTS_DIR_NAME

    def on_error(path: Path, error: Exception) -> None:
        console.print(f"[yellow]Warning: Could not parse {path}: {error}[/yellow]")

    return load_all_experiment_results(results_root, on_error=on_error)


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
) -> tuple[CompletedExperiment | FailedExperiment, list[TaskEvent]]:
    """
    Process a single experiment task by running the agent and returning the result.

    Args:
        task: The experiment task to process
        instance: The benchmark instance to process
        submission_id: Unique identifier for the submission
        started_at: When the task processing started

    Returns:
        Tuple containing the experiment outcome and emitted task events
    """
    dt_factory = dt_factory or (lambda: datetime.now(UTC))
    started_at = dt_factory()
    task_logger = bind_run_context(
        instance=task.instance_id,
        model=task.agent_config.display_name,
        submission=submission_id,
    )

    events: list[TaskEvent] = []

    def add_event(
        status: Literal["started", "completed", "failed"],
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        events.append(
            TaskEvent(
                status=status,
                message=message,
                instance_id=task.instance_id,
                agent_display_name=task.agent_config.display_name,
                submission_id=submission_id,
                details=details,
            )
        )

    add_event("started", "Starting agent run")
    task_logger.info("Starting agent run")

    try:
        agent_result = agent.run(submission_id=submission_id)
        if not agent_result.model_patch or not agent_result.model_patch.strip():
            raise RuntimeError(
                "Agent produced an empty diff; submission aborted before evaluation."
            )
        submission = Submission(
            submission_id=SubmissionID(submission_id),
            instance_id=InstanceID(task.instance_id),
            model_patch=agent_result.model_patch,
        )
        artifacts = None
        if agent_result.artifacts is not None:
            artifacts = agent_result.artifacts
        completed = CompletedExperiment(
            task=task,
            submission=submission,
            started_at=started_at,
            ended_at=dt_factory(),
            artifacts=artifacts,
        )
        details: dict[str, Any] = {}
        artifacts = completed.artifacts
        if artifacts is not None:
            if artifacts.cost_usd is not None:
                details["cost_usd"] = artifacts.cost_usd
            if artifacts.execution_trace is not None:
                try:
                    details["steps"] = len(artifacts.execution_trace.steps)
                    duration_td = artifacts.execution_trace.total_duration
                    details["duration"] = _format_timedelta(duration_td)
                except Exception:
                    pass
        add_event("completed", "Task completed successfully", details=details or None)
        task_logger.success("Task completed successfully")
        return completed, events
    except Exception as e:
        task_logger.opt(exception=True).error("Task failed during agent run")
        agent_artifacts = agent.collect_artifacts()
        failed = FailedExperiment(
            task=task,
            error=str(e),
            started_at=started_at,
            ended_at=dt_factory(),
            artifacts=agent_artifacts,
        )
        add_event("failed", "Task failed", details={"error": str(e)})
        return failed, events


def execute_task(
    *,
    task: ExperimentTask,
    instance: BenchmarkInstance,
    submission_id: str,
    results_dir: Path,
    dt_factory: Callable[[], datetime] | None = None,
    provider_lock: Semaphore | None = None,
) -> tuple[ExperimentResult, Path, list[TaskEvent]]:
    """Run a single task end-to-end and persist the result JSON.

    When ``provider_lock`` is provided we deliberately serialize execution for
    agents that talk to the same external model provider. This prevents two
    MiniSWE runs from hammering the same LLM API concurrently, which helps us
    respect provider rate limits even if the global worker pool is larger.
    """
    assert task.instance_id == instance.instance_id

    task_logger = bind_run_context(
        instance=task.instance_id,
        model=task.agent_config.display_name,
        submission=submission_id,
    )

    if provider_lock is None:
        agent = create_agent(instance, task.agent_config)
        task_logger.debug("Agent created for task")
        result, events = process_single_task(
            task,
            agent,
            submission_id,
            dt_factory=dt_factory,
        )
    else:
        with provider_lock:
            agent = create_agent(instance, task.agent_config)
            task_logger.debug("Agent created for task")
            result, events = process_single_task(
                task,
                agent,
                submission_id,
                dt_factory=dt_factory,
            )

    result_wrapper = ExperimentResult(root=result)
    result_path = save_experiment_result(result_wrapper, results_dir)

    task_logger.info(
        "Persisted result to {results_path}",
        results_path=str(result_path),
    )

    return result_wrapper, result_path, events


def render_task_event(event: TaskEvent) -> None:
    """Render a task event to the console."""

    header = f"{event.instance_id} ({event.agent_display_name})"

    if event.status == "started":
        console.print(f"[cyan]Running agent:[/cyan] {header}")
        return

    if event.status == "completed":
        cost = event.details.get("cost_usd") if event.details else None
        steps = event.details.get("steps") if event.details else None
        duration = event.details.get("duration") if event.details else None

        price_str = f"${cost:.4f}" if isinstance(cost, int | float) else "unknown"
        steps_str = str(steps) if isinstance(steps, int) else "unknown"
        duration_str = duration if isinstance(duration, str) else "unknown"

        console.print(
            "[green]✓ Completed:[/green] "
            f"{header} — price: {price_str}, steps: {steps_str}, total: {duration_str}"
        )
        return

    if event.status == "failed":
        error = "unknown error"
        if event.details and isinstance(event.details.get("error"), str):
            error = event.details["error"]
        console.print(f"[bold red]Error processing {header}: {error}[/bold red]")
        return

    console.print(event.message)


def resolve_provider_key(task: ExperimentTask) -> str:
    """Return a coarse provider identifier used for concurrency throttling."""
    agent_config = task.agent_config
    match agent_config:
        case MiniSweAgentConfig():
            model_name = agent_config.model_name
            return model_name.split("/", 1)[0]
        case AngularSchematicsConfig():
            return "angular-schematics"
        case _:
            return "unknown"


def dispatch_tasks(
    *,
    tasks: list[ExperimentTask],
    instances: dict[str, BenchmarkInstance],
    results_dir: Path,
    progress: Progress,
    progress_task_id: int,
    max_workers: int,
    provider_limit: int,
) -> None:
    """Execute tasks with global and per-provider concurrency limits.

    ``max_workers`` bounds the total number of live Docker containers, while
    ``provider_limit`` caps how many workers can hit the same provider at once.
    Keeping this small prevents a single LLM backend from triggering API
    throttling without blocking unrelated providers.
    """
    total_tasks = len(tasks)
    completed = 0
    max_workers = max(1, max_workers)
    provider_limit = max(1, provider_limit)
    executor = ThreadPoolExecutor(max_workers=max_workers)
    futures: dict[
        Future[tuple[ExperimentResult, Path, list[TaskEvent]]],
        tuple[ExperimentTask, str],
    ] = {}
    provider_locks: dict[str, Semaphore] = {}
    cancelled = False
    try:
        for task in tasks:
            instance = instances[task.instance_id]
            submission_id = str(uuid7())

            provider_key = resolve_provider_key(task)
            provider_lock = provider_locks.get(provider_key)
            if provider_lock is None:
                # Many hosted LLM APIs (OpenAI, Anthropic, Groq, etc.) impose
                # shared tenant rate limits; keeping this semaphore small
                # reduces the likelihood of 429 responses while still allowing
                # other providers to progress in parallel.
                provider_lock = Semaphore(provider_limit)
                provider_locks[provider_key] = provider_lock

            future = executor.submit(
                execute_task,
                task=task,
                instance=instance,
                submission_id=submission_id,
                results_dir=results_dir,
                provider_lock=provider_lock,
            )
            futures[future] = (task, submission_id)

        for future in as_completed(futures):
            task, submission_id = futures[future]
            try:
                _, _, events = future.result()
            except Exception as exc:
                logger.opt(exception=True).error(
                    "Task execution crashed: instance={instance} model={model}",
                    instance=task.instance_id,
                    model=task.agent_config.display_name,
                )
                render_task_event(
                    TaskEvent(
                        status="failed",
                        message="Task crashed",
                        instance_id=task.instance_id,
                        agent_display_name=task.agent_config.display_name,
                        submission_id=submission_id,
                        details={"error": str(exc)},
                    )
                )
            else:
                for event in events:
                    render_task_event(event)
            finally:
                completed += 1
                remaining = total_tasks - completed
                progress.advance(TaskID(progress_task_id))
                progress.update(
                    TaskID(progress_task_id),
                    description=(
                        f"[cyan]Processing Tasks ({remaining} remaining)[/cyan]"
                    ),
                )
    except KeyboardInterrupt:
        cancelled = True
        for future in futures:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    finally:
        if not cancelled:
            executor.shutdown(wait=True)


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
    results_dir: Path | None = typer.Option(  # noqa: B008
        None,
        "--results-dir",
        "-r",
        help=f"Directory where experiment JSON files will be written. Default: {settings.experiments_dir}/results/run_<timestamp>",
    ),
    instance_ids: list[str] | None = typer.Option(  # noqa: B008
        None,
        "--instance-id",
        help="Filter by specific instance ID(s). Can be used multiple times.",
    ),
    swe_mini_config_yaml: Path = typer.Option(  # noqa: B008
        Path("experiments/prompts/mini_swe_agent/minimal.yaml"),
        "--swe-mini-config-yaml",
        help="Path to the swe-mini config YAML file.",
    ),
    step_limit: int | None = typer.Option(
        100,
        "--step-limit",
        help="Maximum number of steps for the swe-agent-mini scaffold.",
    ),
    cost_limit_usd: float | None = typer.Option(
        1.0,
        "--cost-limit-usd",
        help="Maximum cost in USD for the swe-agent-mini scaffold.",
    ),
    max_workers: int = typer.Option(
        1,
        "--max-workers",
        help="Maximum number of experiment tasks processed concurrently.",
    ),
    provider_workers: int = typer.Option(
        1,
        "--provider-workers",
        help="Maximum concurrent tasks per provider.",
    ),
) -> None:
    """
    Runs an LLM-powered agent on BenchMAC instances and generates a submission file.
    """
    settings.initialize_directories()
    console.print("[bold green]Starting BenchMAC Experiment Runner[/bold green]")

    selected_scaffolds = scaffolds or list(DEFAULT_SCAFFOLDS)

    # Validate model names early if swe-agent-mini is selected
    if any(s.strip().lower() == "swe-agent-mini" for s in selected_scaffolds):
        _validate_model_names_or_exit(model_names)

    swe_mini_config = yaml.safe_load(swe_mini_config_yaml.read_text())
    task_templates = swe_mini_config["task_template"]
    agent_settings = swe_mini_config["agent_settings"]

    agent_configs = build_agent_configs(
        scaffolds=selected_scaffolds,
        model_names=model_names,
        task_templates=task_templates,
        agent_settings=agent_settings,
        step_limit=step_limit,
        cost_limit_usd=cost_limit_usd,
    )

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

    if max_workers < 1:
        raise typer.BadParameter("--max-workers must be greater than 0")

    if provider_workers < 1:
        raise typer.BadParameter("--provider-workers must be greater than 0")

    console.print(
        "🔒 Per-provider worker limit: "
        f"[cyan]{provider_workers}[/cyan] concurrent task(s)"
        " (override with --provider-workers)"
    )

    if results_dir is None:
        results_dir = create_results_run_dir(settings.experiments_dir)
    else:
        if results_dir.exists() and not results_dir.is_dir():
            msg = f"Provided results path is not a directory: {results_dir}"
            raise typer.BadParameter(msg)
        results_dir.mkdir(parents=True, exist_ok=True)

    log_path = setup_experiment_logging(results_dir.name)
    console.print(f"🗒 Logging to [cyan]{log_path}[/cyan]")
    logger.info("Initialized experiment logging -> {log_path}", log_path=str(log_path))
    console.print(f"📁 Writing results to [cyan]{results_dir}[/cyan]")

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

            dispatch_tasks(
                tasks=tasks,
                instances=instances,
                results_dir=results_dir,
                progress=progress,
                progress_task_id=task_progress,
                max_workers=max_workers,
                provider_limit=provider_workers,
            )

    except KeyboardInterrupt:
        console.print(
            "\n[bold yellow]Experiment interrupted by user (KeyboardInterrupt).[/bold yellow]"
        )
        console.print(f"Partial results saved in [cyan]{results_dir}[/cyan]")
        return

    console.print("\n[bold green]✅ Experiment finished![/bold green]")
    console.print(f"Results saved in [cyan]{results_dir}[/cyan]")


if __name__ == "__main__":
    app()
