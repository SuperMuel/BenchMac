# ruff: noqa: E501
# this is temporary a Typer app, but we'll fusion this with the main CLI later

import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from textwrap import dedent
from typing import Optional

import litellm
import typer
import yaml
from dotenv import load_dotenv
from minisweagent.agents.default import DefaultAgent
from minisweagent.models.litellm_model import LitellmModel
from minisweagent.run.utils.save import (
    save_traj,  # type: ignore[reportUnknownReturnType]
)
from pydantic import ValidationError
from rich.console import Console
from rich.progress import Progress

from bench_mac.docker.manager import DockerManager
from bench_mac.models import (
    BenchmarkInstance,
    Submission,
    SubmissionMetadata,
)
from experiments.environment import InstanceEnv
from experiments.models import PatchGenerationTask
from src.bench_mac.config import settings
from src.bench_mac.utils import load_instances

# --- Configuration ---
app = typer.Typer(add_completion=False)
console = Console()

# Default model names
DEFAULT_MODEL_NAMES = ["mistral/devstral-medium-2507"]


load_dotenv()
# assert os.getenv("LANGSMITH_API_KEY"), "LANGSMITH_API_KEY is not set"
# litellm.callbacks = ["langsmith"]
# litellm.langsmith_batch_size = 1

litellm.callbacks = ["logfire"]


assert os.getenv("MISTRAL_API_KEY"), "MISTRAL_API_KEY is not set"
settings.initialize_directories()


def get_submissions_file_path(
    experiments_dir: Path, now: datetime | None = None
) -> Path:
    if now is None:
        now = datetime.now(UTC)
    assert experiments_dir.is_dir() and experiments_dir.exists()
    submissions_dir = experiments_dir / "submissions"
    submissions_dir.mkdir(parents=True, exist_ok=True)

    return submissions_dir / f"submissions_{now.strftime('%Y-%m-%d_%H-%M-%S')}.jsonl"


def generate_task_prompt(instance: BenchmarkInstance) -> str:
    """Generates a detailed, structured prompt for the agent."""
    return dedent(
        f"""\
        ## Goal
        Migrate the application from Angular version {instance.source_angular_version} to {instance.target_angular_version}.

        ## Context
        - The codebase is available in `/app/project`
        - The project is already cloned in the current directory. You do not need to clone it.
        - NPM is already installed.
        - Commands hints:
            - Install dependencies: {instance.commands.install}
            - Build the project: {instance.commands.build}

        ## Rules
        Do not change any application logic or functionality. Your focus is only on making the code compatible with the target Angular version.

        ## Recommended Workflow

        This workflows should be done step-by-step so that you can iterate on your changes and any possible problems.

        1. Analyze the codebase by finding and reading relevant files
        2. Edit the source code or run any command to migrate the codebase to the target Angular version
        3. Test the application by running the build command
        4. Submit your changes and finish your work by issuing the following command: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.
        Do not combine it with any other command. <important>After this command, you cannot continue working on this task.</important>
        """
    )


AGENT_CONFIG = yaml.safe_load(Path("experiments/agent_config.yaml").read_text())


# --- Patch Generation Task Models ---


def collect_tasks(
    instances: dict[str, BenchmarkInstance], model_names: list[str]
) -> list[PatchGenerationTask]:
    """
    Collect all patch generation tasks by combining instances with agent configurations.

    Args:
        instances: Dictionary of benchmark instances keyed by instance_id
        model_names: List of model names to use for patch generation

    Returns:
        List of PatchGenerationTask objects, one for each instance-model_name combination
    """
    return [
        PatchGenerationTask(instance_id=instance_id, model_name=model_name)
        for instance_id in instances
        for model_name in model_names
    ]


def collect_old_submissions(experiments_dir: Path) -> list[Submission]:
    """
    Collect all old submissions from the experiments directory.

    Args:
        experiments_dir: Path to the experiments directory containing submissions

    Returns:
        List of all Submission objects found in submission files
    """
    submissions: list[Submission] = []
    submissions_dir = experiments_dir / "submissions"

    if not submissions_dir.exists():
        return submissions

    # Find all JSONL files in the submissions directory
    submission_files = list(submissions_dir.glob("*.jsonl"))

    for submission_file in submission_files:
        try:
            with submission_file.open("r") as f:
                for line in f:
                    line = line.strip()
                    if line:  # Skip empty lines
                        submission_data = json.loads(line)
                        submission = Submission.model_validate(submission_data)
                        submissions.append(submission)
        except (json.JSONDecodeError, ValidationError) as e:
            console.print(
                f"[yellow]Warning: Could not parse {submission_file}: {e}[/yellow]"
            )
            continue

    return submissions


def filter_completed_tasks(
    tasks: list[PatchGenerationTask], old_submissions: list[Submission]
) -> list[PatchGenerationTask]:
    """
    Filter out tasks that have already been completed based on old submissions.

    A task is considered completed if there's already a submission with the same
    instance_id and model_name combination.

    Args:
        tasks: List of tasks to potentially run
        old_submissions: List of existing submissions

    Returns:
        List of tasks that haven't been completed yet
    """
    # Create a set of completed (instance_id, model_name) combinations
    completed_combinations = {
        (submission.instance_id, submission.metadata.model_name)
        for submission in old_submissions
        if submission.metadata.model_name is not None
    }

    # Filter tasks that haven't been completed
    filtered_tasks: list[PatchGenerationTask] = []
    skipped_count = 0

    for task in tasks:
        task_combination = (task.instance_id, task.model_name)
        if task_combination in completed_combinations:
            console.print(
                f"[blue]Skipping already completed task: {task.instance_id} ({task.model_name})[/blue]"
            )
            skipped_count += 1
        else:
            filtered_tasks.append(task)

    if skipped_count > 0:
        console.print(f"[blue]Skipped {skipped_count} already completed tasks[/blue]")

    return filtered_tasks


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
    submissions_file: Optional[Path] = typer.Option(  # noqa: B008, UP045
        None,
        "--submissions-file",
        "-s",
        help="Path to the submissions file.",
    ),
    instance_ids: Optional[list[str]] = typer.Option(  # noqa: B008, UP045
        None,
        "--instance-id",
        help="Filter by specific instance ID(s). Can be used multiple times.",
    ),
) -> None:
    """
    Runs an LLM-powered agent on BenchMAC instances and generates a submission file.
    """
    console.print("[bold green]Starting BenchMAC Experiment Runner[/bold green]")
    model_names_str = ", ".join(f"[cyan]{m}[/cyan]" for m in model_names)
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

    if submissions_file is None:
        submissions_file = get_submissions_file_path(settings.experiments_dir)

    # Collect old submissions to avoid re-running completed tasks
    old_submissions = collect_old_submissions(settings.experiments_dir)
    console.print(f"Found {len(old_submissions)} existing submissions")

    # Collect all tasks
    tasks = collect_tasks(instances, model_names)
    console.print(f"Generated {len(tasks)} potential tasks")

    # Filter out already completed tasks
    tasks = filter_completed_tasks(tasks, old_submissions)

    if not tasks:
        console.print("No tasks to process. Exiting...")
        return

    docker_manager = DockerManager()

    with Progress(console=console) as progress:
        task_progress = progress.add_task("[cyan]Processing Tasks...", total=len(tasks))

        for task in tasks:
            instance = instances[task.instance_id]
            submission_id = str(uuid.uuid4())
            progress.update(
                task_progress,
                description=f"[cyan]Processing: {task.instance_id} ({task.model_name})[/cyan]",
            )

            # Create environment
            console.print("Creating environment")
            with InstanceEnv(instance, docker_manager) as env:
                console.print("Environment created")
                test_env = env.execute("ls -la")
                console.print(f"Test environment: {test_env}")
                assert "package.json" in str(test_env.get("output", "")), (
                    "package.json not found in the environment. The repository is not cloned correctly, or the current directory is not the project directory."
                )

                console.print("Creating model")
                model = LitellmModel(model_name=task.model_name)
                console.print("Creating agent")
                agent = DefaultAgent(
                    model,
                    env,
                    **AGENT_CONFIG["agent"],
                )
                task_prompt = generate_task_prompt(instance)

                try:
                    # Run the agent
                    console.print("Running agent")
                    exit_status, result = agent.run(task_prompt)  # type: ignore[reportUnknownReturnType]
                    print(f"Agent output: {result}")  # noqa: T201

                    save_traj(
                        agent,
                        settings.experiments_dir
                        / "swe_agent_mini"
                        / Path(f"{submission_id}.traj.json"),
                        exit_status=exit_status,
                        result=result,
                        extra_info={
                            "instance_id": task.instance_id,
                            "submission_id": submission_id,
                            "model_name": task.model_name,
                        },
                    )
                except Exception as e:
                    console.print(
                        f"[bold red]Error processing {task.instance_id} ({task.model_name}): {e}[/bold red]"
                    )
                    progress.console.print(
                        f"[yellow]Skipping {task.instance_id} ({task.model_name}) due to error.[/yellow]"
                    )
                    progress.advance(task_progress)
                    continue

                # Generate patch
                model_patch = env.diff_with_base_commit()
                # Write submission
                submission = Submission(
                    submission_id=submission_id,
                    instance_id=task.instance_id,
                    model_patch=model_patch,
                    metadata=SubmissionMetadata(model_name=task.model_name),
                )
                with submissions_file.open("a") as f:
                    f.write(submission.model_dump_json() + "\n")

            progress.advance(task_progress)

    console.print("\n[bold green]✅ Experiment finished![/bold green]")
    console.print(f"Submissions saved to [cyan]{submissions_file}[/cyan]")
    console.print(
        "You can now run the evaluation with:\n"
        f"[bold]uv run benchmac eval {submissions_file}[/bold]"
    )


if __name__ == "__main__":
    app()
