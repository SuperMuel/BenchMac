# ruff: noqa: E501
# this is temporary a Typer app, but we'll fusion this with the main CLI later

import json
import os
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
from pydantic import BaseModel, Field
from rich.console import Console
from rich.progress import Progress

from bench_mac.docker.manager import DockerManager
from bench_mac.models import (
    BenchmarkInstance,
    Submission,
    SubmissionMetadata,
)
from experiments.environment import InstanceEnv
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


class PatchGenerationTask(BaseModel):
    """
    A task for generating patches using an agent configuration on a specific instance.

    This represents a single unit of work in the patch generation process.
    """

    instance_id: str = Field(
        ...,
        description="Unique identifier for the benchmark instance to process.",
    )
    model_name: str = Field(
        ...,
        description="The name of the model to use for patch generation "
        "(e.g., 'mistral/devstral-medium-2507').",
    )


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

    else:
        # Check if the file contains some submissions, so we can skip the instances that are already processed
        existing_submissions = {
            json.loads(line)["instance_id"]
            for line in submissions_file.read_text().splitlines()
        }
        console.print(
            f"Skipping {len(existing_submissions)} instances that are already processed"
        )
        instances = {
            instance_id: instance
            for instance_id, instance in instances.items()
            if instance_id not in existing_submissions
        }

    # Collect all tasks
    tasks = collect_tasks(instances, model_names)

    docker_manager = DockerManager()

    with Progress(console=console) as progress:
        task_progress = progress.add_task("[cyan]Processing Tasks...", total=len(tasks))

        for task in tasks:
            instance = instances[task.instance_id]
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
                    output: tuple[str, str] = agent.run(task_prompt)  # type: ignore[reportUnknownReturnType]
                    print(f"Agent output: {output}")  # noqa: T201

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
