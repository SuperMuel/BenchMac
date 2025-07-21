"""
High-Level Integration Test for Silver Patch Validation.

This test suite serves as the primary validation gate for the benchmark's
"ground truth" data. Its purpose is to ensure that every known-good "silver"
patch can be successfully processed by the harness and passes all core
evaluation metrics.

It is NOT a test of the harness's ability to handle failures; rather, it is
a test of the data's correctness and the harness's "happy path" functionality.

This suite is parameterized to automatically discover and run a test for every
single `.patch` file found in the `data/silver_patches` directory. If this
test suite passes, we can have high confidence that our silver patches are
correct and that our harness integration tests can reliably use them as
a baseline for success.
"""

import json

import pytest
from loguru import logger

from bench_mac.config import settings
from bench_mac.docker.manager import DockerManager
from bench_mac.executor import execute_submission
from bench_mac.models import BenchmarkInstance, ExecutionJob, Submission

# --- Test Data Generation ---


def _load_all_instances() -> dict[str, BenchmarkInstance]:
    """
    Loads all benchmark instances from the instances.jsonl file into a
    dictionary, keyed by instance_id for efficient lookups.
    """
    instances_map: dict[str, BenchmarkInstance] = {}
    instances_file = settings.instances_file
    if not instances_file.exists():
        # Use pytest.fail for clear test setup errors
        pytest.fail(f"Instances file not found at: {instances_file}")

    with instances_file.open("r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            instance = BenchmarkInstance.model_validate(data)
            instances_map[instance.instance_id] = instance
    return instances_map


def discover_silver_tasks() -> list[ExecutionJob]:
    """
    Scans the silver_patches directory, matches each patch with its
    corresponding benchmark instance, and returns a list of executable
    ExecutionJob objects for pytest to parameterize.
    """
    tasks: list[ExecutionJob] = []
    instances_map = _load_all_instances()
    silver_patch_dir = settings.silver_patches_dir

    if not silver_patch_dir.exists() or not any(silver_patch_dir.iterdir()):
        # If the directory is missing or empty, it's a setup issue.
        # We return an empty list, and pytest will skip the tests.
        logger.warning(
            f"Silver patches directory not found or is empty: {silver_patch_dir}.\n"
            "Please generate them with `uv run scripts/generate_silvers.py`"
        )
        return []

    for patch_file in sorted(silver_patch_dir.glob("*.patch")):
        instance_id = patch_file.stem
        if instance_id in instances_map:
            submission = Submission(
                instance_id=instance_id, model_patch=patch_file.read_text()
            )
            tasks.append(
                ExecutionJob(instance=instances_map[instance_id], submission=submission)
            )
        else:
            logger.warning(
                f"Skipping silver patch '{patch_file.name}': "
                "No corresponding instance found in instances.jsonl."
            )
    return tasks


# --- The Test Suite ---


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize(
    "task",
    discover_silver_tasks(),
    # This lambda function provides clean, readable test names in the output
    ids=lambda t: t.instance.instance_id if isinstance(t, ExecutionJob) else "",
)
def test_silver_patch_passes_full_evaluation(
    task: ExecutionJob, docker_manager: DockerManager
) -> None:
    """
    Verifies that a specific silver patch successfully completes the entire
    evaluation pipeline, from patch application through to passing tests.

    This test is parameterized to run for every silver patch found, effectively
    acting as an automated validation suite for the ground truth data.
    """
    # --- ACT ---
    # Execute the full evaluation pipeline for the given silver submission.
    trace = execute_submission(
        instance=task.instance,
        submission=task.submission,
        docker_manager=docker_manager,
        logger=logger,  # Use the global logger configured by pytest
    )

    # --- ASSERT ---
    # Define the sequence of commands that MUST succeed for a silver patch.
    # We dynamically pull the commands from the instance definition to ensure
    # this test respects any custom commands.
    required_successful_commands = [
        "git apply -p0",  # The patch must apply.
        task.instance.commands.install,
        task.instance.commands.build,
        task.instance.commands.lint,
        task.instance.commands.test,
    ]

    # Verify each required command was executed and was successful.
    for command_substring in required_successful_commands:
        # Find the corresponding step in the execution trace.
        step_found = next(
            (s for s in trace.steps if command_substring in s.command), None
        )

        # 1. Assert that the command was actually run.
        # If it's None, the evaluation halted prematurely.
        assert step_found is not None, (
            f"Validation failed: "
            f"Command containing '{command_substring}' was not executed."
        )

        # 2. Assert that the command succeeded.
        # Provide a rich failure message for easy debugging.
        assert step_found.success, f"""
Validation failed: Command '{step_found.command}' failed.
  - Instance ID: {task.instance.instance_id}
  - Exit Code: {step_found.exit_code}
  - Stdout:
{step_found.stdout[-500:]}
  - Stderr:
{step_found.stderr[-1000:]}
"""
