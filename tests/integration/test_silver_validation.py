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

import pytest
from loguru import logger

from bench_mac.config import settings
from bench_mac.docker.manager import DockerManager
from bench_mac.executor import execute_submission
from bench_mac.metrics import calculate_metrics
from bench_mac.models import EvaluationTask, Submission
from bench_mac.utils import load_instances

# --- Test Data Generation ---


def discover_silver_tasks() -> list[EvaluationTask]:
    """
    Scans the silver_patches directory, matches each patch with its
    corresponding benchmark instance, and returns a list of executable
    EvaluationTask objects for pytest to parameterize.
    """
    tasks: list[EvaluationTask] = []
    instances_map = load_instances(settings.instances_file, strict=True)
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
                EvaluationTask(
                    instance=instances_map[instance_id], submission=submission
                )
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
    ids=lambda t: t.instance.instance_id if isinstance(t, EvaluationTask) else "",
)
def test_silver_patch_passes_full_evaluation(
    task: EvaluationTask, docker_manager: DockerManager
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

    # 1. The patch MUST apply cleanly. This is non-negotiable.
    patch_apply_step = next(
        (s for s in trace.steps if "git apply -p0" in s.command), None
    )
    assert patch_apply_step is not None, (
        "Validation failed: Patch apply command was not executed."
    )
    assert patch_apply_step.success, f"""
Validation failed: Silver patch command '{patch_apply_step.command}' failed.
  - Instance ID: {task.instance.instance_id}
  - Exit Code: {patch_apply_step.exit_code}
  - Stderr:
{patch_apply_step.stderr[-1000:]}
"""

    # 2. An install command MUST succeed. This can be the original or the fallback.
    install_steps = [s for s in trace.steps if "npm ci" in s.command]
    assert len(install_steps) > 0, "Validation failed: No install command was executed."

    assert any(step.success for step in install_steps), f"""
Validation failed: All install attempts failed for the silver patch.
  - Instance ID: {task.instance.instance_id}
  - Last attempt command: '{install_steps[-1].command}'
  - Last attempt exit code: {install_steps[-1].exit_code}
  - Last attempt stderr:
{install_steps[-1].stderr[-1000:]}
"""

    # 3. The build command MUST be reached and succeed. This is also non-negotiable.
    build_step = next(
        (s for s in trace.steps if task.instance.commands.build in s.command), None
    )
    assert build_step is not None, "Validation failed: Build command was not executed."
    assert build_step.success, f"""
Validation failed: Silver patch build command '{build_step.command}' failed.
  - Instance ID: {task.instance.instance_id}
  - Exit Code: {build_step.exit_code}
  - Stderr:
{build_step.stderr[-1000:]}
"""

    # 4. The target version MUST be achieved.
    # We run the same metrics calculation that the main runner would.
    metrics = calculate_metrics(trace, task.instance)
    assert metrics.target_version_achieved is True, f"""
Validation failed: Target version was not achieved for the silver patch.
  - Instance ID: {task.instance.instance_id}
  - Expected Major Version: {task.instance.target_angular_version}
  - Metric Result from Trace: {metrics.target_version_achieved}
"""
