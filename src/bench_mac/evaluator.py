import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from loguru import Logger

from bench_mac.docker.builder import prepare_environment
from bench_mac.docker.manager import DockerManager
from bench_mac.models import (
    BenchmarkInstance,
    EvaluationResult,
    MetricsReport,
    Submission,
)


def _create_evaluation_result(
    instance_id: str,
    metrics_data: dict[str, bool],
    logs: dict[str, str],
) -> EvaluationResult:
    """Helper to assemble the final EvaluationResult object."""
    metrics = MetricsReport(
        patch_application_success=metrics_data["patch_application_success"],
        # Add other metrics here as they are implemented
    )
    return EvaluationResult(instance_id=instance_id, metrics=metrics, logs=logs)


def evaluate_submission(
    instance: BenchmarkInstance,
    submission: Submission,
    docker_manager: DockerManager,
    *,
    logger: "Logger",
) -> EvaluationResult:
    """
    Orchestrates the end-to-end evaluation for a single submission.

    This function performs the following steps:
    1. Prepares the specific Docker environment for the instance.
    2. Runs a container from the environment.
    3. Copies and applies the submitted patch.
    4. (Future) Runs install, build, lint, and test commands.
    5. Collects logs and metrics.
    6. Cleans up all Docker resources.

    Parameters
    ----------
    instance
        The benchmark instance to evaluate against.
    submission
        The SUT's submitted solution.
    docker_manager
        An initialized DockerManager to interact with Docker.

    Returns
    -------
    An EvaluationResult object containing all metrics and logs.
    """
    logs: dict[str, str] = {}
    metrics_data: dict[str, bool] = {}
    container = None

    with tempfile.TemporaryDirectory() as tmpdir:
        submission_patch_file_name = f"{instance.instance_id}.patch"
        patch_file_path = Path(tmpdir) / submission_patch_file_name
        patch_file_path.write_text(submission.model_patch, encoding="utf-8")

        try:
            # 1. Prepare Environment
            logger.debug(f"Preparing environment for instance: {instance.instance_id}")
            instance_image_tag = prepare_environment(instance, docker_manager)

            # 2. Run Container
            logger.debug(f"Starting container from image: {instance_image_tag}")
            container = docker_manager.run_container(instance_image_tag)

            # 3. Apply Patch
            container_patch_path = f"/tmp/{submission_patch_file_name}"
            docker_manager.copy_to_container(
                container, src_path=patch_file_path, dest_path=container_patch_path
            )

            # Use `git apply --check` first for a non-destructive failure check
            # but for simplicity in the MVP, we'll apply directly.
            patch_command = f"git apply -p0 {container_patch_path}"
            logger.debug(f"Executing patch command: {patch_command}")
            exit_code, stdout, stderr = docker_manager.execute_in_container(
                container, patch_command
            )

            # TODO: keep the logs structured
            logs["patch_apply"] = f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"

            if exit_code != 0:
                logger.error("❌ Patch application failed. Halting evaluation.")
                metrics_data["patch_application_success"] = False
                return _create_evaluation_result(
                    instance.instance_id, metrics_data, logs
                )

            metrics_data["patch_application_success"] = True
            logger.success("✅ Patch applied successfully.")

            # --- (Future steps will be added here) ---
            # For now, we stop here and return a successful patch application result.

        except Exception as e:
            logger.exception(
                f"An unexpected error occurred during evaluation: {e.__class__.__name__}: {e}"  # noqa: E501
            )
            logs["harness_error"] = str(e)
            # Ensure metrics reflect a total failure in case of a crash
            metrics_data["patch_application_success"] = False

        finally:
            # 4. Cleanup
            if container:
                logger.debug(f"Cleaning up container {container.short_id}...")
                docker_manager.cleanup_container(container)

    return _create_evaluation_result(instance.instance_id, metrics_data, logs)
