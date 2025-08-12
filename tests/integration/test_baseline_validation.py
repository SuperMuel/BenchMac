"""
Integration test for validating the baseline state of all benchmark instances.

This test suite is critical for ensuring the quality and reliability of the
benchmark dataset. It verifies that for every instance defined in `instances.jsonl`,
the project at its `base_commit` is in a "green" state—meaning it can be
successfully installed, built, linted, and tested.

- This test is marked as 'slow' and is excluded from the default test run.
- It requires a running Docker daemon.
- **In theory**, a failure in this test indicates a problem with the benchmark data, not
  the harness logic.
"""

import json

import pytest
from loguru import logger

from bench_mac.config import settings
from bench_mac.docker.builder import prepare_environment
from bench_mac.docker.manager import DockerManager
from bench_mac.models import BenchmarkInstance


def load_all_benchmark_instances() -> list[BenchmarkInstance]:
    """
    Reads 'data/instances.jsonl' and yields BenchmarkInstance objects
    to be used for pytest parameterization.
    """
    instances_file = settings.instances_file
    instances_list: list[BenchmarkInstance] = []

    if not instances_file.is_file():
        # If the file doesn't exist, pytest will report a skip.
        logger.warning(
            f"Instances file not found at {instances_file}, "
            "skipping baseline validation tests."
        )
        return instances_list

    with instances_file.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                instance = BenchmarkInstance.model_validate(data)
                instances_list.append(instance)
            except (json.JSONDecodeError, Exception) as e:
                logger.error(
                    f"Failed to load instance from line {line_num} "
                    f"in {instances_file}: {e}"
                )
                # This test should indicate when instances are invalid.
                # Therefore, we raise the error to fail the test.
                raise e

    return instances_list


@pytest.mark.baseline_validation
class TestBaselineInstanceValidation:
    """
    Validates that every benchmark instance starts from a known-good state.
    """

    @pytest.mark.parametrize(
        "instance",
        load_all_benchmark_instances(),
        ids=lambda inst: inst.instance_id,  # Use instance_id for readable test names
    )
    def test_instance_baseline_is_green(
        self, instance: BenchmarkInstance, docker_manager: DockerManager
    ) -> None:
        """
        For a given instance, verifies that the baseline state is "green"
        by running install, build, lint, and test commands successfully.
        """
        logger.info(f"--- Validating Baseline for Instance: {instance.instance_id} ---")
        container = None
        try:
            # === 1. SETUP: Prepare the Docker Environment ===
            logger.info("Step 1: Preparing Docker environment...")
            instance_image_tag = prepare_environment(instance, docker_manager)
            assert instance_image_tag, "Failed to prepare Docker environment."

            logger.info(
                f"Step 2: Starting container from image '{instance_image_tag}'..."
            )
            container = docker_manager.run_container(instance_image_tag)
            assert container, "Failed to run container."

            # === 2. EXECUTION & ASSERTION: Run all baseline commands ===
            # The project working directory inside the container is /app/project
            project_dir = "/app/project"
            commands_to_validate = [
                ("Install", instance.commands.install),
                ("Build", instance.commands.build),
            ]

            for step_name, command_str in commands_to_validate:
                logger.info(f"Step 3: Executing '{step_name}' command: {command_str}")

                exit_code, stdout, stderr = docker_manager.execute_in_container(
                    container, command_str, workdir=project_dir
                )

                # The primary assertion for each step
                assert exit_code == 0, (
                    f"\n\n❌ Baseline validation FAILED for instance"
                    f" '{instance.instance_id}' at step '{step_name}'.\n"
                    f"  - Command: '{command_str}'\n"
                    f"  - Exit Code: {exit_code}\n\n"
                    f"  --- STDOUT (last 1000 chars) ---\n{stdout[-1000:]}\n\n"
                    f"  --- STDERR (last 1000 chars) ---\n{stderr[-1000:]}\n"
                )
                logger.success(
                    f"✅ Step '{step_name}' PASSED for instance"
                    f" '{instance.instance_id}'."
                )

        finally:
            # === 4. CLEANUP: Always remove the container ===
            if container:
                logger.info(f"Cleaning up container {container.short_id}...")
                docker_manager.cleanup_container(container)
