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

import pytest
from loguru import logger

from bench_mac.config import settings
from bench_mac.docker.builder import prepare_environment
from bench_mac.docker.manager import DockerManager
from bench_mac.models import BenchmarkInstance
from bench_mac.utils import load_instances


@pytest.mark.instance_validation
class TestBaselineInstanceValidation:
    """
    Validates that every benchmark instance starts from a known-good state.
    """

    @pytest.mark.parametrize(
        "instance",
        load_instances(settings.instances_file, strict=True).values(),
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
            # TODO: this test shouldn't use the lower-level prepare_environment function
            # but rather use the higher-level InstanceEnv class
            instance_image_tag = prepare_environment(instance, docker_manager)
            assert instance_image_tag, "Failed to prepare Docker environment."

            logger.info(
                f"Step 2: Starting container from image '{instance_image_tag}'..."
            )
            container = docker_manager.run_container(instance_image_tag)
            assert container, "Failed to run container."

            project_dir = "/app/project"

            # === 2a. VALIDATE GIT STATE ===
            logger.info("Step 2a: Validating Git history sanitization...")

            # Ensure there's no leaked history.
            # `git rev-list --count --all` counts all commits in the repository.
            # After `curl | tar | git init`, this count should be 0. If an initial
            # commit is added in the Dockerfile, it should be 1. We'll check for <= 1.
            git_count_cmd = "git rev-list --count --all"
            exit_code, stdout, stderr = docker_manager.execute_in_container(
                container, git_count_cmd, workdir=project_dir
            )

            assert exit_code == 0, f"Git count command failed: {stderr}"
            commit_count = int(stdout.strip())

            assert commit_count == 1, (
                f"\n\n❌ History validation FAILED for instance "
                f"'{instance.instance_id}'.\n"
                f"   - Expected exactly 1 baseline commit, but found {commit_count}.\n"
                f"   - If {commit_count} > 1: FULL git history was included (INVALID).\n"  # noqa: E501
                f"   - If {commit_count} == 0: No baseline commit created (INVALID).\n"
                f"   - The Dockerfile should create exactly 1 baseline commit after "
                f"extracting source code."
            )
            logger.success("✅ Git history is properly sanitized.")

            # === 2b. VALIDATE BASELINE COMMIT AND TAG ===
            logger.info("Step 2b: Validating baseline commit and tag...")

            # 1. Verify the baseline tag exists
            tag_check_cmd = "git tag -l baseline"
            exit_code, stdout, stderr = docker_manager.execute_in_container(
                container, tag_check_cmd, workdir=project_dir
            )
            assert exit_code == 0, f"Git tag check failed: {stderr}"
            assert stdout.strip() == "baseline", (
                f"Missing 'baseline' tag for instance '{instance.instance_id}'. "
                f"Expected 'baseline', got: '{stdout.strip()}'. "
                f"The Dockerfile should create a 'baseline' tag after the initial commit."  # noqa: E501
            )

            # 2. Verify the baseline tag points to the current HEAD
            head_hash_cmd = "git rev-parse HEAD"
            tag_hash_cmd = "git rev-parse baseline"
            exit_code, head_hash, _ = docker_manager.execute_in_container(
                container, head_hash_cmd, workdir=project_dir
            )
            assert exit_code == 0, "Failed to get HEAD hash"

            exit_code, tag_hash, _ = docker_manager.execute_in_container(
                container, tag_hash_cmd, workdir=project_dir
            )
            assert exit_code == 0, "Failed to get baseline tag hash"

            assert head_hash.strip() == tag_hash.strip(), (
                f"Baseline tag doesn't point to HEAD for "
                f"instance '{instance.instance_id}'. "
                f"HEAD: {head_hash.strip()}, baseline: {tag_hash.strip()}"
            )

            # === 3. EXECUTION & ASSERTION: Run all baseline commands ===
            # The project working directory inside the container is /app/project
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
                logger.success(f"✅ Step '{step_name}' PASSED")
            logger.success(
                f"✅ Baseline validation PASSED for instance: {instance.instance_id}"
            )
        finally:
            # === 4. CLEANUP: Always remove the container ===
            if container:
                logger.info(f"Cleaning up container {container.short_id}...")
                docker_manager.cleanup_container(container)
