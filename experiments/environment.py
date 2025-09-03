"""
Custom Environment for mini-swe-agent tailored to BenchMAC instances.
"""

from __future__ import annotations

from typing import Any

from docker.models.containers import Container
from loguru import logger

from src.bench_mac.docker.builder import prepare_environment
from src.bench_mac.docker.manager import DockerManager
from src.bench_mac.models import BenchmarkInstance


class InstanceEnv:
    """
    An agent environment that uses the BenchMAC Docker infrastructure.

    This class acts as a bridge between the generic agent framework
    (like mini-swe-agent) and the specific, reproducible environments
    defined by BenchMAC. It uses the project's own `DockerManager`
    to prepare and interact with the instance-specific container.

    It is designed to be used as a context manager to ensure proper cleanup.
    """

    def __init__(self, instance: BenchmarkInstance, docker_manager: DockerManager):
        """
        Initializes and starts the Docker environment for a specific instance.

        Args:
            instance: The BenchmarkInstance defining the task and environment.
            docker_manager: The initialized DockerManager to handle Docker operations.
        """
        self.instance = instance
        self.docker_manager = docker_manager
        self.container: Container | None = None
        self._project_workdir = (
            "/app/project"  # Consistent with executor.py #TODO: make it configurable
        )

        self._start_environment()

    def _start_environment(self) -> None:
        """Prepares the Docker image and starts the container."""

        logger.info(
            f"[{self.instance.instance_id}] "
            "Preparing environment and starting container..."
        )
        try:
            # 1. Prepare the content-hashed Docker image for the instance.
            # This reuses your benchmark's core logic to ensure reproducibility.
            image_tag = prepare_environment(self.instance, self.docker_manager)

            # 2. Run a container from the prepared image.
            self.container = self.docker_manager.run_container(image_tag)
            logger.success(
                f"[{self.instance.instance_id}] Environment ready. "
                f"Container ID: {self.container.short_id}"
            )
        except Exception:
            logger.exception(
                f"[{self.instance.instance_id}] Failed to start environment."
            )
            self.close()  # Attempt cleanup on failure
            raise

    def execute(self, command: str, cwd: str = "") -> dict[str, str]:
        """
        Executes a shell command inside the instance's container.

        This method fulfills the interface required by mini-swe-agent.

        Args:
            command: The command string to execute.
            cwd: The working directory to run the command in. If empty, uses
                 the default project directory (`/app/project`).

        Returns:
            A dictionary containing the combined output and the return code.
        """
        if not self.container:
            raise RuntimeError(
                "Container is not running. The environment may have failed to start "
                "or has been closed."
            )

        workdir = cwd or self._project_workdir
        exit_code, stdout, stderr = self.docker_manager.execute_in_container(
            self.container, command, workdir=workdir
        )
        if "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in stdout:
            return {
                "output": f"COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n{stdout}\n{stderr}",
                "returncode": str(exit_code),
            }

        return {
            "output": f"<stdout>\n{stdout}\n</stdout>\n<stderr>\n{stderr}\n</stderr>\n",
            "returncode": str(exit_code),
        }

    def diff_with_base_commit(self) -> str:
        """
        Diff the current state of the container with the base commit.
        """
        if not self.container:
            raise RuntimeError(
                "Container is not running. The environment may have failed to start "
                "or has been closed."
            )

        diff_command = "git diff --no-prefix baseline"
        _, stdout, _ = self.docker_manager.execute_in_container(
            self.container, diff_command, workdir=self._project_workdir
        )
        return stdout

    def close(self) -> None:
        """Stops and removes the container, cleaning up all resources."""
        if self.container:
            logger.info(
                f"[{self.instance.instance_id}] "
                f"Cleaning up container {self.container.short_id}..."
            )
            self.docker_manager.cleanup_container(self.container)
            self.container = None
            logger.success(f"[{self.instance.instance_id}] Environment cleaned up.")

    def get_template_vars(self) -> dict[str, Any]:
        """Provides template variables for prompts, fulfilling the agent's API."""
        return {
            "instance_id": self.instance.instance_id,
            "repo": self.instance.repo,
            "base_commit": self.instance.base_commit,
            "source_version": self.instance.source_angular_version,
            "target_version": self.instance.target_angular_version,
            "cwd": self._project_workdir,
        }

    @property
    def config(self) -> dict[str, Any]:
        """Configuration for the environment, required by the Environment protocol."""
        return {}

    def __enter__(self) -> InstanceEnv:
        """Enables use as a context manager."""
        return self

    def __exit__(self, *_: Any, **__: Any) -> None:
        """Ensures cleanup is called when exiting the context."""
        self.close()
