"""Adapter around the core DockerExecutionEnvironment for experiment agents."""

from __future__ import annotations

import platform
from typing import Any

from loguru import logger

from bench_mac.core.config import settings
from bench_mac.core.models import BenchmarkInstance, ExecutionTrace
from bench_mac.docker.manager import DockerManager
from bench_mac.environments import DockerExecutionEnvironment


class MiniSweAgentEnvironmentAdapter:
    """Adapter that exposes DockerExecutionEnvironment
    via the mini-swe-agent interface."""

    def __init__(self, instance: BenchmarkInstance, docker_manager: DockerManager):
        self.instance = instance
        self._project_workdir = settings.project_workdir
        self._env = DockerExecutionEnvironment(
            instance,
            docker_manager,
            project_dir=self._project_workdir,
            logger=logger,
        )
        # Start immediately so agents can issue commands before entering a context.
        self._env.start()

    def execute(self, command: str, cwd: str = "") -> dict[str, str]:
        """Execute a command inside the container and return mini-swe-agent output."""
        workdir = cwd or self._project_workdir
        result = self._env.exec(command, workdir=workdir)

        stdout = result.stdout
        stderr = result.stderr
        exit_code = result.exit_code

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
        """Return the diff between the working tree and the baseline tag."""
        return self._env.diff_from_baseline().stdout

    def execution_trace(self) -> ExecutionTrace:
        """Expose the execution trace recorded by the DockerExecutionEnvironment."""
        return self._env.trace()

    def close(self) -> None:
        """Close the underlying environment."""
        self._env.close()

    def get_template_vars(self) -> dict[str, Any]:
        """Provides template variables for prompts, fulfilling the agent's API."""
        return {
            "instance_id": self.instance.instance_id,
            "repo": self.instance.repo,
            "base_commit": self.instance.base_commit,
            "source_version": self.instance.source_angular_version,
            "target_version": self.instance.target_angular_version,
            "cwd": self._project_workdir,
            # Provide the OS name for Jinja conditionals in templates
            "system": platform.system(),
        }

    @property
    def config(self) -> dict[str, Any]:
        """Configuration for the environment, required by the Environment protocol."""
        return {}

    def __enter__(self) -> MiniSweAgentEnvironmentAdapter:
        return self

    def __exit__(self, *_: Any, **__: Any) -> None:
        self.close()
