"""Environment helpers for running Qwen Code inside BenchMAC containers."""

from __future__ import annotations

import json
import shlex
from typing import Any

from docker.models.containers import Container
from loguru import logger

from bench_mac.docker.manager import DockerManager
from bench_mac.models import (
    BenchmarkInstance,
    CommandResult,
    ExecutionTrace,
    utc_now,
)

from .docker import build_qwen_overlay


class QwenInstanceEnv:
    """BenchMAC-aware environment that bundles Qwen Code tooling."""

    _project_workdir = "/app/project"

    def __init__(self, instance: BenchmarkInstance, docker_manager: DockerManager):
        self.instance = instance
        self.docker_manager = docker_manager
        self.container: Container | None = None
        self._steps: list[CommandResult] = []

        self._start_environment()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def _start_environment(self) -> None:
        logger.info(
            "[{instance}] Preparing Qwen Code environment...",
            instance=self.instance.instance_id,
        )
        try:
            image_tag = build_qwen_overlay(self.instance, self.docker_manager)
            self.container = self.docker_manager.run_container(image_tag)
            logger.success(
                "[{instance}] Environment ready (container={container}).",
                instance=self.instance.instance_id,
                container=self.container.short_id if self.container else "n/a",
            )
        except Exception:
            logger.exception(
                "[{instance}] Failed to start Qwen Code environment.",
                instance=self.instance.instance_id,
            )
            self.close()
            raise

    # ------------------------------------------------------------------
    # Core execution helpers
    # ------------------------------------------------------------------
    def execute(
        self, command: str, cwd: str = "", env: dict[str, str] | None = None
    ) -> CommandResult:
        if not self.container:
            raise RuntimeError("Container is not running.")

        workdir = cwd or self._project_workdir
        start_time = utc_now()
        exit_code, stdout, stderr = self.docker_manager.execute_in_container(
            self.container,
            command,
            workdir=workdir,
            environment=env,
        )
        end_time = utc_now()
        result = CommandResult(
            command=command,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            start_time=start_time,
            end_time=end_time,
        )
        self._steps.append(result)
        return result

    def diff_with_base_commit(self) -> str:
        if not self.container:
            raise RuntimeError("Container is not running.")

        _, stdout, _ = self.docker_manager.execute_in_container(
            self.container,
            "git diff --no-prefix baseline",
            workdir=self._project_workdir,
        )
        return stdout

    def execution_trace(self) -> ExecutionTrace:
        return ExecutionTrace(steps=list(self._steps))

    def close(self) -> None:
        if self.container:
            logger.info(
                "[{instance}] Cleaning up container {container}...",
                instance=self.instance.instance_id,
                container=self.container.short_id,
            )
            self.docker_manager.cleanup_container(self.container)
            self.container = None
            logger.success(
                "[{instance}] Environment cleaned up.",
                instance=self.instance.instance_id,
            )

    # ------------------------------------------------------------------
    # Qwen-specific helpers
    # ------------------------------------------------------------------
    def run_qwen(
        self,
        *,
        prompt: str,
        model_name: str,
        openrouter_api_key: str,
        openrouter_model: str,
        max_session_turns: int,
    ) -> CommandResult:
        """Run the Qwen CLI in non-interactive YOLO mode and capture the output."""
        settings = {
            "security": {"auth": {"selectedType": "use_openai", "useExternal": False}},
            "model": {
                "name": model_name,
                "maxSessionTurns": max_session_turns,
            },
            "telemetry": {"enabled": False},
            "tools": {"sandbox": False},
        }

        self._write_settings(settings)
        prompt_path = "/tmp/benchmac_prompt.txt"
        self._write_file(prompt_path, prompt)

        env_vars = {
            "HOME": "/home/node",
            "NO_COLOR": "1",
            "OPENAI_API_KEY": openrouter_api_key,
            "OPENAI_BASE_URL": "https://openrouter.ai/api/v1",
            "OPENAI_MODEL": openrouter_model,
            "OPENROUTER_API_KEY": openrouter_api_key,
            "OPENROUTER_MODEL": openrouter_model,
        }

        command = (
            f'qwen --prompt "$(cat {prompt_path})" '
            f"--approval-mode yolo --model {shlex.quote(model_name)}"
        )
        return self.execute(command, cwd=self._project_workdir, env=env_vars)

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    def _write_settings(self, settings: dict[str, Any]) -> None:
        settings_json = json.dumps(settings, indent=2)
        self.execute("mkdir -p /home/node/.qwen", cwd="/")
        self._write_file("/home/node/.qwen/settings.json", settings_json)

    def _write_file(self, path: str, content: str) -> None:
        sentinel = "__BENCHMAC_EOF__"
        command = f"cat <<'{sentinel}' > {path}\n{content}\n{sentinel}"
        self.execute(command, cwd="/")

    def get_template_vars(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance.instance_id,
            "repo": self.instance.repo,
            "base_commit": self.instance.base_commit,
            "source_version": self.instance.source_angular_version,
            "target_version": self.instance.target_angular_version,
            "cwd": self._project_workdir,
        }

    def __enter__(self) -> QwenInstanceEnv:
        return self

    def __exit__(self, *_: Any, **__: Any) -> None:
        self.close()
