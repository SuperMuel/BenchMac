from __future__ import annotations

import os

from loguru import logger

from bench_mac.config import settings
from bench_mac.docker.manager import DockerManager
from bench_mac.models import BenchmarkInstance
from experiments.agents.base import (
    AgentRunArtifacts,
    AgentRunResult,
    BaseAgent,
)
from experiments.agents.mini_swe_agent.agent import generate_task_prompt
from experiments.agents.qwen_code.environment import QwenInstanceEnv
from experiments.models import AgentConfig

OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
DEFAULT_MAX_SESSION_TURNS = 64


class QwenCodeAgent(BaseAgent):
    """Runs the Qwen Code CLI against a BenchMAC instance."""

    def __init__(
        self,
        instance: BenchmarkInstance,
        agent_config: AgentConfig,
        docker_manager: DockerManager,
    ) -> None:
        self.instance = instance
        self.agent_config = agent_config
        self.docker_manager = docker_manager
        self.env = QwenInstanceEnv(instance, docker_manager)
        self.task_prompt = generate_task_prompt(instance)

        smoke = self.env.execute("ls -la")
        if "package.json" not in smoke.stdout:
            raise RuntimeError(
                "package.json not found in the container workdir; "
                "environment bootstrap failed."
            )

    def run(
        self,
        *,
        submission_id: str,
    ) -> AgentRunResult:
        logger.info(
            "Running Qwen Code agent for instance {instance}",
            instance=self.instance.instance_id,
        )
        openrouter_api_key = os.environ.get(OPENROUTER_API_KEY_ENV)
        if not openrouter_api_key:
            raise RuntimeError(
                f"Environment variable {OPENROUTER_API_KEY_ENV} is required "
                "for Qwen Code runs."
            )

        log_dir = settings.experiments_dir / "qwen_code"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{submission_id}.log"

        with self.env:
            result = self.env.run_qwen(
                prompt=self.task_prompt,
                model_name=self.agent_config.model_name,
                openrouter_api_key=openrouter_api_key,
                openrouter_model=self.agent_config.model_name,
                max_session_turns=DEFAULT_MAX_SESSION_TURNS,
            )

            combined_output = result.stdout or ""
            if result.stderr:
                combined_output = (
                    f"{combined_output}\n{result.stderr}"
                    if combined_output
                    else result.stderr
                )
            log_path.write_text(combined_output, encoding="utf-8")

            model_patch = self.env.diff_with_base_commit()
            artifacts = AgentRunArtifacts(
                execution_trace=self.env.execution_trace(),
                cost_usd=None,
                n_calls=None,
            )

            return AgentRunResult(model_patch=model_patch, artifacts=artifacts)

    def collect_artifacts(self) -> AgentRunArtifacts | None:
        try:
            return AgentRunArtifacts(
                execution_trace=self.env.execution_trace(),
                cost_usd=None,
                n_calls=None,
            )
        except Exception:
            return None
