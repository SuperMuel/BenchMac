from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from bench_mac.docker.manager import DockerManager
from bench_mac.models import BenchmarkInstance
from experiments.agents.base import AgentRunArtifacts, AgentRunResult, BaseAgent
from experiments.agents.mini_swe_agent.environment import InstanceEnv
from experiments.models import AngularSchematicsConfig


@dataclass(frozen=True)
class _PlannedCommand:
    description: str
    command: str


class AngularSchematicsAgent(BaseAgent):
    """Deterministic baseline that applies Angular CLI schematics."""

    def __init__(
        self,
        instance: BenchmarkInstance,
        agent_config: AngularSchematicsConfig,
        docker_manager: DockerManager,
    ) -> None:
        self.instance = instance
        self.agent_config = agent_config
        self.env = InstanceEnv(instance, docker_manager)
        self._plan = self._build_plan()

    def _build_plan(self) -> list[_PlannedCommand]:
        target_version = self.instance.target_angular_version
        update_command = self.agent_config.update_command_template.format(
            target_version=target_version
        )

        return [
            _PlannedCommand(
                description="Install project dependencies",
                command=self.instance.commands.install,
            ),
            _PlannedCommand(
                description="Run Angular update schematics",
                command=update_command,
            ),
            _PlannedCommand(
                description="Build project for verification",
                command=self.instance.commands.build,
            ),
        ]

    def _run_command(self, planned: _PlannedCommand) -> None:
        logger.info("[{}] {}", self.instance.instance_id, planned.description)
        self.env.execute(planned.command)

    def run(
        self,
        *,
        submission_id: str,
    ) -> AgentRunResult:
        logger.info(
            "Running Angular schematics baseline for instance: {}",
            self.instance.instance_id,
        )

        with self.env:
            for planned in self._plan:
                self._run_command(planned)

            model_patch = self.env.diff_with_base_commit()
            artifacts = AgentRunArtifacts(execution_trace=self.env.execution_trace())

        return AgentRunResult(model_patch=model_patch, artifacts=artifacts)

    def collect_artifacts(self) -> AgentRunArtifacts | None:
        try:
            return AgentRunArtifacts(execution_trace=self.env.execution_trace())
        except Exception:  # pragma: no cover - best effort fallback
            return None
