from dataclasses import dataclass

from loguru import logger

from bench_mac.core.models import BenchmarkInstance
from bench_mac.docker.manager import DockerManager
from bench_mac.environments import DockerExecutionEnvironment
from experiments.agents.base import AgentRunResult, BaseAgent
from experiments.models import AngularSchematicsConfig, ExperimentArtifacts


@dataclass(frozen=True)
class _PlannedCommand:
    description: str
    command: str
    error_msg_if_fails: str | None = None


_INSTALL_FAILED_MSG = """Install stage failed during Angular Schematics run.
This is not expected to happen, as per the 'green baseline' approach.
To reproduce the issue, you may run the baseline_validation integration \
test for that instance :
`uv run pytest -m instance_validation -k "{instance_id}".
"""


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
        self.env = DockerExecutionEnvironment(instance, docker_manager)
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
                error_msg_if_fails=_INSTALL_FAILED_MSG.format(
                    instance_id=self.instance.instance_id
                ),
            ),
            _PlannedCommand(
                description="Run Angular update schematics",
                command=update_command,
            ),
        ]

    def _run_command(self, planned: _PlannedCommand) -> bool:
        logger.info("[{}] {}", self.instance.instance_id, planned.description)
        result = self.env.exec(planned.command)
        if result.success:
            return True

        stdout_tail = result.stdout[-400:]
        stderr_tail = result.stderr[-400:]
        logger.error(
            (
                "[{}] Command failed. command={}, exit_code={}, "
                "stdout_tail={}, stderr_tail={}"
            ),
            self.instance.instance_id,
            planned.command,
            result.exit_code,
            stdout_tail,
            stderr_tail,
        )
        return False

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
                success = self._run_command(planned)
                # TODO: Raise if the command fails for harness reasons -
                # e.g network errors
                if not success and planned.error_msg_if_fails:
                    raise RuntimeError(planned.error_msg_if_fails)

                if not success:
                    # Don't raise, as we want to capture in the
                    # results when this method fails
                    break

            model_patch = self.env.diff_from_baseline().stdout
            artifacts = ExperimentArtifacts(execution_trace=self.env.trace())

        return AgentRunResult(model_patch=model_patch, artifacts=artifacts)

    def collect_artifacts(self) -> ExperimentArtifacts | None:
        try:
            return ExperimentArtifacts(execution_trace=self.env.trace())
        except Exception:  # pragma: no cover - best effort fallback
            return None
