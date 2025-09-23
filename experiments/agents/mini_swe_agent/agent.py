from pathlib import Path
from textwrap import dedent

from jinja2 import Environment, StrictUndefined
from loguru import logger
from minisweagent.agents.default import DefaultAgent
from minisweagent.run.utils.save import save_traj

from bench_mac.core.config import settings
from bench_mac.core.models import BenchmarkInstance
from bench_mac.docker.manager import DockerManager
from experiments.agents.base import AgentRunResult, BaseAgent
from experiments.agents.mini_swe_agent.environment import (
    MiniSweAgentEnvironmentAdapter,
)
from experiments.models import ExperimentArtifacts, MiniSweAgentConfig

from .tracing_model import TracingModel

_TASK_TEMPLATE_ENV = Environment(
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
)


def _render_task_prompt(template: str, instance: BenchmarkInstance) -> str:
    """Render the instance-specific task prompt using a strict Jinja template."""
    rendered = _TASK_TEMPLATE_ENV.from_string(template).render(instance=instance)
    return dedent(rendered).strip()


class MiniSweAgent(BaseAgent):
    """
    Mini SWE Agent implementation for BenchMAC.

    This agent wraps the minisweagent.DefaultAgent and provides a clean interface
    that returns a Submission object with the migration patch.
    """

    def __init__(
        self,
        instance: BenchmarkInstance,
        agent_config: MiniSweAgentConfig,
        docker_manager: DockerManager,
    ) -> None:
        self.instance = instance
        self.agent_config = agent_config

        # Use tracing subclass to collect raw LiteLLM responses for analysis
        self.model = TracingModel(model_name=agent_config.model_name)
        self.env = MiniSweAgentEnvironmentAdapter(instance, docker_manager)

        test_env = self.env.execute("ls -la")
        assert "package.json" in str(test_env.get("output", "")), (
            "package.json not found in the environment. "
            "The repository is not cloned correctly, or the current directory "
            "is not the project directory."
        )

        self.task_prompt = _render_task_prompt(
            agent_config.task_template, instance=self.instance
        )

        agent_kwargs = dict(agent_config.agent_settings)
        if agent_config.step_limit is not None:
            agent_kwargs["step_limit"] = int(agent_config.step_limit)
        if agent_config.cost_limit_usd is not None:
            agent_kwargs["cost_limit"] = float(agent_config.cost_limit_usd)

        self.agent = DefaultAgent(
            self.model,
            self.env,
            **agent_kwargs,
        )

    def run(
        self,
        *,
        submission_id: str,
    ) -> AgentRunResult:
        """Execute the agent and return the generated patch and artifacts."""
        logger.info(
            "Running Mini SWE Agent for instance: {instance_id} "
            "using minisweagent {swe_agent_mini_version}",
            instance_id=self.instance.instance_id,
            swe_agent_mini_version=self.agent_config.swe_agent_mini_version,
        )

        with self.env:
            exit_status, result = self.agent.run(task=self.task_prompt)

            cost_usd = self.agent.model.cost
            n_calls = self.agent.model.n_calls

            save_traj(
                self.agent,
                settings.experiments_dir
                / "swe_agent_mini"
                / Path(f"{submission_id}.traj.json"),
                exit_status=exit_status,
                result=result,
                extra_info={
                    "instance_id": self.instance.instance_id,
                    "submission_id": submission_id,
                    "agent_config": self.agent_config.model_dump(),
                    "cost_usd": cost_usd,
                    "n_calls": n_calls,
                },
            )

            if exit_status != "Submitted":
                raise RuntimeError(
                    f"Mini SWE Agent stopped before submission: {exit_status}: {result}"
                )

            # Generate patch and write completed result
            model_patch = self.env.diff_with_base_commit()
            artifacts = ExperimentArtifacts(
                execution_trace=self.env.execution_trace(),
                cost_usd=cost_usd,
                n_calls=n_calls,
                model_responses=list(self.model.responses),
            )

            return AgentRunResult(model_patch=model_patch, artifacts=artifacts)

    def collect_artifacts(self) -> ExperimentArtifacts | None:
        try:
            return ExperimentArtifacts(
                execution_trace=self.env.execution_trace(),
                cost_usd=self.agent.model.cost,
                n_calls=self.agent.model.n_calls,
                model_responses=list(self.model.responses),
            )
        except Exception:
            return None
