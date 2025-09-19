from pathlib import Path
from textwrap import dedent

import yaml
from loguru import logger
from minisweagent.agents.default import DefaultAgent
from minisweagent.models.litellm_model import LitellmModel
from minisweagent.run.utils.save import save_traj

from bench_mac.config import settings
from bench_mac.docker.manager import DockerManager
from bench_mac.models import BenchmarkInstance
from experiments.agents.base import (
    AgentRunArtifacts,
    AgentRunResult,
    BaseAgent,
)
from experiments.agents.mini_swe_agent.environment import (
    MiniSweAgentEnvironmentAdapter,
)
from experiments.models import MiniSweAgentConfig


def generate_task_prompt(instance: BenchmarkInstance) -> str:
    """Generates a detailed, structured prompt for the agent."""
    return dedent(
        f"""\
        ## Goal
        Migrate the application from Angular version from {instance.source_angular_version} to {instance.target_angular_version}.

        ## Context
        - The codebase is available in `/app/project`
        - The project is already cloned in the current directory. You do not need to clone it.
        - NPM is already installed.
        - Commands hints:
            - Install dependencies: {instance.commands.install}
            - Build the project: {instance.commands.build}

        ## Rules
        Do not change any application logic or functionality. Your focus is only on making the code compatible with the target Angular version.

        ## Recommended Workflow

        This workflows should be done step-by-step so that you can iterate on your changes and any possible problems.

        1. Analyze the codebase by finding and reading relevant files
        2. Edit the source code or run any command to migrate the codebase to the target Angular version
        3. Test the application by running the build command
        4. Submit your changes and finish your work by issuing the following command: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.
        Do not combine it with any other command. <important>After this command, you cannot continue working on this task.</important>
        """  # noqa: E501
    )


AGENT_CONFIG = yaml.safe_load(
    Path("experiments/agents/mini_swe_agent/mini_swe_agent_config.yaml").read_text()
)


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
        model = LitellmModel(model_name=agent_config.model_name)
        self.env = MiniSweAgentEnvironmentAdapter(instance, docker_manager)

        test_env = self.env.execute("ls -la")
        assert "package.json" in str(test_env.get("output", "")), (
            "package.json not found in the environment. "
            "The repository is not cloned correctly, or the current directory "
            "is not the project directory."
        )
        self.task_prompt = generate_task_prompt(instance)

        self.agent = DefaultAgent(
            model,
            self.env,
            **AGENT_CONFIG["agent"],
        )

    def run(
        self,
        *,
        submission_id: str,
    ) -> AgentRunResult:
        """Execute the agent and return the generated patch and artifacts."""
        logger.info(f"Running Mini SWE Agent for instance: {self.instance.instance_id}")

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

            # Generate patch and write completed result
            model_patch = self.env.diff_with_base_commit()
            artifacts = AgentRunArtifacts(
                execution_trace=self.env.execution_trace(),
                cost_usd=cost_usd,
                n_calls=n_calls,
            )

            return AgentRunResult(model_patch=model_patch, artifacts=artifacts)

    def collect_artifacts(self) -> AgentRunArtifacts | None:
        try:
            return AgentRunArtifacts(
                execution_trace=self.env.execution_trace(),
                cost_usd=self.agent.model.cost,
                n_calls=self.agent.model.n_calls,
            )
        except Exception:
            return None
