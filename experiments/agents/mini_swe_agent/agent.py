import json
from pathlib import Path
from textwrap import dedent
from typing import Any, cast

import litellm
from jinja2 import Environment, StrictUndefined
from loguru import logger
from minisweagent.agents.default import DefaultAgent
from minisweagent.models import GLOBAL_MODEL_STATS
from minisweagent.models.litellm_model import LitellmModel
from minisweagent.run.utils.save import save_traj

from bench_mac.config import settings
from bench_mac.docker.manager import DockerManager
from bench_mac.models import BenchmarkInstance
from experiments.agents.base import AgentRunResult, BaseAgent
from experiments.agents.mini_swe_agent.environment import (
    MiniSweAgentEnvironmentAdapter,
)
from experiments.models import ExperimentArtifacts, MiniSweAgentConfig


class _TracingLitellmModel(LitellmModel):
    """
    Wrapper around minisweagent.models.litellm_model.LitellmModel that records
    the full LiteLLM response object for each call, while preserving the
    expected return shape for DefaultAgent (a dict with "content").
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._raw_responses: list[dict[str, Any]] = []

    def _serialize_response(self, response: Any) -> dict[str, Any]:
        # Best-effort conversion to a JSON-serializable dict
        try:
            if hasattr(response, "model_dump_json"):
                # Pydantic v2 style
                return cast(dict[str, Any], json.loads(response.model_dump_json()))  # type: ignore[name-defined]
        except Exception:
            pass
        try:
            if hasattr(response, "model_dump"):
                return response.model_dump()  # type: ignore[no-any-return]
        except Exception:
            pass
        try:
            if hasattr(response, "dict"):
                return response.dict()  # type: ignore[no-any-return]
        except Exception:
            pass
        try:
            if hasattr(response, "to_dict"):
                return response.to_dict()  # type: ignore[no-any-return]
        except Exception:
            pass
        if isinstance(response, dict):
            return response
        # Fallback: store a string representation
        return {"_unserializable_response": str(response)}

    def query(self, messages: list[dict[str, str]], **kwargs: Any) -> dict:
        response = self._query(messages, **kwargs)
        # Maintain counters & cost exactly as upstream
        cost = litellm.cost_calculator.completion_cost(response)
        self.n_calls += 1
        self.cost += cost
        GLOBAL_MODEL_STATS.add(cost)
        # Record the full response for artifact collection
        try:
            serialized = self._serialize_response(response)
            self._raw_responses.append(serialized)
        except Exception:
            self._raw_responses.append({"_serialization_error": True})
        # Return the reduced content shape expected by DefaultAgent
        return {
            "content": response.choices[0].message.content or "",  # type: ignore[attr-defined]
        }

    def collect_raw_responses(self) -> list[dict[str, Any]]:
        return list(self._raw_responses)


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
        model = _TracingLitellmModel(model_name=agent_config.model_name)
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
            model,
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
            # Collect full raw responses from the tracing model
            model_responses: list[dict[str, Any]] | None = (
                self.agent.model.collect_raw_responses()  # type: ignore[attr-defined]
                if hasattr(self.agent.model, "collect_raw_responses")
                else None
            )

            artifacts = ExperimentArtifacts(
                execution_trace=self.env.execution_trace(),
                cost_usd=cost_usd,
                n_calls=n_calls,
                model_responses=model_responses,
            )

            return AgentRunResult(model_patch=model_patch, artifacts=artifacts)

    def collect_artifacts(self) -> ExperimentArtifacts | None:
        try:
            model_responses: list[dict[str, Any]] | None = (
                self.agent.model.collect_raw_responses()  # type: ignore[attr-defined]
                if hasattr(self.agent.model, "collect_raw_responses")
                else None
            )

            return ExperimentArtifacts(
                execution_trace=self.env.execution_trace(),
                cost_usd=self.agent.model.cost,
                n_calls=self.agent.model.n_calls,
                model_responses=model_responses,
            )
        except Exception:
            return None
