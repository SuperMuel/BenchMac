from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

from bench_mac.models import ExecutionTrace


class AgentRunArtifacts(BaseModel):
    execution_trace: ExecutionTrace | None = Field(
        default=None,
        description="Ordered command trace captured during the agent run.",
    )


class AgentRunResult(BaseModel):
    model_patch: str = Field(
        ...,
        description="A string containing the full, unified diff (.patch format) "
        "of all changes.",
    )
    artifacts: AgentRunArtifacts | None = Field(
        default=None,
        description="Optional structured artifacts captured during the agent run.",
    )


class BaseAgent(ABC):
    @abstractmethod
    def run(
        self,
        *,
        submission_id: str,
    ) -> AgentRunResult:
        """Execute the agent and return the generated patch and optional artifacts."""
        pass

    def collect_artifacts(self) -> AgentRunArtifacts | None:
        """Return any artifacts gathered outside of a successful run."""
        return None
