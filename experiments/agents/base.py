from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

from experiments.models import ExperimentArtifacts


class AgentRunResult(BaseModel):
    model_patch: str = Field(
        ...,
        description="A string containing the full, unified diff (.patch format) "
        "of all changes.",
    )
    artifacts: ExperimentArtifacts | None = Field(
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

    def collect_artifacts(self) -> ExperimentArtifacts | None:
        """Return any artifacts gathered outside of a successful run."""
        return None
