import uuid
from datetime import timedelta
from typing import Annotated, Literal, NewType

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, RootModel

from bench_mac.models import ExecutionTrace, Submission


class AgentConfig(BaseModel):
    """
    Configuration for an agent.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    scaffold: Literal["swe-agent-mini"] = "swe-agent-mini"

    model_name: str = Field(
        ...,
        description="The name of the model to use for patch generation "
        "(e.g., 'mistral/devstral-medium-2507').",
    )

    @property
    def key(self) -> str:
        return f"{self.scaffold}/{self.model_name}"

    def __hash__(self) -> int:
        return hash(self.key)


class ExperimentTask(BaseModel):
    """
    A task for generating patches using an agent configuration on a specific instance.

    This represents a single unit of work in the patch generation process.
    """

    instance_id: str = Field(
        ...,
        description="Unique identifier for the benchmark instance to process.",
    )
    agent_config: AgentConfig = Field(
        ...,
        description="The agent configuration to use for patch generation.",
    )


ExperimentID = NewType("ExperimentID", str)


class ExperimentArtifacts(BaseModel):
    execution_trace: ExecutionTrace | None = Field(
        default=None,
        description="Ordered command trace captured during the experiment run.",
    )
    cost_usd: float | None = Field(
        default=None,
        description="The cost of the agent run in USD.",
    )
    n_calls: int | None = Field(
        default=None,
        description="The number of calls to the model.",
    )


class CompletedExperiment(BaseModel):
    id: ExperimentID = Field(
        default_factory=lambda: ExperimentID(str(uuid.uuid4())),
        description="Unique identifier for the experiment.",
    )
    status: Literal["completed"] = "completed"
    task: ExperimentTask
    submission: Submission
    started_at: AwareDatetime
    ended_at: AwareDatetime
    artifacts: ExperimentArtifacts | None = Field(
        default=None,
        description="Optional structured artifacts captured during the run.",
    )

    @property
    def duration(self) -> timedelta:
        return self.ended_at - self.started_at


class FailedExperiment(BaseModel):
    id: ExperimentID = Field(
        default_factory=lambda: ExperimentID(str(uuid.uuid4())),
        description="Unique identifier for the experiment.",
    )
    status: Literal["failed"] = "failed"
    task: ExperimentTask
    error: str
    started_at: AwareDatetime
    ended_at: AwareDatetime
    artifacts: ExperimentArtifacts | None = Field(
        default=None,
        description="Optional structured artifacts captured during the run,"
        " possibly empty or incomplete.",
    )


Discriminated = Annotated[
    CompletedExperiment | FailedExperiment,
    Field(discriminator="status"),
]


class ExperimentResult(RootModel[Discriminated]):
    @property
    def is_completed(self) -> bool:
        return self.root.status == "completed"

    @property
    def is_failed(self) -> bool:
        return self.root.status == "failed"
