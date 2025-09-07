import uuid
from typing import Annotated, Literal, NewType

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, RootModel

from bench_mac.models import Submission


class AgentConfig(BaseModel):
    """
    Configuration for an agent.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_name: str = Field(
        ...,
        description="The name of the model to use for patch generation "
        "(e.g., 'mistral/devstral-medium-2507').",
    )

    @property
    def key(self) -> str:
        return f"{self.model_name}"

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


Discriminated = Annotated[
    CompletedExperiment | FailedExperiment,
    Field(discriminator="status"),
]


class ExperimentResult(RootModel[Discriminated]):
    pass
