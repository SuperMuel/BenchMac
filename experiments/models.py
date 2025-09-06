import uuid
from typing import Annotated, Literal

from pydantic import AwareDatetime, BaseModel, Field, RootModel

from bench_mac.models import Submission


class AgentConfig(BaseModel):
    """
    Configuration for an agent.
    """

    model_name: str = Field(
        ...,
        description="The name of the model to use for patch generation "
        "(e.g., 'mistral/devstral-medium-2507').",
    )


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


class CompletedExperiment(BaseModel):
    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the experiment.",
    )
    status: Literal["completed"] = "completed"
    task: ExperimentTask
    submission: Submission
    started_at: AwareDatetime
    ended_at: AwareDatetime


class FailedExperiment(BaseModel):
    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
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
