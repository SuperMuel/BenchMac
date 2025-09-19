import hashlib
from datetime import timedelta
from typing import Annotated, Literal, NewType

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, RootModel
from uuid6 import uuid7

from bench_mac.models import ExecutionTrace, Submission


class MiniSweAgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    scaffold: Literal["swe-agent-mini"] = "swe-agent-mini"
    model_name: str = Field(
        ...,
        min_length=1,
        description="The name of the model to use for patch generation "
        "(e.g., 'mistral/devstral-medium-2507').",
    )

    @property
    def key(self) -> str:
        return f"{self.scaffold}/{self.model_name}"

    @property
    def display_name(self) -> str:
        return self.model_name

    def __hash__(self) -> int:
        return hash(self.key)


class AngularSchematicsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    scaffold: Literal["angular-schematics"] = "angular-schematics"
    update_command_template: str = Field(
        default=(
            "npx --yes @angular/cli@{target_version} update "
            "@angular/core@{target_version} @angular/cli@{target_version} "
            "--allow-dirty --force --verbose"
        ),
        description=(
            "Template for the Angular update command. "
            "Use {target_version} as placeholder."
        ),
    )

    @property
    def key(self) -> str:
        # Hash the update_command_template for uniqueness
        template_hash = hashlib.sha256(
            self.update_command_template.encode("utf-8")
        ).hexdigest()[:8]
        return f"{self.scaffold}/{template_hash}"

    @property
    def display_name(self) -> str:
        return self.key

    def __hash__(self) -> int:
        return hash(self.key)


AgentConfig = Annotated[
    MiniSweAgentConfig | AngularSchematicsConfig,
    Field(discriminator="scaffold"),
]


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
        default_factory=lambda: ExperimentID(str(uuid7())),
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
        default_factory=lambda: ExperimentID(str(uuid7())),
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
