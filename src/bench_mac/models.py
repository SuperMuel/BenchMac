import re
import uuid
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any, Literal, NewType, Self
from urllib.parse import urlparse

from pydantic import (
    AwareDatetime,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
    model_validator,
)

from bench_mac.config import settings


def utc_now() -> datetime:
    """Returns the current time in UTC."""
    return datetime.now(UTC)


def validate_angular_version(value: str) -> str:
    """
    Validate an Angular version string.

    Accepts either a bare major (e.g., "16") or a semantic version with up to
    three numeric components (e.g., "16.2" or "16.2.1"). Major must be >= 2.
    """
    # Allow semantic versions (e.g., "16.2.1") or just major versions (e.g., "16")
    semver_pattern = r"^\d+(\.\d+){0,2}$"
    if not re.match(semver_pattern, value):
        raise ValueError(
            "Angular version must be in semantic version format (e.g., '16', '16.2', '16.2.1')"  # noqa: E501
        )

    # Check if major version is a known Angular version
    major_version = int(value.split(".")[0])
    if major_version < 2:
        raise ValueError("Angular version must be 2 or higher")

    return value


# Custom types for versions
AngularVersion = Annotated[str, BeforeValidator(validate_angular_version)]


class InstanceCommands(BaseModel):
    install: str = Field(
        ...,
        description="The command to install dependencies.",
    )
    build: str = Field(
        ...,
        description="The command to build the target project.",
    )


# --- Core Benchmark & SUT Models ---


InstanceID = NewType("InstanceID", str)


class BenchmarkInstance(BaseModel):
    """
    A single migration task to evaluate.

    Captures the repository, the starting commit/version, the target version,
    and the per-instance commands used during evaluation. Docker build context
    comes from either `override_dockerfile_content` or a file located at
    `data/dockerfiles/<instance_id>`.
    """

    instance_id: InstanceID = Field(
        ...,
        description="Unique identifier for the benchmark instance.",
        min_length=1,
    )
    repo: str = Field(
        ...,
        description="The repository URL or owner/name (e.g., 'owner/repo').",
    )

    @field_validator("repo")
    @classmethod
    def validate_repo_format(cls, v: str) -> str:
        # Check if it's a valid URL
        parsed = urlparse(v)
        if parsed.scheme in ("http", "https") and parsed.netloc:
            return v

        # Check if it's in owner/repo format
        owner_repo_pattern = r"^[a-zA-Z0-9]([a-zA-Z0-9._-]*[a-zA-Z0-9])?/[a-zA-Z0-9]([a-zA-Z0-9._-]*[a-zA-Z0-9])?$"  # noqa: E501
        if re.match(owner_repo_pattern, v):
            return v

        raise ValueError(
            "repo must be either a valid URL (http/https) or in 'owner/repo' format"
        )

    base_commit: str = Field(
        ...,
        description="The specific Git commit hash to start the migration from.",
    )

    @field_validator("base_commit")
    @classmethod
    def validate_commit_hash(cls, v: str) -> str:
        # Git commit hashes are hexadecimal and typically 7-40 characters
        commit_hash_pattern = r"^[a-fA-F0-9]{7,40}$"
        if not re.match(commit_hash_pattern, v):
            raise ValueError(
                "base_commit must be a valid Git commit hash (7-40 hexadecimal characters)"  # noqa: E501
            )
        return v

    source_angular_version: AngularVersion = Field(
        ...,
        description="The starting major Angular version (e.g., '15').",
    )
    target_angular_version: AngularVersion = Field(
        ...,
        description="The desired target major Angular version (e.g., '16').",
    )

    commands: InstanceCommands = Field(
        ...,
        description="Custom commands for evaluation, overriding defaults.",
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata like tags, difficulty, etc.",
    )
    override_dockerfile_content: str | None = Field(
        default=None,
        description="The Dockerfile content for the instance, overriding any "
        "Dockerfile in `.data/dockerfiles/<instance_id>`",
    )

    @field_validator("override_dockerfile_content")
    @classmethod
    def validate_override_dockerfile_content(cls, v: str | None) -> str | None:
        if v is not None and not v.strip():
            raise ValueError("override_dockerfile_content is empty.")
        return v

    @model_validator(mode="after")
    def validate_dockerfile_availability(self) -> Self:
        if self.override_dockerfile_content is not None:
            return self

        # Check if dockerfile file exists at ./data/dockerfiles/<instance_id>
        dockerfile_path = settings.dockerfiles_dir / self.instance_id
        if not dockerfile_path.exists():
            raise ValueError(
                "Either override_dockerfile_content must be provided"
                f" or dockerfile must exist at {dockerfile_path}"
            )

        # Ensure the Dockerfile is not empty
        content = dockerfile_path.read_text()
        if not content.strip():
            raise ValueError(f"Dockerfile at {dockerfile_path} is empty.")

        return self

    @property
    def dockerfile_content(self) -> str:
        if self.override_dockerfile_content is not None:
            return self.override_dockerfile_content

        dockerfile_path = settings.dockerfiles_dir / self.instance_id
        assert dockerfile_path.exists(), (
            "Dockerfile should exist at .data/dockerfiles/<instance_id>"
            " since override_dockerfile_content wasn't provided. It indicates"
            " a programming error in the Pydantic model validation."
        )
        content = dockerfile_path.read_text()
        if not content.strip():
            raise ValueError(f"Dockerfile at {dockerfile_path} is empty.")

        return content


class SubmissionMetadata(BaseModel):
    """Metadata for a submission."""

    model_config = ConfigDict(extra="allow")

    created_at: AwareDatetime = Field(
        default_factory=utc_now,
        description="The timestamp when the submission was created.",
    )


SubmissionID = NewType("SubmissionID", str)
EvaluationID = NewType("EvaluationID", str)


class Submission(BaseModel):
    """
    A solution candidate for a specific instance.

    `model_patch` must be a unified diff (git patch) applying all changes that
    implement the migration from source to target.
    """

    submission_id: SubmissionID = Field(
        default_factory=lambda: SubmissionID(str(uuid.uuid4())),
        description="The unique identifier of the submission.",
    )

    instance_id: InstanceID = Field(
        ...,
        description="The unique identifier of the instance being solved.",
    )
    model_patch: str = Field(
        ...,
        description="A string containing the full, unified diff (.patch format) of all changes.",  # noqa: E501
    )

    metadata: SubmissionMetadata = Field(
        default_factory=SubmissionMetadata,
        description="Metadata for the submission.",
    )


# --- Evaluation & Metrics Models ---


class MetricsReport(BaseModel):
    """
    Quantitative evaluation outcomes derived from an `ExecutionTrace`.

    Fields are tri-state:
      - True/False when the harness can determine success or failure;
      - None when indeterminate (step not run or insufficient data).
    """

    patch_application_success: bool | None = Field(
        default=None,
        description="Did the SUT's patch apply cleanly without conflicts?",
    )
    target_version_achieved: bool | None = Field(
        default=None,
        description="Do key @angular/* packages match the target version?",
    )
    build_success: bool | None = Field(
        default=None,
        description="Did the build command complete successfully?",
    )
    install_success: bool | None = Field(
        default=None,
        description="Did the install command complete successfully?",
    )
    # no_new_critical_lint_errors: bool = Field(
    #     ...,
    #     description="Did the 'ng lint' command pass without new critical errors?",
    # )
    # lint_error_delta: int = Field(
    #     ...,
    #     description="The change in the number of lint errors (new - resolved).",
    # )
    # test_pass_rate: float = Field(
    #     ...,
    #     ge=0.0,
    #     le=1.0,
    #     description="Percentage of baseline passing tests that still pass after "
    #     "migration.",
    # )
    # test_coverage_delta: float = Field(
    #     ...,
    #     description="The percentage point change in test coverage "
    #     "(new_coverage - base_coverage).",
    # )
    # no_superficial_fixes: bool = Field(
    #     ...,
    #     description="Did the SUT avoid 'cheating' (e.g., commenting out tests)?",
    # )
    # qualitative_code_score: float | None = Field(
    #     default=None,
    #     description="(Future) LLM-as-judge score for code style, idiomaticity, etc.",
    # )


class EvaluationTask(BaseModel):
    """
    A schedulable unit of work: (instance, submission).

    The runner distributes these to worker processes for execution.
    """

    instance: BenchmarkInstance
    submission: Submission


class CommandResult(BaseModel):
    """
    Result of a single command executed inside the container.

    Timestamps are expected to be timezone-aware (UTC). Use `.success` for a
    semantic success check (exit_code == 0).
    """

    command: str = Field(..., description="The exact command that was executed.")
    exit_code: int = Field(
        ..., description="The exit code of the command. 0 typically means success."
    )
    stdout: str = Field(
        default="", description="The captured standard output (stdout) of the command."
    )
    stderr: str = Field(
        default="", description="The captured standard error (stderr) of the command."
    )
    start_time: AwareDatetime = Field(
        ..., description="The UTC timestamp when the command started."
    )
    end_time: AwareDatetime = Field(
        ..., description="The UTC timestamp when the command finished."
    )

    @property
    def duration(self) -> timedelta:
        return self.end_time - self.start_time

    @property
    def success(self) -> bool:
        """Convenience predicate: True iff `exit_code == 0`."""
        return self.exit_code == 0


class ExecutionTrace(BaseModel):
    """
    Ordered list of `CommandResult` steps produced during evaluation.

    Typical order on the happy path:
      1) git apply --check
      2) git apply -p0
      3) install command(s)
      4) version check (npm ls … --json)
      5) build command
    """

    steps: list[CommandResult]

    @property
    def total_duration(self) -> timedelta:
        """
        Returns the total duration of all command steps in the execution trace.
        If there are no steps, returns timedelta(0).
        """
        if not self.steps:
            return timedelta(0)
        return sum((step.duration for step in self.steps), timedelta(0))


class EvaluationReport(BaseModel):
    """
    Final structured record for a completed evaluation.

    Contains the raw `ExecutionTrace` (source of truth) and the derived
    `MetricsReport` (interpretation).
    """

    instance_id: InstanceID = Field(...)
    submission_id: SubmissionID = Field(
        ..., description="The unique identifier of the submission that was evaluated."
    )

    # The raw, unprocessed data from the execution environment.
    # This is the "source of truth".
    execution: ExecutionTrace = Field(
        ...,
        description="A structured trace of all commands executed during the evaluation.",  # noqa: E501
    )

    # The metrics derived from the execution trace.
    # This is the "interpretation" of the raw data.
    metrics: MetricsReport = Field(
        ..., description="The detailed performance metrics for the submission."
    )
    created_at: AwareDatetime = Field(
        default_factory=utc_now,
        description="The timestamp when the evaluation report was created.",
    )


class EvaluationCompleted(BaseModel):
    """
    Wrapper for a successful harness run.

    Use this when the evaluation executed to completion (regardless of
    metric values) and produced an `EvaluationReport`.
    """

    id: EvaluationID = Field(
        default_factory=lambda: EvaluationID(str(uuid.uuid4())),
        description="The unique identifier of the evaluation.",
    )
    status: Literal["completed"] = "completed"
    result: EvaluationReport
    started_at: AwareDatetime = Field(
        description="The timestamp when the evaluation started.",
    )
    ended_at: AwareDatetime = Field(
        description="The timestamp when the evaluation ended.",
    )


class EvaluationFailed(BaseModel):
    """
    Wrapper for a harness/system failure.

    Use this when the evaluation could not be completed (e.g., Docker not
    available, process crash). No `ExecutionTrace` is available here.
    """

    id: EvaluationID = Field(
        default_factory=lambda: EvaluationID(str(uuid.uuid4())),
        description="The unique identifier of the evaluation.",
    )
    status: Literal["failed"] = "failed"
    instance_id: InstanceID
    submission_id: SubmissionID = Field(
        ..., description="The unique identifier of the submission that failed."
    )
    started_at: AwareDatetime = Field(
        description="The timestamp when the evaluation started.",
    )
    ended_at: AwareDatetime = Field(
        description="The timestamp when the evaluation ended.",
    )
    error: str


EvaluationResult = Annotated[
    EvaluationCompleted | EvaluationFailed,
    Field(discriminator="status"),
]

EvaluationResultAdapter = TypeAdapter(EvaluationResult)
