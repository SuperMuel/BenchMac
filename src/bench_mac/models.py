import re
from datetime import datetime
from typing import Annotated, Any, Literal, Self
from urllib.parse import urlparse

from pydantic import BaseModel, BeforeValidator, Field, field_validator, model_validator

from bench_mac.config import settings


def validate_angular_version(value: str) -> str:
    """Validate Angular version format."""
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


class CommandsConfig(BaseModel):
    """Specifies custom evaluation commands, overriding harness defaults."""

    install: str = Field(
        ...,
        description="The command to install dependencies.",
    )
    build: str = Field(
        ...,
        description="The command to build the target project.",
    )


# --- Core Benchmark & SUT Models ---


class BenchmarkInstance(BaseModel):
    """Represents a single, specific task within the benchmark."""

    instance_id: str = Field(
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
        """Validate that repo is either a valid URL or owner/repo format."""
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
        """Validate that base_commit is a valid Git commit hash."""
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

    commands: CommandsConfig = Field(
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
        """Validate that override_dockerfile_content is not empty if provided."""
        if v is not None and not v.strip():
            raise ValueError("override_dockerfile_content is empty.")
        return v

    @model_validator(mode="after")
    def validate_dockerfile_availability(self) -> Self:
        """Validate that either dockerfile_content is provided or
        dockerfile file exists.
        """
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
        """The Dockerfile content for the instance."""
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


class Submission(BaseModel):
    """Represents a single solution submitted by a SUT for a task instance."""

    instance_id: str = Field(
        ...,
        description="The unique identifier of the instance being solved.",
    )
    model_patch: str = Field(
        ...,
        description="A string containing the full, unified diff (.patch format) of all changes.",  # noqa: E501
    )


# --- Evaluation & Metrics Models ---


class MetricsReport(BaseModel):
    """A detailed, quantifiable report of the SUT's performance on an instance."""

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


class ExecutionJob(BaseModel):
    """A job to evaluate a single submission."""

    instance: BenchmarkInstance
    submission: Submission


class CommandOutput(BaseModel):
    """
    Represents the detailed result of a single command executed in the container.
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
    start_time: datetime = Field(
        ..., description="The UTC timestamp when the command started."
    )
    end_time: datetime = Field(
        ..., description="The UTC timestamp when the command finished."
    )

    @property
    def duration_seconds(self) -> float:
        """The total duration of the command execution in seconds."""
        return (self.end_time - self.start_time).total_seconds()

    @property
    def success(self) -> bool:
        """A convenience property to check if the command succeeded
        (exit_code == 0)
        """
        return self.exit_code == 0


class ExecutionTrace(BaseModel):
    steps: list[CommandOutput]


class EvaluationReport(BaseModel):
    """The final, comprehensive result of evaluating a single submission."""

    instance_id: str = Field(...)

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


class RunSuccess(BaseModel):
    """Represents a successfully completed evaluation run for an instance."""

    status: Literal["success"] = "success"
    result: EvaluationReport


class RunFailure(BaseModel):
    """Represents a harness-level failure during an evaluation run."""

    status: Literal["failure"] = "failure"
    instance_id: str
    error: str


EvaluationResult = RunSuccess | RunFailure
