import re
from typing import Annotated, Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, BeforeValidator, Field, field_validator


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


def validate_node_version(value: str) -> str:
    """Validate Node.js version format."""
    # Allow semantic versions (e.g., "18.13.0") or partial versions (e.g., "18", "18.13")  # noqa: E501
    semver_pattern = r"^\d+(\.\d+){0,2}$"
    if not re.match(semver_pattern, value):
        raise ValueError(
            "Node.js version must be in semantic version format (e.g., '18', '18.13', '18.13.0')"  # noqa: E501
        )

    return value


# Custom types for versions
AngularVersion = Annotated[str, BeforeValidator(validate_angular_version)]
NodeVersion = Annotated[str, BeforeValidator(validate_node_version)]


class CommandsConfig(BaseModel):
    """Specifies custom evaluation commands, overriding harness defaults."""

    install: str = Field(
        default="npm ci",
        description="The command to install dependencies.",
    )
    build: str = Field(
        default="ng build --configuration production",
        description="The command to build the target project.",
    )
    lint: str = Field(
        default="ng lint",
        description="The command to lint the target project.",
    )
    test: str = Field(
        default="ng test --watch=false --browsers=ChromeHeadless",
        description="The command to run tests for the target project.",
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

    target_node_version: NodeVersion = Field(
        ...,
        description="The Node.js version for the evaluation environment "
        "(e.g., '18.13.0').",
    )

    commands: CommandsConfig = Field(
        default_factory=lambda: CommandsConfig(),
        description="Custom commands for evaluation, overriding defaults.",
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata like tags, difficulty, etc.",
    )


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

    patch_application_success: bool = Field(
        ...,
        description="Did the SUT's patch apply cleanly without conflicts?",
    )
    # target_version_achieved: bool = Field(
    #     ...,
    #     description="Do key @angular/* packages match the target version?",
    # )
    # build_success: bool = Field(
    #     ...,
    #     description="Did the 'ng build' command complete successfully?",
    # )
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


class EvaluationJob(BaseModel):
    """A job to evaluate a single submission."""

    instance: BenchmarkInstance
    submission: Submission


class EvaluationReport(BaseModel):
    """The final, comprehensive result of evaluating a single submission."""

    instance_id: str = Field(
        ...,
        description="The unique identifier of the evaluated instance.",
    )

    metrics: MetricsReport = Field(
        ...,
        description="The detailed performance metrics for the submission.",
    )
    logs: dict[str, str] = Field(
        default_factory=dict,
        description="A dictionary containing raw logs from evaluation steps (e.g., 'build', 'test').",  # noqa: E501
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


RunOutcome = RunSuccess | RunFailure
