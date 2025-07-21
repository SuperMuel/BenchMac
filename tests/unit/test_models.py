from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from bench_mac.models import (
    BenchmarkInstance,
    CommandOutput,
    CommandsConfig,
    EvaluationReport,
    ExecutionTrace,
    MetricsReport,
    Submission,
)


@pytest.mark.unit
class TestBenchmarkInstance:
    """Tests for the BenchmarkInstance model, including its complex validation."""

    # A dictionary of valid data to be used as a base for tests
    VALID_INSTANCE_DATA = {
        "instance_id": "my-project_v15_to_v16",
        "repo": "SuperMuel/BenchMAC",
        "base_commit": "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0",
        "source_angular_version": "15.0.0",
        "target_angular_version": "16.1.0",
        "target_node_version": "18.13.0",
    }

    def test_instantiation_with_valid_data(self) -> None:
        """Test successful creation of a BenchmarkInstance with valid data."""
        instance = BenchmarkInstance.model_validate(self.VALID_INSTANCE_DATA)
        assert instance.instance_id == self.VALID_INSTANCE_DATA["instance_id"]
        assert instance.repo == self.VALID_INSTANCE_DATA["repo"]
        assert isinstance(instance.commands, CommandsConfig)
        assert instance.metadata == {}

    @pytest.mark.parametrize(
        "valid_repo",
        [
            "owner/repo-name",
            "SuperMuel/BenchMAC",
            "https://github.com/angular/angular-cli.git",
            "http://gitlab.com/my-org/my-project",
        ],
    )
    def test_repo_format_valid_cases(self, valid_repo: str) -> None:
        """Test that valid repo formats are accepted."""
        instance = BenchmarkInstance.model_validate(
            {**self.VALID_INSTANCE_DATA, "repo": valid_repo}
        )
        assert instance.repo == valid_repo

    @pytest.mark.parametrize(
        "invalid_repo",
        [
            "owner-repo-name",  # Missing slash
            "owner/repo/extra",  # Too many slashes
            "ftp://invalid.scheme/repo",  # Invalid scheme
            "just-a-string",
        ],
    )
    def test_repo_format_invalid_cases(self, invalid_repo: str) -> None:
        """Test that invalid repo formats raise a ValueError."""
        with pytest.raises(ValidationError, match="repo must be either"):
            BenchmarkInstance.model_validate(
                {**self.VALID_INSTANCE_DATA, "repo": invalid_repo}
            )

    @pytest.mark.parametrize(
        "valid_hash",
        ["a1b2c3d", "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0"],
    )
    def test_commit_hash_valid_cases(self, valid_hash: str) -> None:
        """Test that valid Git commit hashes are accepted."""
        instance = BenchmarkInstance.model_validate(
            {**self.VALID_INSTANCE_DATA, "base_commit": valid_hash}
        )
        assert instance.base_commit == valid_hash

    @pytest.mark.parametrize(
        "invalid_hash",
        [
            "",
            " ",
            "g1h2i3j",
            "a1b2c",
            "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0a",
            "random",
        ],
    )
    def test_commit_hash_invalid_cases(self, invalid_hash: str) -> None:
        """Test that invalid Git commit hashes raise a ValueError."""
        with pytest.raises(ValidationError, match="base_commit must be a valid"):
            BenchmarkInstance.model_validate(
                {**self.VALID_INSTANCE_DATA, "base_commit": invalid_hash}
            )

    @pytest.mark.parametrize(
        "field_name", ["source_angular_version", "target_angular_version"]
    )
    @pytest.mark.parametrize("valid_version", ["15", "16.1", "17.2.3", "2"])
    def test_angular_version_valid_cases(
        self, field_name: str, valid_version: str
    ) -> None:
        """Test that valid Angular versions are accepted for both source and target fields."""  # noqa: E501
        instance = BenchmarkInstance.model_validate(
            {**self.VALID_INSTANCE_DATA, field_name: valid_version}
        )
        assert getattr(instance, field_name) == valid_version

    @pytest.mark.parametrize(
        "field_name", ["source_angular_version", "target_angular_version"]
    )
    @pytest.mark.parametrize(
        "invalid_version", ["", " ", "v15", "1", "15.a.b", "15..0"]
    )
    def test_angular_version_invalid_cases(
        self, field_name: str, invalid_version: str
    ) -> None:
        """Test that invalid Angular versions raise a ValueError for both source and target fields."""  # noqa: E501
        with pytest.raises(ValidationError, match="Angular version must be"):
            BenchmarkInstance.model_validate(
                {**self.VALID_INSTANCE_DATA, field_name: invalid_version}
            )

    @pytest.mark.parametrize("invalid_version", ["", " ", "v18", "18.a.b", "18..0"])
    def test_node_version_invalid_cases(self, invalid_version: str) -> None:
        """Test that invalid Node.js versions raise a ValueError."""
        with pytest.raises(ValidationError, match="Node.js version must be"):
            BenchmarkInstance.model_validate(
                {**self.VALID_INSTANCE_DATA, "target_node_version": invalid_version}
            )

    @pytest.mark.parametrize("valid_version", ["1", "16", "18.13", "20.5.1"])
    def test_node_version_valid_cases(self, valid_version: str) -> None:
        """Test that valid Node.js versions are accepted."""
        instance = BenchmarkInstance.model_validate(
            {**self.VALID_INSTANCE_DATA, "target_node_version": valid_version}
        )
        assert instance.target_node_version == valid_version

    def test_missing_required_field_raises_error(self) -> None:
        """Test that omitting a required field raises a ValidationError."""
        # Create a dict without the required field (can't delete from TypedDict)
        invalid_data = {
            k: v for k, v in self.VALID_INSTANCE_DATA.items() if k != "repo"
        }
        with pytest.raises(ValidationError, match="Field required"):
            BenchmarkInstance.model_validate(invalid_data)


@pytest.mark.unit
class TestSubmission:
    """Tests for the Submission model."""

    def test_instantiation_with_valid_data(self) -> None:
        """Test successful creation of a Submission with valid data."""
        patch_content = "diff --git a/file.ts b/file.ts\n--- a/file.ts\n+++ b/file.ts"
        submission = Submission(
            instance_id="my-project_v15_to_v16", model_patch=patch_content
        )
        assert submission.instance_id == "my-project_v15_to_v16"
        assert submission.model_patch == patch_content

    def test_missing_model_patch_raises_error(self) -> None:
        """Test that omitting the model_patch field raises a ValidationError."""
        with pytest.raises(ValidationError, match="Field required"):
            Submission.model_validate({"instance_id": "some-id"})


@pytest.mark.unit
class TestMetricsReport:
    """Tests for the MetricsReport model."""

    def test_instantiation_with_valid_data(self) -> None:
        """Test successful creation of a MetricsReport."""
        report = MetricsReport(patch_application_success=True)
        assert report.patch_application_success is True


@pytest.mark.unit
class TestEvaluationResult:
    """Tests for the EvaluationResult model."""

    def test_instantiation_with_valid_data(self) -> None:
        """Test successful creation of an EvaluationResult."""
        # Create a mock execution trace
        command_output = CommandOutput(
            command="ng build",
            exit_code=1,
            stdout="",
            stderr="Build failed with 1 error.",
            start_time=datetime.now(UTC),
            end_time=datetime.now(UTC),
        )
        execution = ExecutionTrace(steps=[command_output])
        metrics = MetricsReport(patch_application_success=False)

        result = EvaluationReport(
            instance_id="my-project_v15_to_v16",
            execution=execution,
            metrics=metrics,
        )
        assert result.instance_id == "my-project_v15_to_v16"
        assert result.metrics.patch_application_success is False
        assert len(result.execution.steps) == 1
        assert result.execution.steps[0].command == "ng build"

    def test_empty_execution_trace(self) -> None:
        """Verify that an execution trace can be empty."""
        execution = ExecutionTrace(steps=[])
        metrics = MetricsReport(patch_application_success=True)
        result = EvaluationReport(
            instance_id="some-id", execution=execution, metrics=metrics
        )
        assert len(result.execution.steps) == 0
