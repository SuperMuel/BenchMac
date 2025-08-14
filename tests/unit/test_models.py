from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from bench_mac.models import (
    CommandOutput,
    EvaluationReport,
    ExecutionTrace,
    MetricsReport,
    Submission,
)


@pytest.mark.unit
class TestBenchmarkInstance:
    """Tests for the BenchmarkInstance model, including its complex validation."""

    @pytest.mark.parametrize(
        "valid_repo",
        [
            "owner/repo-name",
            "SuperMuel/BenchMAC",
            "https://github.com/angular/angular-cli.git",
            "http://gitlab.com/my-org/my-project",
        ],
    )
    def test_repo_format_valid_cases(
        self, instance_factory: Any, valid_repo: str
    ) -> None:
        """Test that valid repo formats are accepted."""
        instance = instance_factory.create_instance(repo=valid_repo)
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
    def test_repo_format_invalid_cases(
        self, invalid_repo: str, instance_factory: Any
    ) -> None:
        """Test that invalid repo formats raise a ValueError."""
        with pytest.raises(ValidationError, match="repo must be either"):
            instance_factory.create_instance(repo=invalid_repo)

    @pytest.mark.parametrize(
        "valid_hash",
        ["a1b2c3d", "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0"],
    )
    def test_commit_hash_valid_cases(
        self, instance_factory: Any, valid_hash: str
    ) -> None:
        """Test that valid Git commit hashes are accepted."""
        instance = instance_factory.create_instance(base_commit=valid_hash)
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
    def test_commit_hash_invalid_cases(
        self,
        invalid_hash: str,
        instance_factory: Any,
    ) -> None:
        """Test that invalid Git commit hashes raise a ValueError."""
        with pytest.raises(ValidationError, match="base_commit must be a valid"):
            instance_factory.create_instance(base_commit=invalid_hash)

    @pytest.mark.parametrize(
        "field_name", ["source_angular_version", "target_angular_version"]
    )
    @pytest.mark.parametrize("valid_version", ["15", "16.1", "17.2.3", "2"])
    def test_angular_version_valid_cases(
        self, field_name: str, valid_version: str, instance_factory: Any
    ) -> None:
        """Test that valid Angular versions are accepted for both source and target fields."""  # noqa: E501
        instance = instance_factory.create_instance(**{field_name: valid_version})
        assert getattr(instance, field_name) == valid_version

    @pytest.mark.parametrize(
        "field_name", ["source_angular_version", "target_angular_version"]
    )
    @pytest.mark.parametrize(
        "invalid_version", ["", " ", "v15", "1", "15.a.b", "15..0"]
    )
    def test_angular_version_invalid_cases(
        self, field_name: str, invalid_version: str, instance_factory: Any
    ) -> None:
        """Test that invalid Angular versions raise a ValueError for both source and target fields."""  # noqa: E501
        with pytest.raises(ValidationError, match="Angular version must be"):
            instance_factory.create_instance(**{field_name: invalid_version})


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
        patch_content = "diff --git a/file.ts b/file.ts\n--- a/file.ts\n+++ b/file.ts"
        submission = Submission(
            instance_id="my-project_v15_to_v16", model_patch=patch_content
        )

        data = submission.model_dump()
        del data["model_patch"]

        with pytest.raises(ValidationError, match="Field required"):
            Submission.model_validate(data)


@pytest.mark.unit
class TestCommandOutput:
    """Tests for the CommandOutput model."""

    def test_duration_seconds_calculation(self) -> None:
        """Test that duration_seconds correctly calculates the time difference."""
        start = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        end = datetime(2024, 1, 1, 12, 0, 5, 500000, tzinfo=UTC)  # 5.5 seconds later

        command_output = CommandOutput(
            command="npm install",
            exit_code=0,
            stdout="Dependencies installed",
            stderr="",
            start_time=start,
            end_time=end,
        )

        assert command_output.duration_seconds == 5.5


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


@pytest.mark.unit
class TestBenchmarkInstanceDockerfileValidation:
    def test_allows_override_dockerfile_content(
        self, instance_factory: Any, tmp_path: Any, monkeypatch: Any
    ) -> None:
        # Point dockerfiles_dir to a temp directory that does not contain the file
        from bench_mac.config import settings

        monkeypatch.setattr(settings, "dockerfiles_dir", tmp_path, raising=False)

        instance = instance_factory.create_instance(
            instance_id="no-file-needed",
            override_dockerfile_content="FROM node:18\n",
        )
        assert instance.override_dockerfile_content == "FROM node:18\n"
        assert instance.dockerfile_content == "FROM node:18\n"

    def test_uses_existing_dockerfile_when_present(
        self, instance_factory: Any, tmp_path: Any, monkeypatch: Any
    ) -> None:
        from bench_mac.config import settings

        monkeypatch.setattr(settings, "dockerfiles_dir", tmp_path, raising=False)

        instance_id = "test-instance-has-file"
        dockerfile_path = tmp_path / instance_id
        dockerfile_content = "FROM node:18.19.0-bullseye-slim\n"
        dockerfile_path.write_text(dockerfile_content)

        instance = instance_factory.create_instance(
            instance_id=instance_id,
            override_dockerfile_content=None,
        )
        # Should not raise and should read from the file
        assert instance.override_dockerfile_content is None
        assert instance.dockerfile_content == dockerfile_content

    def test_raises_when_no_override_and_no_file(
        self, instance_factory: Any, tmp_path: Any, monkeypatch: Any
    ) -> None:
        from bench_mac.config import settings

        monkeypatch.setattr(settings, "dockerfiles_dir", tmp_path, raising=False)

        missing_id = "missing-instance"
        with pytest.raises(
            ValidationError, match="Either override_dockerfile_content must be provided"
        ):
            instance_factory.create_instance(
                instance_id=missing_id,
                override_dockerfile_content=None,
            )

    def test_raises_when_dockerfile_is_empty(
        self, instance_factory: Any, tmp_path: Any, monkeypatch: Any
    ) -> None:
        from bench_mac.config import settings

        monkeypatch.setattr(settings, "dockerfiles_dir", tmp_path, raising=False)

        instance_id = "empty-dockerfile-instance"
        dockerfile_path = tmp_path / instance_id
        dockerfile_path.write_text("  \n  ")

        with pytest.raises(ValidationError, match=r"is empty\."):
            instance_factory.create_instance(
                instance_id=instance_id,
                override_dockerfile_content=None,
            )

    def test_raises_when_override_dockerfile_content_is_empty(
        self, instance_factory: Any, tmp_path: Any, monkeypatch: Any
    ) -> None:
        from bench_mac.config import settings

        monkeypatch.setattr(settings, "dockerfiles_dir", tmp_path, raising=False)

        # Test empty string
        with pytest.raises(ValidationError, match=r"is empty\."):
            instance_factory.create_instance(
                instance_id="test-instance",
                override_dockerfile_content="",
            )

        # Test whitespace only
        with pytest.raises(ValidationError, match=r"is empty\."):
            instance_factory.create_instance(
                instance_id="test-instance",
                override_dockerfile_content="  \n  \t  ",
            )
