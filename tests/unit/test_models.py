import pytest
from pydantic import ValidationError
from typing_extensions import TypedDict

from bench_mac.models import (
    BenchmarkInstance,
    CommandsConfig,
    EvaluationResult,
    MetricsReport,
    Submission,
)


class BenchmarkInstanceData(TypedDict):
    """TypedDict for BenchmarkInstance test data - supports IDE refactoring."""

    instance_id: str
    repo: str
    base_commit: str
    source_angular_version: str
    target_angular_version: str
    target_node_version: str


class SubmissionData(TypedDict):
    """TypedDict for Submission test data - supports IDE refactoring."""

    instance_id: str
    model_patch: str


@pytest.mark.unit
class TestCommandsConfig:
    """Tests for the CommandsConfig model."""

    def test_instantiation_with_defaults(self) -> None:
        """Verify that CommandsConfig instantiates with correct default values."""
        config = CommandsConfig()
        assert config.install == "npm ci"
        assert config.build == "ng build --configuration production"
        assert config.lint == "ng lint"
        assert config.test == "ng test --watch=false --browsers=ChromeHeadless"

    def test_instantiation_with_overrides(self) -> None:
        """Verify that default values can be overridden at instantiation."""
        custom_build_command = "nx build my-app"
        config = CommandsConfig(build=custom_build_command)
        assert config.install == "npm ci"  # Should remain default
        assert config.build == custom_build_command  # Should be overridden


@pytest.mark.unit
class TestBenchmarkInstance:
    """Tests for the BenchmarkInstance model, including its complex validation."""

    # A dictionary of valid data to be used as a base for tests
    VALID_INSTANCE_DATA: BenchmarkInstanceData = {
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
        metrics = MetricsReport(patch_application_success=False)
        result = EvaluationResult(
            instance_id="my-project_v15_to_v16",
            metrics=metrics,
            logs={"build": "Build failed with 1 error."},
        )
        assert result.instance_id == "my-project_v15_to_v16"
        assert result.metrics.patch_application_success is False
        assert "build" in result.logs

    def test_default_logs_is_empty_dict(self) -> None:
        """Verify that the 'logs' field defaults to an empty dictionary."""
        metrics = MetricsReport(patch_application_success=True)
        result = EvaluationResult(instance_id="some-id", metrics=metrics)
        assert result.logs == {}


# ============================================================================
# DEMONSTRATION OF DIFFERENT SOLUTIONS TO THE REFACTORING PROBLEM
# ============================================================================


# Solution 3: Use constants for field names
class BenchmarkInstanceFields:
    """Constants for BenchmarkInstance field names - supports IDE refactoring."""

    INSTANCE_ID = "instance_id"
    REPO = "repo"
    BASE_COMMIT = "base_commit"
    SOURCE_ANGULAR_VERSION = "source_angular_version"
    TARGET_ANGULAR_VERSION = "target_angular_version"
    TARGET_NODE_VERSION = "target_node_version"


# Solution 4: Use factory functions
def create_valid_benchmark_instance(**overrides) -> BenchmarkInstance:
    """Factory function to create BenchmarkInstance with overrides."""
    defaults = BenchmarkInstanceData(
        instance_id="my-project_v15_to_v16",
        repo="SuperMuel/BenchMAC",
        base_commit="a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0",
        source_angular_version="15.0.0",
        target_angular_version="16.1.0",
        target_node_version="18.13.0",
    )
    return BenchmarkInstance.model_validate({**defaults, **overrides})


@pytest.mark.unit
class TestRefactoringSolutions:
    """Demonstrates different solutions to the field name refactoring problem."""

    def test_solution_1_typeddict(self) -> None:
        """Solution 1: TypedDict provides type safety and IDE refactoring support."""
        # The TypedDict ensures field names are typed and refactorable
        data: BenchmarkInstanceData = {
            "instance_id": "test-id",
            "repo": "test/repo",
            "base_commit": "a1b2c3d4e5f6a7b8",
            "source_angular_version": "15.0.0",
            "target_angular_version": "16.0.0",
            "target_node_version": "18.13.0",
        }
        instance = BenchmarkInstance.model_validate(data)
        assert instance.instance_id == data["instance_id"]

    def test_solution_2_direct_construction(self) -> None:
        """Solution 2: Direct model construction with keyword arguments."""
        # Field names are in code, not strings, so IDE refactoring works
        instance = BenchmarkInstance(
            instance_id="test-id",
            repo="test/repo",
            base_commit="a1b2c3d4e5f6a7b8",
            source_angular_version="15.0.0",
            target_angular_version="16.0.0",
            target_node_version="18.13.0",
        )
        assert instance.instance_id == "test-id"

    def test_solution_3_field_constants(self) -> None:
        """Solution 3: Use constants for field names that need string references."""
        # Field names as constants can be refactored by IDE
        data = {
            BenchmarkInstanceFields.INSTANCE_ID: "test-id",
            BenchmarkInstanceFields.REPO: "test/repo",
            BenchmarkInstanceFields.BASE_COMMIT: "a1b2c3d4e5f6a7b8",
            BenchmarkInstanceFields.SOURCE_ANGULAR_VERSION: "15.0.0",
            BenchmarkInstanceFields.TARGET_ANGULAR_VERSION: "16.0.0",
            BenchmarkInstanceFields.TARGET_NODE_VERSION: "18.13.0",
        }
        instance = BenchmarkInstance.model_validate(data)

        # Using getattr with constants
        assert getattr(instance, BenchmarkInstanceFields.INSTANCE_ID) == "test-id"

    def test_solution_4_factory_functions(self) -> None:
        """Solution 4: Use factory functions for test data creation."""
        # Factory functions encapsulate field names in code
        instance = create_valid_benchmark_instance(
            instance_id="custom-id", repo="custom/repo"
        )
        assert instance.instance_id == "custom-id"
        assert instance.repo == "custom/repo"
        # Other fields use defaults from factory

    def test_solution_5_model_fields_introspection(self) -> None:
        """Solution 5: Use Pydantic model field introspection."""
        # Get field names dynamically from the model
        field_names = list(BenchmarkInstance.model_fields.keys())
        assert "instance_id" in field_names
        assert "repo" in field_names

        # This approach is useful for generic testing but still has string issues
        for field_name in ["instance_id", "repo"]:
            assert field_name in BenchmarkInstance.model_fields
