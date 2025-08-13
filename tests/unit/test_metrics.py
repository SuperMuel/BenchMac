"""
Unit tests for the metrics calculation module.
"""

import json
from datetime import UTC, datetime

import pytest

from bench_mac.metrics import (
    _calculate_patch_application_success,
    _calculate_target_version_achieved,
    calculate_metrics,
)
from bench_mac.models import (
    BenchmarkInstance,
    CommandsConfig,
    ExecutionTrace,
)
from bench_mac.models import (
    CommandOutput as _RealCommandOutput,
)


def CommandOutput(  # noqa: N802
    command: str,
    exit_code: int,
    stdout: str = "",
    stderr: str = "",
    start_time: datetime | None = None,
    end_time: datetime | None = None,
) -> _RealCommandOutput:
    """Test helper to auto-fill required timestamps for CommandOutput."""
    now = datetime.now(UTC)
    start = start_time or now
    end = end_time or start
    return _RealCommandOutput(
        command=command,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        start_time=start,
        end_time=end,
    )


@pytest.mark.unit
class TestCalculatePatchApplicationSuccess:
    """Unit tests for the _calculate_patch_application_success helper function."""

    def test_returns_true_when_both_steps_succeed(self) -> None:
        patch_check = CommandOutput(command="git apply --check", exit_code=0)
        patch_apply = CommandOutput(command="git apply -p0", exit_code=0)
        assert _calculate_patch_application_success(patch_check, patch_apply) is True

    def test_returns_false_when_check_fails(self) -> None:
        patch_check = CommandOutput(command="git apply --check", exit_code=1)
        patch_apply = CommandOutput(command="git apply -p0", exit_code=0)
        assert _calculate_patch_application_success(patch_check, patch_apply) is False

    def test_returns_false_when_apply_fails(self) -> None:
        patch_check = CommandOutput(command="git apply --check", exit_code=0)
        patch_apply = CommandOutput(command="git apply -p0", exit_code=1)
        assert _calculate_patch_application_success(patch_check, patch_apply) is False

    def test_returns_false_when_only_failed_check_is_present(self) -> None:
        patch_check = CommandOutput(command="git apply --check", exit_code=1)
        assert _calculate_patch_application_success(patch_check, None) is False

    def test_returns_true_when_only_successful_apply_is_present(self) -> None:
        patch_apply = CommandOutput(command="git apply -p0", exit_code=0)
        assert _calculate_patch_application_success(None, patch_apply) is True

    def test_returns_none_when_no_steps_are_present(self) -> None:
        assert _calculate_patch_application_success(None, None) is None


@pytest.mark.unit
class TestCalculateTargetVersionAchieved:
    """Unit tests for the _calculate_target_version_achieved helper function."""

    def test_returns_true_on_major_version_match(self) -> None:
        version_output = {"@angular/core": "16.1.5"}
        version_step = CommandOutput(
            command="npx ng version --json",
            exit_code=0,
            stdout=json.dumps(version_output),
        )
        assert _calculate_target_version_achieved(version_step, "16") is True

    def test_returns_true_on_major_version_match_with_full_target_version(self) -> None:
        version_output = {"@angular/core": "16.1.5"}
        version_step = CommandOutput(
            command="npx ng version --json",
            exit_code=0,
            stdout=json.dumps(version_output),
        )
        assert _calculate_target_version_achieved(version_step, "16.2.1") is True

    def test_returns_false_on_major_version_mismatch(self) -> None:
        version_output = {"@angular/core": "15.2.9"}
        version_step = CommandOutput(
            command="npx ng version --json",
            exit_code=0,
            stdout=json.dumps(version_output),
        )
        assert _calculate_target_version_achieved(version_step, "16") is False

    def test_returns_none_if_command_failed(self) -> None:
        version_step = CommandOutput(
            command="npx ng version --json", exit_code=1, stderr="Command failed"
        )
        # The rationale is that the command could fail for some reason,
        # even if the version is correct.
        assert _calculate_target_version_achieved(version_step, "16") is None

    def test_returns_false_if_key_is_missing(self) -> None:
        version_output = {"@some/other-package": "1.0.0"}
        version_step = CommandOutput(
            command="npx ng version --json",
            exit_code=0,
            stdout=json.dumps(version_output),
        )
        assert _calculate_target_version_achieved(version_step, "16") is False

    def test_returns_none_if_json_is_invalid(self) -> None:
        version_step = CommandOutput(
            command="npx ng version --json", exit_code=0, stdout="this is not json"
        )
        # Invalid output should result in None, as we can't be certain.
        assert _calculate_target_version_achieved(version_step, "16") is None

    def test_returns_none_if_step_is_missing(self) -> None:
        assert _calculate_target_version_achieved(None, "16") is None


@pytest.fixture
def sample_instance() -> BenchmarkInstance:
    """Provides a standard BenchmarkInstance for testing."""
    return BenchmarkInstance(
        instance_id="test-project_v15_to_v16",
        repo="owner/repo",
        base_commit="a" * 40,
        source_angular_version="15.2.0",
        target_angular_version="16.0.0",
        override_dockerfile_content="FROM node:18",  # Dummy content
        commands=CommandsConfig(
            install="npm ci",
            build="ng build",
        ),
    )


@pytest.mark.unit
class TestCalculateMetrics:
    """
    Tests the main calculate_metrics function to ensure it correctly
    composes the results from its helper functions.
    """

    def test_successful_run_calculates_all_metrics_correctly(
        self, sample_instance: BenchmarkInstance
    ) -> None:
        """
        Tests a successful trace where all relevant steps passed.
        """
        trace = ExecutionTrace(
            steps=[
                CommandOutput(command="git apply --check", exit_code=0),
                CommandOutput(command="git apply -p0", exit_code=0),
                CommandOutput(
                    command="npx ng version --json",
                    exit_code=0,
                    stdout=json.dumps({"@angular/core": "16.2.1"}),
                ),
            ]
        )
        metrics = calculate_metrics(trace, sample_instance)
        assert metrics.patch_application_success is True
        assert metrics.target_version_achieved is True

    def test_partial_failure_run_is_captured(
        self, sample_instance: BenchmarkInstance
    ) -> None:
        """
        Tests a trace where the patch succeeded but the version is wrong.
        """
        trace = ExecutionTrace(
            steps=[
                CommandOutput(command="git apply --check", exit_code=0),
                CommandOutput(command="git apply -p0", exit_code=0),
                CommandOutput(
                    command="npx ng version --json",
                    exit_code=0,
                    stdout='{"@angular/core": "15.0.0"}',  # Wrong version
                ),
            ]
        )
        metrics = calculate_metrics(trace, sample_instance)
        assert metrics.patch_application_success is True
        assert metrics.target_version_achieved is False

    def test_empty_trace_returns_all_none(
        self, sample_instance: BenchmarkInstance
    ) -> None:
        """
        Tests that an empty trace results in all metrics being None.
        """
        trace = ExecutionTrace(steps=[])
        metrics = calculate_metrics(trace, sample_instance)
        assert metrics.patch_application_success is None
        assert metrics.target_version_achieved is None
