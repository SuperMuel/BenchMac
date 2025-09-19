"""
Unit tests for the metrics calculation module.
"""

import json

import pytest

from bench_mac.metrics import (
    _calculate_patch_application_success,
    calculate_metrics,
    calculate_target_version_achieved,
)
from bench_mac.models import (
    BenchmarkInstance,
    ExecutionTrace,
    InstanceCommands,
    InstanceID,
)

from ..utils import create_command_output


@pytest.mark.unit
class TestCalculatePatchApplicationSuccess:
    """Unit tests for the _calculate_patch_application_success helper function."""

    def test_returns_true_when_both_steps_succeed(self) -> None:
        patch_check = create_command_output(command="git apply --check", exit_code=0)
        patch_apply = create_command_output(command="git apply -p0", exit_code=0)
        assert _calculate_patch_application_success(patch_check, patch_apply) is True

    def test_returns_false_when_check_fails(self) -> None:
        patch_check = create_command_output(command="git apply --check", exit_code=1)
        patch_apply = create_command_output(command="git apply -p0", exit_code=0)
        assert _calculate_patch_application_success(patch_check, patch_apply) is False

    def test_returns_false_when_apply_fails(self) -> None:
        patch_check = create_command_output(command="git apply --check", exit_code=0)
        patch_apply = create_command_output(command="git apply -p0", exit_code=1)
        assert _calculate_patch_application_success(patch_check, patch_apply) is False

    def test_returns_false_when_only_failed_check_is_present(self) -> None:
        patch_check = create_command_output(command="git apply --check", exit_code=1)
        assert _calculate_patch_application_success(patch_check, None) is False

    def test_returns_true_when_only_successful_apply_is_present(self) -> None:
        patch_apply = create_command_output(command="git apply -p0", exit_code=0)
        assert _calculate_patch_application_success(None, patch_apply) is True

    def test_returns_none_when_no_steps_are_present(self) -> None:
        assert _calculate_patch_application_success(None, None) is None


@pytest.mark.unit
class TestCalculateTargetVersionAchieved:
    """
    Unit tests for the _calculate_target_version_achieved helper function using
    `npm ls`.
    """

    NPM_LS_COMMAND = "npm ls @angular/cli @angular/core --json"

    def test_returns_true_on_major_version_match(self) -> None:
        version_output = {"dependencies": {"@angular/core": {"version": "16.1.5"}}}
        version_step = create_command_output(
            command=self.NPM_LS_COMMAND,
            exit_code=0,
            stdout=json.dumps(version_output),
        )
        assert calculate_target_version_achieved(version_step, "16") is True

    def test_returns_true_even_if_exit_code_is_1_but_json_is_valid(self) -> None:
        # This is the critical test case reflecting reality.
        version_output = {"dependencies": {"@angular/core": {"version": "13.3.12"}}}
        version_step = create_command_output(
            command=self.NPM_LS_COMMAND,
            exit_code=1,  # Command failed due to peer deps
            stdout=json.dumps(version_output),
            stderr="npm ERR! peer dep missing...",
        )
        assert calculate_target_version_achieved(version_step, "13.1.0") is True

    def test_returns_false_on_major_version_mismatch(self) -> None:
        version_output = {"dependencies": {"@angular/core": {"version": "15.2.9"}}}
        version_step = create_command_output(
            command=self.NPM_LS_COMMAND, exit_code=0, stdout=json.dumps(version_output)
        )
        assert calculate_target_version_achieved(version_step, "16") is False

    def test_returns_false_if_angular_core_key_is_missing(self) -> None:
        version_output = {"dependencies": {"@angular/cli": {"version": "16.0.0"}}}
        version_step = create_command_output(
            command=self.NPM_LS_COMMAND, exit_code=0, stdout=json.dumps(version_output)
        )
        assert calculate_target_version_achieved(version_step, "16") is False

    def test_returns_false_if_dependencies_key_is_missing(self) -> None:
        version_output = {"name": "my-project"}  # Missing top-level 'dependencies'
        version_step = create_command_output(
            command=self.NPM_LS_COMMAND, exit_code=0, stdout=json.dumps(version_output)
        )
        assert calculate_target_version_achieved(version_step, "16") is False

    def test_returns_none_if_json_is_invalid(self) -> None:
        version_step = create_command_output(
            command=self.NPM_LS_COMMAND, exit_code=0, stdout="this is not json"
        )
        assert calculate_target_version_achieved(version_step, "16") is None

    def test_returns_none_if_step_is_missing(self) -> None:
        assert calculate_target_version_achieved(None, "16") is None

    def test_returns_none_if_stdout_is_empty(self) -> None:
        version_step = create_command_output(
            command=self.NPM_LS_COMMAND, exit_code=1, stdout=""
        )
        assert calculate_target_version_achieved(version_step, "16") is None

    def test_returns_none_if_version_string_is_malformed(self) -> None:
        version_output = {
            "dependencies": {"@angular/core": {"version": "not-a-version"}}
        }
        version_step = create_command_output(
            command=self.NPM_LS_COMMAND, exit_code=0, stdout=json.dumps(version_output)
        )
        assert calculate_target_version_achieved(version_step, "16") is None


@pytest.fixture
def sample_instance() -> BenchmarkInstance:
    """Provides a standard BenchmarkInstance for testing."""
    return BenchmarkInstance(
        instance_id=InstanceID("test-project_v15_to_v16"),
        repo="owner/repo",
        base_commit="a" * 40,
        source_angular_version="15.2.0",
        target_angular_version="16.0.0",
        override_dockerfile_content="FROM node:18",  # Dummy content
        commands=InstanceCommands(
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

    VERSION_CHECK_COMMAND = "npm ls @angular/cli @angular/core --json"

    def test_successful_run_calculates_all_metrics_correctly(
        self, sample_instance: BenchmarkInstance
    ) -> None:
        """Tests a successful trace where all relevant steps passed."""
        trace = ExecutionTrace(
            steps=[
                create_command_output(command="git apply --check", exit_code=0),
                create_command_output(command="git apply -p0", exit_code=0),
                create_command_output(
                    command="npm ci", exit_code=0
                ),  # Successful install
                create_command_output(
                    command=self.VERSION_CHECK_COMMAND,
                    exit_code=0,
                    stdout=json.dumps(
                        {"dependencies": {"@angular/core": {"version": "16.2.1"}}}
                    ),
                ),
                create_command_output(
                    command="ng build", exit_code=0
                ),  # Successful build
            ]
        )
        metrics = calculate_metrics(trace, sample_instance)
        assert metrics.patch_application_success is True
        assert metrics.install_success is True
        assert metrics.target_version_achieved is True
        assert metrics.build_success is True

    def test_partial_failure_run_is_captured(
        self, sample_instance: BenchmarkInstance
    ) -> None:
        """Tests a trace where build fails but other steps succeed."""
        trace = ExecutionTrace(
            steps=[
                create_command_output(command="git apply --check", exit_code=0),
                create_command_output(command="git apply -p0", exit_code=0),
                create_command_output(command="npm ci", exit_code=0),
                create_command_output(
                    command=self.VERSION_CHECK_COMMAND,
                    exit_code=0,
                    stdout=json.dumps(
                        {"dependencies": {"@angular/core": {"version": "16.0.0"}}}
                    ),
                ),
                create_command_output(
                    command="ng build", exit_code=1, stderr="Build failed!"
                ),  # Failed build
            ]
        )
        metrics = calculate_metrics(trace, sample_instance)
        assert metrics.patch_application_success is True
        assert metrics.install_success is True
        assert metrics.target_version_achieved is True
        assert metrics.build_success is False

    def test_empty_trace_returns_all_none(
        self, sample_instance: BenchmarkInstance
    ) -> None:
        """Tests that an empty trace results in all metrics being None."""
        trace = ExecutionTrace(steps=[])
        metrics = calculate_metrics(trace, sample_instance)
        assert metrics.patch_application_success is None
        assert metrics.install_success is None
        assert metrics.target_version_achieved is None
        assert metrics.build_success is None
