"""
Unit tests for the metrics calculation module.
"""

from datetime import UTC, datetime

import pytest

from bench_mac.metrics import calculate_metrics
from bench_mac.models import CommandOutput, ExecutionTrace


@pytest.mark.unit
class TestCalculateMetrics:
    """Tests for the calculate_metrics function."""

    def test_successful_patch_application(self) -> None:
        """Test metrics calculation for successful patch application."""
        # Create a successful execution trace
        patch_check = CommandOutput(
            command="git apply --check -p0 /tmp/patch.patch",
            exit_code=0,
            stdout="",
            stderr="",
            start_time=datetime.now(UTC),
            end_time=datetime.now(UTC),
        )
        patch_apply = CommandOutput(
            command="git apply -p0 /tmp/patch.patch",
            exit_code=0,
            stdout="Applied patch successfully",
            stderr="",
            start_time=datetime.now(UTC),
            end_time=datetime.now(UTC),
        )

        trace = ExecutionTrace(steps=[patch_check, patch_apply])

        # Calculate metrics
        metrics = calculate_metrics(trace)

        # Assert successful patch application
        assert metrics.patch_application_success is True

    def test_failed_patch_check(self) -> None:
        """Test metrics calculation when patch check fails."""
        patch_check = CommandOutput(
            command="git apply --check -p0 /tmp/patch.patch",
            exit_code=1,
            stdout="",
            stderr="patch does not apply",
            start_time=datetime.now(UTC),
            end_time=datetime.now(UTC),
        )

        trace = ExecutionTrace(steps=[patch_check])

        # Calculate metrics
        metrics = calculate_metrics(trace)

        # Assert failed patch application
        assert metrics.patch_application_success is False

    def test_failed_patch_apply(self) -> None:
        """Test metrics calculation when patch check succeeds but apply fails."""
        patch_check = CommandOutput(
            command="git apply --check -p0 /tmp/patch.patch",
            exit_code=0,
            stdout="",
            stderr="",
            start_time=datetime.now(UTC),
            end_time=datetime.now(UTC),
        )
        patch_apply = CommandOutput(
            command="git apply -p0 /tmp/patch.patch",
            exit_code=1,
            stdout="",
            stderr="apply failed",
            start_time=datetime.now(UTC),
            end_time=datetime.now(UTC),
        )

        trace = ExecutionTrace(steps=[patch_check, patch_apply])

        # Calculate metrics
        metrics = calculate_metrics(trace)

        # Assert failed patch application
        assert metrics.patch_application_success is False

    def test_empty_trace(self) -> None:
        """Test metrics calculation with empty execution trace."""
        trace = ExecutionTrace(steps=[])

        # Calculate metrics
        metrics = calculate_metrics(trace)

        # No step probably means harness error, so we don't know.
        assert metrics.patch_application_success is None

    def test_legacy_patch_apply_only(self) -> None:
        """Test metrics calculation with only patch apply step."""
        patch_apply = CommandOutput(
            command="git apply -p0 /tmp/patch.patch",
            exit_code=0,
            stdout="Applied patch successfully",
            stderr="",
            start_time=datetime.now(UTC),
            end_time=datetime.now(UTC),
        )

        trace = ExecutionTrace(steps=[patch_apply])

        # Calculate metrics
        metrics = calculate_metrics(trace)

        # Assert successful patch application
        assert metrics.patch_application_success is True

    def test_no_patch_steps(self) -> None:
        """Test metrics calculation when no patch-related steps are found."""
        other_command = CommandOutput(
            command="npm install",
            exit_code=0,
            stdout="Installed packages",
            stderr="",
            start_time=datetime.now(UTC),
            end_time=datetime.now(UTC),
        )

        trace = ExecutionTrace(steps=[other_command])

        # Calculate metrics
        metrics = calculate_metrics(trace)

        # No patch steps found, so we don't know.
        assert metrics.patch_application_success is None
