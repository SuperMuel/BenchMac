"""Tests for the trace analysis helpers."""

from __future__ import annotations

import pytest

from bench_mac.models import (
    BenchmarkInstance,
    ExecutionTrace,
    InstanceCommands,
    InstanceID,
)
from bench_mac.trace_analyzer import TraceAnalyzer

from ..utils import create_command_output


@pytest.fixture
def sample_instance() -> BenchmarkInstance:
    return BenchmarkInstance(
        instance_id=InstanceID("test-instance"),
        repo="owner/repo",
        base_commit="a" * 40,
        source_angular_version="15.2.0",
        target_angular_version="16.0.0",
        override_dockerfile_content="FROM node:18",
        commands=InstanceCommands(install="npm ci", build="ng build"),
    )


@pytest.mark.unit
class TestTraceAnalyzer:
    def test_detects_patch_steps(self, sample_instance: BenchmarkInstance) -> None:
        trace = ExecutionTrace(
            steps=[
                create_command_output(
                    command="git apply --check -p0 /tmp/submission.patch",
                    exit_code=0,
                ),
                create_command_output(
                    command="git apply -p0 /tmp/submission.patch",
                    exit_code=0,
                ),
            ]
        )
        analyzer = TraceAnalyzer(trace, sample_instance)

        patch_check = analyzer.patch_check_step()
        patch_apply = analyzer.patch_apply_step()

        assert patch_check is not None
        assert patch_check.command.startswith("git apply --check")
        assert patch_apply is not None
        assert patch_apply.command.startswith("git apply -p0")

    def test_collects_all_install_variants(
        self, sample_instance: BenchmarkInstance
    ) -> None:
        trace = ExecutionTrace(
            steps=[
                create_command_output(command="npm ci", exit_code=1),
                create_command_output(command="npm ci --legacy-peer-deps", exit_code=0),
            ]
        )
        analyzer = TraceAnalyzer(trace, sample_instance)

        installs = analyzer.install_steps()
        final_install = analyzer.final_install_step()

        assert len(installs) == 2
        assert installs[-1].command.endswith("--legacy-peer-deps")
        assert final_install is installs[-1]

    def test_detects_version_and_build_steps(
        self, sample_instance: BenchmarkInstance
    ) -> None:
        trace = ExecutionTrace(
            steps=[
                create_command_output(
                    command="npm ls @angular/cli @angular/core --json",
                    exit_code=0,
                ),
                create_command_output(
                    command="ng build --configuration=prod", exit_code=0
                ),
            ]
        )
        analyzer = TraceAnalyzer(trace, sample_instance)

        version_step = analyzer.version_check_step()
        build_step = analyzer.build_step()

        assert version_step is not None
        assert version_step.command.startswith(
            "npm ls @angular/cli @angular/core --json"
        )
        assert build_step is not None
        assert build_step.command.startswith("ng build")
