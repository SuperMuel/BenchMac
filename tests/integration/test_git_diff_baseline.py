"""Integration test ensuring `InstanceEnvironment.diff_from_baseline` captures all
edits."""

from __future__ import annotations

import shlex

import pytest

from bench_mac.config import settings
from bench_mac.docker.manager import DockerManager
from bench_mac.environment import InstanceEnvironment
from bench_mac.utils import load_instances


@pytest.mark.integration
@pytest.mark.slow
def test_git_diff_baseline_captures_all_changes(docker_manager: DockerManager) -> None:
    """Validate that committed, staged, and unstaged edits appear in the baseline
    diff."""

    instances = load_instances(settings.instances_file, strict=True)
    assert instances, "Expected at least one benchmark instance for the test."

    instance = next(iter(instances.values()))

    with InstanceEnvironment(instance, docker_manager, auto_remove=True) as env:
        tracked_file = env.exec('bash -lc "git ls-files | head -n 1"').stdout.strip()
        assert tracked_file, "Unable to determine an existing tracked file to modify."

        # Safely quote the filename so it can be used in a shell command,
        # even if it contains spaces or special characters.
        tracked_file_quoted = shlex.quote(tracked_file)

        env.exec("bash -lc \"printf 'committed change\\n' > committed_change.txt\"")
        env.exec("git add committed_change.txt")
        env.exec("git commit -m 'test: committed change for diff validation'")

        env.exec("bash -lc \"printf 'staged change\\n' > staged_change.txt\"")
        env.exec("git add staged_change.txt")

        env.exec(f"bash -lc \"printf 'unstaged change\\n' >> {tracked_file_quoted}\"")

        diff_output = env.diff_from_baseline().stdout

        assert "committed_change.txt" in diff_output
        assert "staged_change.txt" in diff_output
        assert tracked_file in diff_output
