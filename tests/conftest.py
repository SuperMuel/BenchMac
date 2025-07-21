"""Shared test fixtures and utilities."""

import pytest

from bench_mac.models import BenchmarkInstance, CommandsConfig, Submission


@pytest.fixture
def default_commands() -> CommandsConfig:
    """Provides a standard CommandsConfig for testing."""
    return CommandsConfig(
        install="npm install",
        build="ng build --prod",
        lint="ng lint",
        test="ng test --watch=false --browsers=ChromeHeadless",
    )


@pytest.fixture
def sample_instance(default_commands: CommandsConfig) -> BenchmarkInstance:
    """Provides a standard BenchmarkInstance for testing."""
    return BenchmarkInstance(
        instance_id="my-project_v15_to_v16",
        repo="SuperMuel/BenchMAC",
        base_commit="a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0",
        source_angular_version="15.0.0",
        target_angular_version="16.1.0",
        target_node_version="18.13.0",
        commands=default_commands,
    )


class InstanceFactory:
    """Factory for creating test instances with custom parameters."""

    def __init__(self, default_commands: CommandsConfig):
        self.default_commands = default_commands

    def create_instance(
        self,
        instance_id: str = "test-instance",
        repo: str = "owner/repo",
        base_commit: str = "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0",
        source_angular_version: str = "15.0.0",
        target_angular_version: str = "16.1.0",
        target_node_version: str = "18.13.0",
        commands: CommandsConfig | None = None,
        metadata: dict[str, str] | None = None,
    ) -> BenchmarkInstance:
        """Create a BenchmarkInstance with custom parameters."""
        return BenchmarkInstance(
            instance_id=instance_id,
            repo=repo,
            base_commit=base_commit,
            source_angular_version=source_angular_version,
            target_angular_version=target_angular_version,
            target_node_version=target_node_version,
            commands=commands or self.default_commands,
            metadata=metadata or {},
        )


@pytest.fixture
def instance_factory(default_commands: CommandsConfig) -> InstanceFactory:
    """Factory fixture for creating custom BenchmarkInstance objects."""
    return InstanceFactory(default_commands)


@pytest.fixture
def sample_submission() -> Submission:
    """Provides a standard Submission for testing."""
    return Submission(
        instance_id="my-project_v15_to_v16",
        model_patch="diff --git a/file.ts b/file.ts\n--- a/file.ts\n+++ b/file.ts",
    )
