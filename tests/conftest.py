"""Shared test fixtures and utilities."""

from typing import cast

import pytest

from bench_mac.models import BenchmarkInstance, InstanceCommands, InstanceID, Submission

_UNSET = object()


@pytest.fixture
def default_commands() -> InstanceCommands:
    """Provides a standard InstanceCommands for testing."""
    return InstanceCommands(
        install="npm install",
        build="ng build --configuration production",
    )


@pytest.fixture
def default_dockerfile_content() -> str:
    """Provides a standard Dockerfile content for testing."""
    return "FROM node:18\n"


@pytest.fixture
def sample_instance(default_commands: InstanceCommands) -> BenchmarkInstance:
    """Provides a standard BenchmarkInstance for testing."""
    return BenchmarkInstance(
        instance_id=InstanceID("my-project_v15_to_v16"),
        repo="SuperMuel/BenchMAC",
        base_commit="a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0",
        source_angular_version="15.0.0",
        target_angular_version="16.1.0",
        commands=default_commands,
    )


class InstanceFactory:
    """Factory for creating test instances with custom parameters."""

    def __init__(
        self, default_commands: InstanceCommands, default_dockerfile_content: str
    ):
        self.default_commands = default_commands
        self.default_dockerfile_content = default_dockerfile_content

    def create_instance(
        self,
        instance_id: InstanceID = InstanceID("test-instance"),  # noqa: B008
        repo: str = "owner/repo",
        base_commit: str = "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0",
        source_angular_version: str = "15.0.0",
        target_angular_version: str = "16.1.0",
        commands: InstanceCommands | None = None,
        metadata: dict[str, str] | None = None,
        override_dockerfile_content: str | None | object = _UNSET,
    ) -> BenchmarkInstance:
        """Create a BenchmarkInstance with custom parameters."""

        if override_dockerfile_content is _UNSET:
            value: str | None = self.default_dockerfile_content
        else:
            value = cast(str | None, override_dockerfile_content)

        return BenchmarkInstance(
            instance_id=instance_id,
            repo=repo,
            base_commit=base_commit,
            source_angular_version=source_angular_version,
            target_angular_version=target_angular_version,
            commands=commands or self.default_commands,
            metadata=metadata or {},
            override_dockerfile_content=value,
        )


@pytest.fixture
def instance_factory(
    default_commands: InstanceCommands,
    default_dockerfile_content: str,
) -> InstanceFactory:
    """Factory fixture for creating custom BenchmarkInstance objects."""
    return InstanceFactory(
        default_commands,
        default_dockerfile_content,
    )


@pytest.fixture
def sample_submission() -> Submission:
    """Provides a standard Submission for testing."""
    return Submission(
        instance_id=InstanceID("my-project_v15_to_v16"),
        model_patch="diff --git a/file.ts b/file.ts\n--- a/file.ts\n+++ b/file.ts",
    )
