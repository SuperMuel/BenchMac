"""
Shared pytest fixtures for integration tests.

This module provides common fixtures used across multiple integration test modules,
particularly for Docker-related functionality.
"""

import pytest
from docker.errors import DockerException

from bench_mac.docker.manager import DockerManager


@pytest.fixture(scope="module")
def docker_manager() -> DockerManager:
    """
    Provides a DockerManager instance for integration tests.

    Skips all tests that require this fixture if the Docker daemon is not running.
    This fixture is shared across all integration test modules.
    """
    try:
        manager = DockerManager()
        return manager
    except DockerException:
        pytest.skip("Docker daemon is not running. Skipping integration tests.")
