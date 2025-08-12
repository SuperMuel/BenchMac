"""
Utilities for working with Dockerfiles in the Docker strategy refactor.

This module provides functions for deriving Dockerfile paths from instance IDs
using the convention-based approach: dockerfiles/{instance_id}
"""

from pathlib import Path


def get_dockerfile_path(instance_id: str, base_dir: str | Path = "data") -> Path:
    """
    Get the Dockerfile path for a given instance ID.

    Args:
        instance_id: The unique identifier for the benchmark instance
        base_dir: The base directory containing the dockerfiles
        directory (default: "data")

    Returns:
        Path to the Dockerfile for the given instance

    Example:
        >>> get_dockerfile_path("test-instance-v15-to-v16")
        PosixPath('data/dockerfiles/test-instance-v15-to-v16')

        >>> get_dockerfile_path("test-instance-v15-to-v16", "tests/fixtures")
        PosixPath('tests/fixtures/dockerfiles/test-instance-v15-to-v16')
    """
    base_path = Path(base_dir)
    return base_path / "dockerfiles" / instance_id


def dockerfile_exists(instance_id: str, base_dir: str | Path = "data") -> bool:
    """
    Check if a Dockerfile exists for the given instance ID.

    Args:
        instance_id: The unique identifier for the benchmark instance
        base_dir: The base directory containing the dockerfiles directory
        (default: "data")

    Returns:
        True if the Dockerfile exists, False otherwise

    Example:
        >>> dockerfile_exists("gothinkster__angular-realworld-example-app_v11_to_v12")
        True

        >>> dockerfile_exists("nonexistent-instance")
        False
    """
    dockerfile_path = get_dockerfile_path(instance_id, base_dir)
    return dockerfile_path.exists()


def validate_dockerfile_exists(instance_id: str, base_dir: str | Path = "data") -> Path:
    """
    Get the Dockerfile path for a given instance ID and validate it exists.

    Args:
        instance_id: The unique identifier for the benchmark instance
        base_dir: The base directory containing the dockerfiles directory
        (default: "data")

    Returns:
        Path to the Dockerfile for the given instance

    Raises:
        FileNotFoundError: If the Dockerfile does not exist

    Example:
        >>> validate_dockerfile_exists("test-instance-v15-to-v16")
        PosixPath('data/dockerfiles/test-instance-v15-to-v16')

        >>> validate_dockerfile_exists("nonexistent-instance")
        FileNotFoundError: Dockerfile not found for instance 'nonexistent-instance'
        at data/dockerfiles/nonexistent-instance
    """
    dockerfile_path = get_dockerfile_path(instance_id, base_dir)

    if not dockerfile_path.exists():
        raise FileNotFoundError(
            f"Dockerfile not found for instance '{instance_id}' at {dockerfile_path}"
        )

    return dockerfile_path
