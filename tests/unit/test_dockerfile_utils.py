"""Tests for dockerfile_utils module."""

from pathlib import Path

import pytest

from bench_mac.dockerfile_utils import (
    dockerfile_exists,
    get_dockerfile_path,
    validate_dockerfile_exists,
)


class TestDockerfileUtils:
    """Test cases for Dockerfile utility functions."""

    def test_get_dockerfile_path_default_base_dir(self):
        """Test getting Dockerfile path with default base directory."""
        instance_id = "test-instance-v15-to-v16"
        expected_path = Path("data/dockerfiles/test-instance-v15-to-v16")

        result = get_dockerfile_path(instance_id)

        assert result == expected_path

    def test_get_dockerfile_path_custom_base_dir(self):
        """Test getting Dockerfile path with custom base directory."""
        instance_id = "test-instance-v15-to-v16"
        base_dir = "tests/fixtures"
        expected_path = Path("tests/fixtures/dockerfiles/test-instance-v15-to-v16")

        result = get_dockerfile_path(instance_id, base_dir)

        assert result == expected_path

    def test_get_dockerfile_path_with_path_object(self):
        """Test getting Dockerfile path with Path object as base directory."""
        instance_id = "test-instance-v15-to-v16"
        base_dir = Path("tests/fixtures")
        expected_path = Path("tests/fixtures/dockerfiles/test-instance-v15-to-v16")

        result = get_dockerfile_path(instance_id, base_dir)

        assert result == expected_path

    def test_dockerfile_exists_true(self):
        """Test dockerfile_exists returns True for existing Dockerfile."""
        instance_id = "test-instance-v15-to-v16"
        base_dir = "tests/fixtures"

        result = dockerfile_exists(instance_id, base_dir)

        assert result is True

    def test_dockerfile_exists_false(self):
        """Test dockerfile_exists returns False for non-existing Dockerfile."""
        instance_id = "nonexistent-instance"
        base_dir = "tests/fixtures"

        result = dockerfile_exists(instance_id, base_dir)

        assert result is False

    def test_validate_dockerfile_exists_success(self):
        """Test validate_dockerfile_exists returns path for existing Dockerfile."""
        instance_id = "test-instance-v15-to-v16"
        base_dir = "tests/fixtures"
        expected_path = Path("tests/fixtures/dockerfiles/test-instance-v15-to-v16")

        result = validate_dockerfile_exists(instance_id, base_dir)

        assert result == expected_path

    def test_validate_dockerfile_exists_raises_error(self):
        """Test validate_dockerfile_exists raises FileNotFoundError
        for non-existing Dockerfile."""
        instance_id = "nonexistent-instance"
        base_dir = "tests/fixtures"

        with pytest.raises(FileNotFoundError) as exc_info:
            validate_dockerfile_exists(instance_id, base_dir)

        assert "Dockerfile not found for instance 'nonexistent-instance'" in str(
            exc_info.value
        )
        assert "tests/fixtures/dockerfiles/nonexistent-instance" in str(exc_info.value)

    def test_production_data_dockerfile_exists(self):
        """Test that production Dockerfiles exist using the utility function."""
        # Test with a known production instance
        instance_id = "gothinkster__angular-realworld-example-app_v11_to_v12"

        result = dockerfile_exists(instance_id, "data")

        assert result is True

    def test_production_data_validate_dockerfile(self):
        """Test validating production Dockerfile exists."""
        # Test with a known production instance
        instance_id = "gothinkster__angular-realworld-example-app_v11_to_v12"
        expected_path = Path(
            "data/dockerfiles/gothinkster__angular-realworld-example-app_v11_to_v12"
        )

        result = validate_dockerfile_exists(instance_id, "data")

        assert result == expected_path
