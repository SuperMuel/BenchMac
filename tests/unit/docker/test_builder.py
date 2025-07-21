"""Tests for bench_mac.docker.builder module."""

import pytest

from bench_mac.docker.builder import _get_instance_image_tag
from bench_mac.models import BenchmarkInstance

# Default commands for testing
DEFAULT_COMMANDS = {
    "install": "npm install",
    "build": "npm run build",
    "lint": "npm run lint",
    "test": "npm test",
}


@pytest.mark.unit
class TestGetInstanceImageTag:
    """Test cases for the get_instance_image_tag function."""

    def test_basic_functionality(self) -> None:
        """Test basic tag generation functionality."""
        # Create a minimal valid instance
        instance = BenchmarkInstance.model_validate(
            {
                "instance_id": "test-instance",
                "repo": "user/repo",
                "base_commit": "abcdef123456789a",
                "source_angular_version": "15.0.0",
                "target_angular_version": "16.1.0",
                "target_node_version": "18.13.0",
                "commands": DEFAULT_COMMANDS,
            }
        )

        tag = _get_instance_image_tag(instance)

        # Should be in format: benchmac-instance:user__repo__abcdef123456789a
        assert tag == "benchmac-instance:user__repo__abcdef123456789a"

    def test_short_commit_hash(self) -> None:
        """Test handling of short commit hashes."""
        instance = BenchmarkInstance.model_validate(
            {
                "instance_id": "test-instance",
                "repo": "user/repo",
                "base_commit": "a1b2c3d",
                "source_angular_version": "15.0.0",
                "target_angular_version": "16.1.0",
                "target_node_version": "18.13.0",
                "commands": DEFAULT_COMMANDS,
            }
        )

        tag = _get_instance_image_tag(instance)

        # Should handle short commit hash properly
        assert tag == "benchmac-instance:user__repo__a1b2c3d"

    @pytest.mark.parametrize(
        "repo,commit,expected_suffix",
        [
            ("owner/repo", "abc123def45", "owner__repo__abc123def45"),
            ("test/project-name", "def456789ab", "test__project-name__def456789ab"),
            (
                "https://github.com/angular/angular-cli.git",
                "abcdef123456789a",
                "gh__angular__angular-cli__abcdef123456789a",
            ),
        ],
    )
    def test_various_repo_commit_combinations(
        self, repo: str, commit: str, expected_suffix: str
    ) -> None:
        """Test various combinations of repository names and commit hashes."""
        instance = BenchmarkInstance.model_validate(
            {
                "instance_id": "test-instance",
                "repo": repo,
                "base_commit": commit,
                "source_angular_version": "15.0.0",
                "target_angular_version": "16.1.0",
                "target_node_version": "18.13.0",
                "commands": DEFAULT_COMMANDS,
            }
        )

        tag = _get_instance_image_tag(instance)

        assert tag == f"benchmac-instance:{expected_suffix}"

    def test_tag_length_exceeds_limit_raises_error(self) -> None:
        """Test that a tag longer than 128 characters raises a ValueError."""
        # Create an instance that will generate a very long tag
        very_long_repo = "https://github.com/very-long-organization-name/extremely-long-repository-name-that-goes-on-and-on.git"
        very_long_commit = "a" * 40  # 40-character commit hash

        instance = BenchmarkInstance.model_validate(
            {
                "instance_id": "test-instance-with-very-long-name",
                "repo": very_long_repo,
                "base_commit": very_long_commit,
                "source_angular_version": "15.0.0",
                "target_angular_version": "16.1.0",
                "target_node_version": "18.13.0",
                "commands": DEFAULT_COMMANDS,
            }
        )

        with pytest.raises(ValueError, match="Generated image tag is too long"):
            _get_instance_image_tag(instance)
