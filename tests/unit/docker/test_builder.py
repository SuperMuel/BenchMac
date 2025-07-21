from typing import Any

import pytest

from bench_mac.docker.builder import (
    _get_environment_image_tag,
    _get_instance_image_tag,
    _normalize_repo_url,
)


@pytest.mark.unit
class TestGetEnvironmentImageTag:
    """Tests for the _get_environment_image_tag function."""

    def test_basic_version_formatting(self) -> None:
        """Test that versions are correctly formatted in the tag."""
        node_version = "18.13.0"
        angular_cli_version = "16"

        result = _get_environment_image_tag(node_version, angular_cli_version)

        assert result == "benchmac-env:node18-13-0-ng16"

    def test_version_sanitization_dots_to_dashes(self) -> None:
        """Test that dots in versions are converted to dashes."""
        node_version = "20.5.1"
        angular_cli_version = "17.2.3"

        result = _get_environment_image_tag(node_version, angular_cli_version)

        assert result == "benchmac-env:node20-5-1-ng17-2-3"

    def test_single_digit_versions(self) -> None:
        """Test handling of single digit versions."""
        node_version = "18"
        angular_cli_version = "16"

        result = _get_environment_image_tag(node_version, angular_cli_version)

        assert result == "benchmac-env:node18-ng16"

    def test_partial_versions(self) -> None:
        """Test handling of partial versions (major.minor)."""
        node_version = "18.13"
        angular_cli_version = "16.1"

        result = _get_environment_image_tag(node_version, angular_cli_version)

        assert result == "benchmac-env:node18-13-ng16-1"

    @pytest.mark.parametrize(
        "node_version,angular_cli_version,expected",
        [
            ("16", "14", "benchmac-env:node16-ng14"),
            ("18.0", "15.0", "benchmac-env:node18-0-ng15-0"),
            ("20.1.2", "17.3.4", "benchmac-env:node20-1-2-ng17-3-4"),
        ],
    )
    def test_various_version_combinations(
        self, node_version: str, angular_cli_version: str, expected: str
    ) -> None:
        """Test various combinations of node and angular versions."""
        result = _get_environment_image_tag(node_version, angular_cli_version)
        assert result == expected

    def test_empty_node_version_raises_error(self) -> None:
        """Test that empty node version raises a ValueError."""
        with pytest.raises(ValueError, match="Node version is required"):
            _get_environment_image_tag("", "16")

    def test_empty_angular_cli_version_raises_error(self) -> None:
        """Test that empty Angular CLI version raises a ValueError."""
        with pytest.raises(ValueError, match="Angular CLI version is required"):
            _get_environment_image_tag("18.13.0", "")


@pytest.mark.unit
class TestGetInstanceImageTag:
    """Tests for the _get_instance_image_tag function."""

    def test_short_commit_hash(self, instance_factory: Any) -> None:
        """Test handling of short commit hashes."""
        instance = instance_factory.create_instance(
            instance_id="test-instance",
            repo="user/repo",
            base_commit="a1b2c3d",
        )

        result = _get_instance_image_tag(instance)

        assert result == "benchmac-instance:user__repo__a1b2c3d"

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
        self, repo: str, commit: str, expected_suffix: str, instance_factory: Any
    ) -> None:
        """Test various combinations of repository names and commit hashes."""
        instance = instance_factory.create_instance(
            instance_id="test-instance",
            repo=repo,
            base_commit=commit,
        )

        result = _get_instance_image_tag(instance)

        assert result == f"benchmac-instance:{expected_suffix}"

    def test_tag_length_exceeds_limit_raises_error(self, instance_factory: Any) -> None:
        """Test that a tag longer than 128 characters raises a ValueError."""
        # Create an instance that will generate a very long tag
        very_long_repo = "https://github.com/very-long-organization-name/extremely-long-repository-name-that-goes-on-and-on.git"
        very_long_commit = "a" * 40  # 40-character commit hash

        instance = instance_factory.create_instance(
            instance_id="test-instance-with-very-long-name",
            repo=very_long_repo,
            base_commit=very_long_commit,
        )

        with pytest.raises(ValueError, match="Generated image tag is too long"):
            _get_instance_image_tag(instance)


@pytest.mark.unit
class TestNormalizeRepoUrl:
    """Tests for the _normalize_repo_url function."""

    def test_https_url_returned_as_is(self) -> None:
        """Test that HTTPS URLs are returned unchanged."""
        url = "https://github.com/angular/angular-cli.git"
        result = _normalize_repo_url(url)
        assert result == url

    def test_http_url_returned_as_is(self) -> None:
        """Test that HTTP URLs are returned unchanged."""
        url = "http://github.com/user/repo.git"
        result = _normalize_repo_url(url)
        assert result == url

    def test_git_ssh_url_returned_as_is(self) -> None:
        """Test that git SSH URLs are returned unchanged."""
        url = "git@github.com:user/repo.git"
        result = _normalize_repo_url(url)
        assert result == url

    def test_git_protocol_url_returned_as_is(self) -> None:
        """Test that git:// URLs are returned unchanged."""
        url = "git://github.com/user/repo.git"
        result = _normalize_repo_url(url)
        assert result == url

    def test_owner_repo_format_converted_to_github_url(self) -> None:
        """Test that owner/repo format is converted to GitHub HTTPS URL."""
        repo_ref = "SuperMuel/angular2-hn"
        result = _normalize_repo_url(repo_ref)
        assert result == "https://github.com/SuperMuel/angular2-hn.git"

    def test_owner_repo_with_underscores_and_dashes(self) -> None:
        """Test owner/repo format with underscores and dashes in names."""
        repo_ref = "my-org_name/my-repo_name"
        result = _normalize_repo_url(repo_ref)
        assert result == "https://github.com/my-org_name/my-repo_name.git"

    def test_owner_repo_with_dots(self) -> None:
        """Test owner/repo format with dots in names."""
        repo_ref = "org.name/repo.name"
        result = _normalize_repo_url(repo_ref)
        assert result == "https://github.com/org.name/repo.name.git"

    def test_empty_string_raises_error(self) -> None:
        """Test that empty string raises ValueError."""
        with pytest.raises(ValueError, match="Repository URL cannot be empty"):
            _normalize_repo_url("")

    def test_whitespace_only_raises_error(self) -> None:
        """Test that whitespace-only string raises ValueError."""
        with pytest.raises(ValueError, match="Repository URL cannot be empty"):
            _normalize_repo_url("   ")

    def test_invalid_format_raises_error(self) -> None:
        """Test that invalid format raises ValueError."""
        with pytest.raises(
            ValueError,
            match="Invalid repository reference.*Expected a git-clonable URL",
        ):
            _normalize_repo_url("not/a/valid/format")

    def test_single_word_raises_error(self) -> None:
        """Test that single word (no slash) raises ValueError."""
        with pytest.raises(
            ValueError,
            match="Invalid repository reference.*Expected a git-clonable URL",
        ):
            _normalize_repo_url("justarepo")

    def test_whitespace_trimmed(self) -> None:
        """Test that whitespace is trimmed from input."""
        repo_ref = "  user/repo  "
        result = _normalize_repo_url(repo_ref)
        assert result == "https://github.com/user/repo.git"
