"""
Integration tests for the entire Docker environment pipeline.

This includes low-level tests for the DockerManager and high-level tests
for the image builder.

These tests require a running Docker daemon to be present on the host machine.
They are marked as 'integration' and will be skipped by default.

To run these tests: `uv run pytest -m integration`
"""

import uuid
from collections.abc import Generator
from pathlib import Path

import pytest
from docker.errors import DockerException
from docker.models.containers import Container

from bench_mac.config import settings
from bench_mac.docker.builder import (
    _get_environment_image_tag,
    _get_instance_image_tag,
    prepare_environment,
)
from bench_mac.docker.manager import DockerManager
from bench_mac.models import BenchmarkInstance

# A simple, minimal Dockerfile for testing the manager in isolation.
SIMPLE_DOCKERFILE = 'FROM alpine:latest\nCMD ["echo", "container is running"]'


@pytest.fixture(scope="module")
def docker_manager() -> DockerManager:
    """
    Provides a DockerManager instance for the test module.

    Skips all tests in this module if the Docker daemon is not running.
    """
    try:
        manager = DockerManager()
        return manager
    except DockerException:
        pytest.skip("Docker daemon is not running. Skipping integration tests.")


@pytest.fixture
def unique_tag(docker_manager: DockerManager) -> Generator[str, None, None]:
    """
    Generates a unique Docker image tag for each test function.

    This ensures that tests do not interfere with each other and handles
    the cleanup of the image after the test has completed.
    """
    tag = f"benchmac-test-image:{uuid.uuid4().hex[:8]}"
    yield tag
    if docker_manager and docker_manager.image_exists(tag):
        print(f"\nCleaning up image: {tag}")
        docker_manager.remove_image(tag)


@pytest.mark.integration
class TestDockerManager:
    """Test suite for the low-level DockerManager class."""

    def test_initialization_succeeds(self, docker_manager: DockerManager) -> None:
        """Verify that the DockerManager can be instantiated successfully."""
        assert docker_manager is not None
        assert docker_manager._client is not None

    def test_build_image_and_cleanup(
        self, docker_manager: DockerManager, unique_tag: str
    ) -> None:
        """Test the full lifecycle of building and removing an image."""
        assert not docker_manager.image_exists(unique_tag)
        docker_manager.build_image(dockerfile_content=SIMPLE_DOCKERFILE, tag=unique_tag)
        assert docker_manager.image_exists(unique_tag)
        docker_manager.remove_image(unique_tag)
        assert not docker_manager.image_exists(unique_tag)

    def test_run_container_and_execute_streams(
        self, docker_manager: DockerManager, unique_tag: str
    ) -> None:
        """Test running a container and capturing stdout and stderr correctly."""
        docker_manager.build_image(dockerfile_content=SIMPLE_DOCKERFILE, tag=unique_tag)
        container = None
        try:
            container = docker_manager.run_container(image_tag=unique_tag)
            assert isinstance(container, Container)

            # Test stdout
            exit_code, stdout, stderr = docker_manager.execute_in_container(
                container, 'echo "hello stdout"'
            )
            assert exit_code == 0
            assert "hello stdout" in stdout
            assert stderr == ""

            # Test stderr
            exit_code, stdout, stderr = docker_manager.execute_in_container(
                container, 'sh -c "echo hello stderr >&2"'
            )
            assert exit_code == 0
            assert stdout == ""
            assert "hello stderr" in stderr

        finally:
            if container:
                docker_manager.cleanup_container(container)

    def test_copy_to_container(
        self, docker_manager: DockerManager, unique_tag: str, tmp_path: Path
    ) -> None:
        """Test copying a local file into a running container."""
        local_file = tmp_path / "test.txt"
        file_content = "hello world"
        local_file.write_text(file_content)

        docker_manager.build_image(dockerfile_content=SIMPLE_DOCKERFILE, tag=unique_tag)
        container = docker_manager.run_container(image_tag=unique_tag)
        try:
            dest_path = "/tmp/test.txt"
            docker_manager.copy_to_container(container, local_file, dest_path)
            exit_code, stdout, _ = docker_manager.execute_in_container(
                container, f"cat {dest_path}"
            )
            assert exit_code == 0
            assert file_content in stdout
        finally:
            if container:
                docker_manager.cleanup_container(container)


@pytest.mark.integration
class TestImageBuilder:
    """Test suite for the high-level image builder pipeline."""

    @pytest.fixture
    def test_instance(self) -> BenchmarkInstance:
        """Provides a standard, real-world benchmark instance for testing."""
        return BenchmarkInstance(
            instance_id="angular2-hn-v10_to_v11",
            repo="SuperMuel/angular2-hn",
            base_commit="60ed37f",
            source_angular_version="10.2.5",
            target_angular_version="11.2.14",
            target_node_version="16.20.2",
        )

    def test_prepare_environment_full_lifecycle(
        self, docker_manager: DockerManager, test_instance: BenchmarkInstance
    ) -> None:
        """
        Tests the full, end-to-end creation of a hierarchical environment.
        This is the primary test for the entire infrastructure pipeline.
        """
        # Predict the tags that will be generated for cleanup
        target_cli_major_version = test_instance.target_angular_version.split(".")[0]
        env_image_tag = _get_environment_image_tag(
            test_instance.target_node_version, target_cli_major_version
        )
        instance_image_tag = _get_instance_image_tag(test_instance)

        container = None
        try:
            # 1. Run the preparation pipeline
            final_tag = prepare_environment(test_instance, docker_manager)
            assert final_tag == instance_image_tag

            # 2. Verify all images were created
            assert docker_manager.image_exists(settings.docker_base_image_name)
            assert docker_manager.image_exists(env_image_tag)
            assert docker_manager.image_exists(instance_image_tag)

            # 3. Run a container and verify the project was cloned correctly
            container = docker_manager.run_container(image_tag=final_tag)
            # Check for a key file from the cloned repository
            exit_code, _, stderr = docker_manager.execute_in_container(
                container, "test -f /app/project/angular.json"
            )
            assert exit_code == 0, f"Verification command failed: {stderr}"

        finally:
            # 4. Cleanup all created resources
            if container:
                docker_manager.cleanup_container(container)
            # Remove images in reverse order of creation
            if docker_manager.image_exists(instance_image_tag):
                docker_manager.remove_image(instance_image_tag)
            if docker_manager.image_exists(env_image_tag):
                docker_manager.remove_image(env_image_tag)
            # Note: We typically don't remove the base image in a real run,
            # but we do here to ensure the test is fully isolated.
            if docker_manager.image_exists(settings.docker_base_image_name):
                docker_manager.remove_image(settings.docker_base_image_name)
