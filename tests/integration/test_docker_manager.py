"""
Integration tests for the DockerManager class.

These tests require a running Docker daemon to be present on the host machine.
They are marked as 'integration' and will be skipped by default in the
standard test run.

To run these tests: `pytest -m integration`
"""

import uuid
from collections.abc import Generator
from pathlib import Path

import pytest
from docker.errors import DockerException
from docker.models.containers import Container

from bench_mac.docker.manager import DockerManager

# A simple, minimal Dockerfile for testing purposes.
# Using alpine as a base makes it small and fast.
SIMPLE_DOCKERFILE = 'FROM alpine:latest\nCMD ["echo", "container is running"]'


@pytest.fixture(scope="module")
def docker_manager() -> DockerManager | None:
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

    This ensures that tests do not interfere with each other. It also handles
    the cleanup of the image after the test has completed.
    """
    # Generate a short, unique tag to avoid collisions
    tag = f"benchmac-test-image:{uuid.uuid4().hex[:8]}"
    yield tag
    # Teardown: This code runs after the test function finishes
    print(f"\nCleaning up image: {tag}")
    if docker_manager and docker_manager.image_exists(tag):
        docker_manager.remove_image(tag)


@pytest.mark.integration
class TestDockerManager:
    """Test suite for the DockerManager class."""

    def test_initialization_succeeds(self, docker_manager: DockerManager) -> None:
        """Verify that the DockerManager can be instantiated successfully."""
        assert docker_manager is not None
        assert docker_manager._client is not None  # type: ignore[reportPrivateUsage]
        print("DockerManager initialized successfully.")

    def test_build_image_and_cleanup(
        self, docker_manager: DockerManager, unique_tag: str
    ) -> None:
        """
        Test the full lifecycle of building and removing an image.
        """
        # 1. Verify the image does not exist initially
        assert not docker_manager.image_exists(unique_tag), (
            f"Image {unique_tag} should not exist before build."
        )

        # 2. Build the image
        print(f"Building image: {unique_tag}")
        docker_manager.build_image(dockerfile_content=SIMPLE_DOCKERFILE, tag=unique_tag)

        # 3. Verify the image now exists
        assert docker_manager.image_exists(unique_tag), (
            f"Image {unique_tag} should exist after build."
        )

        # 4. Remove the image (explicitly testing the remove method)
        docker_manager.remove_image(unique_tag)

        # 5. Verify the image is gone
        assert not docker_manager.image_exists(unique_tag), (
            f"Image {unique_tag} should not exist after removal."
        )

    def test_run_container_and_execute(
        self, docker_manager: DockerManager, unique_tag: str
    ) -> None:
        """
        Test running a container and executing a command inside it.
        """
        # Setup: Build the image first
        docker_manager.build_image(dockerfile_content=SIMPLE_DOCKERFILE, tag=unique_tag)

        container = None
        try:
            # 1. Run the container
            print(f"Running container from image: {unique_tag}")
            container = docker_manager.run_container(image_tag=unique_tag)
            assert isinstance(container, Container)
            container.reload()  # Update container state from daemon
            assert container.status == "running"

            # 2. Execute a simple command
            command = 'echo "hello from the container"'
            print(f"Executing command: {command}")
            exit_code, stdout, stderr = docker_manager.execute_in_container(
                container, command
            )

            # 3. Assert the results
            assert exit_code == 0
            assert "hello from the container" in stdout
            assert stderr == ""  # No error output expected

        finally:
            # 4. Cleanup
            if container:
                print(f"Cleaning up container: {container.short_id}")
                docker_manager.cleanup_container(container)

    def test_copy_to_container(
        self, docker_manager: DockerManager, unique_tag: str, tmp_path: Path
    ) -> None:
        """
        Test copying a local file into a running container.
        """
        # Setup: Create a temporary file to copy
        local_file = tmp_path / "test_patch.txt"
        file_content = "This is a test patch file."
        local_file.write_text(file_content)

        # Setup: Build and run a container
        docker_manager.build_image(dockerfile_content=SIMPLE_DOCKERFILE, tag=unique_tag)
        container = docker_manager.run_container(image_tag=unique_tag)

        try:
            # 1. Copy the file into the container
            container_dest_path = "/tmp/test_patch.txt"
            print(f"Copying {local_file} to {container.short_id}:{container_dest_path}")
            docker_manager.copy_to_container(
                container, src_path=local_file, dest_path=container_dest_path
            )

            # 2. Verify the file exists and has the correct content
            command = f"cat {container_dest_path}"
            print(f"Verifying file content with command: {command}")
            exit_code, stdout, stderr = docker_manager.execute_in_container(
                container, command
            )

            assert exit_code == 0
            assert file_content in stdout
            assert stderr == ""  # No error output expected

        finally:
            # 3. Cleanup
            if container:
                docker_manager.cleanup_container(container)

    def test_execute_command_with_stderr(
        self, docker_manager: DockerManager, unique_tag: str
    ) -> None:
        """
        Test that stderr is properly captured when executing commands.
        """
        # Setup: Build and run a container
        docker_manager.build_image(dockerfile_content=SIMPLE_DOCKERFILE, tag=unique_tag)
        container = docker_manager.run_container(image_tag=unique_tag)

        try:
            # 1. Execute a command that writes to stderr
            command = 'sh -c "echo stdout_message && echo stderr_message >&2"'
            print(
                f"Executing command that outputs to both stdout and stderr: {command}"
            )
            exit_code, stdout, stderr = docker_manager.execute_in_container(
                container, command
            )

            # 2. Assert the results
            assert exit_code == 0
            assert "stdout_message" in stdout
            assert "stderr_message" in stderr

            # 3. Test a command that fails (non-zero exit code)
            fail_command = 'sh -c "echo error_output >&2 && exit 1"'
            print(f"Executing failing command: {fail_command}")
            exit_code, stdout, stderr = docker_manager.execute_in_container(
                container, fail_command
            )

            # 4. Assert failure is captured
            assert exit_code == 1
            assert stdout == ""
            assert "error_output" in stderr

        finally:
            # 5. Cleanup
            if container:
                docker_manager.cleanup_container(container)
