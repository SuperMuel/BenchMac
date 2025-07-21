"""
Integration tests for the entire Docker environment pipeline.

This includes low-level tests for the DockerManager and high-level tests
for the image builder.

These tests require a running Docker daemon to be present on the host machine.
They are marked as 'integration' and will be skipped by default.
"""

import uuid
from collections.abc import Generator

import pytest

from bench_mac.config import settings
from bench_mac.docker.builder import (
    _get_environment_image_tag,
    _get_instance_image_tag,
    prepare_environment,
)
from bench_mac.docker.manager import DockerManager
from bench_mac.models import BenchmarkInstance, CommandsConfig

# A simple, minimal Dockerfile for testing the manager in isolation.
SIMPLE_DOCKERFILE = 'FROM alpine:latest\nCMD ["echo", "container is running"]'


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
            commands=CommandsConfig(
                install="npm install",
                build="ng build --prod",
            ),
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
