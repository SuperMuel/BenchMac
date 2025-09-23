"""
Integration tests for the entire Docker environment pipeline.

This includes low-level tests for the DockerManager and high-level tests
for the image builder.

These tests require a running Docker daemon to be present on the host machine.
They are marked as 'integration' and will be skipped by default.
"""

from collections.abc import Generator

import pytest
from uuid6 import uuid7

from bench_mac.core.config import settings
from bench_mac.core.models import BenchmarkInstance, InstanceCommands, InstanceID
from bench_mac.docker.builder import prepare_environment
from bench_mac.docker.manager import DockerManager


@pytest.fixture
def unique_tag(docker_manager: DockerManager) -> Generator[str, None, None]:
    """
    Generates a unique Docker image tag for each test function.

    This ensures that tests do not interfere with each other and handles
    the cleanup of the image after the test has completed.
    """
    tag = f"benchmac-test-image:{uuid7().hex[:8]}"
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
            instance_id=InstanceID(
                "gothinkster__angular-realworld-example-app_v11_to_v12"
            ),
            repo="gothinkster/angular-realworld-example-app",
            base_commit="4f29e0e",
            source_angular_version="11.2.8",
            target_angular_version="12",
            commands=InstanceCommands(
                install="npm ci",
                build="npx ng build --configuration production",
            ),
        )

    def test_prepare_environment_full_lifecycle(
        self, docker_manager: DockerManager, test_instance: BenchmarkInstance
    ) -> None:
        """
        Tests the full, end-to-end creation of a hierarchical environment.
        This is the primary test for the entire infrastructure pipeline.
        """
        instance_image_tag = prepare_environment(test_instance, docker_manager)

        container = None
        try:
            # 1. Run the preparation pipeline
            final_tag = prepare_environment(test_instance, docker_manager)
            assert final_tag == instance_image_tag

            # 2. Verify the instance image was created
            assert docker_manager.image_exists(instance_image_tag)

            # 3. Run a container and verify the project was cloned correctly
            container = docker_manager.run_container(image_tag=final_tag)
            # Check for a key file from the cloned repository
            workdir = settings.project_workdir
            exit_code, _, stderr = docker_manager.execute_in_container(
                container, f"test -f {workdir}/angular.json"
            )
            assert exit_code == 0, f"Verification command failed: {stderr}"

        finally:
            # 4. Cleanup all created resources
            if container:
                docker_manager.cleanup_container(container)
            # Remove images in reverse order of creation
            if docker_manager.image_exists(instance_image_tag):
                docker_manager.remove_image(instance_image_tag)
