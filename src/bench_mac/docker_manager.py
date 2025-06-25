"""
Docker Interaction Layer for BenchMAC.

This module provides a high-level, clean API for managing Docker resources
(images, containers) required for benchmark evaluations. It encapsulates the
low-level details of the `docker-py` library.
"""

import io
import tarfile
import tempfile
from pathlib import Path

import docker
from bench_mac.config import settings
from docker.errors import DockerException, ImageNotFound, NotFound
from docker.models.containers import Container
from docker.models.images import Image


class DockerManager:
    """
    Manages Docker images and containers for isolated benchmark evaluations.
    """

    def __init__(self) -> None:
        """
        Initializes the Docker client and verifies connection to the daemon.

        Raises
        ------
        DockerException
            If the Docker daemon is not running or cannot be reached.
        """
        try:
            print("Attempting to connect to Docker daemon...")
            if settings.docker_host:
                print(f"  > Using configured host: {settings.docker_host}")
                self._client = docker.DockerClient(base_url=settings.docker_host)
            else:
                print("  > No host configured, using auto-detection (from_env).")
                self._client = docker.from_env()

            # A low-timeout ping is a fast way to check for a running daemon
            if not self._client.ping():  # type: ignore[reportUnknownMemberType]
                raise DockerException(
                    "Docker daemon responded to ping, but in a failed state."
                )
            print("✅ Docker client initialized successfully.")
        except DockerException as e:
            print("❌ Error: Docker is not running or is not configured correctly.")
            print("   Please ensure the Docker daemon is active.")
            # Re-raise the exception to halt execution if Docker is required
            raise e

    def image_exists(self, tag: str) -> bool:
        """Check if a Docker image with the given tag exists locally."""
        try:
            self._client.images.get(tag)
            return True
        except ImageNotFound:
            return False

    def build_image(self, dockerfile_content: str, tag: str) -> Image:
        """
        Builds a Docker image from a string containing Dockerfile content.

        Parameters
        ----------
        dockerfile_content
            A string containing the full content of the Dockerfile.
        tag
            The tag to apply to the built image (e.g., 'my-image:latest').

        Returns
        -------
        The built Docker Image object.
        """
        print(f"Building Docker image with tag: {tag}...")
        try:
            # docker-py needs a file context, so we create a temporary one.
            with tempfile.TemporaryDirectory() as tmpdir:
                dockerfile_path = Path(tmpdir) / "Dockerfile"
                dockerfile_path.write_text(dockerfile_content, encoding="utf-8")

                image, build_log_stream = self._client.images.build(
                    path=str(tmpdir),
                    tag=tag,
                    rm=True,  # Remove intermediate containers
                    forcerm=True,
                )

                # Stream and print build logs for user feedback
                for chunk in build_log_stream:
                    if "stream" in chunk:
                        line = chunk["stream"].strip()
                        if line:
                            print(f"  | {line}")

            print(f"✅ Successfully built image: {tag}")
            return image
        except docker.errors.BuildError as e:
            print(f"❌ Docker build failed for tag {tag}: {e}")
            raise

    def remove_image(self, tag: str) -> None:
        """
        Removes a Docker image by its tag.

        Parameters
        ----------
        tag
            The tag of the image to remove.
        """
        if not self.image_exists(tag):
            print(f"Image {tag} does not exist, no need to remove.")
            return
        try:
            print(f"Removing image: {tag}")
            self._client.images.remove(tag, force=True)
            print(f"✅ Successfully removed image: {tag}")
        except DockerException as e:
            print(f"❌ Failed to remove image {tag}: {e}")
            raise

    def run_container(
        self, image_tag: str, detach: bool = True, auto_remove: bool = False
    ) -> Container:
        """
        Runs a container from a given image tag.

        Parameters
        ----------
        image_tag
            The tag of the image to run.
        detach
            If True (default), run the container in the background.
        auto_remove
            If True, the container will be automatically removed on exit.
            Note: This is False by default to allow for inspection on failure.

        Returns
        -------
        The running Docker Container object.
        """
        print(f"Running container from image: {image_tag}...")
        try:
            container = self._client.containers.run(
                image_tag,
                detach=detach,
                auto_remove=auto_remove,
                # Keep the container alive indefinitely until we stop it
                command="tail -f /dev/null",
            )
            print(f"✅ Container {container.short_id} is running.")
            return container
        except DockerException as e:
            print(f"❌ Failed to run container from image {image_tag}: {e}")
            raise

    def execute_in_container(
        self, container: Container, command: str
    ) -> tuple[int, str]:
        """
        Executes a shell command inside a running container.

        Parameters
        ----------
        container
            The Container object to execute the command in.
        command
            The shell command to execute.

        Returns
        -------
        A tuple containing (exit_code, logs).
        """
        print(f"Executing in {container.short_id}: {command}")
        exit_code, output = container.exec_run(command)
        logs = output.decode("utf-8").strip()
        print(f"  > Exit code: {exit_code}")
        return exit_code, logs

    def copy_to_container(
        self, container: Container, src_path: Path, dest_path: str
    ) -> None:
        """
        Copies a local file or directory into a container.

        Parameters
        ----------
        container
            The target Container object.
        src_path
            The local path to the file or directory to copy.
        dest_path
            The absolute path inside the container where the content will be placed.
        """
        print(f"Copying {src_path} to {container.short_id}:{dest_path}")

        # The docker-py `put_archive` method requires a tarball stream.
        # We create one in memory.
        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            tar.add(str(src_path), arcname=src_path.name)

        # Rewind the stream to the beginning before sending
        tar_stream.seek(0)

        try:
            container.put_archive(path=str(Path(dest_path).parent), data=tar_stream)
            print("✅ Copy successful.")
        except Exception as e:
            print(f"❌ Failed to copy to container: {e}")
            raise

    def cleanup_container(self, container: Container) -> None:
        """
        Stops and removes a container, handling errors gracefully.

        Parameters
        ----------
        container
            The Container object to clean up.
        """
        if not container:
            return
        try:
            # Reload the container's state from the daemon to get the latest status
            container.reload()
            if container.status == "running":
                print(f"Stopping container: {container.short_id}")
                container.stop()
            print(f"Removing container: {container.short_id}")
            container.remove()
            print(f"✅ Container {container.short_id} cleaned up.")
        except NotFound:
            # The container was already removed (e.g., with auto_remove=True)
            print(f"Container {container.short_id} already removed.")
        except DockerException as e:
            print(f"⚠️  Could not clean up container {container.short_id}: {e}")
