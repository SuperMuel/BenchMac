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
from docker import DockerClient
from docker.errors import BuildError, DockerException, ImageNotFound, NotFound
from docker.models.containers import Container
from docker.models.images import Image
from loguru import logger

from bench_mac.config import settings


class DockerManager:
    """
    Manages Docker images and containers for isolated benchmark evaluations.
    """

    @classmethod
    def get_client(cls, quiet: bool = True) -> DockerClient:
        """
        Tests the connection to Docker daemon and returns a Docker client.

        Parameters
        ----------
        quiet
            If True, suppresses debug messages.

        Returns
        -------
        DockerClient
            A configured Docker client instance.

        Raises
        ------
        DockerException
            If the Docker daemon is not running or cannot be reached.
        """

        def maybe_log(msg: str) -> None:
            if not quiet:
                logger.info(msg)

        try:
            maybe_log("Attempting to connect to Docker daemon...")

            if settings.docker_host:
                maybe_log(f"  > Using configured host: {settings.docker_host}")
                client = docker.DockerClient(base_url=settings.docker_host)
            else:
                maybe_log("  > No host configured, using auto-detection (from_env).")
                client = docker.from_env()  # type: ignore[reportUnknownMemberType]

            # A low-timeout ping is a fast way to check for a running daemon
            if not client.ping():  # type: ignore[reportUnknownMemberType]
                raise DockerException(
                    "Docker daemon responded to ping, but in a failed state."
                )

            maybe_log("✅ Docker client initialized successfully.")

            return client

        except DockerException as e:
            raise DockerException(
                "❌ Error: Docker is not running or is not configured correctly."
            ) from e

    def __init__(self, quiet_init: bool = True) -> None:
        """
        Initializes the Docker client and verifies connection to the daemon.

        Parameters
        ----------
        quiet_init
            If True, suppresses debug messages during initialization.

        Raises
        ------
        DockerException
            If the Docker daemon is not running or cannot be reached.
        """
        self._client = self.get_client(quiet=quiet_init)

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
        logger.debug(f"Building Docker image with tag: {tag}...")
        try:
            # docker-py needs a file context, so we create a temporary one.
            with tempfile.TemporaryDirectory() as tmpdir:
                dockerfile_path = Path(tmpdir) / "Dockerfile"
                dockerfile_path.write_text(dockerfile_content, encoding="utf-8")

                image, build_log_stream = self._client.images.build(
                    path=tmpdir,
                    tag=tag,
                    rm=True,  # Remove intermediate containers
                    forcerm=True,
                )

                # Stream and print build logs for user feedback
                # TODO: improve this
                for chunk in build_log_stream:
                    if isinstance(chunk, dict) and "stream" in chunk:
                        line = chunk["stream"]
                        if isinstance(line, str) and line.strip():
                            logger.debug(f"  | {line.strip()}")

            logger.debug(f"✅ Successfully built image: {tag}")
            return image
        except BuildError as e:
            logger.exception(
                f"❌ Docker build failed for tag {tag} ({e.__class__.__name__}: {e})"
            )
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
            logger.debug(f"Image {tag} does not exist, no need to remove.")
            return
        try:
            logger.debug(f"Removing image: {tag}")
            self._client.images.remove(tag, force=True)  # type: ignore[reportUnknownMemberType]
            logger.debug(f"✅ Successfully removed image: {tag}")
        except DockerException as e:
            logger.error(f"❌ Failed to remove image {tag}: {e}")
            raise

    def run_container(self, image_tag: str, auto_remove: bool = False) -> Container:
        """
        Runs a container from a given image tag.

        Parameters
        ----------
        image_tag
            The tag of the image to run.
        auto_remove
            If True, the container will be automatically removed on exit.
            Note: This is False by default to allow for inspection on failure.

        Returns
        -------
        The running Docker Container object.
        """
        logger.debug(f"Starting container from image: {image_tag}")
        try:
            container = self._client.containers.run(
                image_tag,
                # Keep the container alive indefinitely until we stop it
                command="tail -f /dev/null",
                detach=True,
                auto_remove=auto_remove,
            )
            logger.debug(f"✅ Container {container.short_id} is running.")
            return container
        except DockerException as e:
            logger.exception(
                f"❌ Failed to run container from image {image_tag} ({e.__class__.__name__}: {e})"  # noqa: E501
            )
            raise

    def execute_in_container(
        self, container: Container, command: str, workdir: str | None = None
    ) -> tuple[int, str, str]:
        """
        Executes a shell command inside a running container.

        Parameters
        ----------
        container
            The Container object to execute the command in.
        command
            The shell command to execute.
        workdir
            The working directory inside the container to run the command from.

        Returns
        -------
        A tuple containing (exit_code, stdout, stderr).
        """
        # Ensure the command is run within a shell to handle operators like &&
        # and shell builtins like cd.
        shell_command = ["/bin/sh", "-c", command]

        logger.debug(
            f"Executing in {container.short_id}"
            f" (workdir: {workdir or '/'}): {shell_command}"
        )
        exit_code, output = container.exec_run(  # type: ignore[reportUnknownMemberType]
            shell_command,
            # demux=True returns (stdout, stderr) as separate byte streams, not combined
            # For example, running 'sh -c "echo out && echo err >&2"' will yield:
            #   output = (b'out\n', b'err\n')
            demux=True,
            workdir=workdir,
        )

        logger.debug(f"  > Exit code: {exit_code}")

        # When demux=True, output is a tuple of (stdout, stderr)
        # Each can be bytes or None
        stdout = output[0].decode("utf-8") if output[0] else ""
        stderr = output[1].decode("utf-8") if output[1] else ""

        return exit_code, stdout, stderr

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
        logger.debug(f"Copying {src_path} to {container.short_id}:{dest_path}")

        # The docker-py `put_archive` method requires a tarball stream.
        # We create one in memory.
        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            tar.add(str(src_path), arcname=src_path.name)

        # Rewind the stream to the beginning before sending
        tar_stream.seek(0)

        try:
            success = container.put_archive(  # type: ignore[reportUnknownMemberType]
                path=str(Path(dest_path).parent), data=tar_stream
            )
            if not success:
                raise DockerException(
                    f"Failed to copy {src_path} to container {container.short_id}:{dest_path}"  # noqa: E501
                )
            logger.debug("✅ Copy successful.")
        except Exception as e:
            logger.error(f"❌ Failed to copy to container: {e}")
            raise

    def cleanup_container(self, container: Container) -> None:
        """
        Stops and removes a container, handling errors gracefully.

        Parameters
        ----------
        container
            The Container object to clean up.
        """
        try:
            # Reload the container's state from the daemon to get the latest status
            container.reload()
            if container.status == "running":
                logger.debug(f"Stopping container: {container.short_id}")
                container.stop()
            logger.debug(f"Removing container: {container.short_id}")
            container.remove()
            logger.debug(f"✅ Container {container.short_id} cleaned up.")
        except NotFound:  # pragma: no cover
            # The container was already removed (e.g., with auto_remove=True)
            logger.debug(f"Container {container.short_id} already removed.")
        except DockerException as e:  # pragma: no cover
            logger.error(f"⚠️  Could not clean up container {container.short_id}: {e}")
