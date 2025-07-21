# #!/usr/bin/env python
# #
# # Standalone Script to Validate a Single BenchMAC Instance
# #
# # This script simulates the core functionality of the BenchMAC harness for a
# # single, hardcoded benchmark instance. It builds the necessary hierarchical
# # Docker images, runs a container, and executes the defined evaluation commands
# # to verify that a known-good state is achievable.
# #
# # It is self-contained and does not depend on the BenchMAC project structure,
# # making it a perfect tool for isolated testing and debugging of the core
# # Docker and execution logic.
# #
# import logging
# import re
# import tempfile
# from pathlib import Path
# from typing import TYPE_CHECKING

# import docker
# from docker.errors import BuildError, DockerException, ImageNotFound, NotFound
# from slugify import slugify

# if TYPE_CHECKING:
#     from docker import DockerClient
#     from docker.models.containers import Container
#     from docker.models.images import Image

# # --- Configuration: The Benchmark Instance to Test ---
# INSTANCE_DATA = {
#     "instance_id": "spring-petclinic_v10_to_v11",
#     "repo": "spring-petclinic/spring-petclinic-angular",
#     "base_commit": "202b83a8e9a7c5e9c4d77befa99755571c5005d3",
#     "source_angular_version": "10",
#     "target_angular_version": "11",
#     "target_node_version": "12.13.0",
#     "commands": {
#         "install": "npm install",
#         "build": "ng build --prod",
#         "lint": "ng lint",
#         "test": "ng test --watch=false --browsers=ChromeHeadless",
#     },
# }

# # --- Standalone DockerManager (Copied & Adapted from BenchMAC) ---
# # This class is a self-contained version of the DockerManager, with logging
# # adapted to use Python's standard logging module.


# class StandaloneDockerManager:
#     """Manages Docker images and containers for isolated benchmark evaluations."""

#     def __init__(self) -> None:
#         """Initializes the Docker client and verifies connection to the daemon."""
#         try:
#             self._client: DockerClient = docker.DockerClient(
#                 base_url="unix:///Users/s.mallet/.orbstack/run/docker.sock"
#             )
#             if not self._client.ping():
#                 raise DockerException("Docker daemon responded in a failed state.")
#         except DockerException as e:
#             raise DockerException(
#                 "❌ Error: Docker is not running or is not configured correctly."
#             ) from e

#     def image_exists(self, tag: str) -> bool:
#         """Check if a Docker image with the given tag exists locally."""
#         try:
#             self._client.images.get(tag)
#             return True
#         except ImageNotFound:
#             return False

#     def build_image(self, dockerfile_content: str, tag: str) -> "Image":
#         """Builds a Docker image from a string containing Dockerfile content."""
#         logging.info(f"Building Docker image with tag: {tag}...")
#         try:
#             with tempfile.TemporaryDirectory() as tmpdir:
#                 dockerfile_path = Path(tmpdir) / "Dockerfile"
#                 dockerfile_path.write_text(dockerfile_content, encoding="utf-8")
#                 image, build_log_stream = self._client.images.build(
#                     path=tmpdir, tag=tag, rm=True, forcerm=True
#                 )
#                 # Consume logs to ensure build completion
#                 for chunk in build_log_stream:
#                     if "stream" in chunk:
#                         line = chunk["stream"].strip()
#                         if line:
#                             logging.debug(f"  | {line}")
#             logging.info(f"✅ Successfully built image: {tag}")
#             return image
#         except BuildError as e:
#             logging.error(f"❌ Docker build failed for tag {tag}: {e}")
#             raise

#     def remove_image(self, tag: str) -> None:
#         """Removes a Docker image by its tag."""
#         if not self.image_exists(tag):
#             return
#         try:
#             logging.info(f"Removing image: {tag}")
#             self._client.images.remove(tag, force=True)
#         except DockerException as e:
#             logging.error(f"❌ Failed to remove image {tag}: {e}")
#             raise

#     def run_container(self, image_tag: str) -> "Container":
#         """Runs a container from a given image tag."""
#         logging.info(f"Starting container from image: {image_tag}")
#         try:
#             container = self._client.containers.run(
#                 image_tag, command="tail -f /dev/null", detach=True
#             )
#             logging.info(f"✅ Container {container.short_id} is running.")
#             return container
#         except DockerException as e:
#             logging.error(f"❌ Failed to run container from image {image_tag}: {e}")
#             raise

#     def execute_in_container(
#         self, container: "Container", command: str
#     ) -> tuple[int, str, str]:
#         """Executes a shell command inside a running container."""
#         logging.info(f"Executing in {container.short_id}: {command}")
#         exit_code, output = container.exec_run(command, demux=True)
#         stdout = output[0].decode("utf-8") if output[0] else ""
#         stderr = output[1].decode("utf-8") if output[1] else ""
#         logging.info(f"  > Exit code: {exit_code}")
#         return exit_code, stdout, stderr

#     def cleanup_container(self, container: "Container") -> None:
#         """Stops and removes a container, handling errors gracefully."""
#         try:
#             container.reload()
#             if container.status == "running":
#                 logging.info(f"Stopping container: {container.short_id}")
#                 container.stop()
#             logging.info(f"Removing container: {container.short_id}")
#             container.remove()
#         except NotFound:
#             logging.debug(f"Container {container.short_id} already removed.")
#         except DockerException as e:
#             logging.warning(f"⚠️ Could not clean up container {container.short_id}: {e}")


# # --- Standalone Builder Logic (Copied & Adapted from BenchMAC) ---

# BASE_DOCKERFILE_CONTENT = """
# FROM ubuntu:22.04
# ENV DEBIAN_FRONTEND=noninteractive

# # Install basic dependencies
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     ca-certificates \
#     curl \
#     git \
#     python3 \
#     python-is-python3 \
#     wget \
#     gnupg \
#     --no-install-recommends

# # Install Chromium from Ubuntu snap-free source
# RUN echo "deb http://deb.debian.org/debian bullseye main" > /etc/apt/sources.list.d/debian.list && \
#     apt-get update && \
#     apt-get install -y chromium --no-install-recommends && \
#     rm -rf /var/lib/apt/lists/* && \
#     rm /etc/apt/sources.list.d/debian.list

# # Set Chrome binary path for Angular tests
# ENV CHROME_BIN=/usr/bin/chromium

# ENV NVM_DIR /usr/local/nvm
# RUN mkdir -p $NVM_DIR
# RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
# ENV NODE_VERSION 18.13.0
# ENV PATH $NVM_DIR/versions/node/v$NODE_VERSION/bin:$PATH
# WORKDIR /app
# CMD ["/bin/bash"]
# """


# def _get_environment_image_tag(node_version: str, angular_cli_version: str) -> str:
#     node_tag = node_version.replace(".", "-")
#     cli_tag = angular_cli_version.replace(".", "-")
#     return f"benchmac-env:node{node_tag}-ng{cli_tag}"


# def _get_instance_image_tag(repo: str, base_commit: str) -> str:
#     owner, repo_name = repo.split("/")
#     slug = f"{slugify(owner)}__{slugify(repo_name)}"
#     return f"benchmac-instance:{slug}__{base_commit}"


# def _normalize_repo_url(repo_url: str) -> str:
#     if not repo_url or not repo_url.strip():
#         raise ValueError("Repository URL cannot be empty")
#     repo_url = repo_url.strip()
#     valid_prefixes = ("http://", "https://", "git@", "git://")
#     if any(repo_url.startswith(prefix) for prefix in valid_prefixes):
#         return repo_url
#     owner_repo_pattern = re.compile(r"^[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+$")
#     if owner_repo_pattern.match(repo_url):
#         return f"https://github.com/{repo_url}.git"
#     raise ValueError(f"Invalid repository reference: '{repo_url}'")


# def _generate_environment_dockerfile_content(
#     base_image_tag: str, node_version: str, angular_cli_version: str
# ) -> str:
#     return f"""
# FROM {base_image_tag}
# ENV NVM_DIR /usr/local/nvm
# ENV NODE_VERSION {node_version}
# ENV PATH $NVM_DIR/versions/node/v$NODE_VERSION/bin:$PATH
# RUN . "$NVM_DIR/nvm.sh" && \\
#     nvm install $NODE_VERSION && \\
#     nvm use $NODE_VERSION && \\
#     nvm alias default $NODE_VERSION && \\
#     npm install -g @angular/cli@{angular_cli_version}
# RUN . "$NVM_DIR/nvm.sh" && node -v && ng version
# """


# def _generate_instance_dockerfile_content(
#     environment_image_tag: str, repo_url: str, base_commit: str
# ) -> str:
#     normalized_repo_url = _normalize_repo_url(repo_url)
#     return f"""
# FROM {environment_image_tag}
# WORKDIR /app/project
# RUN git clone {normalized_repo_url} . && \\
#     git checkout {base_commit}
# CMD ["/bin/bash"]
# """


# # --- Main Script Logic ---


# def main() -> None:
#     """Main execution function."""
#     logging.basicConfig(
#         level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
#     )

#     # --- 1. Setup ---
#     logging.info(f"Starting validation for instance: {INSTANCE_DATA['instance_id']}")
#     manager = StandaloneDockerManager()
#     container = None

#     # Define image tags for building and cleanup
#     base_image_tag = "benchmac-base"
#     target_cli_major_version = INSTANCE_DATA["target_angular_version"].split(".")[0]
#     env_image_tag = _get_environment_image_tag(
#         INSTANCE_DATA["target_node_version"], target_cli_major_version
#     )
#     instance_image_tag = _get_instance_image_tag(
#         INSTANCE_DATA["repo"], INSTANCE_DATA["base_commit"]
#     )

#     images_to_cleanup = [instance_image_tag, env_image_tag, base_image_tag]

#     try:
#         # --- 2. Build Hierarchical Images ---
#         # Layer 1: Base Image
#         if not manager.image_exists(base_image_tag):
#             manager.build_image(BASE_DOCKERFILE_CONTENT, base_image_tag)
#         else:
#             logging.info(
#                 f"Base image '{base_image_tag}' already exists. Skipping build."
#             )

#         # Layer 2: Environment Image
#         if not manager.image_exists(env_image_tag):
#             env_dockerfile = _generate_environment_dockerfile_content(
#                 base_image_tag,
#                 INSTANCE_DATA["target_node_version"],
#                 target_cli_major_version,
#             )
#             manager.build_image(env_dockerfile, env_image_tag)
#         else:
#             logging.info(f"Env image '{env_image_tag}' already exists. Skipping build.")

#         # Layer 3: Instance Image
#         if not manager.image_exists(instance_image_tag):
#             instance_dockerfile = _generate_instance_dockerfile_content(
#                 env_image_tag,
#                 INSTANCE_DATA["repo"],
#                 INSTANCE_DATA["base_commit"],
#             )
#             manager.build_image(instance_dockerfile, instance_image_tag)
#         else:
#             logging.info(
#                 f"Instance image '{instance_image_tag}' already exists. Skipping build."
#             )

#         # --- 3. Run Container ---
#         container = manager.run_container(instance_image_tag)

#         # --- 4. Execute and Verify Commands ---
#         commands_to_run = INSTANCE_DATA["commands"]
#         for stage, command in commands_to_run.items():
#             logging.info(f"\n--- Running Stage: {stage.upper()} ---")
#             exit_code, stdout, stderr = manager.execute_in_container(container, command)

#             if exit_code != 0:
#                 logging.error(f"❌ Stage '{stage}' FAILED with exit code {exit_code}.")
#                 logging.error("--- STDOUT ---")
#                 print(stdout)
#                 logging.error("--- STDERR ---")
#                 print(stderr)
#                 raise SystemExit(1)
#             else:
#                 logging.info(f"✅ Stage '{stage}' PASSED.")
#                 logging.debug("--- STDOUT ---")
#                 logging.debug(stdout)
#                 if stderr:
#                     logging.warning("--- STDERR (non-fatal) ---")
#                     logging.warning(stderr)

#         logging.info("\n🎉 All validation stages passed successfully!")

#     except (DockerException, SystemExit) as e:
#         logging.error(f"Script failed: {e}")
#     except Exception as e:
#         logging.exception(f"An unexpected error occurred: {e}")
#     finally:
#         # --- 5. Cleanup ---
#         logging.info("\n--- Starting Cleanup ---")
#         if container:
#             manager.cleanup_container(container)
#         for tag in images_to_cleanup:
#             manager.remove_image(tag)
#         logging.info("✅ Cleanup complete.")


# if __name__ == "__main__":
#     main()
