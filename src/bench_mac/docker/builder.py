"""
Manages the creation of hierarchical Docker environments for BenchMAC.

This module orchestrates the building of three layers of Docker images:
1.  Base Image: A static, universal image with OS and core tools (nvm, git).
2.  Environment Image: Built on Base, installs specific Node.js and Angular CLI versions.
3.  Instance Image: Built on Environment, clones a specific project at a specific commit.
"""  # noqa: E501

import re
from urllib.parse import urlparse

from loguru import logger
from slugify import slugify

from bench_mac.config import settings
from bench_mac.docker.manager import DockerManager
from bench_mac.models import BenchmarkInstance

BASE_DOCKERFILE_CONTENT = """
# Base image for all BenchMAC environments.
# It provides a stable OS and universal, version-agnostic tools.
# We use a specific Ubuntu LTS version for long-term reproducibility.
FROM ubuntu:22.04

# Prevent apt-get and other tools from hanging by waiting for user input.
# This is essential for non-interactive builds in Docker.
ENV DEBIAN_FRONTEND=noninteractive


# Install essential system dependencies in a single layer to optimize caching.
#
# - build-essential: Provides C/C++ compilers (gcc, g++) needed by `node-gyp`
#   to build native Node.js addons for performance-critical npm packages.
# - ca-certificates: Provides SSL certificates for tools like curl to make
#   secure HTTPS requests. A critical fix to prevent download failures.
# - curl: Used to download the nvm installation script.
# - git: Required to clone the benchmark project repositories.
# - python3 & python-is-python3: `node-gyp` requires a Python executable to
#   orchestrate its build scripts. Installing both ensures maximum compatibility
#   for any npm package that needs to be compiled during `npm install`.
#
RUN apt-get update && apt-get install -y \
    build-essential \
    ca-certificates \
    curl \
    git \
    python3 \
    python-is-python3 \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# --- NVM (Node Version Manager) Setup ---

# Define a standard, system-wide location for the NVM installation.
# This is more explicit and robust than letting it default to a user's home
# directory (e.g., /root/.nvm), making the setup predictable.
ENV NVM_DIR /usr/local/nvm

# The nvm installer script requires its target directory to exist
# before it will run.
RUN mkdir -p $NVM_DIR

# Download and execute the nvm installation script.
# The script will install nvm into the directory specified by $NVM_DIR.
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash

# Add nvm and a default Node version's bin directory to the system's PATH.
# This makes `nvm`, `node`, and `npm` directly available to subsequent RUN commands
# in this Dockerfile and, crucially, in any child images that build FROM this one.
# Note: This NODE_VERSION is just a sensible default for the base image; it will be
# overridden by the specific version required in the 'environment' image layer.
ENV NODE_VERSION 18.13.0
ENV PATH $NVM_DIR/versions/node/v$NODE_VERSION/bin:$PATH

# Set a default working directory for subsequent commands.
WORKDIR /app

# Provide a default command to get an interactive shell
# if the container is run directly.
CMD ["/bin/bash"]
"""


def _get_environment_image_tag(node_version: str, angular_cli_version: str) -> str:
    """Generates a consistent, unique tag for an Environment Image."""
    # Sanitize versions for use in Docker tags (e.g., 18.13.0 -> 18-13-0)
    if not node_version:
        raise ValueError("Node version is required")
    if not angular_cli_version:
        raise ValueError("Angular CLI version is required")

    node_tag = node_version.replace(".", "-")
    cli_tag = angular_cli_version.replace(".", "-")
    return f"benchmac-env:node{node_tag}-ng{cli_tag}"


def _get_instance_image_tag(instance: BenchmarkInstance) -> str:
    """Generates a consistent, unique tag for an Instance Image."""
    repo = instance.repo

    # Special handling for GitHub URLs
    if repo.startswith("https://github.com/"):
        # Extract owner/repo from GitHub URL like
        # "https://github.com/angular/angular-cli.git" using urllib

        parsed = urlparse(repo)
        # The path is like "/owner/repo.git" or "/owner/repo"
        repo_path = parsed.path.lstrip("/")
        if repo_path.endswith(".git"):
            repo_path = repo_path[: -len(".git")]
        repo_part = repo_path
        slug = f"gh__{repo_part.replace('/', '__')}"
        tag = f"benchmac-instance:{slug}__{instance.base_commit}"
    else:
        # Regular repo handling with double underscores (no slugify)
        owner, repo_name = repo.split("/")
        slug = f"{slugify(owner)}__{slugify(repo_name)}"
        tag = f"benchmac-instance:{slug}__{instance.base_commit}"

    if len(tag) > 128:
        raise ValueError(
            f"Generated image tag is too long: {tag=} ({len(tag)} > 128). "
            f"instance_id={instance.instance_id}"
            "Please kindly ask @SuperMuel to find a solution."
        )
    return tag


def _normalize_repo_url(repo_url: str) -> str:
    """
    Converts a repository reference to a git-clonable URL.

    Parameters
    ----------
    repo_url
        The repository reference, which can be:
        - A full git URL (http://, https://, git@, git://)
        - An owner/repo format (e.g., "angular/angular-cli")

    Returns
    -------
    A git-clonable URL string.

    Raises
    ------
    ValueError
        If the input is not a valid repository reference.
    """
    if not repo_url or not repo_url.strip():
        raise ValueError("Repository URL cannot be empty")

    repo_url = repo_url.strip()

    # Check if it's already a git-clonable URL
    valid_prefixes = ("http://", "https://", "git@", "git://")
    if any(repo_url.startswith(prefix) for prefix in valid_prefixes):
        return repo_url

    # Check if it matches owner/repo format
    owner_repo_pattern = re.compile(r"^[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+$")
    if owner_repo_pattern.match(repo_url):
        return f"https://github.com/{repo_url}.git"

    # If it doesn't match any known format, raise an error
    raise ValueError(
        f"Invalid repository reference: '{repo_url}'. "
        "Expected a git-clonable URL (http://, https://, git@, git://) "
        "or owner/repo format (e.g., 'angular/angular-cli')"
    )


def _generate_environment_dockerfile_content(
    base_image_tag: str, node_version: str, angular_cli_version: str
) -> str:
    """Programmatically generates the Dockerfile content for an Environment Image."""
    return f"""
FROM {base_image_tag}

# Set environment variables for nvm, making them available to subsequent layers
ENV NVM_DIR /usr/local/nvm
ENV NODE_VERSION {node_version}
# Ensure nvm's shims are in the PATH
ENV PATH $NVM_DIR/versions/node/v$NODE_VERSION/bin:$PATH

# Source nvm and install the specified Node.js version and Angular CLI.
# This becomes the new default Node.js for this image and its children.
RUN . "$NVM_DIR/nvm.sh" && \\
    nvm install $NODE_VERSION && \\
    nvm use $NODE_VERSION && \\
    nvm alias default $NODE_VERSION && \\
    npm install -g @angular/cli@{angular_cli_version}

# Verify installations to catch errors early
RUN . "$NVM_DIR/nvm.sh" && node -v && ng version
"""


def _generate_instance_dockerfile_content(
    environment_image_tag: str, repo_url: str, base_commit: str
) -> str:
    """Programmatically generates the Dockerfile content for an Instance Image."""
    # Normalize the repo URL to ensure it's git-clonable
    normalized_repo_url = _normalize_repo_url(repo_url)

    return f"""
FROM {environment_image_tag}

WORKDIR /app/project

# Clone the specific repository and checkout the base commit
# Using --depth 1 can speed up clones for large repos, but we need the full
# history to check out a specific commit.
RUN git clone {normalized_repo_url} . && \\
    git checkout {base_commit}

# Set the default command to open a shell in the project directory
CMD ["/bin/bash"]
"""


def prepare_environment(instance: BenchmarkInstance, manager: DockerManager) -> str:
    """
    Ensures the complete Docker environment for a given instance exists,
    building it if necessary.

    This function is idempotent. It checks if images exist before building them,
    leveraging Docker's layer caching for efficiency.

    Parameters
    ----------
    instance
        The BenchmarkInstance for which to prepare the environment.
    manager
        An initialized DockerManager instance.

    Returns
    -------
    The tag of the final, ready-to-use Instance Image.
    """
    logger.debug(
        f"\n--- Preparing environment for instance: {instance.instance_id} ---"
    )

    # --- Layer 1: Base Image ---
    base_image_tag = settings.docker_base_image_name
    if not manager.image_exists(base_image_tag):
        logger.debug(f"Base image '{base_image_tag}' not found. Building...")
        manager.build_image(
            dockerfile_content=BASE_DOCKERFILE_CONTENT, tag=base_image_tag
        )
    else:
        logger.debug(f"Base image '{base_image_tag}' already exists. Skipping build.")

    # --- Layer 2: Environment Image ---
    # The target Angular version determines the CLI version needed.
    # We use the major version for the CLI installation for simplicity.
    target_cli_major_version = instance.target_angular_version.split(".")[0]
    env_image_tag = _get_environment_image_tag(
        instance.target_node_version, target_cli_major_version
    )

    if not manager.image_exists(env_image_tag):
        logger.debug(f"Environment image '{env_image_tag}' not found. Building...")
        dockerfile_content = _generate_environment_dockerfile_content(
            base_image_tag, instance.target_node_version, target_cli_major_version
        )
        manager.build_image(dockerfile_content=dockerfile_content, tag=env_image_tag)
    else:
        logger.debug(
            f"Environment image '{env_image_tag}' already exists. Skipping build."
        )

    # --- Layer 3: Instance Image ---
    instance_image_tag = _get_instance_image_tag(instance)
    if not manager.image_exists(instance_image_tag):
        logger.debug(f"Instance image '{instance_image_tag}' not found. Building...")
        dockerfile_content = _generate_instance_dockerfile_content(
            env_image_tag, instance.repo, instance.base_commit
        )
        manager.build_image(
            dockerfile_content=dockerfile_content, tag=instance_image_tag
        )
    else:
        logger.debug(
            f"Instance image '{instance_image_tag}' already exists. Skipping build."
        )

    logger.success(f"✅ Environment ready. Final image tag: {instance_image_tag}")
    return instance_image_tag
