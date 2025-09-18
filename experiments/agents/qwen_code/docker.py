from __future__ import annotations

from textwrap import dedent

from loguru import logger

from bench_mac.docker.builder import prepare_environment
from bench_mac.docker.manager import DockerManager
from bench_mac.models import BenchmarkInstance

NODE_VERSION = "20.17.0"
NVM_VERSION = "0.39.7"
QWEN_CODE_VERSION = "0.0.11"


def build_qwen_overlay(
    instance: BenchmarkInstance, docker_manager: DockerManager
) -> str:
    """Return a Docker image tag with Qwen Code tooling baked in."""
    base_tag = prepare_environment(instance, docker_manager)
    overlay_tag = f"{base_tag}-qwen-code"

    if docker_manager.image_exists(overlay_tag):
        logger.debug(
            "Image '{overlay_tag}' already exists; reusing.", overlay_tag=overlay_tag
        )
        return overlay_tag

    logger.info(
        "[{instance}] Building Qwen Code overlay image on top of {base_tag}",
        instance=instance.instance_id,
        base_tag=base_tag,
    )

    dockerfile = dedent(
        f"""
        FROM {base_tag}

        USER root

        ENV NVM_DIR=/usr/local/nvm
        ENV NODE_VERSION={NODE_VERSION}
        ENV NVM_INSTALLER=https://raw.githubusercontent.com/nvm-sh/nvm/v{NVM_VERSION}/install.sh
        ENV QWEN_CODE_PACKAGE=@qwen-code/qwen-code@{QWEN_CODE_VERSION}

        RUN apt-get update \\
            && apt-get install -y --no-install-recommends curl ca-certificates \\
            && rm -rf /var/lib/apt/lists/* \\
            && mkdir -p $NVM_DIR \\
            && curl -fsSL $NVM_INSTALLER | bash

        RUN /bin/bash -c 'set -e && source $NVM_DIR/nvm.sh && nvm install $NODE_VERSION'
        RUN /bin/bash -c 'source $NVM_DIR/nvm.sh && nvm alias default $NODE_VERSION'
        RUN /bin/bash -c 'source $NVM_DIR/nvm.sh && npm install -g $QWEN_CODE_PACKAGE'
        RUN chown -R node:node $NVM_DIR

        ENV PATH="/usr/local/nvm/versions/node/v{NODE_VERSION}/bin:$PATH"

        USER node
        """
    )

    docker_manager.build_image(dockerfile_content=dockerfile, tag=overlay_tag)
    return overlay_tag
