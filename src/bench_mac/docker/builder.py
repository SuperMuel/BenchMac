import hashlib

from loguru import logger
from slugify import slugify

from bench_mac.docker.manager import DockerManager
from bench_mac.models import BenchmarkInstance


# TODO: check if we should use the term "reference" instead of "tag"
def _get_instance_image_tag(
    instance: BenchmarkInstance, dockerfile_content: str
) -> str:
    """
    Generates a consistent, unique tag for an Instance Image using its ID
    and a hash of its Dockerfile content.
    """
    instance_slug = slugify(instance.instance_id)  # slugify for extra security

    # Calculate a hash of the Dockerfile content
    content_hash = hashlib.sha256(dockerfile_content.encode("utf-8")).hexdigest()
    short_hash = content_hash[:8]  # Use the first 8 characters for brevity

    tag = f"benchmac-instance:{instance_slug}-{short_hash}"

    if len(tag) > 128:
        # TODO: if it's actually a reference or name and not a tag,
        # maybe the limit is different
        raise ValueError(f"Generated image tag is too long: {tag=}")

    return tag


def prepare_environment(instance: BenchmarkInstance, manager: DockerManager) -> str:
    """
    Ensures the Docker image for a given instance exists, building it from its
    dedicated Dockerfile if necessary. The image tag includes a hash of the
    Dockerfile's content to ensure changes are automatically rebuilt.

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
    logger.debug(f"Preparing environment for instance: {instance.instance_id}")

    # 1. Get the Dockerfile content first
    dockerfile_content = instance.dockerfile_content

    # 2. Generate the content-aware image tag
    instance_image_tag = _get_instance_image_tag(instance, dockerfile_content)

    # 3. Check if this specific version of the image already exists
    if manager.image_exists(instance_image_tag):
        logger.debug(f"Image '{instance_image_tag}' already exists. Skipping build.")
        return instance_image_tag

    logger.debug(f"Image '{instance_image_tag}' not found. Building...")

    manager.build_image(dockerfile_content=dockerfile_content, tag=instance_image_tag)

    logger.success(f"✅ Environment ready. Final image tag: {instance_image_tag}")
    return instance_image_tag
