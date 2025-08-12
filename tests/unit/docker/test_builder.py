import hashlib
from typing import Any, cast

import pytest
from slugify import slugify

from bench_mac.docker.builder import (
    _get_instance_image_tag,
    prepare_environment,
)
from bench_mac.docker.manager import DockerManager


class FakeDockerManager:
    def __init__(self, existing_tags: set[str] | None = None) -> None:
        self.existing_tags = existing_tags or set()
        self.last_built: dict[str, str] | None = None

    def image_exists(self, tag: str) -> bool:  # pragma: no cover - trivial
        return tag in self.existing_tags

    def build_image(self, dockerfile_content: str, tag: str) -> None:
        self.last_built = {"dockerfile_content": dockerfile_content, "tag": tag}
        self.existing_tags.add(tag)


@pytest.mark.unit
class TestGetInstanceImageTag:
    def test_generates_slug_and_content_hash(self, instance_factory: Any) -> None:
        instance = instance_factory.create_instance(
            instance_id="test-instance",
            override_dockerfile_content="FROM node:18\n",
        )

        expected_slug = slugify(instance.instance_id)
        expected_hash8 = hashlib.sha256(
            instance.dockerfile_content.encode("utf-8")
        ).hexdigest()[:8]

        result = _get_instance_image_tag(instance, instance.dockerfile_content)

        assert result == f"benchmac-instance:{expected_slug}-{expected_hash8}"

    def test_different_dockerfile_contents_produce_different_tags(
        self, instance_factory: Any
    ) -> None:
        a = instance_factory.create_instance(
            instance_id="same-id",
            override_dockerfile_content="FROM node:18\n",
        )
        b = instance_factory.create_instance(
            instance_id="same-id",
            override_dockerfile_content="FROM node:20\n",
        )

        tag_a = _get_instance_image_tag(a, a.dockerfile_content)
        tag_b = _get_instance_image_tag(b, b.dockerfile_content)

        assert tag_a != tag_b
        assert tag_a.startswith("benchmac-instance:same-id-")
        assert tag_b.startswith("benchmac-instance:same-id-")

    def test_tag_length_exceeds_limit_raises_error(self, instance_factory: Any) -> None:
        very_long_id = "x" * 200
        instance = instance_factory.create_instance(
            instance_id=very_long_id,
            override_dockerfile_content="FROM node:18\n",
        )

        with pytest.raises(ValueError, match="Generated image tag is too long"):
            _get_instance_image_tag(instance, instance.dockerfile_content)


@pytest.mark.unit
class TestPrepareEnvironment:
    def test_skips_build_when_image_already_exists(self, instance_factory: Any) -> None:
        instance = instance_factory.create_instance(
            instance_id="exists",
            override_dockerfile_content="FROM node:18\n",
        )
        expected_tag = _get_instance_image_tag(instance, instance.dockerfile_content)
        manager = FakeDockerManager(existing_tags={expected_tag})

        final_tag = prepare_environment(instance, cast(DockerManager, manager))

        assert final_tag == expected_tag
        assert manager.last_built is None

    def test_builds_image_when_missing(self, instance_factory: Any) -> None:
        instance = instance_factory.create_instance(
            instance_id="needs-build",
            override_dockerfile_content="FROM node:20\n",
        )
        manager = FakeDockerManager()

        final_tag = prepare_environment(instance, cast(DockerManager, manager))

        assert manager.last_built is not None
        assert manager.last_built["dockerfile_content"] == instance.dockerfile_content
        assert manager.last_built["tag"] == final_tag
