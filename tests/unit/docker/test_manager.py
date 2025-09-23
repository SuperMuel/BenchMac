import io
import tarfile
from pathlib import Path
from typing import cast

import pytest
from docker import DockerClient
from docker.errors import ImageNotFound, NotFound
from docker.models.containers import Container

from bench_mac.docker.manager import DockerManager


class FakeImage:
    def __init__(self, tag: str) -> None:
        self.tag = tag
        self.short_id = f"img-{tag}"


class FakeImages:
    def __init__(self, existing: set[str] | None = None) -> None:
        self._existing = set(existing or [])
        self.removed: list[tuple[str, bool]] = []
        self.built: list[dict[str, object]] = []

    def get(self, tag: str) -> FakeImage:
        if tag not in self._existing:
            raise ImageNotFound(f"missing: {tag}")
        return FakeImage(tag)

    def build(
        self, path: str, tag: str, rm: bool, forcerm: bool
    ) -> tuple[FakeImage, list[dict[str, str]]]:
        dockerfile = Path(path) / "Dockerfile"
        content = dockerfile.read_text(encoding="utf-8")
        self.built.append(
            {
                "path": Path(path),
                "tag": tag,
                "content": content,
                "rm": rm,
                "forcerm": forcerm,
            }
        )
        self._existing.add(tag)
        return FakeImage(tag), [{"stream": "Step 1"}, {"stream": "Step 2"}]

    def remove(self, tag: str, force: bool) -> None:
        self.removed.append((tag, force))
        self._existing.discard(tag)


class FakeContainer:
    def __init__(self, short_id: str = "abc123", status: str = "created") -> None:
        self.short_id = short_id
        self.status = status
        self.exec_calls: list[tuple[list[str], bool, str | None]] = []
        self.reload_called = False
        self.stop_called = False
        self.remove_called = False
        self._archive_bytes: bytes | None = None

    def exec_run(self, command: list[str], demux: bool, workdir: str | None):
        self.exec_calls.append((command, demux, workdir))
        return 42, (b"stdout\n", b"stderr\n")

    def put_archive(self, path: str, data: io.BytesIO) -> bool:
        data.seek(0)
        self._archive_bytes = data.read()
        return True

    def reload(self) -> None:
        self.reload_called = True

    def stop(self) -> None:
        self.status = "exited"
        self.stop_called = True

    def remove(self) -> None:
        self.remove_called = True

    def raise_not_found(self) -> None:
        raise NotFound("gone")


class FakeContainers:
    def __init__(self, container: FakeContainer | None = None) -> None:
        self.container = container or FakeContainer()
        self.run_calls: list[dict[str, object]] = []

    def run(
        self, image_tag: str, command: str, detach: bool, auto_remove: bool
    ) -> FakeContainer:
        self.run_calls.append(
            {
                "image_tag": image_tag,
                "command": command,
                "detach": detach,
                "auto_remove": auto_remove,
            }
        )
        self.container.status = "running"
        return self.container


class FakeClient:
    def __init__(
        self, images: FakeImages | None = None, containers: FakeContainers | None = None
    ) -> None:
        self.images = images or FakeImages()
        self.containers = containers or FakeContainers()


def make_manager(
    images: FakeImages | None = None, containers: FakeContainers | None = None
) -> DockerManager:
    return DockerManager(
        client=cast(
            DockerClient,
            FakeClient(images=images, containers=containers),
        )
    )


@pytest.mark.unit
def test_image_exists_handles_present_and_missing() -> None:
    images = FakeImages(existing={"bench:latest"})
    manager = make_manager(images=images)

    assert manager.image_exists("bench:latest") is True
    assert manager.image_exists("bench:missing") is False


@pytest.mark.unit
def test_build_image_records_dockerfile_contents() -> None:
    images = FakeImages()
    manager = make_manager(images=images)

    dockerfile = "FROM python:3.11-slim\nRUN echo hi\n"
    image = manager.build_image(dockerfile, tag="bench:test")

    assert image.tag == "bench:test"
    assert images.built, "expected build to be recorded"
    build = images.built[-1]
    assert build["tag"] == "bench:test"
    assert build["content"] == dockerfile
    assert build["rm"] is True and build["forcerm"] is True


@pytest.mark.unit
def test_remove_image_skips_missing_and_forces_known_image() -> None:
    images = FakeImages(existing={"bench:old"})
    manager = make_manager(images=images)

    manager.remove_image("bench:old")
    manager.remove_image("bench:missing")

    assert images.removed == [("bench:old", True)]


@pytest.mark.unit
def test_run_and_execute_in_container() -> None:
    fake_container = FakeContainer()
    containers = FakeContainers(container=fake_container)
    manager = make_manager(containers=containers)

    container = manager.run_container("bench:tag", auto_remove=True)
    exit_code, stdout, stderr = manager.execute_in_container(
        container, "echo test", workdir="/src"
    )

    assert container is fake_container
    assert containers.run_calls and containers.run_calls[0]["auto_remove"] is True
    assert exit_code == 42
    assert stdout == "stdout\n"
    assert stderr == "stderr\n"
    command_call = fake_container.exec_calls[-1]
    assert command_call == (["/bin/sh", "-c", "echo test"], True, "/src")


@pytest.mark.unit
def test_copy_to_container_packages_single_file(tmp_path: Path) -> None:
    payload = tmp_path / "hello.txt"
    payload.write_text("hello docker", encoding="utf-8")

    fake_container = FakeContainer()
    manager = make_manager(containers=FakeContainers(container=fake_container))

    manager.copy_to_container(
        cast(Container, fake_container), payload, "/app/hello.txt"
    )

    assert fake_container._archive_bytes is not None
    tar_stream = io.BytesIO(fake_container._archive_bytes)
    with tarfile.open(fileobj=tar_stream) as tar:
        names = tar.getnames()
        assert names == ["hello.txt"]
        extracted = tar.extractfile("hello.txt")
        assert extracted is not None
        assert extracted.read().decode("utf-8") == "hello docker"


@pytest.mark.unit
def test_cleanup_container_stops_running_container() -> None:
    fake_container = FakeContainer(status="running")
    manager = make_manager()

    manager.cleanup_container(cast(Container, fake_container))

    assert fake_container.reload_called is True
    assert fake_container.stop_called is True
    assert fake_container.remove_called is True
    assert fake_container.status == "exited"


@pytest.mark.unit
def test_cleanup_container_handles_missing_container() -> None:
    class MissingContainer(FakeContainer):
        def reload(self) -> None:  # type: ignore[override]
            self.reload_called = True
            self.raise_not_found()

    fake_container = MissingContainer(status="running")
    manager = make_manager()

    manager.cleanup_container(cast(Container, fake_container))

    assert fake_container.reload_called is True
    assert fake_container.stop_called is False
    assert fake_container.remove_called is False
