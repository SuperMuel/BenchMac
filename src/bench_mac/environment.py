from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger as _default_logger

from bench_mac.docker.builder import prepare_environment
from bench_mac.docker.manager import DockerManager
from bench_mac.models import (
    BenchmarkInstance,
    CommandResult,
    ExecutionTrace,
    utc_now,
)

if TYPE_CHECKING:  # pragma: no cover
    from docker.models.containers import Container


class _EnvState(str, Enum):
    INIT = "init"
    STARTED = "started"
    CLOSED = "closed"


class InstanceEnvironment:
    def __init__(
        self,
        instance: BenchmarkInstance,
        manager: DockerManager,
        *,
        project_dir: str = "/app/project",
        logger: Any | None = None,
        auto_remove: bool = False,
    ) -> None:
        self._instance = instance
        self._manager = manager
        self._project_dir = project_dir
        self._logger = logger or _default_logger
        self._auto_remove = auto_remove

        self._image_tag: str | None = None
        self._container: Container | None = None
        self._steps: list[CommandResult] = []
        self._state: _EnvState = _EnvState.INIT

    # ----------------------- Lifecycle (context manager) -----------------------

    def __enter__(self) -> "InstanceEnvironment":
        return self.start()

    def __exit__(self, *_: Any) -> None:
        self.close()

    def start(self) -> "InstanceEnvironment":
        """Ensure image exists and start a container (idempotent for a *live* env)."""
        if self._state is _EnvState.CLOSED:
            raise RuntimeError(
                "This InstanceEnvironment has been closed and cannot be restarted. "
                "Create a new InstanceEnvironment for another run."
            )
        if self._state is _EnvState.STARTED:
            return self
        self._ensure_image()
        self._ensure_container()
        self._state = _EnvState.STARTED
        return self

    def close(self) -> None:
        """Stop and remove the container if present; make environment terminal."""
        if self._state is _EnvState.CLOSED:
            return
        try:
            if self._container is not None:
                self._manager.cleanup_container(self._container)
        finally:
            self._container = None
            self._state = _EnvState.CLOSED

    # ----------------------- Core operations ----------------------------------

    def exec(
        self,
        cmd: str,
        *,
        workdir: str | None = None,
        trace: bool = True,
    ) -> CommandResult:
        """
        Execute a shell command inside the container and return a CommandResult.
        By default, records the step into the environment's trace.
        """
        self._ensure_container()
        assert self._container is not None
        workdir = workdir or self._project_dir

        self._logger.debug(
            f"Executing in {self._container.short_id} ({workdir=!r}): {cmd=!r}"
        )

        start = utc_now()
        exit_code, stdout, stderr = self._manager.execute_in_container(
            self._container, cmd, workdir=workdir
        )
        end = utc_now()

        result = CommandResult(
            command=cmd,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            start_time=start,
            end_time=end,
        )
        if trace:
            self._steps.append(result)

        self._logger.debug(
            f"Command exit={result.exit_code} (len(stdout)={len(result.stdout)}, "
            f"len(stderr)={len(result.stderr)})"
        )
        return result

    def copy_in(self, local_src_path: Path, container_dest_path: str) -> None:
        """
        Copy a local file or directory *into* the container at container_dest_path.
        """
        if not local_src_path.exists():
            raise FileNotFoundError(f"Local path does not exist: {local_src_path}")
        self._ensure_container()
        assert self._container is not None
        self._manager.copy_to_container(
            self._container, src_path=local_src_path, dest_path=container_dest_path
        )

    # ----------------------- Results ------------------------------------------

    def trace(self) -> ExecutionTrace:
        """Return the accumulated execution trace."""
        return ExecutionTrace(steps=list(self._steps))

    # ----------------------- Introspection ------------------------------------

    @property
    def project_dir(self) -> str:
        return self._project_dir

    @property
    def image_tag(self) -> str | None:
        return self._image_tag

    @property
    def container_id(self) -> str | None:
        if self._container is None:
            return None
        return getattr(self._container, "short_id", None)

    # ----------------------- Internals ----------------------------------------

    def _ensure_image(self) -> str:
        """Build-or-reuse the instance image; memoized."""
        if self._image_tag is not None:
            return self._image_tag
        self._logger.debug(
            f"Preparing image for instance: {self._instance.instance_id}"
        )
        tag = prepare_environment(self._instance, self._manager)
        assert tag, "Failed to prepare environment (empty image tag)."
        self._image_tag = tag
        return tag

    def _ensure_container(self) -> None:
        """Start the container if not already running (only while live)."""
        if self._state is _EnvState.CLOSED:
            raise RuntimeError(
                "InstanceEnvironment is closed; cannot start or use a container."
            )
        if self._container is not None:
            return
        tag = self._ensure_image()
        self._logger.debug(f"Starting container from image: {tag}")
        self._container = self._manager.run_container(
            tag, auto_remove=self._auto_remove
        )

    @property
    def state(self) -> str:
        return self._state.value
