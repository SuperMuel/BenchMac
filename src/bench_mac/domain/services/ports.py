from pathlib import Path
from types import TracebackType
from typing import Protocol, Self

from bench_mac.models import BenchmarkInstance, CommandResult, ExecutionTrace


class ExecutionEnvironment(Protocol):
    """Abstracts the runtime used to execute benchmark commands."""

    @property
    def project_dir(self) -> str:  # pragma: no cover - simple delegation
        """Return the working directory inside the environment."""
        ...

    def __enter__(self) -> Self: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None: ...

    def exec(
        self,
        cmd: str,
        *,
        workdir: str | None = None,
        trace: bool = True,
    ) -> CommandResult:
        """Execute a shell command and return its result."""
        ...

    def copy_in(self, local_src_path: Path, dest_path: str) -> None:
        """Copy a file into the environment."""

    def trace(self) -> ExecutionTrace:
        """Return the accumulated execution trace."""
        ...


class EnvironmentFactory(Protocol):
    """Factory responsible for creating execution environments."""

    def create(self, instance: BenchmarkInstance) -> ExecutionEnvironment:
        """Build an execution environment for the provided instance."""
        ...
