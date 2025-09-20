from pathlib import Path
from types import TracebackType
from typing import Any, Self

from bench_mac.config import settings
from bench_mac.docker.manager import DockerManager
from bench_mac.domain.services.ports import EnvironmentFactory, ExecutionEnvironment
from bench_mac.environment import InstanceEnvironment
from bench_mac.models import BenchmarkInstance, CommandResult, ExecutionTrace


class DockerExecutionEnvironment(ExecutionEnvironment):
    """ExecutionEnvironment backed by an InstanceEnvironment."""

    def __init__(
        self,
        instance: BenchmarkInstance,
        manager: DockerManager,
        *,
        project_dir: str | None = None,
        logger: Any | None = None,
        auto_remove: bool = False,
    ) -> None:
        resolved_project_dir = project_dir or settings.project_workdir
        self._instance_env = InstanceEnvironment(
            instance,
            manager,
            project_dir=resolved_project_dir,
            logger=logger,
            auto_remove=auto_remove,
        )

    def __enter__(self) -> Self:
        self._instance_env.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self._instance_env.__exit__(exc_type, exc_val, exc_tb)

    @property
    def project_dir(self) -> str:
        return self._instance_env.project_dir

    def exec(
        self,
        cmd: str,
        *,
        workdir: str | None = None,
        trace: bool = True,
    ) -> CommandResult:
        return self._instance_env.exec(cmd, workdir=workdir, trace=trace)

    def copy_in(self, local_src_path: Path, dest_path: str) -> None:
        self._instance_env.copy_in(local_src_path, dest_path)

    def trace(self) -> ExecutionTrace:
        return self._instance_env.trace()


class DockerEnvironmentFactory(EnvironmentFactory):
    """EnvironmentFactory that produces Docker-backed environments."""

    def __init__(
        self,
        manager: DockerManager,
        *,
        project_dir: str | None = None,
        logger: Any | None = None,
        auto_remove: bool = False,
    ) -> None:
        self._manager = manager
        self._project_dir = project_dir or settings.project_workdir
        self._logger = logger
        self._auto_remove = auto_remove

    def create(self, instance: BenchmarkInstance) -> ExecutionEnvironment:
        return DockerExecutionEnvironment(
            instance,
            self._manager,
            project_dir=self._project_dir,
            logger=self._logger,
            auto_remove=self._auto_remove,
        )
