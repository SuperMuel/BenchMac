from collections.abc import Callable
from dataclasses import dataclass

from bench_mac.core.models import BenchmarkInstance, CommandResult, ExecutionTrace


@dataclass(frozen=True, slots=True)
class TraceAnalyzer:
    """Domain-oriented helpers for locating meaningful steps in a trace."""

    trace: ExecutionTrace
    instance: BenchmarkInstance

    PATCH_CHECK_PREFIX = "git apply --check"
    PATCH_APPLY_PREFIX = "git apply -p0"
    VERSION_COMMAND = "npm ls @angular/cli @angular/core --json"

    def patch_check_step(self) -> CommandResult | None:
        return self._first(self._matches_patch_check)

    def patch_apply_step(self) -> CommandResult | None:
        return self._first(self._matches_patch_apply)

    def install_steps(self) -> list[CommandResult]:
        install_cmd = self.instance.commands.install.strip()
        return [
            step
            for step in self.trace.steps
            if self._matches_command_with_optional_suffix(step.command, install_cmd)
        ]

    def final_install_step(self) -> CommandResult | None:
        installs = self.install_steps()
        if not installs:
            return None
        return installs[-1]

    def version_check_step(self) -> CommandResult | None:
        return self._first(lambda step: self.VERSION_COMMAND in step.command)

    def build_step(self) -> CommandResult | None:
        build_cmd = self.instance.commands.build.strip()
        return self._first(
            lambda step: self._matches_command_with_optional_suffix(
                step.command, build_cmd
            )
        )

    # ----------------------- internal helpers -----------------------

    def _first(
        self, predicate: Callable[[CommandResult], bool]
    ) -> CommandResult | None:
        for step in self.trace.steps:
            if predicate(step):
                return step
        return None

    def _matches_patch_check(self, step: CommandResult) -> bool:
        return self.PATCH_CHECK_PREFIX in step.command

    def _matches_patch_apply(self, step: CommandResult) -> bool:
        return self.PATCH_APPLY_PREFIX in step.command

    @staticmethod
    def _matches_command_with_optional_suffix(command: str, base_command: str) -> bool:
        if command.strip() == base_command:
            return True
        prefixed = f"{base_command} "
        return command.startswith(prefixed)
