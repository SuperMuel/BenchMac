import json
from pathlib import Path

import pytest
from loguru import logger

from bench_mac.domain.services.ports import EnvironmentFactory, ExecutionEnvironment
from bench_mac.models import (
    BenchmarkInstance,
    CommandResult,
    ExecutionTrace,
    InstanceID,
    Submission,
)
from bench_mac.use_cases.evaluate_submission import evaluate_submission
from tests.conftest import InstanceFactory

from ...utils import create_command_output


class FakeExecutionEnvironment(ExecutionEnvironment):
    """Minimal in-memory environment used to exercise the use case."""

    def __init__(self, result_queue: list[CommandResult]):
        self._result_queue = list(result_queue)
        self._steps: list[CommandResult] = []
        self._copied_files: list[tuple[Path, str]] = []
        self.closed = False

    def __enter__(self) -> "FakeExecutionEnvironment":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.closed = True

    @property
    def project_dir(self) -> str:
        return "/app/project"

    def exec(
        self,
        cmd: str,
        *,
        workdir: str | None = None,
        trace: bool = True,
    ) -> CommandResult:
        if not self._result_queue:
            raise AssertionError(
                "No more command results configured for FakeExecutionEnvironment"
            )

        template = self._result_queue.pop(0)
        result = CommandResult(
            command=cmd,
            exit_code=template.exit_code,
            stdout=template.stdout,
            stderr=template.stderr,
            start_time=template.start_time,
            end_time=template.end_time,
        )
        if trace:
            self._steps.append(result)
        return result

    def copy_in(self, local_src_path: Path, dest_path: str) -> None:
        self._copied_files.append((local_src_path, dest_path))

    def trace(self) -> ExecutionTrace:
        return ExecutionTrace(steps=list(self._steps))

    @property
    def copied_files(self) -> list[tuple[Path, str]]:
        return list(self._copied_files)


class FakeEnvironmentFactory(EnvironmentFactory):
    def __init__(self, environment: ExecutionEnvironment):
        self._environment = environment

    def create(self, instance: BenchmarkInstance) -> ExecutionEnvironment:
        return self._environment


@pytest.mark.unit
class TestEvaluateSubmissionUseCase:
    def test_success_path_executes_all_steps(
        self,
        instance_factory: InstanceFactory,
    ) -> None:
        instance: BenchmarkInstance = instance_factory.create_instance(
            instance_id=InstanceID("my-project_v15_to_v16"),
            override_dockerfile_content="FROM node:18\n",
        )
        submission = Submission(
            instance_id=instance.instance_id,
            model_patch="diff --git a/file.ts b/file.ts\n--- a/file.ts\n+++ b/file.ts",
        )
        version_payload = json.dumps(
            {
                "dependencies": {
                    "@angular/core": {"version": instance.target_angular_version}
                }
            }
        )
        results = [
            create_command_output("git apply --check", 0),
            create_command_output("git apply", 0),
            create_command_output(instance.commands.install, 0),
            create_command_output("npm ls", 0, stdout=version_payload),
            create_command_output(instance.commands.build, 0),
        ]
        env = FakeExecutionEnvironment(results)
        factory = FakeEnvironmentFactory(env)

        trace = evaluate_submission(
            instance,
            submission,
            factory,
            logger=logger,
        )

        executed_commands = [step.command for step in trace.steps]
        assert len(executed_commands) == 5
        assert executed_commands[0].startswith("git apply --check")
        assert executed_commands[-1] == instance.commands.build
        assert env.copied_files  # patch file copied into environment

    def test_halts_when_patch_check_fails(
        self,
        instance_factory: InstanceFactory,
    ) -> None:
        instance: BenchmarkInstance = instance_factory.create_instance(
            instance_id=InstanceID("my-project_v15_to_v16"),
            override_dockerfile_content="FROM node:18\n",
        )
        submission = Submission(
            instance_id=instance.instance_id,
            model_patch="diff --git a/file.ts b/file.ts\n--- a/file.ts\n+++ b/file.ts",
        )
        results = [
            create_command_output(
                "git apply --check",
                exit_code=1,
                stderr="patch failed",
            )
        ]
        env = FakeExecutionEnvironment(results)
        factory = FakeEnvironmentFactory(env)

        trace = evaluate_submission(
            instance,
            submission,
            factory,
            logger=logger,
        )

        assert len(trace.steps) == 1
        assert trace.steps[0].command.startswith("git apply --check")
        assert env.closed is True

    def test_retries_install_with_peer_dependency_flag(
        self,
        instance_factory: InstanceFactory,
    ) -> None:
        instance: BenchmarkInstance = instance_factory.create_instance(
            instance_id=InstanceID("my-project_v15_to_v16"),
            override_dockerfile_content="FROM node:18\n",
        )
        submission = Submission(
            instance_id=instance.instance_id,
            model_patch="diff --git a/file.ts b/file.ts\n--- a/file.ts\n+++ b/file.ts",
        )
        version_payload = json.dumps(
            {
                "dependencies": {
                    "@angular/core": {"version": instance.target_angular_version}
                }
            }
        )
        peer_dep_failure = create_command_output(
            instance.commands.install,
            exit_code=1,
            stderr="ERESOLVE conflicting peer dependency",
        )
        fallback_success = create_command_output(
            f"{instance.commands.install} --legacy-peer-deps",
            exit_code=0,
        )
        results = [
            create_command_output("git apply --check", 0),
            create_command_output("git apply", 0),
            peer_dep_failure,
            fallback_success,
            create_command_output("npm ls", 0, stdout=version_payload),
            create_command_output(instance.commands.build, 0),
        ]
        env = FakeExecutionEnvironment(results)
        factory = FakeEnvironmentFactory(env)

        trace = evaluate_submission(
            instance,
            submission,
            factory,
            logger=logger,
        )

        install_commands = [
            step.command for step in trace.steps if "install" in step.command
        ]
        assert install_commands == [
            instance.commands.install,
            f"{instance.commands.install} --legacy-peer-deps",
        ]
        assert trace.steps[-1].command == instance.commands.build
