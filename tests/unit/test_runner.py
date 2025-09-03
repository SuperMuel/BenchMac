from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from bench_mac.models import (
    CommandResult,
    EvaluationCompleted,
    EvaluationFailed,
    EvaluationReport,
    EvaluationResult,
    EvaluationTask,
    ExecutionTrace,
    MetricsReport,
    Submission,
)
from bench_mac.runner import BenchmarkRunner, WorkerContext

# --- Test Fixtures and Fake Data ---


@pytest.fixture
def sample_tasks(instance_factory: Any) -> list[EvaluationTask]:
    """Provides a simple list of two tasks for testing."""
    instance1 = instance_factory.create_instance(
        instance_id="task-1-success",
        repo="owner/repo",
        base_commit="1234567890",
        source_angular_version="15",
        target_angular_version="16",
    )
    submission1 = Submission(instance_id="task-1-success", model_patch="...")

    instance2 = instance_factory.create_instance(
        instance_id="task-2-failure",
        repo="owner/repo",
        base_commit="1234567890",
        source_angular_version="16",
        target_angular_version="17",
    )
    submission2 = Submission(instance_id="task-2-failure", model_patch="...")

    return [
        EvaluationTask(instance=instance1, submission=submission1),
        EvaluationTask(instance=instance2, submission=submission2),
    ]


def fake_run_single_evaluation_task(context: WorkerContext) -> EvaluationResult:
    """
    A fake worker function that replaces the real one during tests.
    It returns predictable results without any I/O or Docker calls.
    """
    if "success" in context.task.instance.instance_id:
        # Simulate a successful evaluation with a mock execution trace
        successful_command = CommandResult(
            command="git apply -p0 /tmp/patch.patch",
            exit_code=0,
            stdout="Applied patch successfully",
            stderr="",
            start_time=datetime.now(UTC),
            end_time=datetime.now(UTC) + timedelta(seconds=1),
        )
        execution = ExecutionTrace(steps=[successful_command])

        return EvaluationCompleted(
            result=EvaluationReport(
                instance_id=context.task.instance.instance_id,
                submission_id=context.task.submission.submission_id,
                execution=execution,
                metrics=MetricsReport(
                    patch_application_success=True,
                ),
            ),
        )
    else:
        # Simulate a harness-level failure
        return EvaluationFailed(
            instance_id=context.task.instance.instance_id,
            submission_id=context.task.submission.submission_id,
            error="Simulated worker crash",
        )


# --- Test Class ---


@pytest.mark.unit
class TestBenchmarkRunner:
    def test_run_orchestration_and_callbacks(
        self,
        monkeypatch: pytest.MonkeyPatch,
        sample_tasks: list[EvaluationTask],
    ):
        """
        Verify that the runner correctly executes tasks in parallel and
        invokes the on_result and on_progress callbacks for each task.
        """
        # --- ARRANGE ---

        # 1. Replace the real, slow worker function with our fast, fake one.
        monkeypatch.setattr(
            "bench_mac.runner.run_single_evaluation_task",
            fake_run_single_evaluation_task,
        )

        # 2. Prepare lists to capture the data passed to our callbacks.
        results_log: list[EvaluationResult] = []

        def on_result_callback(outcome: EvaluationResult):
            """A fake callback that just records the result."""
            results_log.append(outcome)

        # 3. Instantiate the runner.
        runner = BenchmarkRunner(workers=2)

        # --- ACT ---
        runner.run(
            tasks=sample_tasks,
            log_dir=Path("test_logs"),
            run_id="test-run-123",
            on_result=on_result_callback,
        )

        # --- ASSERT ---

        # 1. Assert on the results.
        assert len(results_log) == 2  # Both tasks should have produced a result.

        # Sort results by instance_id to make assertions deterministic, as
        # the parallel execution order is not guaranteed.
        sorted_results = sorted(
            results_log,
            key=lambda r: r.instance_id
            if isinstance(r, EvaluationFailed)
            else r.result.instance_id,
        )

        # Check the failure case
        failure_result = sorted_results[1]
        assert isinstance(failure_result, EvaluationFailed)
        assert failure_result.instance_id == "task-2-failure"
        assert failure_result.error == "Simulated worker crash"

        # Check the success case
        success_result = sorted_results[0]
        assert isinstance(success_result, EvaluationCompleted)
        assert success_result.result.instance_id == "task-1-success"
        assert success_result.result.metrics.patch_application_success is True
