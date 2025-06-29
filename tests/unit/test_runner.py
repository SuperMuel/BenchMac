from pathlib import Path

import pytest

from bench_mac.models import (
    BenchmarkInstance,
    EvaluationFailure,
    EvaluationResult,
    EvaluationSuccess,
    EvaluationTask,
    MetricsReport,
    RunOutcome,
    Submission,
)
from bench_mac.runner import BenchmarkRunner, WorkerContext

# --- Test Fixtures and Fake Data ---


@pytest.fixture
def sample_tasks() -> list[EvaluationTask]:
    """Provides a simple list of two tasks for testing."""
    instance1 = BenchmarkInstance.model_validate(
        {
            "instance_id": "task-1-success",
            "repo": "owner/repo",
            "base_commit": "1234567890",
            "source_angular_version": "15",
            "target_angular_version": "16",
            "target_node_version": "18.10.0",
        }
    )
    submission1 = Submission(instance_id="task-1-success", model_patch="...")

    instance2 = BenchmarkInstance.model_validate(
        {
            "instance_id": "task-2-failure",
            "repo": "owner/repo",
            "base_commit": "1234567890",
            "source_angular_version": "16",
            "target_angular_version": "17",
            "target_node_version": "18.13.0",
        }
    )
    submission2 = Submission(instance_id="task-2-failure", model_patch="...")

    return [
        EvaluationTask(instance=instance1, submission=submission1),
        EvaluationTask(instance=instance2, submission=submission2),
    ]


def fake_run_single_evaluation_task(context: WorkerContext) -> RunOutcome:
    """
    A fake worker function that replaces the real one during tests.
    It returns predictable results without any I/O or Docker calls.
    """
    if "success" in context.task.instance.instance_id:
        # Simulate a successful evaluation
        return EvaluationSuccess(
            result=EvaluationResult(
                instance_id=context.task.instance.instance_id,
                metrics=MetricsReport(
                    patch_application_success=True,
                ),
                logs={},
            ),
        )
    else:
        # Simulate a harness-level failure
        return EvaluationFailure(
            instance_id=context.task.instance.instance_id,
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
        results_log: list[RunOutcome] = []
        progress_log: list[tuple[int, int]] = []

        def on_result_callback(outcome: RunOutcome):
            """A fake callback that just records the result."""
            results_log.append(outcome)

        def on_progress_callback(completed: int, total: int):
            """A fake callback that just records the progress update."""
            progress_log.append((completed, total))

        # 3. Instantiate the runner.
        runner = BenchmarkRunner(workers=2)

        # --- ACT ---
        runner.run(
            tasks=sample_tasks,
            log_dir=Path("test_logs"),
            run_id="test-run-123",
            on_result=on_result_callback,
            on_progress=on_progress_callback,
        )

        # --- ASSERT ---

        # 1. Assert on the results.
        assert len(results_log) == 2  # Both tasks should have produced a result.

        # Sort results by instance_id to make assertions deterministic, as
        # the parallel execution order is not guaranteed.
        sorted_results = sorted(
            results_log,
            key=lambda r: r.instance_id
            if isinstance(r, EvaluationFailure)
            else r.result.instance_id,
        )

        # Check the failure case
        failure_result = sorted_results[1]
        assert isinstance(failure_result, EvaluationFailure)
        assert failure_result.instance_id == "task-2-failure"
        assert failure_result.error == "Simulated worker crash"

        # Check the success case
        success_result = sorted_results[0]
        assert isinstance(success_result, EvaluationSuccess)
        assert success_result.result.instance_id == "task-1-success"
        assert success_result.result.metrics.patch_application_success is True

        # 2. Assert on the progress.
        assert len(progress_log) == 2  # The callback should have been called twice.
        # The progress updates should be (1, 2) and (2, 2). Sort to ensure order.
        assert sorted(progress_log) == [(1, 2), (2, 2)]
