import os
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed

from bench_mac.docker.manager import DockerManager
from bench_mac.evaluator import evaluate_submission
from bench_mac.models import (
    BenchmarkInstance,
    EvaluationFailure,
    EvaluationSuccess,
    RunOutcome,
    Submission,
)


def run_single_evaluation_task(
    instance: BenchmarkInstance, submission: Submission
) -> RunOutcome:
    """
    A top-level function designed to be run in a separate process.

    It initializes its own DockerManager and calls the core evaluation logic,
    wrapping the result in a RunOutcome object to handle success or failure.
    """
    try:
        docker_manager = DockerManager(quiet_init=True)
        result = evaluate_submission(instance, submission, docker_manager)
        return EvaluationSuccess(result=result)
    except Exception as e:
        # Catch any unexpected crash in the worker process
        return EvaluationFailure(
            instance_id=instance.instance_id,
            error=f"Worker process crashed: {e}",
        )


class BenchmarkRunner:
    """Orchestrates the parallel execution of evaluation tasks."""

    def __init__(self, workers: int | None = None):
        """
        Initializes the runner with a specific number of parallel workers.
        """
        self.workers = workers or os.cpu_count() or 1
        print(f"BenchmarkRunner initialized with {self.workers} worker(s).")

    def run(
        self,
        tasks: Sequence[tuple[BenchmarkInstance, Submission]],
        on_result: Callable[[RunOutcome], None],
        on_progress: Callable[[int, int], None] | None = None,
    ) -> None:
        """
        Executes the evaluation for a given set of tasks.

        Args:
            tasks: An iterable of (instance, submission) tuples to be evaluated.
            on_result: A callback function that is invoked with the RunOutcome
                       of each completed task.
            on_progress: An optional callback function to report progress, taking
                         (completed_count, total_count) as arguments.
        """
        task_list = list(tasks)
        total_tasks = len(task_list)
        if total_tasks == 0:
            print("No tasks to evaluate. Exiting.")
            return

        print(f"Starting evaluation of {total_tasks} tasks...")

        # Fail if the Docker daemon is not running.
        DockerManager.get_client(quiet=False)

        completed_count = 0
        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            futures = {
                executor.submit(run_single_evaluation_task, instance, submission)
                for instance, submission in task_list
            }

            for future in as_completed(futures):
                result_outcome = future.result()
                on_result(result_outcome)  # Invoke the result callback

                completed_count += 1
                if on_progress:
                    on_progress(completed_count, total_tasks)
