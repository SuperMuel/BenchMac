import os
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from loguru import logger

from bench_mac.docker.manager import DockerManager
from bench_mac.evaluator import evaluate_submission
from bench_mac.models import (
    EvaluationFailure,
    EvaluationSuccess,
    EvaluationTask,
    RunOutcome,
)


def run_single_evaluation_task(
    task: EvaluationTask,
    log_dir: Path,
) -> RunOutcome:
    """
    A top-level function designed to be run in a separate process.

    It initializes its own DockerManager and calls the core evaluation logic,
    wrapping the result in a RunOutcome object to handle success or failure.
    """
    instance_log_path = log_dir / "instances" / f"{task.instance.instance_id}.log"

    handler_id = logger.add(instance_log_path, level="DEBUG", serialize=True)
    instance_logger = logger.bind(instance_id=task.instance.instance_id)

    try:
        docker_manager = DockerManager(quiet_init=True)
        result = evaluate_submission(
            task.instance,
            task.submission,
            docker_manager,
            logger=instance_logger,
        )
        return EvaluationSuccess(result=result)
    except Exception as e:
        # Catch any unexpected crash in the worker process
        instance_logger.exception("Worker process for instance crashed unexpectedly.")
        return EvaluationFailure(
            instance_id=task.instance.instance_id,
            error=f"Worker process crashed ({e.__class__.__name__}: {e})\n"
            f"See instance log for details in {instance_log_path}",
        )
    finally:
        logger.remove(handler_id)


class BenchmarkRunner:
    """Orchestrates the parallel execution of evaluation tasks."""

    def __init__(self, workers: int | None = None):
        """
        Initializes the runner with a specific number of parallel workers.
        """
        self.workers = workers or os.cpu_count() or 1
        logger.info(f"BenchmarkRunner initialized with {self.workers} worker(s).")

    def run(
        self,
        tasks: Sequence[EvaluationTask],
        log_dir: Path,
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
            logger.info("No tasks to evaluate. Exiting.")
            return

        logger.info(f"Starting evaluation of {total_tasks} tasks...")

        # Fail if the Docker daemon is not running.
        DockerManager.get_client(quiet=False)

        completed_count = 0
        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            futures = {
                executor.submit(
                    run_single_evaluation_task,
                    task,
                    log_dir,
                )
                for task in task_list
            }

            for future in as_completed(futures):
                result_outcome = future.result()
                on_result(result_outcome)  # Invoke the result callback

                completed_count += 1
                if on_progress:
                    on_progress(completed_count, total_tasks)
