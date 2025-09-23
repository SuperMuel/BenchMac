import os
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from bench_mac.docker.manager import DockerManager
from bench_mac.evaluation import calculate_metrics
from bench_mac.logging_config import get_instance_logger, setup_worker_process_logging
from bench_mac.models import (
    EvaluationCompleted,
    EvaluationFailed,
    EvaluationReport,
    EvaluationResult,
    EvaluationTask,
    utc_now,
)

from .submission import run_submission_in_docker


@dataclass
class WorkerContext:
    """Context passed to each worker process."""

    task: EvaluationTask
    log_dir: Path
    run_id: str


def run_single_evaluation_task(context: WorkerContext) -> EvaluationResult:
    """
    A top-level function designed to be run in a separate process.

    It initializes its own DockerManager and calls the core evaluation logic,
    wrapping the result in a EvaluationResult object to handle success or failure.
    """
    # First thing: configure logging for this worker process
    setup_worker_process_logging(
        run_id=context.run_id,
        instance_id=context.task.instance.instance_id,
        logs_dir=context.log_dir,
    )

    # Get a logger bound with instance context
    instance_logger = get_instance_logger(context.task.instance.instance_id)

    # Record the start time for this worker's evaluation
    started_at = utc_now()

    try:
        docker_manager = DockerManager(quiet_init=True)

        # 1. Get the raw execution trace
        trace = run_submission_in_docker(
            context.task.instance,
            context.task.submission,
            docker_manager,
            logger=instance_logger,
        )

        # 2. Calculate metrics from the trace
        metrics = calculate_metrics(trace, context.task.instance)

        # 3. Assemble the final, comprehensive report
        report = EvaluationReport(
            instance_id=context.task.instance.instance_id,
            submission_id=context.task.submission.submission_id,
            execution=trace,
            metrics=metrics,
        )

        return EvaluationCompleted(
            result=report,
            started_at=started_at,
            ended_at=utc_now(),
        )
    except Exception as e:
        # Catch any unexpected crash in the worker process
        instance_logger.exception("Worker process for instance crashed unexpectedly.")
        instance_log_path = (
            context.log_dir / "instances" / f"{context.task.instance.instance_id}.log"
        )
        return EvaluationFailed(
            instance_id=context.task.instance.instance_id,
            submission_id=context.task.submission.submission_id,
            error=f"Worker process crashed ({e.__class__.__name__}: {e})\n"
            f"See instance log for details in {instance_log_path}",
            started_at=started_at,
            ended_at=utc_now(),
        )


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
        run_id: str,
        on_result: Callable[[EvaluationResult], None],
    ) -> None:
        """
        Executes the evaluation for a given set of tasks.

        Args:
            tasks: An iterable of (instance, submission) tuples to be evaluated.
            on_result: A callback function that is invoked with the EvaluationResult
                       of each completed task.
        """
        task_list = list(tasks)
        total_tasks = len(task_list)
        if total_tasks == 0:
            logger.info("No tasks to evaluate. Exiting.")
            return

        logger.info(f"Starting evaluation of {total_tasks} tasks...")

        # Fail if the Docker daemon is not running.
        DockerManager.get_client(quiet=False)

        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            futures = {
                executor.submit(
                    run_single_evaluation_task,
                    WorkerContext(task=task, log_dir=log_dir, run_id=run_id),
                )
                for task in task_list
            }

            for future in as_completed(futures):
                result = future.result()
                on_result(result)
