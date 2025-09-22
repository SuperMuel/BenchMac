"""Compatibility layer that wires infrastructure to the evaluation use case."""

from typing import TYPE_CHECKING, Any

from bench_mac.docker.manager import DockerManager
from bench_mac.environments import DockerEnvironmentFactory
from bench_mac.models import BenchmarkInstance, ExecutionTrace, Submission
from bench_mac.use_cases.evaluate_submission import (
    evaluate_submission as evaluate_submission_use_case,
)
from bench_mac.use_cases.evaluate_submission import (
    is_peer_dep_error,
)

if TYPE_CHECKING:  # pragma: no cover
    from loguru import Logger
else:  # pragma: no cover
    Logger = Any

__all__ = ["execute_submission", "is_peer_dep_error"]


def execute_submission(
    instance: BenchmarkInstance,
    submission: Submission,
    docker_manager: DockerManager,
    *,
    logger: Logger,
) -> ExecutionTrace:
    """Delegate evaluation to the domain use case using Docker infrastructure."""
    environment_factory = DockerEnvironmentFactory(docker_manager)
    return evaluate_submission_use_case(
        instance,
        submission,
        environment_factory,
        logger=logger,
    )
