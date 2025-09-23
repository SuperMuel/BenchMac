"""Compatibility layer that wires infrastructure to the evaluation use case."""

from typing import TYPE_CHECKING, Any

from bench_mac.docker.manager import DockerManager
from bench_mac.environments import DockerEnvironmentFactory
from bench_mac.models import BenchmarkInstance, ExecutionTrace, Submission

from .workflow import evaluate_submission, is_peer_dep_error

if TYPE_CHECKING:  # pragma: no cover
    from loguru import Logger
else:  # pragma: no cover
    Logger = Any

__all__ = ["is_peer_dep_error", "run_submission_in_docker"]


def run_submission_in_docker(
    instance: BenchmarkInstance,
    submission: Submission,
    docker_manager: DockerManager,
    *,
    logger: Logger,
) -> ExecutionTrace:
    environment_factory = DockerEnvironmentFactory(docker_manager)
    return evaluate_submission(
        instance,
        submission,
        environment_factory,
        logger=logger,
    )
