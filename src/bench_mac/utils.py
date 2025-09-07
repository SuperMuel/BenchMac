import json
from pathlib import Path

from loguru import logger

from bench_mac.models import (
    BenchmarkInstance,
    EvaluationID,
    EvaluationResult,
    SubmissionID,
)


def load_instances(
    instances_path: Path, *, strict: bool = True
) -> dict[str, BenchmarkInstance]:
    """Loads benchmark instances into a dict for fast lookup.

    Args:
        instances_path: Path to the JSONL file containing benchmark instances
        strict: If True, raise an error on invalid instances. If False, emit warnings.

    Returns:
        Dictionary mapping instance_id to BenchmarkInstance objects

    Raises:
        ValueError: If strict=True and invalid instances are found
    """
    instances: dict[str, BenchmarkInstance] = {}
    invalid_count = 0

    with instances_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                instance = BenchmarkInstance.model_validate(data)
                instances[instance.instance_id] = instance
            except (json.JSONDecodeError, Exception) as e:
                invalid_count += 1
                error_msg = f"Invalid instance on line {line_num}: {e}"

                if strict:
                    raise ValueError(error_msg) from e
                else:  # pragma: no cover
                    logger.warning(f"⚠️ Warning: Skipping {error_msg}")

    if invalid_count > 0 and not strict:  # pragma: no cover
        logger.warning(f"⚠️ Skipped {invalid_count} invalid instances")

    return instances


def collect_network_error_details(
    eval_results: list[EvaluationResult],
) -> list[tuple[EvaluationID, SubmissionID, list[str]]]:
    """Return list of tuples with evaluation_id, submission_id, and affected cmds."""
    network_error_keywords = [
        "socket timeout",
        "econnreset",
        "etimedout",
        "network is unreachable",
        "failed to fetch",
        "proxy",
    ]

    results: list[tuple[EvaluationID, SubmissionID, list[str]]] = []

    for result in eval_results:
        if result.status != "completed":
            continue

        evaluation_id = result.id
        submission_id = result.result.submission_id

        affected_commands: list[str] = []

        # Check execution steps for network errors
        for step in result.result.execution.steps:
            if (
                any(
                    keyword in step.stderr.lower() for keyword in network_error_keywords
                )
                and step.command not in affected_commands
            ):
                affected_commands.append(step.command)

        if affected_commands:
            results.append((evaluation_id, submission_id, affected_commands))

    return results
