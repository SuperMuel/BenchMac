"""Shared helpers for saving and loading experiment results."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from experiments.models import ExperimentResult

ErrorCallback = Callable[[Path, Exception], None]

RESULTS_DIR_NAME = "results"
_RUN_PREFIX = "run_"


def ensure_results_root(experiments_dir: Path) -> Path:
    """Ensure the canonical experiments/results directory exists and return it."""

    experiments_dir.mkdir(parents=True, exist_ok=True)
    results_root = experiments_dir / RESULTS_DIR_NAME
    results_root.mkdir(parents=True, exist_ok=True)
    return results_root


def create_results_run_dir(experiments_dir: Path, now: datetime | None = None) -> Path:
    """Create a unique run directory under experiments/results/."""

    if now is None:
        now = datetime.now(UTC)

    results_root = ensure_results_root(experiments_dir)

    base_name = f"{_RUN_PREFIX}{now.strftime('%Y-%m-%d_%H-%M-%S')}"
    run_dir = results_root / base_name
    suffix = 1
    # Avoid clashes when multiple runs start within the same second.
    while run_dir.exists():
        run_dir = results_root / f"{base_name}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _result_file_path(run_dir: Path, result: ExperimentResult) -> Path:
    return run_dir / f"{result.root.id}.json"


def save_experiment_result(result: ExperimentResult, run_dir: Path) -> Path:
    """Persist an ExperimentResult as <experiment_id>.json within run_dir."""

    run_dir.mkdir(parents=True, exist_ok=True)
    path = _result_file_path(run_dir, result)
    path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_experiment_result(path: Path) -> ExperimentResult:
    """Load a single ExperimentResult from a JSON file."""

    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return ExperimentResult.model_validate(data)


def iter_experiment_results(
    results_dir: Path, *, on_error: ErrorCallback | None = None
) -> Iterator[tuple[ExperimentResult, Path]]:
    """Yield (ExperimentResult, file_path) pairs discovered under results_dir."""

    if not results_dir.exists():
        return

    for file_path in sorted(results_dir.rglob("*.json")):
        if not file_path.is_file():
            continue
        try:
            yield (load_experiment_result(file_path), file_path)
        except (json.JSONDecodeError, ValidationError) as exc:
            if on_error is not None:
                on_error(file_path, exc)
            continue


def load_all_experiment_results(
    results_dir: Path, *, on_error: ErrorCallback | None = None
) -> list[ExperimentResult]:
    """Collect and return every ExperimentResult found under results_dir."""

    return [
        result for result, _ in iter_experiment_results(results_dir, on_error=on_error)
    ]


__all__ = [
    "RESULTS_DIR_NAME",
    "create_results_run_dir",
    "ensure_results_root",
    "iter_experiment_results",
    "load_all_experiment_results",
    "load_experiment_result",
    "save_experiment_result",
]
