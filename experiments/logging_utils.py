import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger

from bench_mac.config import settings

DEFAULT_ROTATION = "10 MB"
DEFAULT_RETENTION = "7 days"


def _default_format() -> str:
    return (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | "
        "instance={extra[instance]!s} model={extra[model]!s} "
        "submission={extra[submission]!s} | {message}"
    )


def setup_experiment_logging(run_label: str | None = None) -> Path:
    """Configure loguru sinks for experiment runs and return the file path."""

    logs_dir = settings.experiments_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    if run_label is None:
        run_label = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S")

    log_path = logs_dir / f"experiment_{run_label}.log"

    logger.remove()
    logger.configure(
        handlers=[
            {
                "sink": sys.stderr,
                "level": settings.cli_default_log_level,
                "format": "<level>{message}</level>",
            },
            {
                "sink": log_path,
                "level": "DEBUG",
                "format": _default_format(),
                "rotation": DEFAULT_ROTATION,
                "retention": DEFAULT_RETENTION,
                "enqueue": True,
                "backtrace": True,
                "diagnose": True,
            },
        ],
        extra={"instance": "-", "model": "-", "submission": "-"},
    )

    return log_path


def bind_run_context(**extra: Any) -> Any:
    """Bind additional context to the logger for downstream calls."""

    return logger.bind(**{k: v for k, v in extra.items() if v is not None})
