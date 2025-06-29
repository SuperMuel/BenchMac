"""
Centralized Logging Configuration for BenchMAC.

This module provides clean, centralized logging configuration that handles
multiprocessing properly with loguru.
"""

import sys
from pathlib import Path
from typing import Any

from loguru import logger

from bench_mac.config import settings


def setup_main_process_logging(run_id: str, logs_dir: Path) -> None:
    """
    Configure logging for the main CLI process.

    Sets up:
    - Console output at INFO level
    - Central run.log at DEBUG level
    - Global run_id context
    """
    # Remove default handler first
    logger.remove()

    logger.configure(
        handlers=[
            {
                "sink": sys.stderr,
                "level": settings.cli_default_log_level,
                "format": "<level>{message}</level>",
            },
            {
                "sink": logs_dir / "run.log",
                "level": "DEBUG",
                "serialize": True,
                "enqueue": True,  # Thread-safe
                "backtrace": True,
                "diagnose": True,
            },
        ],
        extra={"run_id": run_id},
    )


def setup_worker_process_logging(run_id: str, instance_id: str, logs_dir: Path) -> None:
    """
    Configure logging for a worker process to log *only* to files.

    This prevents collision with the main process's rich progress bar by ensuring
    worker processes never write to stderr/console directly.

    Sets up:
    - Central run.log at DEBUG level (append mode)
    - Instance-specific log at DEBUG level
    - Global run_id and instance_id context
    """
    instance_log_path = logs_dir / "instances" / f"{instance_id}.log"
    instance_log_path.parent.mkdir(exist_ok=True)

    # Remove the default handler to ensure a clean slate
    logger.remove()

    logger.configure(
        handlers=[
            # NO stderr sink for worker processes - this eliminates log collision
            {
                "sink": logs_dir / "run.log",
                "level": "DEBUG",
                "serialize": True,
                "enqueue": True,
                "mode": "a",  # Append mode for worker processes
            },
            {
                "sink": instance_log_path,
                "level": "DEBUG",
                "enqueue": True,  # Thread-safe for multiprocessing
            },
        ],
        extra={"run_id": run_id, "instance_id": instance_id},
    )


def get_instance_logger(instance_id: str) -> Any:
    """
    Get a logger bound with instance context.

    This assumes the worker process logging has already been configured.
    """
    return logger.bind(instance_id=instance_id)


def get_log_format() -> str:
    """Get appropriate log format based on environment."""
    # For now, we'll use the human-friendly format by default
    # This can be extended later when environment settings are added
    return "<level>{message}</level>"
