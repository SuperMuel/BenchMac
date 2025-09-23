"""Test utilities and helper functions."""

from datetime import datetime

from bench_mac.core.models import CommandResult, utc_now


def create_command_output(
    command: str,
    exit_code: int,
    stdout: str = "",
    stderr: str = "",
    start_time: datetime | None = None,
    end_time: datetime | None = None,
) -> CommandResult:
    """Test helper to auto-fill required timestamps for CommandResult."""
    now = utc_now()
    start = start_time or now
    end = end_time or start
    return CommandResult(
        command=command,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        start_time=start,
        end_time=end,
    )
