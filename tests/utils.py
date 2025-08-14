"""Test utilities and helper functions."""

from datetime import UTC, datetime

from bench_mac.models import CommandOutput


def create_command_output(
    command: str,
    exit_code: int,
    stdout: str = "",
    stderr: str = "",
    start_time: datetime | None = None,
    end_time: datetime | None = None,
) -> CommandOutput:
    """Test helper to auto-fill required timestamps for CommandOutput."""
    now = datetime.now(UTC)
    start = start_time or now
    end = end_time or start
    return CommandOutput(
        command=command,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        start_time=start,
        end_time=end,
    )
