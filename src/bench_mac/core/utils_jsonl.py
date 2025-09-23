from collections.abc import Iterable, Iterator
from pathlib import Path

from loguru import logger


def iter_lines_from_jsonl_files(
    files: Iterable[Path],
) -> Iterable[tuple[Path, int, str]]:
    """Iterate over non-empty lines from JSONL files.

    Yields tuples of (file_path, line_num, line) for each non-empty line.
    Logs a warning if a file cannot be read.
    """

    def _gen() -> Iterator[tuple[Path, int, str]]:
        for file_path in files:
            try:
                with file_path.open("r", encoding="utf-8") as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        yield (file_path, line_num, line)
            except Exception as e:  # pragma: no cover - defensive logging
                logger.warning(f"Unable to read {file_path}: {e}")

    return _gen()
