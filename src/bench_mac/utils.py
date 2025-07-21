import json
from pathlib import Path

from loguru import logger

from bench_mac.models import BenchmarkInstance


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
                else:
                    logger.warning(f"⚠️ Warning: Skipping {error_msg}")

    if invalid_count > 0 and not strict:
        logger.warning(f"⚠️ Skipped {invalid_count} invalid instances")

    return instances
