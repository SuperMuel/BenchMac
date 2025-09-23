# src/bench_mac/versions.py
from importlib.metadata import PackageNotFoundError, version


def harness_version() -> str:
    try:
        return version("bench-mac")
    except PackageNotFoundError:
        return "0.0.0+dev"
