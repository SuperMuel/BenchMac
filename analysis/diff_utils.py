"""Utilities for analyzing unified diffs produced by BenchMAC runs.

This module provides helpers to split statistics between lockfiles
and non-lockfiles so downstream analyses can focus on meaningful
code changes instead of dependency lock churn.

All functions are pure and safe to import in notebooks or scripts.
"""

from __future__ import annotations

from collections.abc import Iterable

# Common lockfile names across npm/yarn/pnpm. Extend as needed.
DEFAULT_LOCKFILE_NAMES: tuple[str, ...] = (
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
)


def is_lockfile_path(
    path: str, *, lockfile_names: Iterable[str] = DEFAULT_LOCKFILE_NAMES
) -> bool:
    """Return True if the given repository-relative path points to a lockfile.

    The check is suffix-based and case-sensitive (as in Git paths).
    """
    return any(path.endswith(name) for name in lockfile_names)


def diff_stats_split(
    patch: str,
    *,
    lockfile_names: Iterable[str] = DEFAULT_LOCKFILE_NAMES,
) -> dict[str, int]:
    """Compute file and line statistics from a unified diff, split by lockfiles.

    The function expects a Git-style unified diff where file sections begin with
    lines like: ``diff --git a/<path> b/<path>``.

    It counts lines added/removed while skipping file header lines (``+++``, ``---``).

    Returns a dictionary with the following integer fields:
      - files_changed: total number of files touched
      - files_changed_nonlock: number of non-lockfiles touched
      - lines_added: total lines starting with '+' (excluding '+++')
      - lines_removed: total lines starting with '-' (excluding '---')
      - lines_added_lock / lines_removed_lock: counts inside lockfiles
      - lines_added_nonlock / lines_removed_nonlock: counts outside lockfiles
    """
    files_all: set[str] = set()
    files_nonlock: set[str] = set()

    lines_added_lock = 0
    lines_removed_lock = 0
    lines_added_nonlock = 0
    lines_removed_nonlock = 0

    current_is_lock = False
    in_section = False

    for line in patch.splitlines():
        if line.startswith("diff --git "):
            in_section = True
            parts = line.split()
            # Expected: ['diff', '--git', 'a/<path>', 'b/<path>']
            path_b = parts[3] if len(parts) >= 4 else ""
            if path_b.startswith("b/"):
                path_b = path_b[2:]
            files_all.add(path_b)
            current_is_lock = is_lockfile_path(path_b, lockfile_names=lockfile_names)
            if not current_is_lock:
                files_nonlock.add(path_b)
            continue

        if not in_section:
            # Ignore any preamble before the first diff header
            continue

        # Skip file header markers inside sections
        if line.startswith("+++") or line.startswith("---"):
            continue

        if line.startswith("+"):
            if current_is_lock:
                lines_added_lock += 1
            else:
                lines_added_nonlock += 1
        elif line.startswith("-"):
            if current_is_lock:
                lines_removed_lock += 1
            else:
                lines_removed_nonlock += 1

    return {
        "files_changed": len(files_all),
        "files_changed_nonlock": len(files_nonlock),
        "lines_added": lines_added_lock + lines_added_nonlock,
        "lines_removed": lines_removed_lock + lines_removed_nonlock,
        "lines_added_lock": lines_added_lock,
        "lines_removed_lock": lines_removed_lock,
        "lines_added_nonlock": lines_added_nonlock,
        "lines_removed_nonlock": lines_removed_nonlock,
    }


def filter_patch_excluding_lockfiles(
    patch_text: str, *, lockfile_names: Iterable[str] = DEFAULT_LOCKFILE_NAMES
) -> str:
    """Return the diff text with lockfile sections removed.

    Keeps the original order and any preamble before the first section.
    """
    lines = patch_text.splitlines()
    if not lines:
        return patch_text

    filtered: list[str] = []
    include_current = True
    in_any_section = False

    for line in lines:
        if line.startswith("diff --git "):
            in_any_section = True
            parts = line.split()
            path_b = parts[3] if len(parts) >= 4 else ""
            if path_b.startswith("b/"):
                path_b = path_b[2:]
            include_current = not is_lockfile_path(
                path_b, lockfile_names=lockfile_names
            )
            if include_current:
                filtered.append(line)
            continue

        if not in_any_section:
            # Preserve any preamble before the first section
            filtered.append(line)
            continue

        if include_current:
            filtered.append(line)

    return "\n".join(filtered)


if __name__ == "__main__":  # Simple self-checks when running as a script
    # Synthetic diff with one lockfile and one code file
    sample = (
        "preamble\n"
        "diff --git a/package-lock.json b/package-lock.json\n"
        "index 111..222 100644\n"
        "--- a/package-lock.json\n"
        "+++ b/package-lock.json\n"
        "+added-in-lock\n"
        "-removed-in-lock\n"
        "diff --git a/src/app.ts b/src/app.ts\n"
        "index 333..444 100644\n"
        "--- a/src/app.ts\n"
        "+++ b/src/app.ts\n"
        "+added-in-code\n"
        "+added-in-code-2\n"
        "-removed-in-code\n"
    )

    stats = diff_stats_split(sample)
    assert stats["files_changed"] == 2
    assert stats["files_changed_nonlock"] == 1
    assert stats["lines_added_lock"] == 1
    assert stats["lines_removed_lock"] == 1
    assert stats["lines_added_nonlock"] == 2
    assert stats["lines_removed_nonlock"] == 1
    assert stats["lines_added"] == 3
    assert stats["lines_removed"] == 2

    filtered = filter_patch_excluding_lockfiles(sample)
    assert "package-lock.json" not in filtered
    assert "src/app.ts" in filtered

    print("diff_utils self-checks passed.")
