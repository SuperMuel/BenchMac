from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.models import ExperimentResult


@dataclass
class Step:
    number: int
    command: str
    exit_code: int
    output: str
    error: str


@dataclass
class Trace:
    file_path: Path
    experiment_id: str
    metadata: dict[str, Any]
    steps: list[Step]


STEP_OPEN_RE = re.compile(r"<step number=\"(\d+)\">", re.IGNORECASE)
STEP_CLOSE_RE = re.compile(r"</step>", re.IGNORECASE)
TAG_RE = re.compile(
    r"<(?P<tag>command|exit_code|output|error)>\\n<!\[CDATA\[(?P<content>[\s\S]*?)\]\]>\\n\s*</(?P=tag)>",
    re.IGNORECASE,
)
EXIT_CODE_RE = re.compile(r"<exit_code>(-?\d+)</exit_code>", re.IGNORECASE)


def _read_file_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _extract_metadata(md_text: str) -> dict[str, Any]:
    # Metadata is embedded as first fenced block: ```json ... ```
    fence = re.compile(r"```json\n([\s\S]*?)\n```", re.IGNORECASE)
    match = fence.search(md_text)
    if not match:
        return {}
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return {}


def _normalize_whitespace(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _normalize_command(cmd: str) -> str:
    # Conservative normalization to preserve signal
    cmd = _normalize_whitespace(cmd)
    return cmd


def parse_trace_file(path: Path) -> Trace | None:
    md_text = _read_file_text(path)
    metadata = _extract_metadata(md_text)

    experiment_id = metadata.get("experiment_id")
    if not isinstance(experiment_id, str) or not experiment_id:
        # fallback: filename stem
        experiment_id = path.stem

    # Parse steps by scanning <step> blocks
    steps: list[Step] = []

    # Chunk by step using regex positions
    step_spans: list[tuple[int, int, int]] = []  # (start, end, number)
    for open_match in STEP_OPEN_RE.finditer(md_text):
        number = int(open_match.group(1))
        start = open_match.end()
        close_match = STEP_CLOSE_RE.search(md_text, start)
        if not close_match:
            break
        end = close_match.start()
        step_spans.append((start, close_match.end(), number))

    for start, end, number in step_spans:
        block = md_text[start:end]

        # Prefer structured TAG_RE blocks
        command = ""
        output = ""
        error = ""
        exit_code_val: int | None = None

        for tag_match in TAG_RE.finditer(block):
            tag = tag_match.group("tag").lower()
            content = tag_match.group("content")
            if tag == "command":
                command = content
            elif tag == "output":
                output = content
            elif tag == "error":
                error = content

        # Fallback for exit code if not captured via TAG_RE
        exit_m = EXIT_CODE_RE.search(block)
        if exit_m:
            exit_code_val = int(exit_m.group(1))
        else:
            # Sometimes exit code is emitted also via TAG_RE; try last resort
            for tag_match in re.finditer(
                r"<exit_code>(-?\d+)</exit_code>", block, re.IGNORECASE
            ):
                exit_code_val = int(tag_match.group(1))

        if exit_code_val is None:
            exit_code_val = -9999

        steps.append(
            Step(
                number=number,
                command=command,
                exit_code=exit_code_val,
                output=output,
                error=error,
            )
        )

    return Trace(
        file_path=path, experiment_id=experiment_id, metadata=metadata, steps=steps
    )


def _extract_steps_from_result_json(obj: dict[str, Any]) -> tuple[str, list[Step]]:
    """
    Extract steps from ExperimentResult/EvaluationReport JSON shapes.
    Returns (experiment_id, steps).
    """
    experiment_id = str(obj.get("id") or obj.get("experiment_id") or "")

    steps_payload: list[dict[str, Any]] | None = None

    artifacts = obj.get("artifacts")
    if isinstance(artifacts, dict):
        exec_trace = artifacts.get("execution_trace")
        if isinstance(exec_trace, dict):
            maybe_steps = exec_trace.get("steps")
            if isinstance(maybe_steps, list):
                steps_payload = maybe_steps  # type: ignore[assignment]

    if steps_payload is None:
        execution = obj.get("execution")
        if isinstance(execution, dict):
            maybe_steps = execution.get("steps")
            if isinstance(maybe_steps, list):
                steps_payload = maybe_steps  # type: ignore[assignment]

    steps: list[Step] = []
    if steps_payload:
        for idx, s in enumerate(steps_payload, start=1):
            if not isinstance(s, dict):
                continue
            command = str(s.get("command", ""))
            exit_code_val = int(s.get("exit_code", -9999))
            stdout = str(s.get("stdout", ""))
            stderr = str(s.get("stderr", ""))
            steps.append(
                Step(
                    number=idx,
                    command=command,
                    exit_code=exit_code_val,
                    output=stdout,
                    error=stderr,
                )
            )

    return experiment_id, steps


def parse_json_trace_file(path: Path) -> Trace | None:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None

    if not isinstance(obj, dict):
        return None

    experiment_id, steps = _extract_steps_from_result_json(obj)
    if not experiment_id:
        experiment_id = path.stem

    metadata: dict[str, Any] = {
        k: obj.get(k)
        for k in (
            "id",
            "status",
            "started_at",
            "ended_at",
        )
        if k in obj
    }

    return Trace(
        file_path=path, experiment_id=experiment_id, metadata=metadata, steps=steps
    )


def detect_tail_loop(
    commands: list[str], max_cycle: int = 5, window: int = 50
) -> dict[str, Any]:
    tail = commands[-window:] if window > 0 else commands[:]
    if not tail:
        return {"has_loop": False}

    best: dict[str, Any] = {
        "has_loop": False,
        "cycle_length": 0,
        "repeats": 0,
        "sequence": [],
    }

    n = len(tail)
    for cycle_len in range(1, min(max_cycle, n) + 1):
        seq = tail[-cycle_len:]
        repeats = 1
        i = n - cycle_len
        while i - cycle_len >= 0 and tail[i - cycle_len : i] == seq:
            repeats += 1
            i -= cycle_len
        if repeats >= 2 and (
            not best["has_loop"]
            or repeats * cycle_len > best["repeats"] * best["cycle_length"]
        ):
            best = {
                "has_loop": True,
                "cycle_length": cycle_len,
                "repeats": repeats,
                "sequence": seq[:],
            }

    return best


def compute_metrics(trace: Trace) -> dict[str, Any]:
    steps = sorted(trace.steps, key=lambda s: s.number)
    norm_cmds = [_normalize_command(s.command) for s in steps]

    # Frequency
    counter = Counter(norm_cmds)
    most_common_cmd, most_common_count = ("", 0)
    if counter:
        most_common_cmd, most_common_count = counter.most_common(1)[0]

    # Longest consecutive run
    longest_run_len = 0
    longest_run_cmd = ""
    current_run_len = 0
    current_cmd = None
    for cmd in norm_cmds:
        if cmd == current_cmd:
            current_run_len += 1
        else:
            current_cmd = cmd
            current_run_len = 1
        if current_run_len > longest_run_len:
            longest_run_len = current_run_len
            longest_run_cmd = cmd

    tail_window = 30
    tail_cmds = norm_cmds[-tail_window:]
    tail_distinct = len(set(tail_cmds))

    loop_info = detect_tail_loop(norm_cmds, max_cycle=5, window=50)

    metrics: dict[str, Any] = {
        "experiment_id": trace.experiment_id,
        "file": str(trace.file_path),
        "step_count": len(steps),
        "unique_commands": len(counter),
        "most_common_command": most_common_cmd,
        "most_common_count": most_common_count,
        "longest_consecutive_command": longest_run_cmd,
        "longest_consecutive_len": longest_run_len,
        "tail_distinct_commands_30": tail_distinct,
        "tail_loop": loop_info.get("has_loop", False),
        "tail_loop_cycle_len": loop_info.get("cycle_length", 0),
        "tail_loop_repeats": loop_info.get("repeats", 0),
        "tail_loop_sequence": " | ".join(loop_info.get("sequence", [])),
    }
    return metrics


def write_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    if not rows:
        out_path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(
    rows: list[dict[str, Any]],
    aggregates: dict[str, Any],
    out_path: Path,
) -> None:
    lines: list[str] = []
    lines.append("# Not Completed Traces Analysis\n")
    lines.append("\n")
    lines.append("## Aggregates\n")
    lines.append("\n")
    lines.append("```json\n" + json.dumps(aggregates, indent=2) + "\n```\n\n")

    lines.append("## Per-Trace Tail Findings (Top 10 by tail_loop_repeats)\n\n")
    rows_sorted = sorted(
        rows, key=lambda r: (r["tail_loop_repeats"], r["longest_consecutive_len"])
    )
    rows_sorted = list(reversed(rows_sorted))[:10]
    for r in rows_sorted:
        lines.append(
            f"- **{r['experiment_id']}**: steps={r['step_count']}, "
            f"unique_cmds={r['unique_commands']}, "
            f"longest_consecutive={r['longest_consecutive_len']} x `"
            f"{r['longest_consecutive_command']}`; tail_loop="
            f"{r['tail_loop']} (len={r['tail_loop_cycle_len']}, repeats="
            f"{r['tail_loop_repeats']})\n"
        )
        if r.get("tail_loop_sequence"):
            seq = r["tail_loop_sequence"]
            lines.append(f"  - sequence: `{seq}`\n")

    lines.append("\n## Notes\n\n")
    lines.append(
        "- Tail loop detection checks for repeated cycles up to length 5 in the last "
        "50 steps.\n"
    )
    lines.append("- Longest consecutive run considers the entire trace.\n")

    out_path.write_text("".join(lines), encoding="utf-8")


def compute_aggregates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "trace_count": 0,
        }

    tail_loops = sum(1 for r in rows if r.get("tail_loop"))
    avg_tail_repeats = sum(r.get("tail_loop_repeats", 0) for r in rows) / len(rows)
    avg_longest_consecutive = sum(
        r.get("longest_consecutive_len", 0) for r in rows
    ) / len(rows)
    # Top commands across traces by most_common_command aggregation
    cmd_counter: Counter[str] = Counter(
        r.get("most_common_command", "") for r in rows if r.get("most_common_command")
    )
    top_cmds = cmd_counter.most_common(5)

    return {
        "trace_count": len(rows),
        "traces_with_tail_loop": tail_loops,
        "tail_loop_ratio": tail_loops / len(rows),
        "avg_tail_loop_repeats": avg_tail_repeats,
        "avg_longest_consecutive_run": avg_longest_consecutive,
        "top_most_common_commands": top_cmds,
    }


def find_trace_files(input_dir: Path) -> list[Path]:
    # Only JSON experiment result files; recurse into run_*/ directories.
    return sorted([p for p in input_dir.rglob("*.json") if p.is_file()])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze not completed agent traces for tail loops and repeats.",
    )
    parser.add_argument(
        "--dir",
        dest="input_dir",
        type=str,
        default=str(Path(".benchmac/experiments/results").resolve()),
        help="Directory containing experiment result JSON files (searches recursively).",
    )
    parser.add_argument(
        "--out-prefix",
        dest="out_prefix",
        type=str,
        default=str(Path("analysis").resolve() / "not_completed_traces"),
        help="Prefix path for output files (without extension).",
    )
    parser.add_argument(
        "--only-step-limit-failures",
        dest="only_step_limit_failures",
        action="store_true",
        help="Filter to failed experiments likely due to step-limit (artifacts.run_limits.step_limit reached)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Input dir not found: {input_dir}")
        return 2

    trace_files = find_trace_files(input_dir)
    traces: list[Trace] = []
    for f in trace_files:
        try:
            # Load via Pydantic model for robust schema handling
            result = ExperimentResult.model_validate_json(
                f.read_text(encoding="utf-8")
            ).root
        except Exception:
            # Fallback to permissive JSON/dict parsing if needed
            t = parse_json_trace_file(f)
            if t is not None and t.steps:
                traces.append(t)
            continue

        # Extract steps from artifacts.execution_trace if available
        steps: list[Step] = []
        experiment_id = getattr(result, "id", f.stem)
        status = getattr(result, "status", None)
        artifacts = getattr(result, "artifacts", None)
        step_limit: int | None = None
        if artifacts is not None:
            run_limits = getattr(artifacts, "run_limits", None)
            if run_limits is not None:
                step_limit = getattr(run_limits, "step_limit", None)
            exec_trace = getattr(artifacts, "execution_trace", None)
            if exec_trace is not None:
                for idx, s in enumerate(
                    getattr(exec_trace, "steps", []) or [], start=1
                ):
                    steps.append(
                        Step(
                            number=idx,
                            command=getattr(s, "command", ""),
                            exit_code=int(getattr(s, "exit_code", -9999)),
                            output=getattr(s, "stdout", ""),
                            error=getattr(s, "stderr", ""),
                        )
                    )

        metadata: dict[str, Any] = {
            "status": status,
            "step_limit": step_limit,
        }

        # Optional filter: only failed because of step limits
        if args.only_step_limit_failures and (
            not steps
            or status != "failed"
            or step_limit is None
            or len(steps) < step_limit
        ):
            continue

        if steps:
            traces.append(
                Trace(
                    file_path=f,
                    experiment_id=str(experiment_id),
                    metadata=metadata,
                    steps=steps,
                )
            )

    rows = [compute_metrics(t) for t in traces]
    aggregates = compute_aggregates(rows)

    out_prefix = Path(args.out_prefix)
    out_csv = out_prefix.with_suffix(".csv")
    out_md = out_prefix.with_suffix(".md")
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    write_csv(rows, out_csv)
    write_markdown(rows, aggregates, out_md)

    print(f"Wrote CSV: {out_csv}")
    print(f"Wrote MD:  {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
