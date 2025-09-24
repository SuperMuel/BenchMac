"""
Reusable analysis utilities for Experiment/Evaluation log review.

This module provides:
- Loaders for experiments and evaluations (mirroring explorer/app.py and cli.py)
- Matching helpers to select a single (experiment, evaluation) pair
- Formatters to render traces and metrics into the ANALYSIS_PROMPT template
- A DI-friendly stub to run an LLM via LangChain (or any object with .invoke)

Usage example (script-ish):
    from pathlib import Path
    from analysis.analysis import (
        load_experiments,
        load_evaluations,
        select_latest_pair,
        build_analysis_prompt,
    )

    experiments = load_experiments()
    completed, _ = load_evaluations()
    exp, evalc = select_latest_pair(experiments, completed)
    prompt = build_analysis_prompt(exp, evalc)
    print(prompt)




"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from langchain_core.runnables import RunnableConfig
from pydantic import ValidationError
from uuid6 import uuid7

from bench_mac.core.config import settings
from bench_mac.core.models import (
    CommandResult,
    EvaluationCompleted,
    EvaluationFailed,
    EvaluationResultAdapter,
    ExecutionTrace,
    MetricsReport,
)
from bench_mac.core.utils import load_instances
from bench_mac.core.utils_jsonl import iter_lines_from_jsonl_files
from experiments.models import CompletedExperiment, ExperimentResult

from .prompt import ANALYSIS_PROMPT

# --- Public API dataclasses & protocols ---


class SupportsInvoke(Protocol):
    """Minimal protocol for LLM-like objects (LangChain-compatible).

    Any object exposing `invoke(input: str) -> Any` is accepted.
    """

    def invoke(self, input: str) -> Any:  # pragma: no cover - thin adapter
        ...


@dataclass(frozen=True)
class AnalysisPair:
    experiment: CompletedExperiment
    evaluation: EvaluationCompleted


# --- Loaders ---


def load_experiments(experiments_dir: Path | None = None) -> list[ExperimentResult]:
    """Load all experiment results (agent runs).

    Mirrors explorer/app.py: expects JSONL files under `<experiments_dir>/results`.
    """
    base_dir = experiments_dir or settings.experiments_dir
    results_dir = base_dir / "results"
    if not base_dir.exists() or not results_dir.exists():
        return []

    results: list[ExperimentResult] = []
    for _, _, line in iter_lines_from_jsonl_files(results_dir.glob("*.jsonl")):
        try:
            data = json.loads(line)
            results.append(ExperimentResult.model_validate(data))
        except (json.JSONDecodeError, ValidationError):
            # Keep loader resilient; callers can choose to surface warnings
            continue
    return results


def load_evaluations(
    evaluations_dir: Path | None = None,
) -> tuple[list[EvaluationCompleted], list[EvaluationFailed]]:
    """Load all evaluations from any `*.jsonl` under the directory recursively.

    Mirrors explorer/app.py and cli helpers.
    Returns (completed_list, failed_list).
    """
    base_dir = evaluations_dir or settings.evaluations_dir
    if not base_dir.exists():
        return ([], [])

    completed: list[EvaluationCompleted] = []
    failed: list[EvaluationFailed] = []

    jsonl_files = list(base_dir.rglob("*.jsonl"))
    for _, _, line in iter_lines_from_jsonl_files(jsonl_files):
        try:
            result = EvaluationResultAdapter.validate_json(line)
            if result.status == "completed":
                completed.append(result)
            elif result.status == "failed":
                failed.append(result)
        except (json.JSONDecodeError, ValidationError, Exception):
            # Skip invalid lines; upstream tools already warn when needed
            continue

    return (completed, failed)


# --- Selection / Matching helpers ---


def _iter_completed_experiments(
    results: Iterable[ExperimentResult],
) -> Iterable[CompletedExperiment]:
    for er in results:
        if er.is_completed:
            yield er.root  # type: ignore[return-value]


def select_latest_pair(
    experiments: Sequence[ExperimentResult],
    evaluations: Sequence[EvaluationCompleted],
) -> AnalysisPair:
    """Select the latest (experiment, evaluation) by evaluation end time.

    - Only pairs where evaluation.submission_id matches experiment.submission.id
    - If multiple evaluations exist per submission, the latest one wins globally.
    Raises ValueError when no pair can be formed.
    """
    # Build lookup: submission_id -> CompletedExperiment
    by_submission: dict[str, CompletedExperiment] = {}
    for ce in _iter_completed_experiments(experiments):
        # For now, only consider swe-agent-mini experiments
        if getattr(ce.task.agent_config, "scaffold", None) != "swe-agent-mini":
            continue
        by_submission[ce.submission.submission_id] = ce

    # Filter evaluations that match a known completed experiment
    matching_evals: list[EvaluationCompleted] = [
        ev for ev in evaluations if ev.result.submission_id in by_submission
    ]
    if not matching_evals:
        raise ValueError("No matching (experiment, evaluation) pair found.")

    # Select the latest by ended_at
    latest_eval = max(matching_evals, key=lambda e: e.ended_at)
    exp = by_submission[latest_eval.result.submission_id]
    return AnalysisPair(experiment=exp, evaluation=latest_eval)


# --- Formatting helpers ---


def _truncate(text: str, max_len: int) -> str:
    if max_len <= 0:
        return ""
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "\u2026"


# --- Mini SWE Agent trajectory helpers ---


def load_mini_swe_messages(
    submission_id: str,
    *,
    experiments_dir: Path | None = None,
) -> list[dict[str, Any]] | None:
    """Load messages from swe-agent-mini trajectory file if present.

    File path: `<experiments_dir>/swe_agent_mini/<submission_id>.traj.json`
    Returns list of message dicts with at least `role` and `content`.
    """
    base_dir = experiments_dir or settings.experiments_dir
    traj_path = base_dir / "swe_agent_mini" / f"{submission_id}.traj.json"
    if not traj_path.exists():
        return None
    try:
        data = json.loads(traj_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    messages = data.get("messages")
    if not isinstance(messages, list):
        return None
    # basic shape validation
    out: list[dict[str, Any]] = []
    for m in messages:
        if isinstance(m, dict) and "role" in m and "content" in m:
            out.append(m)
    return out or None


def format_agent_messages(
    messages: Sequence[dict[str, Any]],
    *,
    max_messages: int | None = None,
) -> str:
    """Render agent messages as a sequential transcript.

    We keep the original content verbatim (it already includes code fences or
    XML-like wrappers for command outputs when present). We only add lightweight
    role markers.
    """
    if max_messages is not None and len(messages) > max_messages:
        messages = messages[:max_messages]

    rendered: list[str] = []
    for msg in messages:
        role = str(msg.get("role", "unknown")).strip()
        content = str(msg.get("content", "")).rstrip()
        rendered.append(f"<role>{role}</role>\n{content}")
    return "\n\n".join(rendered) if rendered else "<empty>"


def format_command_result(
    step: CommandResult,
    *,
    truncate_stdout: int = 2000,
    truncate_stderr: int = 1200,
) -> str:
    """Single step pretty-print for inclusion in a plain-text prompt."""
    stdout = _truncate(step.stdout, truncate_stdout) if step.stdout else ""
    stderr = _truncate(step.stderr, truncate_stderr) if step.stderr else ""

    duration_seconds = step.duration.total_seconds()
    parts = [
        f"<command>{step.command}</command>",
        f"exit_code={step.exit_code}",
        f"duration_seconds={duration_seconds:.3f}",
    ]
    if stdout:
        parts.append("<stdout>\n" + stdout + "\n</stdout>")
    if stderr:
        parts.append("<stderr>\n" + stderr + "\n</stderr>")
    return "\n".join(parts)


def format_execution_trace(
    trace: ExecutionTrace | None,
    *,
    header: str,
    max_steps: int | None = None,
) -> str:
    if trace is None:
        return f"{header}\n<no trace available>"

    steps = trace.steps
    if max_steps is not None and len(steps) > max_steps:
        steps = steps[:max_steps]

    rendered = [format_command_result(s) for s in steps]
    body = "\n\n".join(rendered) if rendered else "<empty>"
    return (f"{header}\n" + body) if header else body


def format_metrics_report(metrics: MetricsReport) -> str:
    payload = metrics.model_dump()
    return json.dumps(payload, indent=2, sort_keys=True)


# --- Patch filtering helpers ---


def filter_patch_excluding_lockfiles(patch_text: str) -> str:
    """Remove unified diff sections for common lockfiles to save space.

    Skips sections where target path ends with:
    - package-lock.json
    - yarn.lock
    - pnpm-lock.yaml
    - bun.lockb
    """
    if not patch_text:
        return patch_text

    lines = patch_text.splitlines()
    if not lines:
        return patch_text

    lockfile_suffixes = (
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "bun.lockb",
    )

    filtered: list[str] = []
    include_current = True
    in_any_section = False

    for line in lines:
        if line.startswith("diff --git "):
            in_any_section = True
            parts = line.split()
            # Expected: ['diff', '--git', 'a/path', 'b/path']
            path_b = parts[3] if len(parts) >= 4 else ""
            if path_b.startswith("b/"):
                path_b = path_b[2:]
            include_current = not any(path_b.endswith(suf) for suf in lockfile_suffixes)
            if include_current:
                filtered.append(line)
            continue

        if not in_any_section:
            # Keep any preamble before first 'diff --git'
            filtered.append(line)
            continue

        if include_current:
            filtered.append(line)

    return "\n".join(filtered)


def build_analysis_prompt(
    experiment: CompletedExperiment,
    evaluation: EvaluationCompleted,
    *,
    max_agent_steps: int | None = 60,
    max_eval_steps: int | None = None,
) -> str:
    """Render ANALYSIS_PROMPT with formatted agent/eval traces and metrics."""
    # Prefer mini-SWE trajectory messages when available
    formatted_agent_trace: str
    agent_cfg = experiment.task.agent_config
    assert agent_cfg.scaffold == "swe-agent-mini"
    messages = load_mini_swe_messages(experiment.submission.submission_id)
    if messages:
        formatted_agent_trace = format_agent_messages(
            messages, max_messages=max_agent_steps
        )
    else:
        # Fallback to artifacts execution trace if no trajectory present
        agent_trace = (
            experiment.artifacts.execution_trace if experiment.artifacts else None
        )
        formatted_agent_trace = format_execution_trace(
            agent_trace, header="", max_steps=max_agent_steps
        )

    formatted_evaluation_trace = format_execution_trace(
        evaluation.result.execution,
        header="Evaluation Trace",
        max_steps=max_eval_steps,
    )

    formatted_metrics_report = format_metrics_report(evaluation.result.metrics)

    # Resolve instance metadata for task definition section
    instance_id = str(evaluation.result.instance_id)
    instances_map = load_instances(settings.instances_file)
    inst = instances_map.get(instance_id)
    assert inst is not None
    repo_url = f"https://github.com/{inst.repo}"
    base_commit = inst.base_commit
    source_version = inst.source_angular_version
    target_version = inst.target_angular_version

    # Prepare generated patch (filtered + optionally truncated to 10k chars)
    raw_patch = experiment.submission.model_patch or ""
    filtered_patch = filter_patch_excluding_lockfiles(raw_patch)
    max_chars = 10_000
    if len(filtered_patch) > max_chars:
        truncated = filtered_patch[:max_chars]
        trunc_note = (
            f"\n\n[TRUNCATED: original patch length {len(filtered_patch)} chars; "
            f"showing first {max_chars} chars]"
        )
        eventually_truncated_generated_patch = truncated + trunc_note
    else:
        eventually_truncated_generated_patch = filtered_patch

    return ANALYSIS_PROMPT.format(
        formatted_agent_trace=formatted_agent_trace,
        formatted_evaluation_trace=formatted_evaluation_trace,
        formatted_metrics_report=formatted_metrics_report,
        eventually_truncated_generated_patch=eventually_truncated_generated_patch,
        repo_url=repo_url,
        base_commit=base_commit,
        source_version=source_version,
        target_version=target_version,
    )


# --- LLM integration  ---

import os  # noqa: E402

from dotenv import load_dotenv  # noqa: E402
from langchain.chat_models import init_chat_model  # noqa: E402, F401
from langchain_openai.chat_models import ChatOpenAI  # noqa: E402

load_dotenv()

assert os.getenv("OPENAI_API_KEY")
assert os.getenv("LANGSMITH_TRACING")

model_name = "gpt-5"
llm = ChatOpenAI(model=model_name)


if __name__ == "__main__":
    # uv run -m analysis.analysis

    print("Loading experiments and evaluations...")
    experiments = load_experiments()
    completed, _ = load_evaluations()
    print("Selecting latest experiment/evaluation pair...")
    pair = select_latest_pair(experiments, completed)
    exp, evalc = pair.experiment, pair.evaluation

    print(f"Building analysis prompt for experiment {exp.id}, evaluation {evalc.id}...")
    trace_id = uuid7()
    print(f"Generated trace_id: {trace_id}")
    prompt = build_analysis_prompt(exp, evalc)
    prompt_dir = Path("analysis") / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    out_path = prompt_dir / f"trace_{trace_id}_eval_{evalc.id}_exp_{exp.id}.md"
    print(f"Writing prompt to {out_path}")
    with out_path.open("w", encoding="utf-8") as f:
        f.write(prompt)

    metadata = {
        "experiment_id": exp.id,
        "submission_id": exp.submission.submission_id,
        "instance_id": exp.submission.instance_id,
        "evaluation_id": evalc.id,
        "scaffold": exp.task.agent_config.scaffold,
        "model_name": exp.task.agent_config.model_name
        if exp.task.agent_config.scaffold == "swe-agent-mini"
        else None,
    }

    config = RunnableConfig(
        metadata=metadata,
        run_id=trace_id,
        run_name="analysis",
    )

    print(f"Invoking {model_name} for analysis...")
    result = llm.invoke(prompt, config=config)
    result_dir = Path("analysis/results")
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / f"trace_{trace_id}_eval_{evalc.id}_exp_{exp.id}.md"

    print(f"Writing LLM result to {result_path}")
    with result_path.open("w", encoding="utf-8") as f:
        f.write(result.text())

    print(f"Analysis complete. Output written to {result_path}")
    print(result + "\n")
