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

import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from tqdm import tqdm
from uuid6 import UUID, uuid7

from bench_mac.core.config import settings
from bench_mac.core.models import (
    CommandResult,
    EvaluationCompleted,
    EvaluationFailed,
    EvaluationResultAdapter,
    ExecutionTrace,
    MetricsReport,
    SubmissionID,
)
from bench_mac.core.utils import load_instances
from bench_mac.core.utils_jsonl import iter_lines_from_jsonl_files
from experiments.models import CompletedExperiment, ExperimentResult

from .prompt import ANALYSIS_PROMPT

# --- Public API dataclasses & protocols ---


class AnalysisTask(BaseModel):
    """
    A task for analyzing an experiment/evaluation pair using an LLM.

    Represents the work needed to generate an analysis of agent behavior,
    evaluation results, and migration outcomes for a specific benchmark instance.
    """

    model_config = ConfigDict(frozen=True)

    experiment: CompletedExperiment = Field(
        ...,
        description="The completed experiment with agent run data and generated patch.",
    )
    evaluation: EvaluationCompleted = Field(
        ...,
        description="The completed evaluation containing execution trace and metrics.",
    )
    trace_id: str = Field(
        default_factory=lambda: str(uuid7()),
        description="Unique identifier for tracing this analysis task.",
    )
    model_name: str = Field(
        default="gpt-5", description="Name of the LLM model to use for analysis."
    )


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


def match_experiments_and_evaluations(
    completed_experiments: Sequence[CompletedExperiment],
    completed_evaluations: Sequence[EvaluationCompleted],
) -> list[tuple[CompletedExperiment, EvaluationCompleted]]:
    """Match experiments and evaluations by submission_id."""
    all_submission_ids: set[SubmissionID] = {
        exp.submission.submission_id for exp in completed_experiments
    } | {ev.result.submission_id for ev in completed_evaluations}

    sub_to_evals: dict[SubmissionID, list[EvaluationCompleted]] = {
        sub_id: [
            ev for ev in completed_evaluations if ev.result.submission_id == sub_id
        ]
        for sub_id in all_submission_ids
    }

    submissions_without_evals: set[SubmissionID] = {
        sub_id for sub_id in all_submission_ids if not sub_to_evals[sub_id]
    }

    if submissions_without_evals:
        print(
            f"Warning: {len(submissions_without_evals)} submissions have no evaluations"
        )

    # Build lookup of the latest CompletedExperiment per submission_id,
    # while preserving the first-seen order of submission_ids from the input
    # experiments sequence for deterministic output ordering.
    exp_by_sub: dict[SubmissionID, CompletedExperiment] = {}
    order: list[SubmissionID] = []
    for exp in completed_experiments:
        sub_id = exp.submission.submission_id
        if sub_id not in exp_by_sub:
            order.append(sub_id)
            exp_by_sub[sub_id] = exp
        else:
            if exp.ended_at > exp_by_sub[sub_id].ended_at:
                exp_by_sub[sub_id] = exp

    # For each submission_id, keep only the latest EvaluationCompleted
    latest_eval_by_sub: dict[SubmissionID, EvaluationCompleted] = {}
    for sub_id, evals in sub_to_evals.items():
        if not evals:
            continue
        latest_eval_by_sub[sub_id] = max(evals, key=lambda e: e.ended_at)

    # Warn about evaluations that have no matching experiment
    exp_sub_ids = set(exp_by_sub.keys())
    eval_only_submissions = set(latest_eval_by_sub.keys()) - exp_sub_ids
    if eval_only_submissions:
        count = len(eval_only_submissions)
        print(f"Warning: {count} evaluations have no matching experiment")

    # Build (experiment, evaluation) pairs in experiment input order, skipping
    # submissions with no evaluation.
    pairs: list[tuple[CompletedExperiment, EvaluationCompleted]] = []
    for sub_id in order:
        ev = latest_eval_by_sub.get(sub_id)
        if ev is None:
            continue
        exp = exp_by_sub[sub_id]
        pairs.append((exp, ev))

    assert all(
        pair[0].submission.submission_id == pair[1].result.submission_id
        for pair in pairs
    )

    return pairs


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


def truncate_xml_with_warning(body: str, *, tag: str, max_len: int) -> str:
    """
    Truncate the content inside an XML tag if it exceeds max_len,
    appending a warning comment.

    Args:
        body: The string content to wrap and possibly truncate.
        tag: The XML tag name (without angle brackets).
        max_len: The maximum allowed length for the body.

    Returns:
        The XML string with the body (possibly truncated) and
        a warning comment if truncated.
    """
    assert max_len > 0, "Max length must be positive"

    if len(body) > max_len:
        truncated = True
        display_body = body[: max_len - 1] + "\u2026"
    else:
        truncated = False
        display_body = body

    xml = f"<{tag}>\n{display_body}\n</{tag}>"

    if truncated:
        warning = "WARNING: truncated to fit display"
        warning += f" ({max_len - 1} out of {len(body)} chars displayed)"

        xml += f" <!-- {warning} -->"

        print(
            "Warning: some of the output was truncated to fit display."
            f" ({max_len - 1} out of {len(body)} chars displayed)"
            f"({tag=})"
        )

    return xml


# --- Mini SWE Agent trajectory helpers ---


def load_mini_swe_messages(
    submission_id: str,
    *,
    experiments_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Load messages from swe-agent-mini trajectory file if present.

    File path: `<experiments_dir>/swe_agent_mini/<submission_id>.traj.json`
    Returns list of message dicts with at least `role` and `content`.
    Raises FileNotFoundError, ValueError, or json.JSONDecodeError on error.
    """
    base_dir = experiments_dir or settings.experiments_dir
    traj_path = base_dir / "swe_agent_mini" / f"{submission_id}.traj.json"
    if not traj_path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {traj_path}")
    try:
        data = json.loads(traj_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(
            f"Failed to load or parse trajectory file: {traj_path}"
        ) from exc
    messages = data.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"Trajectory file missing 'messages' list: {traj_path}")
    # basic shape validation
    out: list[dict[str, Any]] = []
    for m in messages:
        if isinstance(m, dict) and "role" in m and "content" in m:
            out.append(m)
        else:
            print(f"Warning: Invalid message in trajectory file: {traj_path}")
    if not out:
        raise ValueError(f"No valid messages found in trajectory file: {traj_path}")
    return out


def format_agent_messages(
    messages: Sequence[dict[str, Any]],
) -> str:
    """Render agent messages as a sequential transcript.

    We keep the original content verbatim (it already includes code fences or
    XML-like wrappers for command outputs when present). We only add lightweight
    role markers.
    """
    assert messages, "Empty messages list given to format_agent_messages"
    assert all(m.get("role") for m in messages), "Some messages are missing role"
    assert all("content" in m for m in messages), "Some messages are missing content"

    rendered: list[str] = []
    for msg in messages:
        role = str(msg.get("role", "unknown"))
        content = str(msg.get("content"))
        rendered.append(f"<{role}>\n{content}\n</{role}>")

    return "\n\n".join(rendered)


def format_command_result(
    step: CommandResult,
    *,
    truncate_stdout: int = 10_000,
    truncate_stderr: int = 10_000,
) -> str:
    """Single step pretty-print for inclusion in a plain-text prompt."""
    formatted_stdout = truncate_xml_with_warning(
        step.stdout, tag="stdout", max_len=truncate_stdout
    )
    formatted_stderr = truncate_xml_with_warning(
        step.stderr, tag="stderr", max_len=truncate_stderr
    )

    duration_seconds = step.duration.total_seconds()
    parts = [
        f"<command>\n{step.command}\n</command>",
        f"exit_code={step.exit_code}",
        f"duration_seconds={duration_seconds:.3f}",
    ]
    parts.append(formatted_stdout)
    parts.append(formatted_stderr)
    return "\n".join(parts)


def format_execution_trace(
    trace: ExecutionTrace | None,
) -> str:
    assert trace, "Empty trace given to format_execution_trace"

    steps = trace.steps

    rendered = [format_command_result(s) for s in steps]
    return "\n\n".join(rendered)


def format_metrics_report(metrics: MetricsReport) -> str:
    payload = metrics.model_dump()
    return json.dumps(payload, indent=2)


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
) -> str:
    """Render ANALYSIS_PROMPT with formatted agent/eval traces and metrics."""
    # Prefer mini-SWE trajectory messages when available
    formatted_agent_trace: str
    agent_cfg = experiment.task.agent_config
    assert agent_cfg.scaffold == "swe-agent-mini"
    messages = load_mini_swe_messages(experiment.submission.submission_id)
    assert messages

    formatted_agent_trace = format_agent_messages(messages)

    formatted_evaluation_trace = format_execution_trace(evaluation.result.execution)

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
    dockerfile_content = inst.dockerfile_content

    # Prepare generated patch (filtered + optionally truncated to 10k chars)
    raw_patch = experiment.submission.model_patch
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
        dockerfile_content=dockerfile_content,
    )


# --- LLM integration  ---

import os  # noqa: E402

from dotenv import load_dotenv  # noqa: E402
from langchain.chat_models import init_chat_model  # noqa: E402, F401
from langchain_openai.chat_models import ChatOpenAI  # noqa: E402

load_dotenv()

assert os.getenv("OPENAI_API_KEY")
assert os.getenv("LANGSMITH_TRACING")


ANALYSIS_MODEL_NAME = "gpt-5"
llm = ChatOpenAI(model=ANALYSIS_MODEL_NAME)


def run_analysis_task(
    task: AnalysisTask,
    *,
    llm: BaseChatModel,
    on_prompt: Callable[[str, AnalysisTask], None] | None = None,
) -> BaseMessage:
    """
    Run analysis for the given task.

    Returns an updated AnalysisTask with results or error status.
    """

    # Build prompt
    prompt = build_analysis_prompt(task.experiment, task.evaluation)

    if on_prompt:
        on_prompt(prompt, task)

    # Set up metadata for LLM call
    metadata = {
        "experiment_id": task.experiment.id,
        "submission_id": task.experiment.submission.submission_id,
        "instance_id": task.experiment.submission.instance_id,
        "evaluation_id": task.evaluation.id,
        "scaffold": task.experiment.task.agent_config.scaffold,
        "model_name": task.experiment.task.agent_config.model_name
        if task.experiment.task.agent_config.scaffold == "swe-agent-mini"
        else None,
    }

    config = RunnableConfig(
        metadata=metadata,
        run_id=UUID(task.trace_id),
        run_name="analysis",
    )

    # Invoke LLM with config
    result = llm.invoke(prompt, config=config)

    return result


if __name__ == "__main__":
    # uv run -m analysis.analysis

    print("Loading experiments and evaluations...")
    experiments = load_experiments()
    completed_evaluations = load_evaluations()[0]
    completed_experiments = [
        er.root for er in experiments if er.root.status == "completed"
    ]

    pairs = match_experiments_and_evaluations(
        completed_experiments, completed_evaluations
    )

    tasks = [
        AnalysisTask(
            experiment=experiment,
            evaluation=evaluation,
            model_name=ANALYSIS_MODEL_NAME,
        )
        for experiment, evaluation in pairs
    ]
    print(f"Created {len(tasks)} tasks")

    # Read the results directory and parse filenames to extract eval_id and exp_id
    results_dir = Path("analysis") / "results"
    result_files = list(results_dir.glob("trace_*_eval_*_exp_*.md"))
    parsed_ids: list[tuple[str, str]] = []
    for file in result_files:
        # Example filename: trace_<trace_id>_eval_<eval_id>_exp_<exp_id>.md
        parts = file.stem.split("_")
        try:
            eval_idx = parts.index("eval")
            exp_idx = parts.index("exp")
            eval_id = parts[eval_idx + 1]
            exp_id = parts[exp_idx + 1]
            parsed_ids.append((eval_id, exp_id))
        except (ValueError, IndexError):
            continue

    # Filter out tasks for which we already have a result
    completed_pairs = set(parsed_ids)
    filtered_tasks = [
        task
        for task in tasks
        if (str(task.evaluation.id), str(task.experiment.id)) not in completed_pairs
    ]
    tasks = filtered_tasks
    print(
        f"Skipped {len(tasks) - len(filtered_tasks)} tasks that already have a result"
    )
    print(f"Filtered to {len(tasks)} tasks")

    for task in tqdm(tasks, desc="Analyzing tasks"):
        experiment, evaluation = pairs[0]

        print(f"Created task with trace_id: {task.trace_id}")

        tasks_dir = Path("analysis") / "tasks"
        tasks_dir.mkdir(parents=True, exist_ok=True)
        task_path = tasks_dir / f"{task.trace_id}.json"
        task_path.write_text(task.model_dump_json(indent=2))

        print(
            f"Running analysis for experiment {task.experiment.id}, "
            f"evaluation {task.evaluation.id}..."
        )

        def _on_prompt(prompt: str, task: AnalysisTask) -> None:
            prompt_dir = Path("analysis") / "prompts"
            prompt_dir.mkdir(parents=True, exist_ok=True)
            prompt_path = (
                prompt_dir
                / f"trace_{task.trace_id}_eval_{task.evaluation.id}_exp_{task.experiment.id}.md"  # noqa: E501
            )
            prompt_path.write_text(prompt)

        result = run_analysis_task(
            task,
            llm=llm,
            # llm=FakeChatModel(),
            on_prompt=_on_prompt,
        )

        result_dir = Path("analysis") / "results"
        result_dir.mkdir(parents=True, exist_ok=True)
        base_name = (
            f"trace_{task.trace_id}_eval_{task.evaluation.id}_exp_{task.experiment.id}"
        )
        result_md_path = result_dir / f"{base_name}.md"
        result_md_path.write_text(result.text())
        result_json_path = result_dir / f"{base_name}.json"
        result_json_path.write_text(result.model_dump_json(indent=2))

        print(result)
