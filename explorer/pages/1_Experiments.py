from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast

import pandas as pd
import streamlit as st

from bench_mac.config import settings
from experiments.models import (
    CompletedExperiment,
    ExperimentArtifacts,
    ExperimentResult,
    FailedExperiment,
)
from explorer.app import display_execution_steps, load_all_experiments

st.set_page_config(page_title="Experiments", page_icon="🧪")

st.title("🧪 Experiments")

with st.sidebar:
    if st.button("🔄 Reload Data", width="stretch"):
        st.cache_data.clear()
        st.rerun()

st.caption("Browse agent runs loaded from results under experiments/results/*.jsonl.")


def filter_patch_excluding_package_lock(
    patch_text: str, include_package_lock: bool
) -> str:
    """Return patch text, optionally excluding package-lock.json file diffs.

    Assumes git-style unified diff sections starting with 'diff --git a/... b/...'.
    """
    if include_package_lock:
        return patch_text

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
            # Expected: ['diff', '--git', 'a/path', 'b/path']
            path_b = parts[3] if len(parts) >= 4 else ""
            # Strip leading 'b/' if present
            if path_b.startswith("b/"):
                path_b = path_b[2:]
            include_current = not path_b.endswith("package-lock.json")
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


@dataclass
class ExperimentMetrics:
    total_s: float
    commands_s: float | None
    model_only_s: float | None
    steps: int
    cost_usd: float | None
    n_calls: int | None


def compute_metrics(
    started_at: datetime,
    ended_at: datetime,
    artifacts: ExperimentArtifacts | None,
) -> ExperimentMetrics:
    total_s = (ended_at - started_at).total_seconds()
    commands_s: float | None = None
    steps = 0
    cost_usd: float | None = None
    n_calls: int | None = None

    if artifacts:
        trace = artifacts.execution_trace
        if trace:
            commands_s = trace.total_duration.total_seconds()
            steps = len(trace.steps)
        if artifacts.cost_usd is not None:
            cost_usd = round(artifacts.cost_usd, 2)
        if artifacts.n_calls is not None:
            n_calls = int(artifacts.n_calls)

    model_only_s = max(0.0, total_s - commands_s) if commands_s is not None else None

    return ExperimentMetrics(
        total_s=total_s,
        commands_s=commands_s,
        model_only_s=model_only_s,
        steps=steps,
        cost_usd=cost_usd,
        n_calls=n_calls,
    )


def build_table_row(
    experiment: ExperimentResult,
) -> tuple[dict[str, object], ExperimentResult] | None:
    if experiment.is_completed:
        completed = cast(CompletedExperiment, experiment.root)
        metrics = compute_metrics(
            completed.started_at, completed.ended_at, completed.artifacts
        )
        row = {
            "instance_id": completed.task.instance_id,
            "agent": completed.task.agent_config.key,
            "status": "completed",
            "submission_id": str(completed.submission.submission_id),
            "started_at": completed.started_at,
            "ended_at": completed.ended_at,
            "duration_s": round(metrics.total_s, 2),
            "commands_s": (
                round(metrics.commands_s, 2) if metrics.commands_s is not None else None
            ),
            "model_only_s": (
                round(metrics.model_only_s, 2)
                if metrics.model_only_s is not None
                else None
            ),
            "cost_usd": metrics.cost_usd,
            "n_calls": metrics.n_calls,
            "steps": metrics.steps,
        }
        return row, experiment

    if experiment.is_failed:
        failed = cast(FailedExperiment, experiment.root)
        metrics = compute_metrics(failed.started_at, failed.ended_at, failed.artifacts)
        row = {
            "instance_id": failed.task.instance_id,
            "agent": failed.task.agent_config.key,
            "status": "failed",
            "submission_id": "",
            "started_at": failed.started_at,
            "ended_at": failed.ended_at,
            "duration_s": round(metrics.total_s, 2),
            "commands_s": (
                round(metrics.commands_s, 2) if metrics.commands_s is not None else None
            ),
            "model_only_s": (
                round(metrics.model_only_s, 2)
                if metrics.model_only_s is not None
                else None
            ),
            "cost_usd": metrics.cost_usd,
            "n_calls": metrics.n_calls,
            "steps": metrics.steps,
        }
        return row, experiment

    return None


def format_seconds(value: float | None) -> str:
    return f"{value:.2f}s" if value is not None else "N/A"


def format_currency(value: float | None) -> str:
    return f"${value:.2f}" if value is not None else "N/A"


def format_int(value: int | None) -> str:
    return str(value) if value is not None else "N/A"


def render_overview_section(
    status: str,
    metrics: ExperimentMetrics,
    *,
    started_at: datetime,
    ended_at: datetime,
    agent_key: str,
    instance_id: str,
    submission_id: str | None = None,
    error: str | None = None,
) -> None:
    top_cols = st.columns(4)
    top_cols[0].metric("Status", status)
    top_cols[1].metric("Duration", f"{metrics.total_s:.2f}s")
    top_cols[2].metric("Commands", format_seconds(metrics.commands_s))
    top_cols[3].metric("Model-only", format_seconds(metrics.model_only_s))

    bottom_cols = st.columns(3)
    bottom_cols[0].metric("Steps", str(metrics.steps))
    bottom_cols[1].metric("Cost", format_currency(metrics.cost_usd))
    bottom_cols[2].metric("Calls", format_int(metrics.n_calls))

    st.write(f"**Started:** {started_at}")
    st.write(f"**Ended:** {ended_at}")
    st.write("**Agent:** `" + agent_key + "`")
    st.write("**Instance:** `" + instance_id + "`")

    if submission_id is not None:
        st.write("**Submission ID:** `" + submission_id + "`")
    if error:
        st.error(error)


def render_completed_experiment(experiment: CompletedExperiment) -> None:
    metrics = compute_metrics(
        experiment.started_at, experiment.ended_at, experiment.artifacts
    )
    trace = experiment.artifacts.execution_trace if experiment.artifacts else None
    steps_available = bool(trace and trace.steps)
    diff_available = bool(experiment.submission.model_patch)

    tab_labels: list[str] = ["Overview"]
    if steps_available:
        tab_labels.append("Steps")
    if diff_available:
        tab_labels.append("Diff")

    tabs = st.tabs(tab_labels)

    tab_idx = 0
    with tabs[tab_idx]:
        render_overview_section(
            "completed",
            metrics,
            started_at=experiment.started_at,
            ended_at=experiment.ended_at,
            agent_key=experiment.task.agent_config.key,
            instance_id=experiment.task.instance_id,
            submission_id=str(experiment.submission.submission_id),
        )

    if steps_available and trace:
        tab_idx += 1
        with tabs[tab_idx]:
            display_execution_steps(trace.steps)

    if diff_available:
        tab_idx += 1
        with tabs[tab_idx]:
            st.subheader("Model patch")
            include_package_lock = st.toggle(
                "Show package-lock.json changes", value=False
            )
            patch_to_show = filter_patch_excluding_package_lock(
                experiment.submission.model_patch, include_package_lock
            )
            patch_lines = patch_to_show.splitlines()
            st.subheader(f"Unified diff ({len(patch_lines)} lines)")
            st.download_button(
                "Download patch",
                data=patch_to_show,
                file_name=f"{experiment.submission.submission_id}.patch",
                mime="text/x-diff",
            )
            st.code(
                patch_to_show,
                language="diff",
                wrap_lines=True,
            )


def render_failed_experiment(experiment: FailedExperiment) -> None:
    metrics = compute_metrics(
        experiment.started_at, experiment.ended_at, experiment.artifacts
    )
    trace = experiment.artifacts.execution_trace if experiment.artifacts else None
    steps_available = bool(trace and trace.steps)

    tab_labels = ["Overview"]
    if steps_available:
        tab_labels.append("Steps")

    tabs = st.tabs(tab_labels)

    tab_idx = 0
    with tabs[tab_idx]:
        render_overview_section(
            "failed",
            metrics,
            started_at=experiment.started_at,
            ended_at=experiment.ended_at,
            agent_key=experiment.task.agent_config.key,
            instance_id=experiment.task.instance_id,
            error=experiment.error,
        )

    if steps_available and trace:
        tab_idx += 1
        with tabs[tab_idx]:
            display_execution_steps(trace.steps)


def extract_selected_rows(event: Any) -> list[int] | None:
    if event is None:
        return None

    selection = getattr(event, "selection", None)
    if selection is None and isinstance(event, dict):
        selection = event.get("selection")

    if selection is None:
        return None

    rows_attr = getattr(selection, "rows", None)
    if isinstance(rows_attr, list):
        return rows_attr

    if isinstance(selection, dict):
        rows_val = selection.get("rows")
        if isinstance(rows_val, list):
            return rows_val

    return None


experiments: list[ExperimentResult] = load_all_experiments(settings.experiments_dir)

if not experiments:
    st.info("No experiment results found yet.")
else:
    st.write(f"Found {len(experiments)} result(s).")
    st.divider()

    rows: list[dict[str, object]] = []
    row_experiments: list[ExperimentResult] = []

    for er in experiments:
        table_entry = build_table_row(er)
        if table_entry is None:
            continue

        row, experiment_result = table_entry
        rows.append(row)
        row_experiments.append(experiment_result)

    df = pd.DataFrame(rows)

    event: Any = st.dataframe(
        df,
        hide_index=True,
        column_config={
            "instance_id": st.column_config.TextColumn("Instance"),
            "agent": st.column_config.TextColumn("Agent"),
            "status": st.column_config.TextColumn("Status"),
            "submission_id": st.column_config.TextColumn("Submission"),
            "started_at": st.column_config.DatetimeColumn("Started"),
            "ended_at": st.column_config.DatetimeColumn("Ended"),
            "duration_s": st.column_config.NumberColumn("Duration (s)", format="%.2f"),
            "commands_s": st.column_config.NumberColumn("Commands (s)", format="%.2f"),
            "model_only_s": st.column_config.NumberColumn(
                "Model-only (s)", format="%.2f"
            ),
            "cost_usd": st.column_config.NumberColumn("Cost (USD)", format="$%.2f"),
            "n_calls": st.column_config.NumberColumn("Calls"),
            "steps": st.column_config.NumberColumn("Steps"),
        },
        on_select="rerun",
        selection_mode="single-row",
        width="stretch",
    )

    selected_rows = extract_selected_rows(event)

    if selected_rows:
        sel_idx = int(selected_rows[0])
        if 0 <= sel_idx < len(row_experiments):
            st.subheader("Selected experiment")
            sel = row_experiments[sel_idx]
            if sel.is_completed:
                render_completed_experiment(cast(CompletedExperiment, sel.root))
            elif sel.is_failed:
                render_failed_experiment(cast(FailedExperiment, sel.root))
