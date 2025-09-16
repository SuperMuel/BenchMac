from typing import Any, cast

import pandas as pd
import streamlit as st

from bench_mac.config import settings
from experiments.models import CompletedExperiment, ExperimentResult, FailedExperiment
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


experiments: list[ExperimentResult] = load_all_experiments(settings.experiments_dir)

if not experiments:
    st.info("No experiment results found yet.")
else:
    st.write(f"Found {len(experiments)} result(s).")
    st.divider()

    rows: list[dict[str, object]] = []
    row_experiments: list[ExperimentResult] = []

    for er in experiments:
        if er.is_completed:
            c = cast(CompletedExperiment, er.root)
            total_s = c.duration.total_seconds()
            commands_s = (
                c.artifacts.execution_trace.total_duration.total_seconds()
                if c.artifacts and c.artifacts.execution_trace
                else None
            )
            model_only_s = (
                max(0.0, total_s - commands_s) if commands_s is not None else None
            )

            rows.append(
                {
                    "status": "completed",
                    "instance_id": c.task.instance_id,
                    "agent": c.task.agent_config.key,
                    "submission_id": str(c.submission.submission_id),
                    "started_at": c.started_at,
                    "ended_at": c.ended_at,
                    "duration_s": round(total_s, 2),
                    "commands_s": (
                        round(commands_s, 2) if commands_s is not None else None
                    ),
                    "model_only_s": (
                        round(model_only_s, 2) if model_only_s is not None else None
                    ),
                    "steps": (
                        len(c.artifacts.execution_trace.steps)
                        if c.artifacts and c.artifacts.execution_trace
                        else 0
                    ),
                }
            )
            row_experiments.append(er)
        elif er.is_failed:
            f = cast(FailedExperiment, er.root)
            total_s = (f.ended_at - f.started_at).total_seconds()
            commands_s = (
                f.artifacts.execution_trace.total_duration.total_seconds()
                if f.artifacts and f.artifacts.execution_trace
                else None
            )
            row_experiments.append(er)
            model_only_s = (
                max(0.0, total_s - commands_s) if commands_s is not None else None
            )

            rows.append(
                {
                    "status": "failed",
                    "instance_id": f.task.instance_id,
                    "agent": f.task.agent_config.key,
                    "submission_id": "",
                    "started_at": f.started_at,
                    "ended_at": f.ended_at,
                    "duration_s": round(total_s, 2),
                    "commands_s": (
                        round(commands_s, 2) if commands_s is not None else None
                    ),
                    "model_only_s": (
                        round(model_only_s, 2) if model_only_s is not None else None
                    ),
                    "steps": (
                        len(f.artifacts.execution_trace.steps)
                        if f.artifacts and f.artifacts.execution_trace
                        else 0
                    ),
                }
            )

    df = pd.DataFrame(rows)

    event: Any = st.dataframe(
        df,
        hide_index=True,
        column_config={
            "status": st.column_config.TextColumn("Status"),
            "instance_id": st.column_config.TextColumn("Instance"),
            "agent": st.column_config.TextColumn("Agent"),
            "submission_id": st.column_config.TextColumn("Submission"),
            "started_at": st.column_config.DatetimeColumn("Started"),
            "ended_at": st.column_config.DatetimeColumn("Ended"),
            "duration_s": st.column_config.NumberColumn("Duration (s)", format="%.2f"),
            "commands_s": st.column_config.NumberColumn("Commands (s)", format="%.2f"),
            "model_only_s": st.column_config.NumberColumn(
                "Model-only (s)", format="%.2f"
            ),
            "steps": st.column_config.NumberColumn("Steps"),
        },
        on_select="rerun",
        selection_mode="single-row",
        width="stretch",
    )

    selection = None
    if event is not None:
        selection = getattr(event, "selection", None)
        if selection is None and isinstance(event, dict):
            selection = event.get("selection")

    selected_rows: list[int] | None = None
    if selection is not None:
        rows_attr = getattr(selection, "rows", None)
        if isinstance(rows_attr, list):
            selected_rows = rows_attr
        elif isinstance(selection, dict):
            rows_val = selection.get("rows")
            if isinstance(rows_val, list):
                selected_rows = rows_val

    if selected_rows:
        sel_idx = int(selected_rows[0])
        if 0 <= sel_idx < len(row_experiments):
            st.subheader("Selected experiment")
            sel = row_experiments[sel_idx]
            if sel.is_completed:
                c = cast(CompletedExperiment, sel.root)
                total_s = c.duration.total_seconds()
                commands_s = (
                    c.artifacts.execution_trace.total_duration.total_seconds()
                    if c.artifacts and c.artifacts.execution_trace
                    else None
                )
                model_only_s = (
                    max(0.0, total_s - commands_s) if commands_s is not None else None
                )

                cols = st.columns(5)
                cols[0].metric("Status", "completed")
                cols[1].metric("Duration", f"{total_s:.2f}s")
                cols[2].metric(
                    "Commands",
                    f"{commands_s:.2f}s" if commands_s is not None else "N/A",
                )
                cols[3].metric(
                    "Model-only",
                    f"{model_only_s:.2f}s" if model_only_s is not None else "N/A",
                )
                cols[4].metric(
                    "Steps",
                    str(
                        len(c.artifacts.execution_trace.steps)
                        if c.artifacts and c.artifacts.execution_trace
                        else 0
                    ),
                )

                st.write(f"**Started:** {c.started_at}")
                st.write(f"**Ended:** {c.ended_at}")
                st.write("**Agent:** `" + c.task.agent_config.key + "`")
                st.write("**Instance:** `" + c.task.instance_id + "`")
                st.write("**Submission ID:** `" + str(c.submission.submission_id) + "`")

                if c.artifacts and c.artifacts.execution_trace:
                    st.subheader("Execution steps")
                    display_execution_steps(c.artifacts.execution_trace.steps)

                # Model patch viewer
                if c.submission.model_patch:
                    st.subheader("Model patch")
                    include_package_lock = st.toggle(
                        "Show package-lock.json changes", value=False
                    )
                    patch_to_show = filter_patch_excluding_package_lock(
                        c.submission.model_patch, include_package_lock
                    )
                    patch_lines = patch_to_show.splitlines()
                    with st.expander(
                        f"Unified diff ({len(patch_lines)} lines)", expanded=False
                    ):
                        st.download_button(
                            "Download patch",
                            data=patch_to_show,
                            file_name=f"{c.submission.submission_id}.patch",
                            mime="text/x-diff",
                        )
                        st.code(
                            patch_to_show,
                            language="diff",
                            wrap_lines=True,
                        )
            elif sel.is_failed:
                f = cast(FailedExperiment, sel.root)
                total_s = (f.ended_at - f.started_at).total_seconds()
                commands_s = (
                    f.artifacts.execution_trace.total_duration.total_seconds()
                    if f.artifacts and f.artifacts.execution_trace
                    else None
                )
                model_only_s = (
                    max(0.0, total_s - commands_s) if commands_s is not None else None
                )

                cols = st.columns(5)
                cols[0].metric("Status", "failed")
                cols[1].metric("Duration", f"{total_s:.2f}s")
                cols[2].metric(
                    "Commands",
                    f"{commands_s:.2f}s" if commands_s is not None else "N/A",
                )
                cols[3].metric(
                    "Model-only",
                    f"{model_only_s:.2f}s" if model_only_s is not None else "N/A",
                )
                cols[4].metric(
                    "Steps",
                    str(
                        len(f.artifacts.execution_trace.steps)
                        if f.artifacts and f.artifacts.execution_trace
                        else 0
                    ),
                )

                st.write(f"**Started:** {f.started_at}")
                st.write(f"**Ended:** {f.ended_at}")
                st.write("**Agent:** `" + f.task.agent_config.key + "`")
                st.write("**Instance:** `" + f.task.instance_id + "`")
                st.error(f.error)
