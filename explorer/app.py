"""
BenchMAC Results Explorer - A Streamlit app for exploring evaluation results.

This app aggregates all evaluation results found under the configured
`settings.evaluations_dir`, matches them to submissions under
`settings.experiments_dir`, and provides interactive visualizations.
"""

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px  # type: ignore
import streamlit as st
from pydantic import ValidationError

from bench_mac.config import settings
from bench_mac.models import (
    CommandResult,
    EvaluationCompleted,
    EvaluationFailed,
    MetricsReport,
    Submission,
)


def _iter_lines_from_jsonl_files(
    files: Iterable[Path],
) -> Iterable[tuple[Path, int, str]]:
    for file_path in files:
        try:
            with file_path.open("r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    yield (file_path, line_num, line)
        except Exception as e:  # pragma: no cover - defensive
            st.warning(f"Unable to read {file_path}: {e}")


@st.cache_data(show_spinner=False)
def load_all_outcomes() -> tuple[list[EvaluationCompleted], list[EvaluationFailed]]:
    """Load and validate all outcomes from every results.jsonl under evaluations_dir."""
    base_dir = settings.evaluations_dir
    if not base_dir.exists():
        return ([], [])

    jsonl_files = list(base_dir.rglob("*.jsonl"))
    successes: list[EvaluationCompleted] = []
    failures: list[EvaluationFailed] = []

    for file_path, line_num, line in _iter_lines_from_jsonl_files(jsonl_files):
        try:
            # Peek status without double-parsing
            status = json.loads(line).get("status")
            if status == "success":
                successes.append(EvaluationCompleted.model_validate_json(line))
            elif status == "failure":
                failures.append(EvaluationFailed.model_validate_json(line))
            else:
                st.warning(
                    f"Unknown status in {file_path.name} line {line_num}: {status}"
                )
        except (json.JSONDecodeError, ValidationError, Exception) as e:
            st.warning(
                f"Skipping invalid result in {file_path.name} line {line_num}: {e}"
            )

    return successes, failures


@st.cache_data(show_spinner=False)
def load_all_submissions() -> dict[str, Submission]:
    """
    Load all submissions from experiments_dir/submissions/*.jsonl keyed by
    submission_id.
    """
    subs_dir = settings.experiments_dir / "submissions"
    if not subs_dir.exists():
        return {}

    mapping: dict[str, Submission] = {}
    for file_path, line_num, line in _iter_lines_from_jsonl_files(
        subs_dir.glob("*.jsonl")
    ):
        try:
            sub = Submission.model_validate_json(line)
            mapping[sub.submission_id] = sub
        except (json.JSONDecodeError, ValidationError, Exception) as e:
            st.warning(
                f"Skipping invalid submission in {file_path.name} line {line_num}: {e}"
            )
    return mapping


def _latest_timestamp_for(success: EvaluationCompleted) -> float:
    steps = success.result.execution.steps
    if not steps:
        return 0.0
    return steps[-1].end_time.timestamp()


def group_successes_by_instance_and_model(
    successes: list[EvaluationCompleted], submissions_by_id: dict[str, Submission]
) -> dict[str, dict[str, EvaluationCompleted]]:
    """Return latest success per (instance_id, model_name) as nested mapping
    model->instance->result.
    """
    grouped: dict[str, dict[str, EvaluationCompleted]] = {}
    for s in successes:
        sub = submissions_by_id.get(s.result.submission_id)
        model_name = (
            sub.metadata.model_name if sub and sub.metadata.model_name else "(unknown)"
        )
        inst_id = s.result.instance_id

        bucket = grouped.setdefault(model_name, {})
        current = bucket.get(inst_id)
        if current is None or _latest_timestamp_for(s) >= _latest_timestamp_for(
            current
        ):
            bucket[inst_id] = s
    return grouped


def extract_summary_data(
    grouped_successes: dict[str, dict[str, EvaluationCompleted]],
    failures: list[EvaluationFailed],
) -> dict[str, Any]:
    """Compute top-level summary from grouped successes (latest only) and failures."""
    successes_list = [
        s for by_instance in grouped_successes.values() for s in by_instance.values()
    ]

    total_evaluations = len(successes_list) + len(failures)
    successful_evaluations = len(successes_list)
    failed_evaluations = len(failures)

    metrics_fields = [
        "patch_application_success",
        "target_version_achieved",
        "install_success",
        "build_success",
    ]

    metrics_summary: dict[str, dict[str, float | int]] = {}
    for field in metrics_fields:
        values: list[bool] = []
        for s in successes_list:
            value = getattr(s.result.metrics, field)
            if value is not None:
                values.append(bool(value))
        if values:
            success_count = sum(1 for v in values if v)
            total_count = len(values)
            metrics_summary[field] = {
                "success_count": success_count,
                "total_count": total_count,
                "success_rate": success_count / total_count if total_count > 0 else 0,
            }

    instance_ids = [s.result.instance_id for s in successes_list]

    return {
        "total_evaluations": total_evaluations,
        "successful_evaluations": successful_evaluations,
        "failed_evaluations": failed_evaluations,
        "success_rate": successful_evaluations / total_evaluations
        if total_evaluations > 0
        else 0,
        "metrics_summary": metrics_summary,
        "instance_ids": sorted(set(instance_ids)),
    }


def create_metrics_chart(metrics_summary: dict[str, Any]) -> Any:
    """Create a bar chart showing metrics success rates."""
    if not metrics_summary:
        return None

    metrics_names = {
        "patch_application_success": "Patch Application",
        "target_version_achieved": "Target Version Achieved",
        "install_success": "Install Success",
        "build_success": "Build Success",
    }

    data: list[dict[str, Any]] = []
    for field, info in metrics_summary.items():
        data.append(
            {
                "Metric": metrics_names.get(field, field),
                "Success Rate": float(info["success_rate"]) * 100,
                "Success Count": int(info["success_count"]),
                "Total Count": int(info["total_count"]),
            }
        )

    if not data:
        return None

    df = pd.DataFrame(data)

    fig = px.bar(  # type: ignore
        df,
        x="Metric",
        y="Success Rate",
        text=df["Success Rate"].round(1).astype(str) + "%",  # type: ignore
        title="Metrics Success Rates",
        color="Success Rate",
        color_continuous_scale="RdYlGn",
    )

    fig.update_traces(textposition="outside")  # type: ignore
    fig.update_layout(yaxis_title="Success Rate (%)", xaxis_title="", showlegend=False)  # type: ignore

    return fig


def get_status_emoji(metrics: MetricsReport) -> str:
    """Return a status emoji based on metrics.

    - Green (✅): all available metrics are True
    - Orange (🟠): at least one True, but not all
    - Red (❌): none are True (including when none are available)
    """
    metric_values: list[bool | None] = [
        metrics.patch_application_success,
        metrics.target_version_achieved,
        metrics.install_success,
        metrics.build_success,
    ]

    valid_values: list[bool] = [v for v in metric_values if v is not None]
    if not valid_values:
        return "❌"

    passing_count = sum(1 for v in valid_values if v)
    if passing_count == len(valid_values):
        return "✅"
    if passing_count > 0:
        return "🟠"
    return "❌"


def display_execution_steps(steps: list[CommandResult]) -> None:
    """Display execution steps in a nice format."""
    if not steps:
        st.write("No execution steps available.")
        return

    for i, step in enumerate(steps, 1):
        with st.expander(f"Step {i}: {step.command}", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                st.metric("Exit Code", step.exit_code)
                duration_s = (step.end_time - step.start_time).total_seconds()
                st.metric("Duration", f"{duration_s:.2f}s")

            with col2:
                st.write(f"**Start:** {step.start_time}")
                st.write(f"**End:** {step.end_time}")

            if step.stdout:
                st.subheader("📤 Standard Output")
                st.code(step.stdout, language="bash")

            if step.stderr:
                st.subheader("⚠️ Standard Error")
                st.code(step.stderr, language="bash")


def main() -> None:
    st.set_page_config(
        page_title="BenchMAC Results Explorer", page_icon="🔍", layout="wide"
    )

    st.title("🔍 BenchMAC Results Explorer")
    st.markdown(
        "Explore your BenchMAC evaluation results aggregated from your evaluations "
        "directory."
    )

    with st.spinner("Loading outcomes and submissions..."):
        successes, failures = load_all_outcomes()
        submissions_by_id = load_all_submissions()

    grouped = group_successes_by_instance_and_model(successes, submissions_by_id)
    summary = extract_summary_data(grouped, failures)

    # Harness failures section
    if summary["failed_evaluations"] > 0:
        st.header("🚨 Harness Failures")
        st.metric("Number of Failures", summary["failed_evaluations"])
        for f in failures:
            with st.expander(
                f"❌ {f.instance_id} (submission: {f.submission_id})",
                expanded=False,
            ):
                st.error(f.error)

    # Overview section for grouped successes
    st.header("📊 Overview (Successful Evaluations Only)")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Successful", summary["successful_evaluations"])
    with col2:
        st.metric("Failed (ignored below)", summary["failed_evaluations"])
    with col3:
        st.metric("Success Rate", f"{summary['success_rate'] * 100:.1f}%")

    st.subheader("📈 Metrics Breakdown")
    if summary["metrics_summary"]:
        metrics_chart = create_metrics_chart(summary["metrics_summary"])
        if metrics_chart:
            st.plotly_chart(metrics_chart, use_container_width=True)  # type: ignore

        # Detailed metrics table
        st.subheader("Detailed Metrics")
        metrics_data: list[dict[str, Any]] = []
        for field, info in summary["metrics_summary"].items():
            metrics_data.append(
                {
                    "Metric": field.replace("_", " ").title(),
                    "Success": info["success_count"],
                    "Total": info["total_count"],
                    "Success Rate": f"{info['success_rate'] * 100:.1f}%",
                }
            )

        if metrics_data:
            df = pd.DataFrame(metrics_data)  # type: ignore
            st.dataframe(df, use_container_width=True)  # type: ignore
    else:
        st.info("No metrics data available for analysis.")

    # Detailed results by agent configuration (model_name)
    st.header("🔬 Detailed Results by Agent Configuration")

    model_names = sorted(grouped.keys())
    for model_name in model_names:
        st.subheader(f"🤖 {model_name}")
        instances_map = grouped[model_name]

        st.write(f"{len(instances_map)} result(s)")

        for instance_id, success in sorted(instances_map.items()):
            metrics = success.result.metrics
            status_emoji = get_status_emoji(metrics)
            with st.expander(f"{status_emoji} {instance_id}", expanded=False):
                cols = st.columns(4)
                metric_fields = [
                    ("Patch Application", metrics.patch_application_success),
                    ("Target Version Achieved", metrics.target_version_achieved),
                    ("Install Success", metrics.install_success),
                    ("Build Success", metrics.build_success),
                ]

                for i, (label, value) in enumerate(metric_fields):
                    if value is None:
                        cols[i].metric(label, "N/A")
                    else:
                        cols[i].metric(label, "✅" if value else "❌")

                st.subheader("Execution Steps")
                display_execution_steps(success.result.execution.steps)


if __name__ == "__main__":
    main()
