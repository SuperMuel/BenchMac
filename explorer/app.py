"""
BenchMAC Results Explorer - A Streamlit app for exploring evaluation results.

This app aggregates all evaluation results found under the configured
`settings.evaluations_dir`, matches them to submissions under
`settings.experiments_dir`, and provides interactive visualizations.
"""

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import pandas as pd
import plotly.express as px  # type: ignore
import streamlit as st
from pydantic import ValidationError

from bench_mac.config import settings
from bench_mac.models import (
    CommandResult,
    EvaluationCompleted,
    EvaluationFailed,
    EvaluationResultAdapter,
    InstanceID,
    MetricsReport,
    Submission,
    SubmissionID,
)
from experiments.models import (
    AgentConfig,
    CompletedExperiment,
    ExperimentResult,
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
def load_all_evaluations(
    base_dir: Path,
) -> tuple[list[EvaluationCompleted], list[EvaluationFailed]]:
    """Load and validate all evaluations from every results.jsonl
    under base_dir."""
    if not base_dir.exists():
        return ([], [])

    jsonl_files = list(base_dir.rglob("*.jsonl"))
    successes: list[EvaluationCompleted] = []
    failures: list[EvaluationFailed] = []

    for file_path, line_num, line in _iter_lines_from_jsonl_files(jsonl_files):
        try:
            outcome = EvaluationResultAdapter.validate_json(line)
            match outcome.status:
                case "completed":
                    successes.append(outcome)
                case "failed":
                    failures.append(outcome)

        except (json.JSONDecodeError, ValidationError, Exception) as e:
            st.warning(
                f"Skipping invalid result in {file_path.name} line {line_num}: {e}"
            )

    return successes, failures


@st.cache_data(show_spinner=False)
def load_all_experiments(experiments_dir: Path) -> list[ExperimentResult]:
    """Load all experiment results (agent runs) from experiments_dir/results/*.jsonl."""
    if not experiments_dir.exists():
        return []

    results: list[ExperimentResult] = []
    results_dir = experiments_dir / "results"
    if not results_dir.exists():
        return results

    for file_path, line_num, line in _iter_lines_from_jsonl_files(
        results_dir.glob("*.jsonl")
    ):
        try:
            data = json.loads(line)
            result = ExperimentResult.model_validate(data)
            results.append(result)
        except (json.JSONDecodeError, ValidationError) as e:
            st.warning(
                f"Skipping invalid experiment result in {file_path.name} "
                f"line {line_num}: {e}"
            )

    return results


def group_completed_by_instance_and_model(
    completed: list[EvaluationCompleted],
    agent_by_submission_id: dict[SubmissionID, AgentConfig],
) -> dict[AgentConfig, dict[InstanceID, EvaluationCompleted]]:
    """Return latest completed evaluation per (instance, AgentConfig).

    The top-level key is a stable, human-readable AgentConfig label.
    """
    grouped: dict[AgentConfig, dict[InstanceID, EvaluationCompleted]] = {}
    for s in completed:
        inst_id = s.result.instance_id
        submission_id = s.result.submission_id

        # Determine agent_config for this submission
        agent_cfg = agent_by_submission_id.get(submission_id)
        assert agent_cfg is not None

        bucket = grouped.setdefault(agent_cfg, {})
        current = bucket.get(inst_id)
        if current is None or s.ended_at >= current.ended_at:
            bucket[inst_id] = s
    return grouped


def extract_summary_data(
    grouped_completed: dict[AgentConfig, dict[InstanceID, EvaluationCompleted]],
    harness_failures_list: list[EvaluationFailed],
) -> dict[str, Any]:
    """Compute top-level summary from grouped completed and harness failures.

    Only the latest result per (instance, model) is kept in `grouped_completed`.
    """
    completed_list = [
        s for by_instance in grouped_completed.values() for s in by_instance.values()
    ]

    total_evaluations = len(completed_list) + len(harness_failures_list)
    completed_evaluations = len(completed_list)
    harness_failures = len(harness_failures_list)

    metrics_summary = _compute_metrics_summary(completed_list)

    instance_ids = [s.result.instance_id for s in completed_list]

    return {
        "total_evaluations": total_evaluations,
        "completed_evaluations": completed_evaluations,
        "harness_failures": harness_failures,
        "harness_success_rate": completed_evaluations / total_evaluations
        if total_evaluations > 0
        else 0,
        "metrics_summary": metrics_summary,
        "instance_ids": sorted(set(instance_ids)),
    }


def _compute_metrics_summary(
    completed_list: list[EvaluationCompleted],
) -> dict[str, dict[str, float | int]]:
    """Compute success counts and rates for each metric field.

    Returns a mapping: metric_field -> {success_count, total_count, success_rate}.
    """
    metrics_fields = [
        "patch_application_success",
        "target_version_achieved",
        "install_success",
    ]

    metrics_summary: dict[str, dict[str, float | int]] = {}

    # Compute standard metrics (independent)
    for field in metrics_fields:
        values: list[bool] = []
        for s in completed_list:
            value = getattr(s.result.metrics, field)
            if value is not None:
                values.append(bool(value))
        total_count = len(completed_list) if field == "install_success" else len(values)
        success_count = sum(1 for v in values if v)
        if total_count > 0:
            metrics_summary[field] = {
                "success_count": success_count,
                "total_count": total_count,
                "success_rate": success_count / total_count,
            }

    # Compute build_success constrained by install_success and over all instances
    total_instances = len(completed_list)
    if total_instances > 0:
        constrained_build_success = 0
        for s in completed_list:
            m = s.result.metrics
            if m.install_success is True and m.build_success is True:
                constrained_build_success += 1
        metrics_summary["build_success"] = {
            "success_count": constrained_build_success,
            "total_count": total_instances,
            "success_rate": constrained_build_success / total_instances,
        }

    return metrics_summary


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

    fig = px.bar(  # type: ignore[reportUnknownReturnType]
        df,
        x="Metric",
        y="Success Rate",
        text=df["Success Rate"].round(1).astype(str) + "%",  # type: ignore
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

    for step in steps:
        # Show command in red if exit code is 1, otherwise normal
        command_display = (
            f":red[{step.command}]"
            if step.exit_code != 0
            else f":green[{step.command}]"
        )
        with st.expander(command_display, expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                st.metric("Exit Code", step.exit_code)
                duration_s = (step.end_time - step.start_time).total_seconds()
                st.metric("Duration", f"{duration_s:.2f}s")

            with col2:
                st.write(f"**Start:** {step.start_time}")
                st.write(f"**End:** {step.end_time}")

            if step.stdout:
                st.subheader("📤 STDOUT")
                st.code(step.stdout, language="bash")

            if step.stderr:
                st.subheader("⚠️ STDERR")
                st.code(step.stderr, language="bash")


def get_swe_mini_inspector_command(
    submission_id: str,
    experiments_dir: Path = settings.experiments_dir,
) -> tuple[str, bool]:
    """Get the inspector command for a submission and check if trajectory file exists.

    Returns:
        A tuple of (command, file_exists)
    """
    trajectory_path = experiments_dir / "swe_agent_mini" / f"{submission_id}.traj.json"
    trajectory_path_full = str(trajectory_path.resolve())
    command = f"uvx --from mini-swe-agent mini-extra inspector {trajectory_path_full}"
    return command, trajectory_path.exists()


def main() -> None:
    st.set_page_config(
        page_title="BenchMAC Results Explorer",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Sidebar
    with st.sidebar:
        st.header("🔄 Data Controls")
        if st.button(
            "🔄 Reload Data",
            help="Clear cache and reload all data",
            width="stretch",
        ):
            st.cache_data.clear()
            st.rerun()

    st.title("🔍 BenchMAC Results Explorer")

    evaluations_dir = settings.evaluations_dir
    experiments_dir = settings.experiments_dir

    st.markdown(
        f"""
        <div style="background-color: #f0f2f6; padding: 0.75em 1em; border-radius: 0.5em; margin-bottom: 1em;">
            <b>📂 Evaluations directory:</b> <code>{evaluations_dir}</code><br>
            <b>📂 Experiments directory:</b> <code>{experiments_dir}</code>
        </div>
        """,  # noqa: E501
        unsafe_allow_html=True,
    )

    with st.spinner("Loading evaluations and experiments..."):
        completed, harness_failures_list = load_all_evaluations(evaluations_dir)
        experiments = load_all_experiments(experiments_dir)

    # Build lookups from experiments
    submissions_by_id: dict[SubmissionID, Submission] = {}
    agent_by_submission_id: dict[SubmissionID, AgentConfig] = {}
    for er in experiments:
        if er.is_completed:
            er_completed = cast(CompletedExperiment, er.root)
            submissions_by_id[er_completed.submission.submission_id] = (
                er_completed.submission
            )
            agent_by_submission_id[er_completed.submission.submission_id] = (
                er_completed.task.agent_config
            )
        elif er.is_failed:
            continue

    grouped = group_completed_by_instance_and_model(completed, agent_by_submission_id)
    summary = extract_summary_data(grouped, harness_failures_list)

    # Harness failures section
    if summary["harness_failures"] > 0:
        st.header("🚨 Harness Failures")
        st.metric("Number of Failures", summary["harness_failures"])
        for f in harness_failures_list:
            with st.expander(
                f"❌ `{f.instance_id}` (submission: `{f.submission_id}`)",
                expanded=False,
            ):
                st.error(f.error)

    # Check if no evaluation results are available
    if not completed:
        st.info(
            "No evaluation results found. Please run some evaluations "
            "to see metrics and analysis."
        )
        return

    st.subheader("📈 Metrics Breakdown")
    if summary["metrics_summary"]:
        metrics_chart = create_metrics_chart(summary["metrics_summary"])
        if metrics_chart:
            st.plotly_chart(  # type: ignore
                metrics_chart, use_container_width=True, key="metrics_overall"
            )

        # Per-agent-config breakdown charts
        agent_configs = sorted(grouped.keys(), key=lambda x: x.key)
        if agent_configs:
            st.subheader("📊 Metrics Breakdown by Agent Configuration")
            tab_labels = [config.key for config in agent_configs]
            tabs = st.tabs(tab_labels)
            for tab, agent_config in zip(tabs, agent_configs, strict=True):
                by_instance = grouped[agent_config]
                per_agent_summary = _compute_metrics_summary(list(by_instance.values()))
                agent_chart = create_metrics_chart(per_agent_summary)
                with tab:
                    if agent_chart:
                        st.plotly_chart(  # type: ignore
                            agent_chart,
                            use_container_width=True,
                            key=f"metrics_agent_{agent_config}",
                        )
                    else:
                        st.info("No metrics data available for this agent.")

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
            df = pd.DataFrame(metrics_data)
            st.dataframe(df, width="stretch")  # type: ignore
    else:
        st.info("No metrics data available for analysis.")

    # Detailed results by agent configuration
    st.header("🔬 Results by Agent Configuration")

    agent_configs = sorted(grouped.keys(), key=lambda x: x.key)
    for agent_config in agent_configs:
        st.subheader(f"🤖 `{agent_config.key}`")
        instances_map = grouped[agent_config]

        st.write(f"{len(instances_map)} result(s)")

        for instance_id, success in sorted(instances_map.items()):
            metrics = success.result.metrics
            status_emoji = get_status_emoji(metrics)
            with st.expander(f"{status_emoji} `{instance_id}`", expanded=False):
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

                # Agent Inspector Section
                if agent_config.scaffold == "swe-agent-mini":
                    st.subheader("🔍 Agent Inspector")
                    inspector_command, file_exists = get_swe_mini_inspector_command(
                        str(success.result.submission_id)
                    )

                    if file_exists:
                        st.text(
                            "You can check the trajectory of this agent by running "
                            "the following command:"
                        )
                        st.code(
                            inspector_command,
                            language="bash",
                            wrap_lines=True,
                        )
                    else:
                        st.markdown(
                            f"<span style='color: grey;'>Trajectory file not found: "
                            f"`{success.result.submission_id}.traj.json`</span>",
                            unsafe_allow_html=True,
                        )


if __name__ == "__main__":
    main()
