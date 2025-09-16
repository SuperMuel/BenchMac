"""
BenchMAC Results Explorer - A Streamlit app for exploring evaluation results.

This app aggregates all evaluation results found under the configured
`settings.evaluations_dir`, matches them to submissions under
`settings.experiments_dir`, and provides interactive visualizations.
"""

import json
from pathlib import Path
from typing import Any, cast

import pandas as pd
import plotly.express as px  # type: ignore
import plotly.graph_objects as go  # type: ignore
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
from bench_mac.utils import collect_network_error_details, iter_lines_from_jsonl_files
from experiments.models import (
    AgentConfig,
    CompletedExperiment,
    ExperimentResult,
)


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

    for file_path, line_num, line in iter_lines_from_jsonl_files(jsonl_files):
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

    for file_path, line_num, line in iter_lines_from_jsonl_files(
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


def create_agent_radar_chart(
    grouped_completed: dict[AgentConfig, dict[InstanceID, EvaluationCompleted]],
) -> Any:
    """Create a radar chart showing all agent configurations'
    metrics on the same graph."""
    if not grouped_completed:
        return None

    metrics_names = {
        "patch_application_success": "Patch Application",
        "target_version_achieved": "Target Version Achieved",
        "install_success": "Install Success",
        "build_success": "Build Success",
    }

    # Prepare data for radar chart
    metrics = list(metrics_names.keys())
    metric_labels = list(metrics_names.values())

    fig = go.Figure()  # type: ignore[reportUnknownReturnType]

    # Color palette for different agents
    colors = [
        "#1f77b4",  # Blue
        "#ff7f0e",  # Orange
        "#2ca02c",  # Green
        "#d62728",  # Red
        "#9467bd",  # Purple
        "#8c564b",  # Brown
        "#e377c2",  # Pink
        "#7f7f7f",  # Gray
    ]

    for i, (agent_config, instances) in enumerate(grouped_completed.items()):
        completed_list = list(instances.values())
        if not completed_list:
            continue

        metrics_summary = _compute_metrics_summary(completed_list)

        # Extract success rates for each metric
        values = []
        for metric in metrics:
            if metric in metrics_summary:
                success_rate = float(metrics_summary[metric]["success_rate"]) * 100
            else:
                success_rate = 0.0
            values.append(success_rate)

        # Add the first value again to close the radar chart
        values.append(values[0])
        categories = [*metric_labels, metric_labels[0]]

        color = colors[i % len(colors)]

        # Convert hex color to rgba for transparency
        if color.startswith("#"):
            # Convert hex to rgba with 0.3 alpha
            r = int(color[1:3], 16)
            g = int(color[3:5], 16)
            b = int(color[5:7], 16)
            fill_color = f"rgba({r}, {g}, {b}, 0.3)"
        else:
            fill_color = color

        fig.add_trace(
            go.Scatterpolar(  # type: ignore[reportUnknownArgumentType]
                r=values,
                theta=categories,
                name=agent_config.key,
                line={"color": color, "width": 2},
                fill="toself",
                fillcolor=fill_color,
                hovertemplate="<b>%{fullData.name}</b><br>"
                + "%{theta}: %{r:.1f}%<extra></extra>",
            )
        )

    if not fig.data:
        return None

    fig.update_layout(  # type: ignore
        polar={
            "radialaxis": {
                "visible": True,
                "range": [0, 100],
                "tickfont": {"size": 10},
                "title": {"text": "Success Rate (%)", "font": {"size": 12}},
            },
            "angularaxis": {
                "tickfont": {"size": 11},
                "rotation": 90,
                "direction": "clockwise",
            },
        },
        showlegend=True,
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.1,
            "xanchor": "center",
            "x": 0.5,
        },
        title={
            "text": "Agent Configuration Performance Comparison",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 16},
        },
        height=600,
        margin={"t": 100, "b": 100, "l": 80, "r": 80},
    )

    return fig


def compute_agent_scores(
    grouped_completed: dict[AgentConfig, dict[InstanceID, EvaluationCompleted]],
) -> dict[AgentConfig, dict[str, float | int]]:
    """Compute ranking scores for each agent configuration.

    Scoring priorities (in order):
    1. build_success (most important)
    2. install_success (second most important)
    3. patch_application_success (third)
    4. target_version_achieved (fourth)

    CRITICAL: If target_version_achieved success rate is 0%, the overall score is 0.

    Returns a dict mapping agent configs to their performance metrics.
    """
    agent_scores: dict[AgentConfig, dict[str, float | int]] = {}

    for agent_config, instances in grouped_completed.items():
        completed_list = list(instances.values())

        # Compute individual metrics
        metrics_summary = _compute_metrics_summary(completed_list)

        # Check if target_version_achieved is 0% - if so, score is automatically 0
        target_success_rate = metrics_summary.get("target_version_achieved", {}).get(
            "success_rate", 0
        )

        if target_success_rate == 0:
            overall_score = 0.0
        else:
            # Compute weighted overall score (prioritizing key metrics)
            weights = {
                "build_success": 0.4,  # Most important
                "install_success": 0.3,  # Second most important
                "patch_application_success": 0.2,  # Third
                "target_version_achieved": 0.1,  # Fourth
            }

            overall_score = 0.0
            total_weight = 0.0

            for metric_name, weight in weights.items():
                if metric_name in metrics_summary:
                    metric_info = metrics_summary[metric_name]
                    overall_score += float(metric_info["success_rate"]) * weight
                    total_weight += weight

            if total_weight > 0:
                overall_score /= total_weight

        agent_scores[agent_config] = {
            "overall_score": overall_score * 100,  # Convert to percentage
            "total_evaluations": len(completed_list),
            "patch_success_rate": (
                metrics_summary.get("patch_application_success", {}).get(
                    "success_rate", 0
                )
                * 100
            ),
            "target_success_rate": target_success_rate * 100,
            "install_success_rate": (
                metrics_summary.get("install_success", {}).get("success_rate", 0) * 100
            ),
            "build_success_rate": (
                metrics_summary.get("build_success", {}).get("success_rate", 0) * 100
            ),
        }

    return agent_scores


def display_agent_leaderboard(
    grouped: dict[AgentConfig, dict[InstanceID, EvaluationCompleted]],
) -> None:
    """Display the agent configuration leaderboard with detailed rankings table.

    Only displays if there are multiple agent configurations to compare.
    """
    agent_configs = sorted(grouped.keys(), key=lambda x: x.key)
    if len(agent_configs) <= 1:
        return

    st.header("🏆 Leaderboard")

    agent_scores = compute_agent_scores(grouped)

    # Detailed leaderboard table
    sorted_agents = sorted(
        agent_scores.items(), key=lambda x: x[1]["overall_score"], reverse=True
    )

    leaderboard_data: list[dict[str, Any]] = []
    for rank, (agent_config, scores) in enumerate(sorted_agents, 1):
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"#{rank}")
        leaderboard_data.append(
            {
                "Rank": f"{medal}",
                "Agent Configuration": agent_config.key,
                "Overall Score": f"{scores['overall_score']:.1f}%",
                "Total Evaluations": int(scores["total_evaluations"]),
                "Patch Success": f"{scores['patch_success_rate']:.1f}%",
                "Install Success": f"{scores['install_success_rate']:.1f}%",
                "Target Success": f"{scores['target_success_rate']:.1f}%",
                "Build Success": f"{scores['build_success_rate']:.1f}%",
            }
        )

    if leaderboard_data:
        df = pd.DataFrame(leaderboard_data)

        # Best performing agent highlight
        best_agent = sorted_agents[0][0]
        best_score = sorted_agents[0][1]["overall_score"]
        st.success(
            f"🎯 **Top Performing Agent:** `{best_agent.key}` with "
            f"**{best_score:.1f}%** overall success rate"
        )

        st.dataframe(df, width="stretch", hide_index=True)


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
        # Show command in ❌ if exit code is 1, otherwise ✅
        escaped_command = step.command.replace("`", "\\`")
        command_display = (
            f"❌`{escaped_command}`"
            if step.exit_code != 0
            else f"✅`{escaped_command}`"
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

    # Network error detection
    network_error_details = collect_network_error_details(completed)  # type: ignore[arg-type]
    if network_error_details:
        st.header("⚠️ Network Errors Detected")
        st.warning(
            "One or more evaluations failed due to potential network issues "
            "(e.g., 'Socket timeout', 'Connection reset', 'Network is unreachable'). "
            "These failures may be transient and not reflective of the "
            "submission's quality. Consider re-running the evaluation with a "
            "stable network connection."
        )

        with st.expander("📋 Network Error Details", expanded=False):
            for evaluation_id, submission_id, commands in network_error_details:
                st.write(f"**Evaluation:** `{evaluation_id}`")
                st.write(f"**Submission:** `{submission_id}`")
                st.write("**Affected Commands:**")
                for cmd in commands:
                    st.code(cmd, language="bash")
                st.divider()

    # Check if no evaluation results are available
    if not completed:
        st.info(
            "No evaluation results found. Please run some evaluations "
            "to see metrics and analysis."
        )
        return
    # Agent Configuration Leaderboard
    display_agent_leaderboard(grouped)

    # Spider graph for agent comparison
    if grouped:
        st.subheader("🕸️ Metrics Breakdown by Agent Configuration")
        radar_chart = create_agent_radar_chart(grouped)
        if radar_chart:
            st.plotly_chart(  # type: ignore
                radar_chart,
                width="stretch",
                key="agent_radar_chart",
            )

        else:
            st.info("No metrics data available for analysis.")
    else:
        st.info("No agent configuration data available for analysis.")

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
                st.markdown(
                    f"**Evaluation ID:** `{success.id}`  |  "
                    f" **Submission ID:** `{success.result.submission_id}`"
                )
                st.divider()
                cols = st.columns(4)
                metric_fields = [
                    ("Patch Application", metrics.patch_application_success),
                    ("Install Success", metrics.install_success),
                    ("Target Version Achieved", metrics.target_version_achieved),
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
