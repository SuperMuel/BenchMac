"""
BenchMAC Instance Analysis - Analysis grouped by instance/task difficulty.

This page aggregates evaluation results by instance (task) to show which tasks
are easiest vs hardest for agents to solve. Results are aggregated across all
agent configurations that attempted each instance.
"""

import json
from pathlib import Path
from typing import Any, cast

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pydantic import ValidationError

from bench_mac.core.config import settings
from bench_mac.core.models import (
    CommandResult,
    EvaluationCompleted,
    EvaluationFailed,
    EvaluationResultAdapter,
    InstanceID,
    MetricsReport,
    SubmissionID,
)
from bench_mac.core.utils_jsonl import iter_lines_from_jsonl_files
from experiments.models import (
    AgentConfig,
    CompletedExperiment,
    ExperimentResult,
)

st.set_page_config(page_title="Instance Analysis", page_icon="🎯")


@st.cache_data(show_spinner=False)
def load_all_evaluations(
    base_dir: Path,
) -> tuple[list[EvaluationCompleted], list[EvaluationFailed]]:
    """Load and validate all evaluations from every results.jsonl under base_dir."""
    if not base_dir.exists():
        return ([], [])

    jsonl_files = list(base_dir.rglob("*.jsonl"))
    successes: list[EvaluationCompleted] = []
    failures: list[EvaluationFailed] = []

    for file_path, line_num, line in iter_lines_from_jsonl_files(jsonl_files):
        try:
            result = EvaluationResultAdapter.validate_json(line)
            match result.status:
                case "completed":
                    successes.append(result)
                case "failed":
                    failures.append(result)

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


def group_completed_by_instance(
    completed: list[EvaluationCompleted],
    agent_by_submission_id: dict[SubmissionID, AgentConfig],
) -> dict[InstanceID, dict[AgentConfig, EvaluationCompleted]]:
    """Group completed evaluations by instance, then by agent configuration.

    Returns latest evaluation per (instance, agent) pair.
    """
    grouped: dict[InstanceID, dict[AgentConfig, EvaluationCompleted]] = {}

    for evaluation in completed:
        inst_id = evaluation.result.instance_id
        submission_id = evaluation.result.submission_id

        # Determine agent_config for this submission
        agent_cfg = agent_by_submission_id.get(submission_id)
        if agent_cfg is None:
            st.warning(f"No agent config for submission: {submission_id}")
            continue

        instance_bucket = grouped.setdefault(inst_id, {})
        current = instance_bucket.get(agent_cfg)
        if current is None or evaluation.ended_at >= current.ended_at:
            instance_bucket[agent_cfg] = evaluation

    return grouped


def compute_instance_metrics(
    evaluations: dict[AgentConfig, EvaluationCompleted],
) -> dict[str, dict[str, float | int]]:
    """Compute aggregated metrics for an instance across all agents.

    Returns metrics summary with success counts and rates.
    """
    if not evaluations:
        return {}

    metrics_fields = [
        "patch_application_success",
        "target_version_achieved",
        "install_success",
    ]

    metrics_summary: dict[str, dict[str, float | int]] = {}

    # Compute standard metrics
    for field in metrics_fields:
        values: list[bool] = []
        for evaluation in evaluations.values():
            value = getattr(evaluation.result.metrics, field)
            if value is not None:
                values.append(bool(value))

        total_count = len(evaluations)
        success_count = sum(1 for v in values if v)
        if total_count > 0:
            metrics_summary[field] = {
                "success_count": success_count,
                "total_count": total_count,
                "success_rate": success_count / total_count,
            }

    # Compute build_success constrained by install_success
    total_instances = len(evaluations)
    if total_instances > 0:
        constrained_build_success = 0
        for evaluation in evaluations.values():
            m = evaluation.result.metrics
            if m.install_success is True and m.build_success is True:
                constrained_build_success += 1
        metrics_summary["build_success"] = {
            "success_count": constrained_build_success,
            "total_count": total_instances,
            "success_rate": constrained_build_success / total_instances,
        }

    return metrics_summary


def compute_instance_difficulty_score(
    metrics_summary: dict[str, dict[str, float | int]],
) -> float:
    """Compute a difficulty score for an instance based on success rates.

    Lower score = harder instance (fewer successes).
    Higher score = easier instance (more successes).

    Scoring weights (in order of importance):
    1. build_success (most important)
    2. install_success
    3. patch_application_success
    4. target_version_achieved
    """
    if not metrics_summary:
        return 0.0

    weights = {
        "build_success": 0.4,  # Most important
        "install_success": 0.3,  # Second most important
        "patch_application_success": 0.2,  # Third
        "target_version_achieved": 0.1,  # Fourth
    }

    difficulty_score = 0.0
    total_weight = 0.0

    for metric_name, weight in weights.items():
        if metric_name in metrics_summary:
            metric_info = metrics_summary[metric_name]
            difficulty_score += float(metric_info["success_rate"]) * weight
            total_weight += weight

    return difficulty_score if total_weight > 0 else 0.0


def create_instance_difficulty_chart(
    instance_data: list[dict[str, Any]],
) -> Any:
    """Create a bar chart showing instance difficulty (easiest to hardest)."""
    if not instance_data:
        return None

    # Sort by difficulty score (ascending - hardest first)
    sorted_data = sorted(instance_data, key=lambda x: x["difficulty_score"])

    df = pd.DataFrame(sorted_data)

    fig = px.bar(
        df,
        x="difficulty_score",
        y="instance_id",
        orientation="h",
        title="Instance Difficulty Ranking",
        labels={
            "difficulty_score": "Success Rate Score (0-1)",
            "instance_id": "Instance ID",
        },
        color="difficulty_score",
        color_continuous_scale="RdYlGn",
    )

    fig.update_layout(
        xaxis_title="Difficulty Score (Higher = Easier)",
        yaxis_title="",
        showlegend=False,
        height=max(400, len(instance_data) * 20),  # Dynamic height
    )

    return fig


def create_instance_radar_chart(
    grouped_by_instance: dict[InstanceID, dict[AgentConfig, EvaluationCompleted]],
) -> Any:
    """Create a radar chart showing performance across metrics for each instance."""
    if not grouped_by_instance:
        return None

    metrics_names = {
        "patch_application_success": "Patch Application",
        "target_version_achieved": "Target Version Achieved",
        "install_success": "Install Success",
        "build_success": "Build Success",
    }

    metrics = list(metrics_names.keys())
    metric_labels = list(metrics_names.values())

    # Color palette for different instances
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    fig = go.Figure()

    # Take top 10 easiest and hardest instances for readability
    instance_scores = []
    for instance_id, evaluations in grouped_by_instance.items():
        metrics_summary = compute_instance_metrics(evaluations)
        difficulty_score = compute_instance_difficulty_score(metrics_summary)
        instance_scores.append((instance_id, difficulty_score, metrics_summary))

    # Sort and take top 5 easiest and top 5 hardest
    sorted_instances = sorted(instance_scores, key=lambda x: x[1], reverse=True)
    selected_instances = (
        sorted_instances[:5] + sorted_instances[-5:]
        if len(sorted_instances) > 10
        else sorted_instances
    )

    for i, (instance_id, _, metrics_summary) in enumerate(selected_instances):
        if not metrics_summary:
            continue

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
            r = int(color[1:3], 16)
            g = int(color[3:5], 16)
            b = int(color[5:7], 16)
            fill_color = f"rgba({r}, {g}, {b}, 0.2)"
        else:
            fill_color = color

        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=categories,
                name=str(instance_id),
                line={"color": color, "width": 2},
                fill="toself",
                fillcolor=fill_color,
                hovertemplate="<b>%{fullData.name}</b><br>"
                + "%{theta}: %{r:.1f}%<extra></extra>",
            )
        )

    if not fig.data:
        return None

    fig.update_layout(
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
            "text": "Instance Performance Comparison (Top 5 Easiest + Top 5 Hardest)",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 16},
        },
        height=600,
        margin={"t": 100, "b": 100, "l": 80, "r": 80},
    )

    return fig


def display_instance_leaderboard(
    grouped_by_instance: dict[InstanceID, dict[AgentConfig, EvaluationCompleted]],
) -> None:
    """Display the instance difficulty leaderboard."""
    if not grouped_by_instance:
        return

    st.header("🎯 Instance Difficulty Analysis")

    # Compute difficulty scores for all instances
    instance_data = []
    for instance_id, evaluations in grouped_by_instance.items():
        metrics_summary = compute_instance_metrics(evaluations)
        difficulty_score = compute_instance_difficulty_score(metrics_summary)

        total_attempts = len(evaluations)

        instance_data.append(
            {
                "instance_id": instance_id,
                "difficulty_score": difficulty_score,
                "total_attempts": total_attempts,
                "patch_success_rate": (
                    metrics_summary.get("patch_application_success", {}).get(
                        "success_rate", 0
                    )
                    * 100
                ),
                "install_success_rate": (
                    metrics_summary.get("install_success", {}).get("success_rate", 0)
                    * 100
                ),
                "target_success_rate": (
                    metrics_summary.get("target_version_achieved", {}).get(
                        "success_rate", 0
                    )
                    * 100
                ),
                "build_success_rate": (
                    metrics_summary.get("build_success", {}).get("success_rate", 0)
                    * 100
                ),
            }
        )

    # Sort by difficulty (ascending - hardest first)
    sorted_instances = sorted(instance_data, key=lambda x: x["difficulty_score"])

    # Show summary stats
    easiest = max(instance_data, key=lambda x: x["difficulty_score"])
    hardest = min(instance_data, key=lambda x: x["difficulty_score"])

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Instances", len(instance_data))
    with col2:
        st.success(
            f"🏆 Easiest: `{easiest['instance_id']}` "
            f"({easiest['difficulty_score']:.2f})"
        )
    with col3:
        st.error(
            f"💀 Hardest: `{hardest['instance_id']}` "
            f"({hardest['difficulty_score']:.2f})"
        )

    # Difficulty chart
    st.subheader("Instance Difficulty Ranking")
    difficulty_chart = create_instance_difficulty_chart(instance_data)
    if difficulty_chart:
        st.plotly_chart(
            difficulty_chart, width="stretch", key="instance_difficulty_chart"
        )

    # Detailed leaderboard table
    st.subheader("Detailed Instance Metrics")

    leaderboard_data = []
    for instance in sorted_instances:
        leaderboard_data.append(
            {
                "Instance ID": instance["instance_id"],
                "Difficulty Score": ".2f",
                "Attempts": int(instance["total_attempts"]),
                "Patch Success": ".1f",
                "Install Success": ".1f",
                "Target Success": ".1f",
                "Build Success": ".1f",
            }
        )

    if leaderboard_data:
        df = pd.DataFrame(leaderboard_data)
        st.dataframe(df, width="stretch", hide_index=True)


def get_status_emoji(metrics: MetricsReport) -> str:
    """Return a status emoji based on metrics."""
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
        escaped_command = (
            step.command.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
        )
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


def main() -> None:
    # Sidebar
    with st.sidebar:
        st.header("🔄 Data Controls")
        if st.button(
            "🔄 Reload Data", help="Clear cache and reload all data", width="stretch"
        ):
            st.cache_data.clear()
            st.rerun()

    st.title("🎯 Instance Analysis")

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

    st.markdown("""
    **Instance Analysis** shows how different tasks perform across all agent
    configurations. This helps identify which instances (Angular upgrade tasks)
    are easiest vs hardest for agents to solve.

    - **Difficulty Score**: Weighted average of success rates across key metrics
    - **Higher scores** = easier instances (more agents succeed)
    - **Lower scores** = harder instances (fewer agents succeed)
    """)

    with st.spinner("Loading evaluations and experiments..."):
        completed_evals, harness_failures_list = load_all_evaluations(evaluations_dir)
        experiments = load_all_experiments(experiments_dir)

    # Build lookups from experiments
    agent_by_submission_id: dict[SubmissionID, AgentConfig] = {}
    for er in experiments:
        if er.is_completed:
            er_completed = cast(CompletedExperiment, er.root)
            agent_by_submission_id[er_completed.submission.submission_id] = (
                er_completed.task.agent_config
            )

    # Remove evaluations that don't correspond to any known experiment
    experiment_submission_ids = set(agent_by_submission_id.keys())
    completed_evals = [
        eval
        for eval in completed_evals
        if eval.result.submission_id in experiment_submission_ids
    ]

    # Group by instance instead of agent
    grouped_by_instance = group_completed_by_instance(
        completed_evals, agent_by_submission_id
    )

    # Show summary
    total_instances = len(grouped_by_instance)
    total_evaluations = sum(len(evals) for evals in grouped_by_instance.values())

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Instances", total_instances)
    with col2:
        st.metric("Total Evaluations", total_evaluations)
    with col3:
        st.metric("Avg Attempts per Instance", ".1f" if total_instances > 0 else "0")

    if not completed_evals:
        st.info(
            "No evaluation results found."
            " Please run some evaluations to see instance analysis."
        )
        return

    # Instance difficulty leaderboard
    display_instance_leaderboard(grouped_by_instance)

    # Instance performance radar chart
    if grouped_by_instance:
        st.subheader("🕸️ Instance Performance Comparison")
        radar_chart = create_instance_radar_chart(grouped_by_instance)
        if radar_chart:
            st.plotly_chart(radar_chart, width="stretch", key="instance_radar_chart")
        else:
            st.info("No metrics data available for analysis.")

    # Detailed results by instance
    st.header("🔬 Detailed Results by Instance")

    # Sort instances by difficulty (easiest first)
    instance_scores = []
    for instance_id, evaluations in grouped_by_instance.items():
        metrics_summary = compute_instance_metrics(evaluations)
        difficulty_score = compute_instance_difficulty_score(metrics_summary)
        instance_scores.append((instance_id, difficulty_score))

    sorted_instances = sorted(instance_scores, key=lambda x: x[1], reverse=True)

    for instance_id, _ in sorted_instances:
        evaluations = grouped_by_instance[instance_id]
        metrics_summary = compute_instance_metrics(evaluations)

        st.subheader(f"🎯 `{instance_id}`")

        # Show difficulty score and attempts
        difficulty_score = compute_instance_difficulty_score(metrics_summary)
        col1, col2 = st.columns([1, 4])
        with col1:
            st.metric("Difficulty Score", ".2f")
        with col2:
            st.write(f"**{len(evaluations)} agent attempt(s)**")

        # Metrics overview
        if metrics_summary:
            cols = st.columns(4)
            metric_labels = [
                ("Patch Application", "patch_application_success"),
                ("Install Success", "install_success"),
                ("Target Version Achieved", "target_version_achieved"),
                ("Build Success", "build_success"),
            ]

            for i, (label, field) in enumerate(metric_labels):
                if field in metrics_summary:
                    info = metrics_summary[field]
                    success_rate = float(info["success_rate"]) * 100
                    cols[i].metric(
                        label,
                        f"{success_rate:.1f}%",
                        f"{int(info['success_count'])}/{int(info['total_count'])}",
                    )
                else:
                    cols[i].metric(label, "N/A")

        # Show results by agent
        st.write("**Agent Performance:**")
        for agent_config, evaluation in sorted(
            evaluations.items(), key=lambda x: x[0].key
        ):
            metrics = evaluation.result.metrics
            status_emoji = get_status_emoji(metrics)

            with st.expander(f"{status_emoji} `{agent_config.key}`", expanded=False):
                st.markdown(
                    f"**Evaluation ID:** `{evaluation.id}`  |  "
                    f"**Submission ID:** `{evaluation.result.submission_id}`"
                )
                st.divider()

                agent_cols = st.columns(4)
                agent_metric_fields = [
                    ("Patch Application", metrics.patch_application_success),
                    ("Install Success", metrics.install_success),
                    ("Target Version Achieved", metrics.target_version_achieved),
                    ("Build Success", metrics.build_success),
                ]

                for j, (label, value) in enumerate(agent_metric_fields):
                    if value is None:
                        agent_cols[j].metric(label, "N/A")
                    else:
                        agent_cols[j].metric(label, "✅" if value else "❌")

                st.subheader("Execution Steps")
                display_execution_steps(evaluation.result.execution.steps)


if __name__ == "__main__":
    main()
