"""
Generate a comprehensive markdown report from analysis results.

This script reads the analysis results and tasks to produce a single markdown report
containing summaries and detailed analysis for each experiment/evaluation pair.

Usage:
    uv run python analysis/report.py  # Generate report for all analyses
    uv run python analysis/report.py --trace-ids ID1 ID2  # Specific trace IDs
"""

import argparse
import json
from pathlib import Path
from typing import Any


def extract_ids_from_filename(filename: str) -> tuple[str, str, str]:
    """Extract trace_id, eval_id, exp_id from result filename.

    Expected format: trace_<trace_id>_eval_<eval_id>_exp_<exp_id>.<ext>
    """
    # Remove extension
    stem = Path(filename).stem

    # Split by underscores and find the IDs
    parts = stem.split("_")

    trace_id: str | None = None
    eval_id: str | None = None
    exp_id: str | None = None

    i = 0
    while i < len(parts):
        if parts[i] == "trace" and i + 1 < len(parts):
            trace_id = parts[i + 1]
            i += 2
        elif parts[i] == "eval" and i + 1 < len(parts):
            eval_id = parts[i + 1]
            i += 2
        elif parts[i] == "exp" and i + 1 < len(parts):
            exp_id = parts[i + 1]
            i += 2
        else:
            i += 1

    if not all([trace_id, eval_id, exp_id]):
        raise ValueError(f"Could not extract IDs from filename: {filename}")

    # At this point we know all are not None due to the check above
    assert trace_id is not None
    assert eval_id is not None
    assert exp_id is not None
    return trace_id, eval_id, exp_id


def load_task_metadata(task_file: Path) -> dict[str, Any]:
    """Load metadata from a task JSON file."""
    with task_file.open(encoding="utf-8") as f:
        data = json.load(f)

    # Extract key metadata
    metadata = {
        "trace_id": data.get("trace_id"),
        "model_name": data.get("model_name"),
        "experiment_id": data["experiment"]["id"],
        "instance_id": data["experiment"]["submission"]["instance_id"],
        "scaffold": data["experiment"]["task"]["agent_config"]["scaffold"],
        "agent_model_name": data["experiment"]["task"]["agent_config"].get(
            "model_name"
        ),
        "evaluation_id": data["evaluation"]["id"],
        "metrics": data["evaluation"]["result"]["metrics"],
    }

    return metadata


def load_analysis_content(md_file: Path) -> str:
    """Load the analysis report content from markdown file."""
    with md_file.open(encoding="utf-8") as f:
        return f.read().strip()


def generate_markdown_report(analysis_data: list[dict[str, Any]]) -> str:
    """Generate the complete markdown report."""

    # Summary section
    summary = f"""# Analysis Report

## Summary

Total analyses performed: {len(analysis_data)}

"""

    # Individual analyses
    analyses_content = []
    for analysis in analysis_data:
        metadata = analysis["metadata"]
        content = analysis["content"]

        analysis_section = f"""
<analysis id="{metadata["trace_id"]}">
<metadata>
- **Analysis ID**: {metadata["trace_id"]}
- **Instance ID**: {metadata["instance_id"]}
- **Experiment ID**: {metadata["experiment_id"]}
- **Evaluation ID**: {metadata["evaluation_id"]}
- **Agent Scaffold**: {metadata["scaffold"]}
- **Agent Model**: {metadata.get("agent_model_name", "N/A")}
- **Analysis Model**: {metadata["model_name"]}
- **Final Metrics**: {json.dumps(metadata["metrics"], indent=2)}
</metadata>

<report>
{content}
</report>
</analysis>

---

"""
        analyses_content.append(analysis_section)

    # Combine all sections
    full_report = summary + "\n".join(analyses_content)

    return full_report


def main() -> None:
    """Main function to generate the analysis report."""

    parser = argparse.ArgumentParser(
        description="Generate markdown report from analysis results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python analysis/report.py
  uv run python analysis/report.py --trace-ids 01997c1a-9dfb-7798-9145-94c2ab009ba3
  uv run python analysis/report.py --trace-ids TRACE1 TRACE2 --output custom_report.md
        """,
    )
    parser.add_argument(
        "--trace-ids",
        nargs="*",
        help="Specific trace IDs to include. If not provided, includes all analyses.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="analysis_report.md",
        help="Output filename for the report (default: analysis_report.md)",
    )

    args = parser.parse_args()

    # Get the script's directory and look for results/tasks relative to it
    script_dir = Path(__file__).parent
    results_dir = script_dir / "results"
    tasks_dir = script_dir / "tasks"

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    if not tasks_dir.exists():
        raise FileNotFoundError(f"Tasks directory not found: {tasks_dir}")

    # Find all markdown result files
    md_files = list(results_dir.glob("*.md"))

    if not md_files:
        print("No analysis result files found.")
        return

    print(f"Found {len(md_files)} analysis result files")

    analysis_data = []

    for md_file in sorted(md_files):
        try:
            # Extract IDs from filename
            trace_id, eval_id, exp_id = extract_ids_from_filename(md_file.name)

            # If trace IDs are specified, skip if this one is not in the list
            if args.trace_ids and trace_id not in args.trace_ids:
                continue

            # Load corresponding task file
            task_file = tasks_dir / f"{trace_id}.json"
            if not task_file.exists():
                print(f"Warning: Task file not found for {trace_id}, skipping")
                continue

            # Load metadata and content
            metadata = load_task_metadata(task_file)
            content = load_analysis_content(md_file)

            analysis_data.append({"metadata": metadata, "content": content})

        except Exception as e:
            print(f"Error processing {md_file}: {e}")
            continue

    if not analysis_data:
        print("No analyses matched the specified criteria.")
        return

    # Generate and save report
    report_content = generate_markdown_report(analysis_data)

    report_file = Path(args.output)
    # If output path is relative, put it in the analysis/reports directory
    if not report_file.is_absolute():
        report_file = script_dir / "reports" / report_file

    with report_file.open("w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"Report generated: {report_file}")
    print(f"Total analyses included: {len(analysis_data)}")

    if args.trace_ids:
        print(f"Filtered to trace IDs: {', '.join(args.trace_ids)}")


if __name__ == "__main__":
    main()
