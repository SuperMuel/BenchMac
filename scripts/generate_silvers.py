"""
Generates Known-Good Patches and a Packaged Submissions File for Harness Testing.

## The Problem

To test the BenchMAC evaluation harness, we need a set of "correct" patches
that are known to successfully migrate a benchmark instance. Without these, we
can't verify that our harness metrics (like BuildSuccess or TestPassRate) are
working correctly.

## The Solution

This script provides a way to generate these known-good patches, which we call
"Silver Patches". It leverages manually migrated projects where each major
version upgrade is a separate commit.

For a given benchmark instance (e.g., "migrate from v14 to v15"), this script:
1.  Identifies the `base_commit` for the starting version (v14).
2.  Looks up the corresponding `solution_commit` for the target version (v15)
    from a hardcoded map.
3.  Clones the repository and computes the `git diff` between these two commits.
4.  Saves this diff as a `.patch` file in `data/silver_patches/`.
5.  Collects all generated patches and packages them into a single
    `data/silver_submissions.jsonl` file for easy use with the main CLI.

The resulting files serve as perfect test inputs for the harness,
simulating the output of a successful AI agent and allowing for robust,
end-to-end testing of the evaluation pipeline.
"""

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import cyclopts

from bench_mac.config import settings
from bench_mac.models import BenchmarkInstance

# --- Configuration ---

# This map defines the "solution" for each instance.
# The key is the instance_id, and the value is the commit hash that represents
# the state of the repository *after* the migration for that instance was
# successfully completed.
SILVER_SOLUTIONS = {
    "angular2-hn_v9_to_v10": "60ed37f",
    "angular2-hn_v10_to_v11": "16c8a35",
    "angular2-hn_v11_to_v12": "c73a6a7",
    "angular2-hn_v12_to_v13": "cb10fa5",
    "angular2-hn_v13_to_v14": "6dc7037",
    "angular2-hn_v14_to_v15": "52b2ec9",
    "angular2-hn_v15_to_v16": "83db8bd",
    "angular2-hn_v16_to_v17": "c6d3b8c",
    "angular2-hn_v17_to_v18": "e5a358f",
    "angular2-hn_v18_to_v19": "30529e8",
}


def run_command(command: list[str], cwd: Path | None = None) -> str:
    """Run a shell command and return its stdout, raising an error on failure."""
    print(f"  > Running: {' '.join(command)}")
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            cwd=cwd,
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Error executing command: {' '.join(command)}")
        print(f"  Exit Code: {e.returncode}")
        print(f"  Stderr: {e.stderr.strip()}")
        raise


def load_instances(file_path: Path) -> dict[str, BenchmarkInstance]:
    """Load instances from JSONL file into a dictionary keyed by instance_id."""
    instances_map: dict[str, BenchmarkInstance] = {}
    print(f"  > Loading instances from {file_path}...")
    try:
        with file_path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                # Parse JSON and create BenchmarkInstance
                instance_data = json.loads(line)
                try:
                    instance = BenchmarkInstance.model_validate(instance_data)
                    instances_map[instance.instance_id] = instance
                except Exception as e:
                    instance_id = instance_data.get("instance_id", "unknown")
                    print(
                        f"  ⚠️ Warning: Failed to validate instance {instance_id} (line {line_number}): {e}"  # noqa: E501
                    )
                    continue

    except FileNotFoundError:
        print(f"  ❌ Error: Instances file not found at {file_path}")
        raise
    except json.JSONDecodeError as e:
        print(f"  ❌ Error parsing JSON in {file_path}: {e}")
        raise
    print(f"  ✅ Loaded {len(instances_map)} instances.")
    return instances_map


def generate_patch_for_instance(
    instance_id: str,
    instance: BenchmarkInstance,
    solution_commit: str,
    output_dir: Path,
    temp_repos_dir: Path,
) -> str:
    """Generate and save a single silver patch."""
    print(f"\n--- Generating patch for: {instance_id} ---")

    repo_url = instance.repo
    base_commit = instance.base_commit

    # Ensure the repo URL is in a format git can clone
    if not repo_url.startswith(("http", "git@")):
        repo_url = f"https://github.com/{repo_url}.git"

    # Create a unique directory for this instance
    instance_temp_dir = temp_repos_dir / instance_id

    # Clean up existing directory if it exists
    if instance_temp_dir.exists():
        print(f"  > Cleaning up existing directory: {instance_temp_dir}")
        shutil.rmtree(instance_temp_dir)

    instance_temp_dir.mkdir(parents=True, exist_ok=True)

    repo_path = instance_temp_dir / "repo"
    base_worktree_path = instance_temp_dir / f"base-{base_commit}"
    solution_worktree_path = instance_temp_dir / f"solution-{solution_commit}"

    # 1. Clone the repository
    print(f"  > Cloning {repo_url}...")
    run_command(["git", "clone", "--bare", repo_url, str(repo_path)])

    # 2. Create worktrees for base and solution commits for efficiency
    print(f"  > Setting up worktrees for {base_commit} and {solution_commit}...")
    run_command(
        ["git", "worktree", "add", str(base_worktree_path), base_commit],
        cwd=repo_path,
    )
    run_command(
        ["git", "worktree", "add", str(solution_worktree_path), solution_commit],
        cwd=repo_path,
    )

    # 3. Generate the diff
    print("  > Generating diff...")
    patch_content = run_command(
        [
            "git",
            "diff",
            "--no-prefix",
            base_commit,
            solution_commit,
        ],
        cwd=repo_path,
    )

    # 4. Save the patch
    output_file = output_dir / f"{instance_id}.patch"
    print(f"  > Saving patch to {output_file}")
    output_file.write_text(patch_content, encoding="utf-8")

    print(f"  > Repository files available for inspection at: {instance_temp_dir}")

    print(f"✅ Successfully generated patch for {instance_id}")
    return patch_content


def main(instance_id: str | None = None) -> None:
    """
    Generates Silver Patches and a packaged silver_submissions.jsonl file.

    A Silver Patch is a known-correct solution for a migration task,
    generated by diffing the `base_commit` of an instance against the
    commit hash of its successfully migrated state.

    Parameters
    ----------
    instance_id
        The specific instance_id to generate a patch for.
        If not provided, patches will be generated for ALL instances
        defined in the SILVER_SOLUTIONS map.
    """
    # Check if git is installed
    if not shutil.which("git"):
        print("❌ Error: 'git' command not found.")
        print("   Please install Git and ensure it is in your system's PATH.")
        return

    instances_file = settings.instances_file
    output_dir = settings.silver_patches_dir
    temp_repos_dir = settings.silver_patches_repos_dir

    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)

    # Load all instance definitions
    try:
        instances_map = load_instances(instances_file)
    except (FileNotFoundError, json.JSONDecodeError):
        return  # Error message is printed in the helper

    # Determine which instances to process
    if instance_id:
        if instance_id not in SILVER_SOLUTIONS:
            print(f"❌ Error: No silver solution defined for '{instance_id}'.")
            print("   Please add it to the SILVER_SOLUTIONS map in this script.")
            return
        target_ids = [instance_id]
    else:
        print("No specific instance_id provided. Generating all available patches...")
        target_ids = list(SILVER_SOLUTIONS.keys())

    all_submissions: list[dict[str, Any]] = []

    for i, target_id in enumerate(target_ids, 1):
        print(f"\n{'=' * 20} Processing {i}/{len(target_ids)} {'=' * 20}")
        if target_id not in instances_map:
            print(
                f"❌ Error: Instance '{target_id}' found in SILVER_SOLUTIONS map "
                "but not in instances.jsonl."
            )
            continue

        instance = instances_map[target_id]
        solution_commit = SILVER_SOLUTIONS[target_id]

        try:
            generate_patch_for_instance(
                target_id,
                instance,
                solution_commit,
                output_dir,
                temp_repos_dir,
            )
        except Exception as e:
            print(
                f"\n❌ An unexpected error occurred while processing {target_id}: {e}"
            )
            continue

    if all_submissions:
        submissions_output_file = settings.data_dir / "silver_submissions.jsonl"
        print("\n--- Writing packaged submissions file ---")
        print(
            f"  > Saving {len(all_submissions)} submissions to {submissions_output_file}"  # noqa: E501
        )
        with submissions_output_file.open("w", encoding="utf-8") as f:
            for submission_data in all_submissions:
                f.write(json.dumps(submission_data) + "\n")
        print(f"✅ Successfully created {submissions_output_file.name}")

    print(f"\n{'=' * 50}")
    print("🎉 Script finished.")


if __name__ == "__main__":
    settings.initialize_directories()
    cyclopts.run(main)
