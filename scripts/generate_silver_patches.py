"""
Generates Known-Good Patches for Harness Testing.

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
4.  Saves this diff as a `.patch` file.

The resulting patch file serves as a perfect test input for the harness,
simulating the output of a successful AI agent and allowing for robust,
end-to-end testing of the evaluation pipeline.
"""

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import cyclopts

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


def load_instances(file_path: Path) -> dict[str, dict[str, Any]]:
    """Load instances from JSONL file into a dictionary keyed by instance_id."""
    instances_map: dict[str, dict[str, Any]] = {}
    print(f"  > Loading instances from {file_path}...")
    try:
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                instance = json.loads(line)
                instance_id = instance.get("instance_id")
                if instance_id:
                    instances_map[instance_id] = instance
                else:
                    print("  ⚠️ Warning: Found instance without an 'instance_id'.")
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
    instance_data: dict[str, Any],
    solution_commit: str,
    output_dir: Path,
    temp_repos_dir: Path,
) -> None:
    """Generate and save a single silver patch."""
    print(f"\n--- Generating patch for: {instance_id} ---")

    repo_url = instance_data["repo"]
    base_commit = instance_data["base_commit"]

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


def main(instance_id: str | None = None, cleanup: bool = False) -> None:
    """
    Generates Silver Patches for specified benchmark instances.

    A Silver Patch is a known-correct solution for a migration task,
    generated by diffing the `base_commit` of an instance against the
    commit hash of its successfully migrated state.

    Parameters
    ----------
    instance_id
        The specific instance_id to generate a patch for.
        If not provided, patches will be generated for ALL instances
        defined in the SILVER_SOLUTIONS map.
    cleanup
        If True, removes the temp_repos directory after processing.
        If False (default), keeps the cloned repositories for inspection.
    """
    # Check if git is installed
    if not shutil.which("git"):
        print("❌ Error: 'git' command not found.")
        print("   Please install Git and ensure it is in your system's PATH.")
        return

    # Determine script's directory to locate other files
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    instances_file = project_root / "data" / "instances.jsonl"
    output_dir = project_root / "data" / "silver_patches"
    temp_repos_dir = project_root / ".benchmac_cache" / "silver_patches_repos"

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

    # Process each target instance
    for i, target_id in enumerate(target_ids, 1):
        print(f"\n{'=' * 20} Processing {i}/{len(target_ids)} {'=' * 20}")
        if target_id not in instances_map:
            print(
                f"❌ Error: Instance '{target_id}' found in SILVER_SOLUTIONS map "
                "but not in instances.jsonl."
            )
            continue

        instance_data = instances_map[target_id]
        solution_commit = SILVER_SOLUTIONS[target_id]

        try:
            generate_patch_for_instance(
                target_id,
                instance_data,
                solution_commit,
                output_dir,
                temp_repos_dir,
            )
        except Exception as e:
            print(
                f"\n❌ An unexpected error occurred while processing {target_id}: {e}"
            )
            print("   Skipping to the next instance.")
            continue

    print(f"\n{'=' * 50}")
    print("🎉 Script finished.")

    # Optional cleanup
    if cleanup:
        temp_repos_dir = project_root / "temp_repos"
        if temp_repos_dir.exists():
            print(f"\n🧹 Cleaning up temp repositories at {temp_repos_dir}...")
            shutil.rmtree(temp_repos_dir)
            print("✅ Cleanup completed.")
    else:
        temp_repos_dir = project_root / "temp_repos"
        if temp_repos_dir.exists():
            print(
                f"\n📁 Cloned repositories available for inspection at: {temp_repos_dir}"  # noqa: E501
            )
            print(
                "   Use --cleanup flag to automatically remove them after processing."
            )


if __name__ == "__main__":
    cyclopts.run(main)
