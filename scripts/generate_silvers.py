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

import shutil
import subprocess
from pathlib import Path

import cyclopts

from bench_mac.core.config import settings
from bench_mac.core.models import BenchmarkInstance, Submission
from bench_mac.core.utils import load_instances

# --- Configuration ---

# This map defines the "solution" for each instance.
# The key is the instance_id, and the value is the commit hash that represents
# the state of the repository *after* the migration for that instance was
# successfully completed.
SILVER_SOLUTIONS = {
    "gothinkster__angular-realworld-example-app_v11_to_v12": "57eeb6a",
    "gothinkster__angular-realworld-example-app_v12_to_v13": "db8c6b0",
    "gothinkster__angular-realworld-example-app_v13_to_v14": "e2f6f4c",
    "gothinkster__angular-realworld-example-app_v14_to_v15": "e28c896",
    "gothinkster__angular-realworld-example-app_v15_to_v16": "bd914dc",
    "gothinkster__angular-realworld-example-app_v16_to_v17": "f218b2f",
    "gothinkster__angular-realworld-example-app_v17_to_v18": "2555e2f",
    "gothinkster__angular-realworld-example-app_v18_to_v19": "a6f16d0",
}  # TODO: export this in config file


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


def generate_patch_for_instance(
    instance: BenchmarkInstance,
    solution_commit: str,
    output_dir: Path,
    temp_repos_dir: Path,
) -> str:
    """Generate and save a single silver patch, returning its content."""
    instance_id = instance.instance_id
    print(f"\n--- Generating patch for: {instance_id} ---")

    repo_url = instance.repo
    base_commit = instance.base_commit

    # Ensure the repo URL is in a format git can clone
    if not repo_url.startswith(("http", "git@")):
        repo_url = f"https://github.com/{repo_url}.git"

    # Create a unique directory for this instance
    instance_temp_dir = temp_repos_dir / instance_id
    if instance_temp_dir.exists():
        shutil.rmtree(instance_temp_dir)
    instance_temp_dir.mkdir(parents=True)

    repo_path = instance_temp_dir / "repo"

    # 1. Clone the repository
    print(f"  > Cloning {repo_url}...")
    run_command(["git", "clone", "--bare", repo_url, str(repo_path)])

    # 2. Generate the diff
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

    # 3. Save the patch
    output_file = output_dir / f"{instance_id}.patch"
    print(f"  > Saving patch to {output_file}")
    output_file.write_text(patch_content, encoding="utf-8")

    print(f"✅ Successfully generated patch for {instance_id}")
    return patch_content


def main(instance_id: str | None = None) -> None:
    """
    Generates Silver Patches and a packaged silver_submissions.jsonl file.
    """
    # --- 1. SETUP ---
    if not shutil.which("git"):
        print("❌ Error: 'git' command not found. Please install Git.")
        return

    instances_file = settings.instances_file
    output_dir = settings.silver_patches_dir
    temp_repos_dir = settings.silver_patches_repos_dir
    output_dir.mkdir(exist_ok=True)

    print(f"  > Loading instances from {instances_file}...")
    instances_map = load_instances(instances_file, strict=True)

    # --- 2. DETERMINE TASKS ---
    if instance_id:
        if instance_id not in SILVER_SOLUTIONS:
            print(f"❌ Error: No silver solution defined for '{instance_id}'.")
            return
        target_ids = [instance_id]
    else:
        print("No specific instance_id provided. Generating all available patches...")
        target_ids = list(SILVER_SOLUTIONS.keys())

    # --- 3. GENERATE PATCHES and COLLECT SUBMISSIONS ---
    all_submissions: list[Submission] = []
    for i, target_id in enumerate(target_ids, 1):
        print(f"\n{'=' * 20} Processing {i}/{len(target_ids)} {'=' * 20}")
        if target_id not in instances_map:
            print(f"❌ Skipping: Instance '{target_id}' not in instances.jsonl.")
            continue

        instance = instances_map[target_id]
        solution_commit = SILVER_SOLUTIONS[target_id]

        try:
            # Generate the .patch file on disk AND get its content
            patch_content = generate_patch_for_instance(
                instance,
                solution_commit,
                output_dir,
                temp_repos_dir,
            )
            # Create a Submission object and add it to our list
            submission = Submission(instance_id=target_id, model_patch=patch_content)
            all_submissions.append(submission)
        except Exception as e:
            print(f"❌ An unexpected error occurred while processing {target_id}: {e}")
            continue

    # --- 4. WRITE THE FINAL SUBMISSIONS FILE ---
    if not all_submissions:
        print("\nNo submissions were generated. Exiting.")
        return

    submissions_output_file = settings.data_dir / "silver_submissions.jsonl"
    print(f"\n--- Writing packaged submissions file to {submissions_output_file} ---")

    with submissions_output_file.open("w", encoding="utf-8") as f:
        for sub in all_submissions:
            f.write(sub.model_dump_json() + "\n")

    print(
        f"✅ Successfully created {submissions_output_file.name} with "
        f"{len(all_submissions)} entries."
    )
    print(f"\n{'=' * 50}")
    print("🎉 Script finished.")


if __name__ == "__main__":
    settings.initialize_directories()
    cyclopts.run(main)
