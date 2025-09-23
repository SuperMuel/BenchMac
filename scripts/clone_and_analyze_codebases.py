"""
Clone benchmark instances and analyze code metrics using tokei.

This script processes each instance from instances.jsonl and:
1. Creates a directory under .benchmac/codebases/{instance_id}
2. Clones the repository at the specified base_commit
3. Runs tokei to compute lines of code and saves output to tokei.txt
4. Generates a consolidated tokei-report.txt with all results

Features:
- Idempotent: skips instances that are already processed
- Consolidated reporting: combines all tokei outputs into a single report
- Error recovery: cleans up failed partial clones

Usage:
    uv run python scripts/clone_and_analyze_codebases.py [--instance-id INSTANCE_ID]
"""

import shutil
import subprocess
from pathlib import Path

import cyclopts

from bench_mac.core.config import settings
from bench_mac.core.models import BenchmarkInstance
from bench_mac.core.utils import load_instances


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


def process_instance(instance: BenchmarkInstance, codebases_dir: Path) -> bool:
    """Process a single instance: clone repo and run tokei analysis.

    Returns True if the instance was processed, False if it was already processed.
    """
    instance_id = instance.instance_id
    print(f"\n--- Processing: {instance_id} ---")

    repo_url = instance.repo
    base_commit = instance.base_commit

    # Ensure the repo URL is in a format git can clone
    if not repo_url.startswith(("http", "git@")):
        repo_url = f"https://github.com/{repo_url}.git"

    # Create instance directory
    instance_dir = codebases_dir / instance_id
    tokei_file = instance_dir / "tokei.txt"

    # Check if instance is already processed (idempotency check)
    if instance_dir.exists() and tokei_file.exists() and tokei_file.stat().st_size > 0:
        print("  > Instance already processed (tokei.txt exists), skipping...")
        print(f"✅ Skipped {instance_id} (already processed)")
        return False

    # Clean up any partial directories
    if instance_dir.exists():
        print("  > Removing partial directory...")
        shutil.rmtree(instance_dir)
    instance_dir.mkdir(parents=True)

    try:
        # 1. Clone the repository
        print(f"  > Cloning {repo_url} into {instance_dir}...")
        run_command(["git", "clone", repo_url, str(instance_dir)])

        # 2. Checkout the specific commit
        print(f"  > Checking out commit {base_commit}...")
        run_command(["git", "checkout", base_commit], cwd=instance_dir)

        # 3. Run tokei and save output
        print("  > Running tokei analysis...")
        tokei_output = run_command(["tokei"], cwd=instance_dir)

        # Save tokei output to file
        tokei_file.write_text(tokei_output, encoding="utf-8")
        print(f"  > Saved tokei output to {tokei_file}")

        print(f"✅ Successfully processed {instance_id}")
        return True

    except Exception as e:
        print(f"❌ Error processing {instance_id}: {e}")
        # Clean up failed directory
        if instance_dir.exists():
            shutil.rmtree(instance_dir)
        raise


def generate_report(codebases_dir: Path) -> None:
    """Generate a consolidated tokei report from all instances."""
    print("\n--- Generating consolidated tokei report ---")

    report_file = codebases_dir / "tokei-report.txt"
    report_content = []

    # Header for the report
    report_content.append("=" * 80)
    report_content.append("BENCHMAC CODEBASES TOKI ANALYSIS REPORT")
    report_content.append(f"Generated: {Path.cwd()}")
    report_content.append("=" * 80)
    report_content.append("")

    # Find all instance directories
    instance_dirs = [
        d for d in codebases_dir.iterdir() if d.is_dir() and (d / "tokei.txt").exists()
    ]

    if not instance_dirs:
        report_content.append("No processed instances found.")
        report_file.write_text("\n".join(report_content), encoding="utf-8")
        print(f"  > Report saved to {report_file}")
        return

    # Sort by instance name for consistent ordering
    instance_dirs.sort(key=lambda x: x.name)

    total_processed = 0
    for instance_dir in instance_dirs:
        instance_id = instance_dir.name
        tokei_file = instance_dir / "tokei.txt"

        try:
            tokei_content = tokei_file.read_text(encoding="utf-8")

            # Add instance header
            report_content.append(f"{'=' * 60}")
            report_content.append(f"INSTANCE: {instance_id}")
            report_content.append(f"{'=' * 60}")
            report_content.append("")

            # Add the tokei output
            report_content.append(tokei_content.strip())
            report_content.append("")
            report_content.append("")

            total_processed += 1

        except Exception as e:
            print(f"  ⚠️  Warning: Could not read tokei.txt for {instance_id}: {e}")
            continue

    # Footer with summary
    report_content.append("=" * 80)
    report_content.append(f"SUMMARY: {total_processed} instances processed")
    report_content.append("=" * 80)

    # Write the report
    report_file.write_text("\n".join(report_content), encoding="utf-8")
    print(f"  > Consolidated report saved to {report_file}")
    print(f"  > Report includes {total_processed} instances")


def main(instance_id: str | None = None) -> None:
    """
    Clone all benchmark instances and run tokei analysis.

    Args:
        instance_id: Optional specific instance ID to process. If not provided,
                    processes all instances.
    """
    # Check for required tools
    if not shutil.which("git"):
        print("❌ Error: 'git' command not found. Please install Git.")
        return

    if not shutil.which("tokei"):
        print("❌ Error: 'tokei' command not found.")
        print("  Install with: cargo install tokei")
        print("  Or use: brew install tokei")
        return

    # Setup directories
    codebases_dir = Path(".benchmac") / "codebases"
    codebases_dir.mkdir(parents=True, exist_ok=True)

    instances_file = settings.instances_file
    print(f"  > Loading instances from {instances_file}...")
    instances_map = load_instances(instances_file, strict=True)

    # Determine which instances to process
    if instance_id:
        if instance_id not in instances_map:
            print(f"❌ Error: Instance '{instance_id}' not found in instances.jsonl.")
            return
        target_ids = [instance_id]
        print(f"Processing single instance: {instance_id}")
    else:
        target_ids = list(instances_map.keys())
        print(f"Processing all {len(target_ids)} instances...")

    # Process each instance
    processed_instances = 0
    skipped_instances = 0
    for i, target_id in enumerate(target_ids, 1):
        print(f"\n{'=' * 20} Processing {i}/{len(target_ids)} {'=' * 20}")

        instance = instances_map[target_id]
        try:
            was_processed = process_instance(instance, codebases_dir)
            if was_processed:
                processed_instances += 1
            else:
                skipped_instances += 1
        except Exception as e:
            print(f"❌ Failed to process {target_id}: {e}")
            continue

    # Generate consolidated report
    generate_report(codebases_dir)

    print(f"\n{'=' * 50}")
    print("🎉 Script finished!")
    print(f"📊 Summary: {processed_instances} processed, {skipped_instances} skipped")
    print(f"📁 Codebases saved to: {codebases_dir}")
    print(f"📄 Report saved to: {codebases_dir / 'tokei-report.txt'}")


if __name__ == "__main__":
    settings.initialize_directories()
    cyclopts.run(main)
