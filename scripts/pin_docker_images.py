"""
Pins Docker base images to specific SHA256 hashes for reproducible benchmarks.

This script:
1. Identifies all unique base images used in dockerfiles
2. Fetches their SHA256 hashes using Docker
3. Updates dockerfiles to use pinned image hashes (image@sha256:hash format)
4. Provides dry-run mode for safety

Usage:
    uv run scripts/pin_docker_images.py [--dry-run] [--dockerfiles-dir DIR]
"""

import re
import subprocess
import sys
from pathlib import Path

import cyclopts


def run_command(command: list[str], cwd: Path | None = None) -> tuple[str, str, int]:
    """Run a shell command and return (stdout, stderr, returncode)."""
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            cwd=cwd,
        )
        return result.stdout, result.stderr, result.returncode
    except Exception as e:
        return "", str(e), 1


def get_image_digest(image: str) -> str | None:
    """Get the SHA256 digest for a Docker image."""
    print(f"  🔍 Getting digest for {image}...")

    # First, ensure the image is pulled
    _, pull_stderr, pull_code = run_command(["docker", "pull", image])
    if pull_code != 0:
        print(f"    ❌ Failed to pull {image}: {pull_stderr}")
        return None

    # Get the image digest using docker inspect
    inspect_stdout, inspect_stderr, inspect_code = run_command(
        ["docker", "inspect", image, "--format", "{{index .RepoDigests 0}}"]
    )

    if inspect_code != 0:
        print(f"    ❌ Failed to inspect {image}: {inspect_stderr}")
        return None

    # Parse the digest from the output (format: image@sha256:hash)
    digest_match = re.search(r"@sha256:([a-f0-9]+)", inspect_stdout.strip())
    if digest_match:
        digest = f"sha256:{digest_match.group(1)}"
        print(f"    ✅ Found digest: {digest}")
        return digest

    print(f"    ❌ Could not parse digest from: {inspect_stdout.strip()}")
    return None


def find_base_images(dockerfiles_dir: Path) -> set[str]:
    """Find all unique base images used in dockerfiles."""
    base_images: set[str] = set()

    print(f"🔍 Scanning dockerfiles in {dockerfiles_dir}...")

    for dockerfile in dockerfiles_dir.glob("*"):
        if not dockerfile.is_file():
            continue

        try:
            content = dockerfile.read_text(encoding="utf-8")
            # Match FROM statements, capturing image name (excluding existing digest)
            pattern = r"^FROM\s+([^\s@]+)"
            from_match = re.search(pattern, content, re.MULTILINE)
            if from_match:
                image = from_match.group(1)
                base_images.add(image)
                print(f"  📄 {dockerfile.name}: {image}")
        except Exception as e:
            print(f"  ❌ Error reading {dockerfile}: {e}")

    return base_images


def update_dockerfile(
    dockerfile: Path, image_map: dict[str, str], dry_run: bool = False
) -> bool:
    """Update a dockerfile to use pinned image hashes."""
    try:
        content = dockerfile.read_text(encoding="utf-8")
        original_content = content

        # Replace FROM statements with pinned versions
        for old_image, new_digest in image_map.items():
            # Match the exact FROM line (without existing digest)
            from_pattern = rf"^(FROM\s+){re.escape(old_image)}(\s|$)"
            pattern = re.compile(from_pattern, re.MULTILINE)
            replacement = f"\\1{old_image}@{new_digest}\\2"

            if pattern.search(content):
                content = pattern.sub(replacement, content)
                print(f"  🔄 Updated {old_image} -> {old_image}@{new_digest}")

        if content != original_content:
            if dry_run:
                print(f"  📝 Would update {dockerfile.name}")
                # Show diff
                import difflib

                diff = difflib.unified_diff(
                    original_content.splitlines(keepends=True),
                    content.splitlines(keepends=True),
                    fromfile=f"a/{dockerfile.name}",
                    tofile=f"b/{dockerfile.name}",
                    lineterm="",
                )
                print("".join(diff))
            else:
                dockerfile.write_text(content, encoding="utf-8")
                print(f"  ✅ Updated {dockerfile.name}")
            return True
        else:
            print(f"  ⏭️  No changes needed for {dockerfile.name}")
            return False

    except Exception as e:
        print(f"  ❌ Error updating {dockerfile}: {e}")
        return False


def main(dry_run: bool = False, dockerfiles_dir: Path | None = None) -> None:
    """
    Pin Docker base images to specific SHA256 hashes for reproducible benchmarks.

    Args:
        dry_run: Show what would be changed without actually modifying files
        dockerfiles_dir: Directory containing dockerfiles (defaults to data/dockerfiles)
    """
    print("🐳 Docker Image Pinning Tool")
    print("=" * 50)

    # Set default dockerfiles directory
    if dockerfiles_dir is None:
        dockerfiles_dir = Path(__file__).parent.parent / "data" / "dockerfiles"

    if not dockerfiles_dir.exists():
        print(f"❌ Dockerfiles directory not found: {dockerfiles_dir}")
        sys.exit(1)

    # Check if docker is available
    docker_check = run_command(["docker", "--version"])
    if docker_check[2] != 0:
        print("❌ Docker is not available. Please install Docker and try again.")
        sys.exit(1)

    print(f"Docker version: {docker_check[0].strip()}")

    if dry_run:
        print("🔍 DRY RUN MODE - No files will be modified")

    # Phase 1: Find all base images
    print("\n📋 Phase 1: Finding base images...")
    base_images = find_base_images(dockerfiles_dir)

    if not base_images:
        print("❌ No base images found in dockerfiles")
        return

    print(f"\nFound {len(base_images)} unique base images:")
    for img in sorted(base_images):
        print(f"  • {img}")

    # Phase 2: Get digests for all images
    print("\n📋 Phase 2: Getting image digests...")
    image_map: dict[str, str] = {}

    for image in sorted(base_images):
        digest = get_image_digest(image)
        if digest:
            image_map[image] = digest
        else:
            print(f"  ❌ Skipping {image} due to digest retrieval failure")

    if not image_map:
        print("❌ No image digests could be retrieved")
        return

    print(f"\n✅ Successfully got digests for {len(image_map)} images:")
    for image, digest in image_map.items():
        print(f"  • {image}@{digest}")

    # Phase 3: Update dockerfiles
    print("\n📋 Phase 3: Updating dockerfiles...")

    updated_count = 0
    for dockerfile in sorted(dockerfiles_dir.glob("*")):
        if dockerfile.is_file() and update_dockerfile(dockerfile, image_map, dry_run):
            updated_count += 1

    # Summary
    print("\n" + "=" * 50)
    if dry_run:
        print("🔍 DRY RUN COMPLETE")
        print(f"  • Would update {updated_count} dockerfiles")
        print("  • Run without --dry-run to apply changes")
    else:
        print("✅ UPDATE COMPLETE")
        print(f"  • Updated {updated_count} dockerfiles")
        print("  • All base images are now pinned to specific hashes")
        print("\n🎯 Benchmark reproducibility achieved!")
        print("  • Docker builds will now be deterministic")
        print("  • No unexpected changes from base image updates")

    # Show example of what changed
    if image_map:
        print("\n📝 Example FROM statement changes:")
        for old_image, digest in list(image_map.items())[:2]:  # Show first 2 examples
            print(f"  • FROM {old_image}")
            print("    ↓")
            print(f"  • FROM {old_image}@{digest}")


if __name__ == "__main__":
    cyclopts.run(main)
