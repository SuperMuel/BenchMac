"""
Script to validate instances.jsonl file.

This script checks that:
1. All fields conform to the BenchmarkInstance model schema
2. Every repository exists on GitHub
3. Every base_commit exists in the respective repository

Warning: Without GITHUB_TOKEN, the script will probably fail.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

import cyclopts
import requests
from dotenv import load_dotenv
from pydantic import ValidationError

from bench_mac.config import settings
from bench_mac.models import BenchmarkInstance

load_dotenv()


def load_instances(file_path: Path) -> list[dict[str, Any]]:
    """Load instances from JSONL file."""
    instances: list[dict[str, Any]] = []
    try:
        with file_path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    instance: dict[str, Any] = json.loads(line)
                    instances.append(instance)
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON on line {line_num}: {e}")
                    sys.exit(1)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        sys.exit(1)
    return instances


def get_github_headers() -> dict[str, str]:
    """Get headers for GitHub API requests, including auth if token is available."""
    headers = {"Accept": "application/vnd.github+json"}

    github_token = os.getenv("GITHUB_TOKEN")
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"

    return headers


def check_github_token() -> None:
    """Check for GitHub token and print warning if not found."""
    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        print("⚠️  Warning: No GITHUB_TOKEN found in environment variables.")
        print(
            "   GitHub API rate limits apply (60 requests/hour for unauthenticated requests)"  # noqa: E501
        )
        print(
            "   Set GITHUB_TOKEN environment variable for higher rate limits (5000 requests/hour)"  # noqa: E501
        )
        print()


def check_repo_exists(repo: str) -> bool:
    """Check if a GitHub repository exists."""
    url = f"https://api.github.com/repos/{repo}"
    headers = get_github_headers()

    try:
        response = requests.get(url, headers=headers, timeout=10)
        return response.status_code == 200
    except requests.exceptions.RequestException as e:
        print(f"Error checking repo {repo}: {e}")
        return False


def check_commit_exists(repo: str, commit: str) -> bool:
    """Check if a commit exists in a GitHub repository."""
    url = f"https://api.github.com/repos/{repo}/commits/{commit}"
    headers = get_github_headers()

    try:
        response = requests.get(url, headers=headers, timeout=10)
        return response.status_code == 200
    except requests.exceptions.RequestException as e:
        print(f"Error checking commit {commit} in repo {repo}: {e}")
        return False


def validate_instance(instance_data: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate a single instance and return (is_valid, errors)."""
    errors: list[str] = []

    # First, validate using Pydantic model
    try:
        instance = BenchmarkInstance(**instance_data)
    except ValidationError as e:
        for error in e.errors():
            field = ".".join(str(loc) for loc in error["loc"])
            message = error["msg"]
            errors.append(f"Field '{field}': {message}")
        return False, errors

    # If Pydantic validation passes, check GitHub existence
    repo = instance.repo
    base_commit = instance.base_commit

    # Check if repo exists
    print(f"Checking repo: {repo}")
    if not check_repo_exists(repo):
        errors.append(f"Repository {repo} does not exist")
        return False, errors

    # Check if commit exists
    print(f"Checking commit: {base_commit} in {repo}")
    if not check_commit_exists(repo, base_commit):
        errors.append(f"Commit {base_commit} does not exist in repository {repo}")

    return len(errors) == 0, errors


def main(instances_file: Path | None = None) -> None:
    """Validate instances.jsonl file.

    Parameters
    ----------
    instances_file
        Path to instances.jsonl file. If not provided, defaults to the default instances file path in ./config.py.
    """  # noqa: E501
    if instances_file is None:
        instances_file = settings.instances_file

    if not instances_file.exists():
        print(f"Error: {instances_file} not found")
        sys.exit(1)

    # Check GitHub token availability
    check_github_token()

    print(f"Loading instances from {instances_file}")
    instances = load_instances(instances_file)
    print(f"Found {len(instances)} instances")

    all_valid = True

    for i, instance in enumerate(instances, 1):
        instance_id = instance.get("instance_id", f"instance_{i}")
        print(f"\n--- Validating {instance_id} ({i}/{len(instances)}) ---")

        is_valid, errors = validate_instance(instance)

        if is_valid:
            print(f"✅ {instance_id} is valid")
        else:
            print(f"❌ {instance_id} has errors:")
            for error in errors:
                print(f"  - {error}")
            all_valid = False

    print(f"\n{'=' * 50}")
    if all_valid:
        print("✅ All instances are valid!")
        sys.exit(0)
    else:
        print("❌ Some instances have validation errors")
        sys.exit(1)


if __name__ == "__main__":
    settings.initialize_directories()
    cyclopts.run(main)
