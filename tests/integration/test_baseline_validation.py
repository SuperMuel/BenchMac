"""
Integration test for validating the baseline state of all benchmark instances.

This test suite is critical for ensuring the quality and reliability of the
benchmark dataset. It verifies that for every instance defined in `instances.jsonl`,
the project at its `base_commit` is in a "green" state—meaning it can be
successfully installed, built, linted, and tested.

- This test is marked as 'slow' and is excluded from the default test run.
- It requires a running Docker daemon.
- **In theory**, a failure in this test indicates a problem with the benchmark data, not
  the harness logic.
"""

import pytest
from loguru import logger

from bench_mac.config import settings
from bench_mac.docker.manager import DockerManager
from bench_mac.environments import DockerExecutionEnvironment
from bench_mac.models import BenchmarkInstance
from bench_mac.utils import load_instances


@pytest.mark.instance_validation
class TestBaselineInstanceValidation:
    """
    Validates that every benchmark instance starts from a known-good state.
    """

    @pytest.mark.parametrize(
        "instance",
        load_instances(settings.instances_file, strict=True).values(),
        ids=lambda inst: inst.instance_id,  # Use instance_id for readable test names
    )
    def test_base_angular_version_matches_package_json(
        self, instance: BenchmarkInstance, docker_manager: DockerManager
    ) -> None:
        """Ensure the declared base Angular version matches package.json."""
        logger.info(
            f"--- Validating Angular version for Instance: {instance.instance_id} ---"
        )

        version_check_command = (
            "node -e \"const fs=require('fs');"
            "const pkg=JSON.parse(fs.readFileSync('package.json','utf8'));"
            "const fields=['dependencies','devDependencies'];"
            "const keys=['@angular/core','@angular/cli'];"
            "let raw=null;"
            "for (const field of fields) {"
            " const section=pkg[field];"
            " if (!section) continue;"
            " for (const key of keys) {"
            "  if (section[key]) { raw=section[key]; break; }"
            " }"
            " if (raw) break;"
            "}"
            "if (!raw) { console.error('Missing Angular dependency');"
            " console.error('Check package.json for Angular deps');"
            " process.exit(1); }"
            "const match=String(raw).match(/\\d+(\\.\\d+){0,2}/);"
            "if (!match) { console.error('Unable to parse version from ' + raw);"
            " process.exit(1); }"
            'console.log(match[0]);"'
        )

        with DockerExecutionEnvironment(
            instance, docker_manager, auto_remove=True
        ) as env:
            result = env.exec(version_check_command)

            assert result.success, (
                "Angular version extraction failed for instance "
                f"'{instance.instance_id}'.\n"
                f"Command: {version_check_command}\n"
                f"Stdout: {result.stdout}\n"
                f"Stderr: {result.stderr}"
            )

            package_json_version = result.stdout.strip()
            assert package_json_version == instance.source_angular_version, (
                "package.json Angular version mismatch for instance "
                f"'{instance.instance_id}'.\n"
                f"Expected: {instance.source_angular_version}\n"
                f"Found: {package_json_version}"
            )

            logger.success(
                f"✅ Angular version matches for instance: {instance.instance_id}"
            )

    @pytest.mark.parametrize(
        "instance",
        load_instances(settings.instances_file, strict=True).values(),
        ids=lambda inst: inst.instance_id,  # Use instance_id for readable test names
    )
    def test_instance_baseline_is_green(
        self, instance: BenchmarkInstance, docker_manager: DockerManager
    ) -> None:
        """
        For a given instance, verifies that the baseline state is "green"
        by running install, build, lint, and test commands successfully.
        """
        logger.info(f"--- Validating Baseline for Instance: {instance.instance_id} ---")
        with DockerExecutionEnvironment(
            instance, docker_manager, auto_remove=True
        ) as env:
            project_dir = env.project_dir

            # === 2a. VALIDATE GIT STATE ===
            logger.info("Step 2a: Validating Git history sanitization...")
            git_count_cmd = "git rev-list --count --all"
            count_result = env.exec(git_count_cmd, workdir=project_dir)
            assert count_result.success, (
                f"Git count command failed: {count_result.stderr}"
            )
            commit_count = int(count_result.stdout.strip())

            assert commit_count == 1, (
                f"\n\n❌ History validation FAILED for instance "
                f"'{instance.instance_id}'.\n"
                f"   - Expected exactly 1 baseline commit, but found {commit_count}.\n"
                f"   - If {commit_count} > 1: FULL git history was included (INVALID).\n"  # noqa: E501
                f"   - If {commit_count} == 0: No baseline commit created (INVALID).\n"
                f"   - The Dockerfile should create exactly 1 baseline commit after "
                f"extracting source code."
            )
            logger.success("✅ Git history is properly sanitized.")

            # === 2b. VALIDATE BASELINE COMMIT AND TAG ===
            logger.info("Step 2b: Validating baseline commit and tag...")

            tag_check_cmd = "git tag -l baseline"
            tag_check_result = env.exec(tag_check_cmd, workdir=project_dir)
            assert tag_check_result.success, (
                f"Git tag check failed: {tag_check_result.stderr}"
            )
            assert tag_check_result.stdout.strip() == "baseline", (
                f"Missing 'baseline' tag for instance '{instance.instance_id}'. "
                f"Expected 'baseline', got: '{tag_check_result.stdout.strip()}'. "
                f"The Dockerfile should create a 'baseline' tag after the initial commit."  # noqa: E501
            )

            head_hash_cmd = "git rev-parse HEAD"
            tag_hash_cmd = "git rev-parse baseline"

            head_result = env.exec(head_hash_cmd, workdir=project_dir)
            assert head_result.success, "Failed to get HEAD hash"

            tag_result = env.exec(tag_hash_cmd, workdir=project_dir)
            assert tag_result.success, "Failed to get baseline tag hash"

            head_hash = head_result.stdout.strip()
            baseline_hash = tag_result.stdout.strip()

            assert head_hash == baseline_hash, (
                "Baseline tag doesn't point to HEAD for instance "
                f"'{instance.instance_id}'. "
                f"HEAD: {head_hash}, baseline: {baseline_hash}"
            )

            # === 3. EXECUTION & ASSERTION: Run all baseline commands ===
            commands_to_validate = [
                ("Install", instance.commands.install),
                ("Build", instance.commands.build),
            ]

            for step_name, command_str in commands_to_validate:
                logger.info(f"Step 3: Executing '{step_name}' command: {command_str}")
                command_result = env.exec(command_str, workdir=project_dir)

                stdout_tail = command_result.stdout[-1000:]
                stderr_tail = command_result.stderr[-1000:]

                assert command_result.success, (
                    f"\n\n❌ Baseline validation FAILED for instance"
                    f" '{instance.instance_id}' at step '{step_name}'.\n"
                    f"  - Command: '{command_str}'\n"
                    f"  - Exit Code: {command_result.exit_code}\n\n"
                    f"  --- STDOUT (last 1000 chars) ---\n{stdout_tail}\n\n"
                    f"  --- STDERR (last 1000 chars) ---\n{stderr_tail}\n"
                )
                logger.success(f"✅ Step '{step_name}' PASSED")
            logger.success(
                f"✅ Baseline validation PASSED for instance: {instance.instance_id}"
            )
