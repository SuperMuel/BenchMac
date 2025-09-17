import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from bench_mac.environment import InstanceEnvironment
from bench_mac.metrics import calculate_target_version_achieved

if TYPE_CHECKING:  # pragma: no cover
    from loguru import Logger

from bench_mac.docker.manager import DockerManager
from bench_mac.models import (
    BenchmarkInstance,
    CommandResult,
    ExecutionTrace,
    Submission,
)


def _is_peer_dep_error(install_output: CommandResult) -> bool:
    """Checks if a failed command output indicates a peer dependency conflict."""
    if install_output.success:
        return False
    stderr = install_output.stderr.lower()
    return "eresolve" in stderr and "conflicting peer dependency" in stderr


def execute_submission(
    instance: BenchmarkInstance,
    submission: Submission,
    docker_manager: DockerManager,
    *,
    logger: "Logger",
) -> ExecutionTrace:
    """
    Orchestrates the end-to-end evaluation for a single submission.

    This function performs the following steps:
    1. Spawns an environment for the instance.
    2. Copies and applies the submitted patch.
    3. Runs install and build commands.
    4. Returns a structured execution trace.

    Parameters
    ----------
    instance
        The benchmark instance to evaluate against.
    submission
        The SUT's submitted solution.
    docker_manager
        An initialized DockerManager to interact with Docker.

    Returns
    -------
    An ExecutionTrace object containing all command outputs.
    """
    try:
        with InstanceEnvironment(instance, docker_manager) as env:
            with tempfile.TemporaryDirectory() as tmpdir:
                submission_patch_file_name = f"{instance.instance_id}.patch"
                patch_file_path = Path(tmpdir) / submission_patch_file_name
                patch_file_path.write_text(submission.model_patch, encoding="utf-8")

                # Copy patch file to environment
                env_patch_path = f"/tmp/{submission_patch_file_name}"
                env.copy_in(patch_file_path, env_patch_path)

            # 4. Check Patch (non-destructive check first)
            check_command = f"git apply --check -p0 {env_patch_path}"
            logger.debug(f"Executing patch check command: {check_command}")
            patch_check_out = env.exec(check_command)

            if not patch_check_out.success:
                logger.info("❌ Patch check failed. Halting evaluation.")
                return env.trace()

            # 5. Apply Patch
            apply_command = f"git apply -p0 {env_patch_path}"
            logger.debug(f"Executing patch apply command: {apply_command}")
            patch_apply_out = env.exec(apply_command)

            if not patch_apply_out.success:
                logger.info("❌ Patch application failed. Halting evaluation.")
                return env.trace()

            logger.info("✅ Patch applied successfully.")

            # 6. Install Dependencies
            logger.debug(f"Executing install command: {instance.commands.install}")
            install_out = env.exec(instance.commands.install)

            if not install_out.success:
                is_peer_dep_error = _is_peer_dep_error(install_out)
                if not is_peer_dep_error:
                    logger.info("❌ Install failed. Halting evaluation.")
                    return env.trace()
                logger.info("❌ Install failed due to peer dependency conflict.")

                original_install_command = instance.commands.install
                if "--legacy-peer-deps" in original_install_command:
                    logger.warning(
                        "❌ Halting evaluation as the original install command "
                        "already had --legacy-peer-deps flag."
                    )
                    return env.trace()

                fixed_install_command = original_install_command + " --legacy-peer-deps"
                logger.info(
                    f"Trying to install dependencies with --legacy-peer-deps flag: "
                    f"{fixed_install_command}"
                )
                install_out = env.exec(fixed_install_command)

                if not install_out.success:
                    logger.info(
                        "❌ Install failed even with --legacy-peer-deps flag. "
                        "Halting evaluation."
                    )
                    return env.trace()

                logger.info(
                    "✅ Dependencies installed successfully with "
                    "--legacy-peer-deps flag."
                )

            logger.info("✅ Dependencies installed successfully.")

            # 7. Check Version
            # TODO: make the command configurable
            version_command = "npm ls @angular/cli @angular/core --json"
            logger.debug(f"Executing version check command: {version_command}")
            version_check_out = env.exec(version_command)

            target_version_achieved = calculate_target_version_achieved(
                version_check_out, instance.target_angular_version
            )

            if not target_version_achieved:
                logger.info("❌ Target version not achieved. Halting evaluation.")
                return env.trace()

            # 8. Build
            logger.debug(f"Executing build command: {instance.commands.build}")
            build_out = env.exec(instance.commands.build)

            if not build_out.success:
                logger.info("❌ Build failed. Halting evaluation.")
                return env.trace()

            logger.info("✅ Build completed successfully.")

    except Exception as e:
        logger.exception(
            f"An unexpected error occurred during evaluation: {e.__class__.__name__}: {e}"  # noqa: E501
        )
        raise e

    return env.trace()
