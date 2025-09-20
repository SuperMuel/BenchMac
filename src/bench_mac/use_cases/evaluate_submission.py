import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from bench_mac.domain.services.ports import EnvironmentFactory
from bench_mac.metrics import calculate_target_version_achieved
from bench_mac.models import (
    BenchmarkInstance,
    CommandResult,
    ExecutionTrace,
    Submission,
)

if TYPE_CHECKING:  # pragma: no cover
    from loguru import Logger


def is_peer_dep_error(install_output: CommandResult) -> bool:
    """Return True when npm install failed because of peer dependency conflicts."""
    if install_output.success:
        return False
    stderr = install_output.stderr.lower()
    return "eresolve" in stderr and "conflicting peer dependency" in stderr


def evaluate_submission(
    instance: BenchmarkInstance,
    submission: Submission,
    environment_factory: EnvironmentFactory,
    *,
    logger: "Logger",
) -> ExecutionTrace:
    """Execute the benchmark workflow using the provided environment factory."""
    try:
        environment = environment_factory.create(instance)
        with environment as env:
            with tempfile.TemporaryDirectory() as tmpdir:
                submission_patch_name = f"{instance.instance_id}.patch"
                patch_file_path = Path(tmpdir) / submission_patch_name
                patch_file_path.write_text(submission.model_patch, encoding="utf-8")

                env_patch_path = f"/tmp/{submission_patch_name}"
                env.copy_in(patch_file_path, env_patch_path)

            check_command = f"git apply --check -p0 {env_patch_path}"
            logger.debug(f"Executing patch check command: {check_command}")
            patch_check_out = env.exec(check_command)

            if not patch_check_out.success:
                logger.info("❌ Patch check failed. Halting evaluation.")
                return env.trace()

            apply_command = f"git apply -p0 {env_patch_path}"
            logger.debug(f"Executing patch apply command: {apply_command}")
            patch_apply_out = env.exec(apply_command)

            if not patch_apply_out.success:
                logger.info("❌ Patch application failed. Halting evaluation.")
                return env.trace()

            logger.info("✅ Patch applied successfully.")

            logger.debug(f"Executing install command: {instance.commands.install}")
            install_out = env.exec(instance.commands.install)

            if not install_out.success:
                if not is_peer_dep_error(install_out):
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
                    "Trying to install dependencies with --legacy-peer-deps flag: "
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

            version_command = "npm ls @angular/cli @angular/core --json"
            logger.debug(f"Executing version check command: {version_command}")
            version_check_out = env.exec(version_command)

            target_version_achieved = calculate_target_version_achieved(
                version_check_out,
                instance.target_angular_version,
            )

            if not target_version_achieved:
                logger.info("❌ Target version not achieved. Halting evaluation.")
                return env.trace()

            logger.debug(f"Executing build command: {instance.commands.build}")
            build_out = env.exec(instance.commands.build)

            if not build_out.success:
                logger.info("❌ Build failed. Halting evaluation.")
                return env.trace()

            logger.info("✅ Build completed successfully.")

    except Exception as e:  # pragma: no cover - defensive logging
        logger.exception(
            "An unexpected error occurred during evaluation: "
            f"{e.__class__.__name__}: {e}"
        )
        raise

    return environment.trace()
