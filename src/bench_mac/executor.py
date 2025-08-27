import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from docker.models.containers import Container
    from loguru import Logger

from bench_mac.docker.builder import prepare_environment
from bench_mac.docker.manager import DockerManager
from bench_mac.models import (
    BenchmarkInstance,
    CommandOutput,
    ExecutionTrace,
    Submission,
)


def _is_peer_dep_error(install_output: CommandOutput) -> bool:
    """Checks if a failed command output indicates a peer dependency conflict."""
    if install_output.success:
        return False
    stderr = install_output.stderr.lower()
    return "eresolve" in stderr and "conflicting peer dependency" in stderr


def _execute_and_capture(
    manager: DockerManager,
    container: "Container",
    command: str,
    workdir: str | None = None,
) -> CommandOutput:
    """Executes a command and captures all relevant output in a CommandOutput object."""
    start_time = datetime.now(UTC)
    exit_code, stdout, stderr = manager.execute_in_container(
        container, command, workdir
    )
    end_time = datetime.now(UTC)

    return CommandOutput(
        command=command,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        start_time=start_time,
        end_time=end_time,
    )


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
    1. Prepares the specific Docker environment for the instance.
    2. Runs a container from the environment.
    3. Copies and applies the submitted patch.
    4. Runs install and build commands.
    5. Returns a structured execution trace.

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
    steps: list[CommandOutput] = []
    container = None

    project_dir = "/app/project"  # TODO: make this configurable

    with tempfile.TemporaryDirectory() as tmpdir:
        submission_patch_file_name = f"{instance.instance_id}.patch"
        patch_file_path = Path(tmpdir) / submission_patch_file_name
        patch_file_path.write_text(submission.model_patch, encoding="utf-8")

        try:
            # 1. Prepare Environment
            logger.debug(f"Preparing environment for instance: {instance.instance_id}")
            instance_image_tag = prepare_environment(instance, docker_manager)

            # 2. Run Container
            logger.debug(f"Starting container from image: {instance_image_tag}")
            container = docker_manager.run_container(instance_image_tag)

            # 3. Copy patch file to container
            container_patch_path = f"/tmp/{submission_patch_file_name}"
            docker_manager.copy_to_container(
                container, src_path=patch_file_path, dest_path=container_patch_path
            )

            # 4. Check Patch (non-destructive check first)
            check_command = f"git apply --check -p0 {container_patch_path}"
            logger.debug(f"Executing patch check command: {check_command}")
            patch_check_out = _execute_and_capture(
                docker_manager,
                container,
                check_command,
                workdir=project_dir,
            )
            steps.append(patch_check_out)

            if not patch_check_out.success:
                logger.info("❌ Patch check failed. Halting evaluation.")
                return ExecutionTrace(steps=steps)

            # 5. Apply Patch
            apply_command = f"git apply -p0 {container_patch_path}"
            logger.debug(f"Executing patch apply command: {apply_command}")
            patch_apply_out = _execute_and_capture(
                docker_manager,
                container,
                apply_command,
                workdir=project_dir,
            )
            steps.append(patch_apply_out)

            if not patch_apply_out.success:
                logger.info("❌ Patch application failed. Halting evaluation.")
                return ExecutionTrace(steps=steps)

            logger.info("✅ Patch applied successfully.")

            # 6. Install Dependencies
            logger.debug(f"Executing install command: {instance.commands.install}")
            install_out = _execute_and_capture(
                docker_manager,
                container,
                instance.commands.install,
                workdir=project_dir,
            )
            steps.append(install_out)

            if not install_out.success:
                is_peer_dep_error = _is_peer_dep_error(install_out)
                if not is_peer_dep_error:
                    logger.info("❌ Install failed. Halting evaluation.")
                    return ExecutionTrace(steps=steps)
                logger.info("❌ Install failed due to peer dependency conflict.")

                original_install_command = instance.commands.install
                if "--legacy-peer-deps" in original_install_command:
                    logger.warning(
                        "❌ Halting evaluation as the original install command "
                        "already had --legacy-peer-deps flag."
                    )
                    return ExecutionTrace(steps=steps)

                fixed_install_command = original_install_command + " --legacy-peer-deps"
                logger.info(
                    f"Trying to install dependencies with --legacy-peer-deps flag: "
                    f"{fixed_install_command}"
                )
                install_out = _execute_and_capture(
                    docker_manager,
                    container,
                    fixed_install_command,
                    workdir=project_dir,
                )
                steps.append(install_out)

                if not install_out.success:
                    logger.info(
                        "❌ Install failed even with --legacy-peer-deps flag. "
                        "Halting evaluation."
                    )
                    return ExecutionTrace(steps=steps)

                logger.info(
                    "✅ Dependencies installed successfully with "
                    "--legacy-peer-deps flag."
                )

            logger.info("✅ Dependencies installed successfully.")

            # 7. Check Version
            # TODO: make the command configurable
            version_command = "npm ls @angular/cli @angular/core --json"
            logger.debug(f"Executing version check command: {version_command}")
            version_check_out = _execute_and_capture(
                docker_manager,
                container,
                version_command,
                workdir=project_dir,
            )
            steps.append(version_check_out)

            if not version_check_out.success:
                logger.info(
                    f"❌ '{version_command}' command failed. Continuing execution."
                )
            # TODO: should we verify the version matches the target version here?
            #  and stop if it doesn't ?
            # continuing the evaluation makes no sense if the angular version is
            # not even  the target version

            # 8. Build
            logger.debug(f"Executing build command: {instance.commands.build}")
            build_out = _execute_and_capture(
                docker_manager,
                container,
                instance.commands.build,
                workdir=project_dir,
            )
            steps.append(build_out)

            if not build_out.success:
                logger.info("❌ Build failed. Halting evaluation.")
                return ExecutionTrace(steps=steps)

            logger.info("✅ Build completed successfully.")

        except Exception as e:
            logger.exception(
                f"An unexpected error occurred during evaluation: {e.__class__.__name__}: {e}"  # noqa: E501
            )
            raise e

        finally:
            # Cleanup
            if container:
                logger.debug(f"Cleaning up container {container.short_id}...")
                docker_manager.cleanup_container(container)

    return ExecutionTrace(steps=steps)
