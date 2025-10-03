import json
import re

from loguru import logger
from packaging.version import InvalidVersion, Version

from bench_mac.core.models import (
    BenchmarkInstance,
    CommandResult,
    ExecutionTrace,
    MetricsReport,
)

from .trace_analyzer import TraceAnalyzer


def _calculate_patch_application_success(
    patch_check_step: CommandResult | None,
    patch_apply_step: CommandResult | None,
) -> bool | None:
    """
    Calculates the definitive success of patch application based on the outcomes
    of the 'git apply --check' and 'git apply' steps.

    This function follows a clear, hierarchical logic:
    1. If both check and apply steps ran, success requires both to pass.
    2. If only the check step ran, its outcome determines the result (as a failure
       would have halted the process).
    3. If only the apply step ran (a legacy or simplified case), its outcome
       determines the result.
    4. If neither step ran, the outcome is unknown.

    Parameters
    ----------
    patch_check_step
        The result of the `git apply --check` command, or None if not executed.
    patch_apply_step
        The result of the `git apply` command, or None if not executed.

    Returns
    -------
    bool | None
        - True: The patch was successfully applied.
        - False: The patch application failed at either the check or apply stage.
        - None: There is not enough information to determine the outcome.
    """
    # Case 1: The standard, modern evaluation path was followed.
    # This is the most common and robust scenario.
    if patch_check_step and patch_apply_step:
        # For a patch to be considered truly successful, the non-destructive
        # check must pass, AND the actual application must also pass.
        return patch_check_step.success and patch_apply_step.success

    # Case 2: Only the check step was executed.
    # This implies the harness ran the check but halted before applying,
    # almost certainly because the check itself failed.
    elif patch_check_step:
        # The success of the whole operation is determined solely by the
        # success of this single, failed step.
        return patch_check_step.success

    # Case 3: Only the apply step was executed.
    # This might represent a legacy evaluation or a simplified run
    # that skips the '--check' optimization.
    elif patch_apply_step:
        # With no check step available, our only source of truth is whether
        # the apply command itself succeeded.
        return patch_apply_step.success

    # Case 4: Neither patch-related step was found in the execution trace.
    else:
        # Without any relevant steps, we cannot determine the outcome.
        # This signals an issue with the harness execution or that the
        # evaluation failed long before the patch stage.
        return None


def _extract_major_version(raw_version: str) -> int | None:
    """Extracts the major version from a version string.

    Tries to parse the version using `packaging.Version` first. If that fails
    because the string is not PEP 440 compliant (e.g. "21.0.0-next.5"),
    falls back to extracting the leading integer prefix.
    Returns ``None`` when neither strategy works.
    """

    try:
        return Version(raw_version).major
    except InvalidVersion:
        match = re.match(r"\d+", raw_version)
        if match:
            return int(match.group())
    except TypeError:
        return None

    return None


def calculate_target_version_achieved(
    version_check_step: CommandResult | None, target_version: str
) -> bool | None:
    """
    Calculates if the target version was achieved by parsing `npm ls --json` output.
    This function INTENTIONALLY ignores the command's exit code, as `npm ls`
    can exit with 1 due to peer dependency issues while still providing valid JSON.
    """
    if not version_check_step:
        return None  # The step was not executed

    if not version_check_step.stdout.strip():
        return None  # No JSON output to parse

    try:
        data = json.loads(version_check_step.stdout)
    except json.JSONDecodeError:
        logger.warning(
            f"Failed to parse JSON from version check command. Output was:\n"
            f"{version_check_step.stdout}"
        )
        return None  # Outcome is uncertain

    # Safely navigate the dependency tree from `npm ls` output
    dependencies = data.get("dependencies", {})
    angular_core_info = dependencies.get("@angular/core")

    if not angular_core_info or "version" not in angular_core_info:
        # The command ran but @angular/core was not found in the dependency tree.
        return False

    achieved_version_str = angular_core_info["version"]
    achieved_major = _extract_major_version(achieved_version_str)
    target_major = _extract_major_version(target_version)
    assert target_major is not None, (
        "Target version must be a valid major version. "
        "Check your instance configuration."
    )

    if achieved_major is None:
        logger.warning(
            f"Could not determine major version from output '{achieved_version_str}'"
        )
        return None

    return achieved_major == target_major


def calculate_metrics(
    trace: ExecutionTrace, instance: BenchmarkInstance, *, empty_patch: bool = False
) -> MetricsReport:
    """
    Analyzes an ExecutionTrace and computes the final performance metrics.

    Args:
        trace: The execution trace containing all command outputs
        instance: The benchmark instance
        empty_patch: Whether the SUT submitted an empty patch

    Returns:
        MetricsReport with calculated metrics based on the trace
    """
    if empty_patch:
        return MetricsReport(
            patch_application_success=False,  # empty patches are considered failures
            install_success=False,
            target_version_achieved=False,
            build_success=False,
        )

    # Patch application metric
    # Check both the patch check and patch apply steps
    analyzer = TraceAnalyzer(trace, instance)

    patch_check_step = analyzer.patch_check_step()
    patch_apply_step = analyzer.patch_apply_step()
    final_install_step = analyzer.final_install_step()
    version_check_step = analyzer.version_check_step()
    build_step = analyzer.build_step()

    patch_application_success = _calculate_patch_application_success(
        patch_check_step, patch_apply_step
    )
    install_success = final_install_step.success if final_install_step else None

    target_version_achieved = calculate_target_version_achieved(
        version_check_step, instance.target_angular_version
    )

    build_success = build_step.success if build_step else None

    return MetricsReport(
        patch_application_success=patch_application_success,
        install_success=install_success,
        target_version_achieved=target_version_achieved,
        build_success=build_success,
    )
