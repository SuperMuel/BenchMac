from .models import CommandOutput, ExecutionTrace, MetricsReport


def _find_step(trace: ExecutionTrace, command_part: str) -> CommandOutput | None:
    """Finds the first step in the trace whose command contains a substring."""
    for step in trace.steps:
        if command_part in step.command:
            return step
    return None


def calculate_metrics(trace: ExecutionTrace) -> MetricsReport:
    """
    Analyzes an ExecutionTrace and computes the final performance metrics.

    Args:
        trace: The execution trace containing all command outputs

    Returns:
        MetricsReport with calculated metrics based on the trace
    """
    # Patch application metric
    # Check both the patch check and patch apply steps
    patch_check_step = _find_step(trace, "git apply --check")
    patch_apply_step = _find_step(trace, "git apply -p0")

    # Only successful if both check and apply succeeded
    if patch_check_step and patch_apply_step:
        patch_application_success = (
            patch_check_step.success and patch_apply_step.success
        )
    elif patch_apply_step:
        # Legacy: only apply step exists (older evaluation format)
        patch_application_success = patch_apply_step.success
    else:
        # No patch steps found
        patch_application_success = False

    # TODO: Add other metrics as they are uncommented in the MetricsReport model
    # For now, we only calculate the patch_application_success metric
    # Future metrics to implement:
    # - build_success: based on ng build step
    # - no_new_critical_lint_errors: based on ng lint step
    # - test_pass_rate: based on ng test step

    return MetricsReport(patch_application_success=patch_application_success)
