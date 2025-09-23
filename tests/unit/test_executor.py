import pytest

from bench_mac.orchestration import is_peer_dep_error

from ..utils import create_command_output

PEER_DEP_ERROR_STDERR = """npm ERR! code ERESOLVE
npm ERR! ERESOLVE unable to resolve dependency tree
npm ERR!
npm ERR! Found: @angular-devkit/build-angular@0.1102.5
npm ERR! node_modules/@angular-devkit/build-angular
npm ERR!   dev @angular-devkit/build-angular@"~0.1102.9" from the root project
npm ERR!
npm ERR! Could not resolve dependency:
npm ERR! dev @angular-devkit/build-angular@"~0.1102.9" from the root project
npm ERR!
npm ERR! Conflicting peer dependency: @angular/localize@11.2.10
npm ERR! node_modules/@angular/localize
npm ERR!   peerOptional @angular/localize@"^11.0.0 || ^11.2.0-next" from @angular-devkit/build-angular@0.1102.9
npm ERR!   node_modules/@angular-devkit/build-angular
npm ERR!     dev @angular-devkit/build-angular@"~0.1102.9" from the root project
npm ERR!
npm ERR! Fix the upstream dependency conflict, or retry
npm ERR! this command with --force, or --legacy-peer-deps
npm ERR! to accept an incorrect (and potentially broken) dependency resolution.
"""  # noqa: E501


@pytest.mark.unit
class TestIsPeerDepError:
    """Unit tests for the _is_peer_dep_error helper function."""

    def test_returns_true_for_standard_peer_dep_error(self) -> None:
        """Verify the function correctly identifies a standard npm ERESOLVE error."""
        failed_output = create_command_output(
            command="npm ci",
            exit_code=1,
            stderr=PEER_DEP_ERROR_STDERR,
        )
        assert is_peer_dep_error(failed_output) is True

    def test_is_case_insensitive(self) -> None:
        """Verify the check is case-insensitive as intended."""
        stderr = "ERESOLVE failed due to a CONFLICTING PEER DEPENDENCY."
        failed_output = create_command_output(
            command="npm ci",
            exit_code=1,
            stderr=stderr,
        )
        assert is_peer_dep_error(failed_output) is True

    def test_returns_false_if_command_was_successful(self) -> None:
        """A successful command should never be a peer dep error,
        even if stderr has the keywords."""
        stderr_with_keywords = (
            "Log: ERESOLVE check for conflicting peer dependency passed."
        )
        successful_output = create_command_output(
            command="npm ci",
            exit_code=0,
            stderr=stderr_with_keywords,
        )
        assert is_peer_dep_error(successful_output) is False

    @pytest.mark.parametrize(
        "other_error_message",
        [
            "npm ERR! code E404\nnpm ERR! 404 Not Found - GET https://registry.npmjs.org/...",
            "npm ERR! command failed",
            "Error: ENOENT: no such file or directory, open 'package.json'",
            "This is just some random error text.",
            "",
        ],
    )
    def test_returns_false_for_other_npm_or_generic_errors(
        self, other_error_message: str
    ) -> None:
        """Verify that other types of errors are not misidentified."""
        failed_output = create_command_output(
            command="npm ci",
            exit_code=1,
            stderr=other_error_message,
        )
        assert is_peer_dep_error(failed_output) is False
