import pytest

from free_claude_code.cli.launchers.claude import build_claude_launcher_command


def test_claude_launcher_defaults_to_non_auto_permission_mode() -> None:
    assert build_claude_launcher_command(
        binary_path="claude", argv=["fix the tests"]
    ) == ["claude", "--permission-mode", "default", "fix the tests"]


@pytest.mark.parametrize(
    "argv",
    [
        ["--permission-mode", "auto", "fix the tests"],
        ["--permission-mode=acceptEdits", "fix the tests"],
        ["--dangerously-skip-permissions", "fix the tests"],
    ],
)
def test_claude_launcher_preserves_explicit_permission_choice(
    argv: list[str],
) -> None:
    assert build_claude_launcher_command(binary_path="claude", argv=argv) == [
        "claude",
        *argv,
    ]


@pytest.mark.parametrize("argv", [["--permission-mode"], ["--permission-mode="]])
def test_claude_launcher_leaves_malformed_permission_options_to_claude(
    argv: list[str],
) -> None:
    assert build_claude_launcher_command(binary_path="claude", argv=argv) == [
        "claude",
        *argv,
    ]


@pytest.mark.parametrize(
    "argv",
    [
        ["explain --permission-mode auto"],
        ["--", "--permission-mode=auto"],
        ["--", "--dangerously-skip-permissions"],
    ],
)
def test_claude_launcher_does_not_treat_prompt_text_as_a_permission_choice(
    argv: list[str],
) -> None:
    assert build_claude_launcher_command(binary_path="claude", argv=argv) == [
        "claude",
        "--permission-mode",
        "default",
        *argv,
    ]
