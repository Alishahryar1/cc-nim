from unittest.mock import patch

import pytest

from free_claude_code.cli.launchers.claude import (
    build_claude_launcher_command,
    launch,
)
from free_claude_code.config.settings import Settings


def _launcher_settings(*, port: int = 9191, token: str = "proxy-token") -> Settings:
    return Settings(
        host="0.0.0.0",
        port=port,
        proxy_auth_enabled=False,
        proxy_auth_token=token,
        model="nvidia_nim/test-model",
        open_admin_browser=True,
    )


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


def test_launch_claude_fails_when_proxy_is_unreachable_and_auto_start_fails(
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = _launcher_settings(port=9393)

    with (
        patch(
            "free_claude_code.cli.launchers.claude.get_settings", return_value=settings
        ),
        patch(
            "free_claude_code.cli.launchers.claude.preflight_proxy",
            return_value="connection refused",
        ),
        patch(
            "free_claude_code.cli.launchers.claude._ensure_fcc_server_running",
            return_value=False,
        ) as ensure_server,
        patch("free_claude_code.cli.launchers.common.subprocess.Popen") as popen,
        pytest.raises(SystemExit) as exc_info,
    ):
        launch([])

    assert exc_info.value.code == 1
    ensure_server.assert_called_once_with("http://127.0.0.1:9393")
    popen.assert_not_called()
    captured = capsys.readouterr()
    assert "http://127.0.0.1:9393" in captured.err
    assert "fcc-server" in captured.err


def test_launch_claude_proceeds_when_proxy_auto_start_recovers() -> None:
    settings = _launcher_settings()

    with (
        patch(
            "free_claude_code.cli.launchers.claude.get_settings", return_value=settings
        ),
        patch(
            "free_claude_code.cli.launchers.claude.preflight_proxy",
            return_value="connection refused",
        ),
        patch(
            "free_claude_code.cli.launchers.claude._ensure_fcc_server_running",
            return_value=True,
        ) as ensure_server,
        patch(
            "free_claude_code.cli.launchers.common.shutil.which",
            return_value="resolved-claude.cmd",
        ),
        patch("free_claude_code.cli.launchers.common.subprocess.Popen") as popen,
        patch("free_claude_code.cli.launchers.common.register_pid") as register_pid,
        patch("free_claude_code.cli.launchers.common.unregister_pid") as unregister_pid,
        pytest.raises(SystemExit) as exc_info,
    ):
        process = popen.return_value
        process.pid = 12345
        process.wait.return_value = 0
        launch(["--model", "sonnet"])

    assert exc_info.value.code == 0
    ensure_server.assert_called_once_with("http://127.0.0.1:9191")
    popen.assert_called_once()
    assert popen.call_args.args[0] == [
        "resolved-claude.cmd",
        "--permission-mode",
        "default",
        "--model",
        "sonnet",
    ]
    register_pid.assert_called_once_with(12345)
    unregister_pid.assert_called_once_with(12345)
