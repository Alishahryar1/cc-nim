from pathlib import Path

from cli.managed.session import ManagedClaudeSession


def test_cli_session_owns_typed_runner_config(tmp_path: Path) -> None:
    session = ManagedClaudeSession(
        workspace_path=str(tmp_path),
        api_url="http://127.0.0.1:8082/v1",
        allowed_dirs=[str(tmp_path)],
        plans_directory=".plans",
        claude_bin="claude-test",
    )

    assert session.config.workspace_path == str(tmp_path)
    assert session.config.api_url == "http://127.0.0.1:8082/v1"
    assert session.config.allowed_dirs == [str(tmp_path)]
    assert session.config.plans_directory == ".plans"
    assert session.config.claude_bin == "claude-test"
