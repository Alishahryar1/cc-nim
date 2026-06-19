from __future__ import annotations

import os

from cli.managed.claude import (
    ManagedClaudeConfig,
    ManagedClaudeParseState,
    ManagedClaudeTaskRequest,
    build_managed_claude_env,
    build_managed_claude_invocation,
    extract_managed_claude_session_id,
    parse_managed_claude_stdout_line,
)


def _config(**overrides: object) -> ManagedClaudeConfig:
    workspace_path = overrides.get("workspace_path", os.path.normpath("/tmp/workspace"))
    api_url = overrides.get("api_url", "http://localhost:8082/v1")
    raw_allowed_dirs = overrides.get("allowed_dirs")
    allowed_dirs: list[str] = []
    if raw_allowed_dirs is not None:
        assert isinstance(raw_allowed_dirs, list)
        for directory in raw_allowed_dirs:
            assert isinstance(directory, str)
            allowed_dirs.append(directory)
    plans_directory = overrides.get("plans_directory")
    claude_bin = overrides.get("claude_bin", "claude")
    auth_token = overrides.get("auth_token", "proxy-token")

    assert isinstance(workspace_path, str)
    assert isinstance(api_url, str)
    assert plans_directory is None or isinstance(plans_directory, str)
    assert isinstance(claude_bin, str)
    assert isinstance(auth_token, str)
    return ManagedClaudeConfig(
        workspace_path=workspace_path,
        api_url=api_url,
        allowed_dirs=allowed_dirs,
        plans_directory=plans_directory,
        claude_bin=claude_bin,
        auth_token=auth_token,
    )


def test_managed_claude_builds_new_task_command_and_env() -> None:
    invocation = build_managed_claude_invocation(
        config=_config(
            allowed_dirs=[os.path.normpath("/tmp/extra")],
            plans_directory=".plans",
        ),
        request=ManagedClaudeTaskRequest(prompt="hello"),
        base_env={"PATH": "keep", "ANTHROPIC_API_KEY": "official"},
    )

    assert invocation.argv[:2] == ("claude", "-p")
    assert "hello" in invocation.argv
    assert "--output-format" in invocation.argv
    assert "stream-json" in invocation.argv
    assert "--add-dir" in invocation.argv
    assert os.path.normpath("/tmp/extra") in invocation.argv
    assert "--settings" in invocation.argv
    assert invocation.env["PATH"] == "keep"
    assert invocation.env["ANTHROPIC_API_URL"] == "http://localhost:8082/v1"
    assert invocation.env["ANTHROPIC_BASE_URL"] == "http://localhost:8082"
    assert invocation.env["ANTHROPIC_AUTH_TOKEN"] == "proxy-token"
    assert "ANTHROPIC_API_KEY" not in invocation.env
    assert invocation.trace_metadata["client_cli_id"] == "claude"
    assert invocation.trace_metadata["claude_binary"] == "claude"


def test_managed_claude_builds_resume_and_fork_commands() -> None:
    resume = build_managed_claude_invocation(
        config=_config(),
        request=ManagedClaudeTaskRequest(prompt="again", session_id="sess_1"),
        base_env={},
    )
    fork = build_managed_claude_invocation(
        config=_config(),
        request=ManagedClaudeTaskRequest(
            prompt="branch", session_id="sess_1", fork_session=True
        ),
        base_env={},
    )

    assert resume.argv[:3] == ("claude", "--resume", "sess_1")
    assert "--fork-session" not in resume.argv
    assert fork.argv[:3] == ("claude", "--resume", "sess_1")
    assert "--fork-session" in fork.argv


def test_managed_claude_env_uses_sentinel_when_proxy_auth_blank() -> None:
    env = build_managed_claude_env(
        api_url="http://localhost:8082/v1",
        auth_token="",
        base_env={"ANTHROPIC_AUTH_TOKEN": "stale"},
    )

    assert env["ANTHROPIC_AUTH_TOKEN"] == "fcc-no-auth"


def test_managed_claude_extracts_session_ids() -> None:
    assert extract_managed_claude_session_id({"session_id": "direct"}) == "direct"
    assert extract_managed_claude_session_id({"sessionId": "camel"}) == "camel"
    assert (
        extract_managed_claude_session_id({"init": {"session_id": "nested"}})
        == "nested"
    )
    assert (
        extract_managed_claude_session_id({"result": {"sessionId": "result"}})
        == "result"
    )
    assert extract_managed_claude_session_id({"conversation": {"id": "conv"}}) == "conv"
    assert extract_managed_claude_session_id({"type": "message"}) is None
    assert extract_managed_claude_session_id("not a dict") is None


def test_managed_claude_parser_emits_session_info_once() -> None:
    state = ManagedClaudeParseState()

    first = list(parse_managed_claude_stdout_line('{"session_id": "sess_1"}', state))
    second = list(parse_managed_claude_stdout_line('{"session_id": "sess_2"}', state))

    assert first == [
        {"type": "session_info", "session_id": "sess_1"},
        {"session_id": "sess_1"},
    ]
    assert second == [{"session_id": "sess_2"}]


def test_managed_claude_parser_returns_raw_for_non_json() -> None:
    events = list(
        parse_managed_claude_stdout_line(
            "not json", ManagedClaudeParseState(log_raw_cli_diagnostics=False)
        )
    )

    assert events == [{"type": "raw", "content": "not json"}]
