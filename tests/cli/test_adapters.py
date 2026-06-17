from __future__ import annotations

import json
from types import SimpleNamespace

from cli.adapters.base import CliParseState, CliTaskRequest
from cli.adapters.claude import CLAUDE_CLI_ADAPTER
from cli.adapters.codex import CODEX_CLI_ADAPTER
from cli.adapters.registry import DEFAULT_CLIENT_CLI_ID, get_client_cli_adapter


def _config(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "workspace_path": "/workspace",
        "api_url": "http://127.0.0.1:8082/v1",
        "allowed_dirs": [],
        "plans_directory": None,
        "claude_bin": "claude-test",
        "auth_token": "",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_registry_returns_default_claude_adapter() -> None:
    assert DEFAULT_CLIENT_CLI_ID == "claude"
    assert get_client_cli_adapter() is CLAUDE_CLI_ADAPTER
    assert get_client_cli_adapter("claude") is CLAUDE_CLI_ADAPTER
    assert get_client_cli_adapter("codex") is CODEX_CLI_ADAPTER


def test_claude_adapter_builds_new_task_command_and_env() -> None:
    invocation = CLAUDE_CLI_ADAPTER.build_task_invocation(
        config=_config(auth_token="proxy-token"),
        request=CliTaskRequest(prompt="hello"),
        base_env={
            "KEEP_ME": "yes",
            "ANTHROPIC_API_KEY": "official-key",
            "ANTHROPIC_AUTH_TOKEN": "stale-token",
        },
    )

    assert invocation.argv == (
        "claude-test",
        "-p",
        "hello",
        "--output-format",
        "stream-json",
        "--dangerously-skip-permissions",
        "--verbose",
    )
    assert invocation.env["KEEP_ME"] == "yes"
    assert invocation.env["ANTHROPIC_API_URL"] == "http://127.0.0.1:8082/v1"
    assert invocation.env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8082"
    assert invocation.env["ANTHROPIC_AUTH_TOKEN"] == "proxy-token"
    assert invocation.env["CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY"] == "1"
    assert invocation.env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] == "190000"
    assert "ANTHROPIC_API_KEY" not in invocation.env
    assert invocation.trace_metadata["client_cli_id"] == "claude"


def test_claude_adapter_builds_resume_fork_command() -> None:
    invocation = CLAUDE_CLI_ADAPTER.build_task_invocation(
        config=_config(),
        request=CliTaskRequest(
            prompt="continue",
            session_id="sess_123",
            fork_session=True,
        ),
        base_env={},
    )

    assert invocation.argv[:4] == (
        "claude-test",
        "--resume",
        "sess_123",
        "--fork-session",
    )
    assert "-p" in invocation.argv
    assert "continue" in invocation.argv
    assert invocation.trace_metadata["resume_session_id"] == "sess_123"
    assert invocation.trace_metadata["fork_session"] is True


def test_claude_adapter_does_not_resume_pending_session() -> None:
    invocation = CLAUDE_CLI_ADAPTER.build_task_invocation(
        config=_config(),
        request=CliTaskRequest(prompt="new", session_id="pending_123"),
        base_env={},
    )

    assert "--resume" not in invocation.argv
    assert invocation.trace_metadata["resume_session_id"] is None


def test_claude_adapter_adds_allowed_dirs_and_plans_directory() -> None:
    invocation = CLAUDE_CLI_ADAPTER.build_task_invocation(
        config=_config(
            allowed_dirs=["/dir1", "/dir2"],
            plans_directory="./agent_workspace/plans",
        ),
        request=CliTaskRequest(prompt="hello"),
        base_env={},
    )

    assert invocation.argv.count("--add-dir") == 2
    assert "/dir1" in invocation.argv
    assert "/dir2" in invocation.argv
    settings_idx = invocation.argv.index("--settings")
    settings = json.loads(invocation.argv[settings_idx + 1])
    assert settings["plansDirectory"] == "./agent_workspace/plans"


def test_claude_adapter_launcher_env_targets_proxy() -> None:
    env = CLAUDE_CLI_ADAPTER.build_launcher_env(
        proxy_root_url="http://127.0.0.1:9191",
        auth_token=" proxy-token ",
        base_env={
            "PATH": "keep",
            "ANTHROPIC_BASE_URL": "https://api.anthropic.com",
            "ANTHROPIC_API_KEY": "official-key",
            "ANTHROPIC_AUTH_TOKEN": "stale-token",
        },
    )

    assert env["PATH"] == "keep"
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:9191"
    assert env["ANTHROPIC_AUTH_TOKEN"] == "proxy-token"
    assert env["CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY"] == "1"
    assert env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] == "190000"
    assert "ANTHROPIC_API_KEY" not in env


def test_claude_adapter_launcher_command_preserves_args() -> None:
    command = CLAUDE_CLI_ADAPTER.build_launcher_command(
        binary_path="claude.cmd",
        argv=["--model", "sonnet"],
        settings=_config(),
        proxy_root_url="http://127.0.0.1:8082",
    )

    assert command == ["claude.cmd", "--model", "sonnet"]


def test_claude_adapter_extracts_supported_session_id_shapes() -> None:
    assert CLAUDE_CLI_ADAPTER.extract_session_id({"session_id": "direct"}) == "direct"
    assert CLAUDE_CLI_ADAPTER.extract_session_id({"sessionId": "camel"}) == "camel"
    assert (
        CLAUDE_CLI_ADAPTER.extract_session_id({"init": {"session_id": "nested"}})
        == "nested"
    )
    assert (
        CLAUDE_CLI_ADAPTER.extract_session_id({"result": {"sessionId": "result"}})
        == "result"
    )
    assert (
        CLAUDE_CLI_ADAPTER.extract_session_id({"conversation": {"id": "conv"}})
        == "conv"
    )
    assert CLAUDE_CLI_ADAPTER.extract_session_id({"type": "message"}) is None
    assert CLAUDE_CLI_ADAPTER.extract_session_id("not a dict") is None


def test_claude_adapter_invalid_stdout_json_becomes_raw_event() -> None:
    events = list(
        CLAUDE_CLI_ADAPTER.parse_stdout_line(
            "Not valid json",
            CliParseState(log_raw_cli_diagnostics=False),
        )
    )

    assert events == [{"type": "raw", "content": "Not valid json"}]


def test_claude_adapter_synthesizes_session_info_once() -> None:
    state = CliParseState()

    first_events = list(
        CLAUDE_CLI_ADAPTER.parse_stdout_line('{"session_id": "sess_1"}', state)
    )
    second_events = list(
        CLAUDE_CLI_ADAPTER.parse_stdout_line('{"session_id": "sess_2"}', state)
    )

    assert first_events == [
        {"type": "session_info", "session_id": "sess_1"},
        {"session_id": "sess_1"},
    ]
    assert second_events == [{"session_id": "sess_2"}]


def test_codex_adapter_builds_new_task_command_and_env() -> None:
    invocation = CODEX_CLI_ADAPTER.build_task_invocation(
        config=_config(auth_token="proxy-token"),
        request=CliTaskRequest(prompt="hello"),
        base_env={
            "KEEP_ME": "yes",
            "OPENAI_API_KEY": "official-key",
            "OPENAI_BASE_URL": "https://api.openai.com/v1",
            "CODEX_API_KEY": "stale-token",
            "CODEX_HOME": "/tmp/codex",
        },
    )

    assert invocation.argv[:3] == ("codex", "exec", "--json")
    assert "--dangerously-bypass-approvals-and-sandbox" in invocation.argv
    assert "-C" in invocation.argv
    assert "/workspace" in invocation.argv
    assert invocation.argv[-1] == "hello"
    assert 'model_provider="fcc"' in invocation.argv
    assert 'model_providers.fcc.wire_api="responses"' in invocation.argv
    assert invocation.env["KEEP_ME"] == "yes"
    assert invocation.env["CODEX_HOME"] == "/tmp/codex"
    assert invocation.env["FCC_CODEX_API_KEY"] == "proxy-token"
    assert "OPENAI_API_KEY" not in invocation.env
    assert "OPENAI_BASE_URL" not in invocation.env
    assert "CODEX_API_KEY" not in invocation.env
    assert invocation.trace_metadata["client_cli_id"] == "codex"
    assert invocation.trace_metadata["codex_binary"] == "codex"
    assert "claude_binary" not in invocation.trace_metadata
    assert CODEX_CLI_ADAPTER.trace_stage == "codex_cli"
    assert CODEX_CLI_ADAPTER.process_launch_event == "codex_cli.process.launch"
    assert CODEX_CLI_ADAPTER.trace_source == "codex_cli"


def test_codex_adapter_uses_explicit_codex_binary_when_provided() -> None:
    invocation = CODEX_CLI_ADAPTER.build_task_invocation(
        config=_config(codex_bin="codex-test"),
        request=CliTaskRequest(prompt="hello"),
        base_env={},
    )

    assert invocation.argv[:3] == ("codex-test", "exec", "--json")
    assert invocation.trace_metadata["codex_binary"] == "codex-test"


def test_codex_adapter_builds_resume_command() -> None:
    invocation = CODEX_CLI_ADAPTER.build_task_invocation(
        config=_config(),
        request=CliTaskRequest(prompt="continue", session_id="sess_123"),
        base_env={},
    )

    assert invocation.argv[:4] == ("codex", "exec", "resume", "--json")
    assert "sess_123" in invocation.argv
    assert invocation.argv[-1] == "continue"
    assert invocation.trace_metadata["resume_session_id"] == "sess_123"


def test_codex_adapter_fork_starts_new_session() -> None:
    invocation = CODEX_CLI_ADAPTER.build_task_invocation(
        config=_config(),
        request=CliTaskRequest(
            prompt="fork",
            session_id="sess_123",
            fork_session=True,
        ),
        base_env={},
    )

    assert "resume" not in invocation.argv
    assert invocation.trace_metadata["resume_session_id"] is None
    assert invocation.trace_metadata["fork_session"] is True


def test_codex_adapter_launcher_command_targets_responses_provider() -> None:
    command = CODEX_CLI_ADAPTER.build_launcher_command(
        binary_path="codex.cmd",
        argv=["exec", "hello"],
        settings=_config(model="nvidia_nim/test-model"),
        proxy_root_url="http://127.0.0.1:8082",
    )

    assert command[:2] == ["codex.cmd", "-c"]
    assert 'model_provider="fcc"' in command
    assert 'model_providers.fcc.base_url="http://127.0.0.1:8082/v1"' in command
    assert 'model_providers.fcc.env_key="FCC_CODEX_API_KEY"' in command
    assert 'model_providers.fcc.wire_api="responses"' in command
    assert 'model="nvidia_nim/test-model"' in command
    assert command[-2:] == ["exec", "hello"]


def test_codex_adapter_launcher_env_strips_openai_credentials() -> None:
    env = CODEX_CLI_ADAPTER.build_launcher_env(
        proxy_root_url="http://127.0.0.1:9191",
        auth_token=" proxy-token ",
        base_env={
            "PATH": "keep",
            "CODEX_HOME": "/tmp/codex",
            "OPENAI_API_KEY": "official-key",
            "OPENAI_BASE_URL": "https://api.openai.com/v1",
            "CODEX_API_KEY": "stale",
            "FCC_CODEX_API_KEY": "old",
        },
    )

    assert env["PATH"] == "keep"
    assert env["CODEX_HOME"] == "/tmp/codex"
    assert env["FCC_CODEX_API_KEY"] == "proxy-token"
    assert "OPENAI_API_KEY" not in env
    assert "OPENAI_BASE_URL" not in env
    assert "CODEX_API_KEY" not in env


def test_codex_adapter_parses_response_text_delta() -> None:
    events = list(
        CODEX_CLI_ADAPTER.parse_stdout_line(
            '{"type":"response.output_text.delta","delta":"hi","thread_id":"t1"}',
            CliParseState(),
        )
    )

    assert events == [
        {"type": "session_info", "session_id": "t1"},
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "hi"},
        },
    ]


def test_codex_adapter_parses_response_reasoning_text_delta() -> None:
    events = list(
        CODEX_CLI_ADAPTER.parse_stdout_line(
            (
                '{"type":"response.reasoning_text.delta","delta":"think",'
                '"output_index":2,"thread_id":"t1"}'
            ),
            CliParseState(),
        )
    )

    assert events == [
        {"type": "session_info", "session_id": "t1"},
        {
            "type": "content_block_delta",
            "index": 2,
            "delta": {"type": "thinking_delta", "thinking": "think"},
        },
    ]


def test_codex_adapter_parses_completed_function_call_item() -> None:
    line = json.dumps(
        {
            "type": "response.output_item.done",
            "item": {
                "type": "function_call",
                "call_id": "call_1",
                "name": "echo",
                "arguments": '{"value":"FCC"}',
            },
        }
    )

    events = list(CODEX_CLI_ADAPTER.parse_stdout_line(line, CliParseState()))

    assert events == [
        {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call_1",
                        "name": "echo",
                        "input": {"value": "FCC"},
                    }
                ]
            },
        }
    ]


def test_codex_adapter_reasoning_summary_delta_remains_raw() -> None:
    events = list(
        CODEX_CLI_ADAPTER.parse_stdout_line(
            (
                '{"type":"response.reasoning_summary_text.delta",'
                '"delta":"summary","thread_id":"t1"}'
            ),
            CliParseState(),
        )
    )

    assert events == [
        {"type": "session_info", "session_id": "t1"},
        {
            "type": "raw",
            "content": (
                '{"type":"response.reasoning_summary_text.delta",'
                '"delta":"summary","thread_id":"t1"}'
            ),
        },
    ]


def test_codex_adapter_unknown_json_becomes_raw_event() -> None:
    events = list(
        CODEX_CLI_ADAPTER.parse_stdout_line(
            '{"type":"thread.started","thread_id":"t1"}',
            CliParseState(),
        )
    )

    assert events == [
        {"type": "session_info", "session_id": "t1"},
        {"type": "raw", "content": '{"type":"thread.started","thread_id":"t1"}'},
    ]
