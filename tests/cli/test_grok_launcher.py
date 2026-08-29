"""Contract tests for the installed `fcc-grok` launcher."""

import json
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from free_claude_code.cli.launchers.model_catalog import ClientModel
from free_claude_code.config.settings import Settings
from free_claude_code.core.json_types import JsonObject


def _settings(*, token: str = "proxy-token") -> Settings:
    return Settings(
        host="0.0.0.0",
        port=9191,
        proxy_auth_enabled=False,
        proxy_auth_token=token,
        model="nvidia_nim/test-model",
    )


def _models() -> tuple[ClientModel, ...]:
    return (
        ClientModel(
            wire_slug="nvidia_nim/vendor/model",
            provider_model_ref="nvidia_nim/vendor/model",
            display_name="Nested model",
            supports_reasoning=True,
        ),
        ClientModel(
            wire_slug="claude-3-freecc-no-thinking/open_router/plain-model",
            provider_model_ref="open_router/plain-model",
            display_name="Plain model",
            supports_reasoning=False,
        ),
    )


def _models_payload() -> JsonObject:
    return {
        "object": "list",
        "data": [
            {
                "id": model.wire_slug,
                "provider_model_ref": model.provider_model_ref,
                "display_name": model.display_name,
            }
            for model in _models()
        ],
    }


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["fix the bug"],
        ["-p", "fix the bug"],
        ["--single=fix the bug"],
        ["--prompt-json", "[]"],
        ["--prompt-file", "prompt.json"],
        ["--cwd", "workspace", "agent", "--model", "nvidia_nim/model", "stdio"],
    ],
)
def test_attached_grok_surfaces_are_routed(argv: list[str]) -> None:
    from free_claude_code.cli.launchers.grok import (
        GrokInvocationMode,
        classify_grok_invocation,
    )

    invocation = classify_grok_invocation(argv)

    assert invocation.mode is GrokInvocationMode.ROUTED
    if "agent" in argv:
        assert invocation.agent_index is not None


@pytest.mark.parametrize("argv", [["models"], ["inspect"], ["inspect", "--json"]])
def test_catalog_surfaces_receive_fcc_configuration(argv: list[str]) -> None:
    from free_claude_code.cli.launchers.grok import (
        GrokInvocationMode,
        classify_grok_invocation,
    )

    assert classify_grok_invocation(argv).mode is GrokInvocationMode.CONFIGURED


@pytest.mark.parametrize(
    "argv",
    [
        ["--help"],
        ["--version"],
        ["version", "--json"],
        ["doctor"],
        ["login", "--oauth"],
        ["logout"],
        ["mcp", "list"],
        ["plugin", "list"],
        ["memory", "status"],
        ["sessions", "list"],
        ["setup"],
        ["share", "session"],
        ["wrap", "git", "status"],
        ["export", "session"],
        ["trace", "list"],
        ["update", "--check"],
        ["completions", "powershell"],
        ["worktree", "list"],
        ["du"],
        ["disk-usage"],
    ],
)
def test_management_surfaces_are_native_passthrough(argv: list[str]) -> None:
    from free_claude_code.cli.launchers.grok import (
        GrokInvocationMode,
        classify_grok_invocation,
    )

    assert classify_grok_invocation(argv).mode is GrokInvocationMode.PASSTHROUGH


@pytest.mark.parametrize(
    "argv",
    [
        ["agent"],
        ["agent", "headless"],
        ["agent", "serve"],
        ["agent", "leader"],
        ["agent", "future-mode"],
        ["leader", "list"],
        ["workspace", "start"],
        ["dashboard"],
    ],
)
def test_detached_surfaces_are_rejected(argv: list[str]) -> None:
    from free_claude_code.cli.launchers.grok import classify_grok_invocation

    with pytest.raises(SystemExit) as exc_info:
        classify_grok_invocation(argv)

    assert exc_info.value.code == 2


@pytest.mark.parametrize(
    "argv",
    [
        ["--leader"],
        ["--oauth"],
        ["--force-login"],
        ["--xai-api-base-url", "https://example.test"],
        ["--xai-api-base-url=https://example.test"],
        ["agent", "--cli-chat-proxy-base-url", "https://example.test", "stdio"],
    ],
)
def test_route_bypass_flags_are_rejected_before_version_probe(argv: list[str]) -> None:
    from free_claude_code.cli.launchers import grok

    with (
        patch.object(grok, "resolve_client_binary", return_value="resolved-grok"),
        patch.object(grok, "require_compatible_grok") as require_version,
        pytest.raises(SystemExit) as exc_info,
    ):
        grok.launch(argv)

    assert exc_info.value.code == 2
    require_version.assert_not_called()


@pytest.mark.parametrize(
    ("output", "return_code", "expected"),
    [
        (
            '{"currentVersion":"1.0.5 (5115b46bc9)","channel":"stable"}',
            0,
            (1, 0, 5),
        ),
        ('{"channel":"stable","currentVersion":"1.2.3+build.1"}', 0, (1, 2, 3)),
        ('{"currentVersion":"1.0.5","channel":"unknown"}', 0, (1, 0, 5)),
        ('{"currentVersion":"1.2.3","channel":"alpha"}', 0, (1, 2, 3)),
        ('{"currentVersion":"1.2.3-alpha.1","channel":"stable"}', 0, None),
        ('{"currentVersion":123,"channel":"stable"}', 0, None),
        ('{"currentVersion":"1.2.3","channel":"stable"}', 1, None),
        ('{"currentVersion":"1.2.3\t(build)","channel":"stable"}', 0, None),
        ("not-json", 0, None),
        ("[]", 0, None),
    ],
)
def test_grok_version_probe(
    output: str,
    return_code: int,
    expected: tuple[int, int, int] | None,
) -> None:
    from free_claude_code.cli.launchers import grok

    with patch.object(
        grok.subprocess,
        "run",
        return_value=SimpleNamespace(stdout=output, returncode=return_code),
    ) as run:
        assert grok.grok_binary_version("resolved-grok") == expected
    run.assert_called_once_with(
        ["resolved-grok", "version", "--json"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=5.0,
    )


def test_grok_version_probe_handles_timeout() -> None:
    from free_claude_code.cli.launchers import grok

    with patch.object(
        grok.subprocess,
        "run",
        side_effect=subprocess.TimeoutExpired("grok", 5),
    ):
        assert grok.grok_binary_version("resolved-grok") is None


@pytest.mark.parametrize("version", [(1, 0, 4), None])
def test_incompatible_grok_fails_before_settings(
    version: tuple[int, int, int] | None,
) -> None:
    from free_claude_code.cli.launchers import grok

    with (
        patch.object(grok, "resolve_client_binary", return_value="resolved-grok"),
        patch.object(grok, "grok_binary_version", return_value=version),
        patch.object(grok, "get_settings") as get_settings,
        pytest.raises(SystemExit) as exc_info,
    ):
        grok.launch([])

    assert exc_info.value.code == 1
    get_settings.assert_not_called()


def test_native_passthrough_needs_no_proxy_or_version() -> None:
    from free_claude_code.cli.launchers import grok

    with (
        patch.object(grok, "resolve_client_binary", return_value="resolved-grok"),
        patch.object(grok, "get_settings") as get_settings,
        patch.object(grok, "require_compatible_grok") as require_version,
        patch.object(grok, "run_client_process") as run_client_process,
    ):
        grok.launch(["plugin", "list"])

    assert run_client_process.call_args.kwargs["command"] == [
        "resolved-grok",
        "plugin",
        "list",
    ]
    get_settings.assert_not_called()
    require_version.assert_not_called()


def test_routed_launch_uses_catalog_model_and_injects_safety_flags() -> None:
    from free_claude_code.cli.launchers import grok

    with (
        patch.object(grok, "resolve_client_binary", return_value="resolved-grok"),
        patch.object(grok, "require_compatible_grok"),
        patch.object(grok, "get_settings", return_value=_settings()),
        patch.object(grok, "preflight_proxy", return_value=None),
        patch.object(
            grok, "fetch_proxy_models_response", return_value=_models_payload()
        ) as fetch_models,
        patch.object(grok, "run_client_process") as run_client_process,
    ):
        grok.launch(["--model", "nvidia_nim/vendor/model", "-p", "hello"])

    fetch_models.assert_called_once_with("http://127.0.0.1:9191", "proxy-token")
    call = run_client_process.call_args.kwargs
    assert call["command"] == [
        "resolved-grok",
        "--no-leader",
        "--disable-web-search",
        "--model",
        "nvidia_nim/vendor/model",
        "-p",
        "hello",
    ]
    assert call["env"]["GROK_DEFAULT_MODEL"] == "nvidia_nim/vendor/model"


def test_agent_stdio_inserts_flags_in_their_parser_scopes() -> None:
    from free_claude_code.cli.launchers import grok

    with (
        patch.object(grok, "resolve_client_binary", return_value="resolved-grok"),
        patch.object(grok, "require_compatible_grok"),
        patch.object(grok, "get_settings", return_value=_settings()),
        patch.object(grok, "preflight_proxy", return_value=None),
        patch.object(
            grok, "fetch_proxy_models_response", return_value=_models_payload()
        ),
        patch.object(grok, "run_client_process") as run_client_process,
    ):
        grok.launch(["--cwd", "agent", "agent", "--always-approve", "stdio"])

    assert run_client_process.call_args.kwargs["command"] == [
        "resolved-grok",
        "--disable-web-search",
        "--cwd",
        "agent",
        "agent",
        "--no-leader",
        "--always-approve",
        "stdio",
    ]


def test_grok_environment_preserves_native_state_and_replaces_owned_route() -> None:
    from free_claude_code.cli.launchers.grok import build_grok_launcher_env

    env = build_grok_launcher_env(
        models=_models(),
        selected_model="nvidia_nim/vendor/model",
        proxy_root_url="http://127.0.0.1:9191",
        auth_token="proxy-token",
        base_env={
            "PATH": "keep",
            "GROK_HOME": "native-home",
            "GROK_TELEMETRY": "native-value",
            "FCC_GROK_STALE": "remove",
            "GROK_XAI_API_BASE_URL": "https://api.x.ai/v1",
            "GROK_MODELS_BASE_URL": "https://native.example/v1",
            "GROK_MODELS_LIST_URL": "https://native.example/models",
            "GROK_DEFAULT_MODEL": "native-model",
            "GROK_IMAGE_DESCRIPTION_MODEL": "native-model",
            "GROK_PROMPT_SUGGESTIONS_MODEL": "native-model",
            "GROK_SESSION_SUMMARY_MODEL": "native-model",
            "GROK_WEB_SEARCH_MODEL": "native-model",
            "XAI_API_KEY": "native-key",
            "GROK_CODE_XAI_API_KEY": "legacy-key",
            "NO_PROXY": "example.com",
        },
    )

    assert env["PATH"] == "keep"
    assert env["GROK_HOME"] == "native-home"
    assert env["GROK_TELEMETRY"] == "native-value"
    assert "FCC_GROK_STALE" not in env
    assert env["GROK_XAI_API_BASE_URL"] == "http://127.0.0.1:9191/v1"
    assert env["GROK_MODELS_BASE_URL"] == "http://127.0.0.1:9191/v1"
    assert (
        env["GROK_MODELS_LIST_URL"] == "http://127.0.0.1:9191/v1/models?view=responses"
    )
    assert env["GROK_DEFAULT_MODEL"] == "nvidia_nim/vendor/model"
    assert env["GROK_IMAGE_DESCRIPTION_MODEL"] == "nvidia_nim/vendor/model"
    assert env["GROK_PROMPT_SUGGESTIONS_MODEL"] == "nvidia_nim/vendor/model"
    assert env["GROK_SESSION_SUMMARY_MODEL"] == "nvidia_nim/vendor/model"
    assert env["GROK_WEB_SEARCH_MODEL"] == "nvidia_nim/vendor/model"
    assert env["XAI_API_KEY"] == "proxy-token"
    assert "GROK_CODE_XAI_API_KEY" not in env
    assert json.loads(env["GROK_CONFIG"]) == {
        "models": {
            "allowed_models": [
                "nvidia_nim/vendor/model",
                "claude-3-freecc-no-thinking/open_router/plain-model",
            ]
        },
        "shell_environment_policy": {"ignore_default_excludes": False},
    }
    assert "proxy-token" not in env["GROK_CONFIG"]
    for key in ("NO_PROXY", "no_proxy"):
        assert "example.com" in env[key]
        assert "127.0.0.1" in env[key]


@pytest.mark.parametrize("key", ["GROK_CONFIG", "GROK_CONFIG_PATH"])
def test_existing_process_overlay_is_rejected(
    key: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from free_claude_code.cli.launchers import grok

    monkeypatch.setenv(key, "owner-config")
    with (
        patch.object(grok, "resolve_client_binary", return_value="resolved-grok"),
        patch.object(grok, "require_compatible_grok") as require_version,
        pytest.raises(SystemExit) as exc_info,
    ):
        grok.launch([])

    assert exc_info.value.code == 2
    require_version.assert_not_called()


@pytest.mark.parametrize(
    "argv",
    [
        ["--model", "missing/model"],
        ["--model=missing/model"],
        ["-mmissing/model"],
        ["--model"],
        ["--model", "nvidia_nim/vendor/model", "-m", "open_router/other"],
    ],
)
def test_invalid_model_override_never_starts_child(argv: list[str]) -> None:
    from free_claude_code.cli.launchers import grok

    with (
        patch.object(grok, "resolve_client_binary", return_value="resolved-grok"),
        patch.object(grok, "require_compatible_grok"),
        patch.object(grok, "get_settings", return_value=_settings()),
        patch.object(grok, "preflight_proxy", return_value=None),
        patch.object(
            grok, "fetch_proxy_models_response", return_value=_models_payload()
        ),
        patch.object(grok, "run_client_process") as run_client_process,
        pytest.raises(SystemExit) as exc_info,
    ):
        grok.launch(argv)

    assert exc_info.value.code == 2
    run_client_process.assert_not_called()


def test_prompt_value_named_model_flag_is_not_parsed_as_an_override() -> None:
    from free_claude_code.cli.launchers import grok

    with (
        patch.object(grok, "resolve_client_binary", return_value="resolved-grok"),
        patch.object(grok, "require_compatible_grok"),
        patch.object(grok, "get_settings", return_value=_settings()),
        patch.object(grok, "preflight_proxy", return_value=None),
        patch.object(
            grok, "fetch_proxy_models_response", return_value=_models_payload()
        ),
        patch.object(grok, "run_client_process") as run_client_process,
    ):
        grok.launch(["--single=--model"])

    assert (
        run_client_process.call_args.kwargs["env"]["GROK_DEFAULT_MODEL"]
        == "nvidia_nim/vendor/model"
    )


def test_proxy_or_catalog_failure_never_starts_grok() -> None:
    from free_claude_code.cli.launchers import grok

    with (
        patch.object(grok, "resolve_client_binary", return_value="resolved-grok"),
        patch.object(grok, "require_compatible_grok"),
        patch.object(grok, "get_settings", return_value=_settings()),
        patch.object(grok, "preflight_proxy", return_value="connection refused"),
        patch.object(grok, "fetch_proxy_models_response") as fetch_models,
        patch.object(grok, "run_client_process") as run_client_process,
        pytest.raises(SystemExit) as exc_info,
    ):
        grok.launch([])

    assert exc_info.value.code == 1
    fetch_models.assert_not_called()
    run_client_process.assert_not_called()


def test_child_exit_code_is_mirrored_by_shared_process_owner() -> None:
    from free_claude_code.cli.launchers import grok

    with (
        patch.object(grok, "resolve_client_binary", return_value="resolved-grok"),
        patch.object(grok, "require_compatible_grok"),
        patch.object(grok, "get_settings", return_value=_settings()),
        patch.object(grok, "preflight_proxy", return_value=None),
        patch.object(
            grok, "fetch_proxy_models_response", return_value=_models_payload()
        ),
        patch.object(grok, "run_client_process", side_effect=SystemExit(23)),
        pytest.raises(SystemExit) as exc_info,
    ):
        grok.launch([])

    assert exc_info.value.code == 23
