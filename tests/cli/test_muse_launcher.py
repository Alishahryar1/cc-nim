"""Contract tests for the installed `fcc-muse` launcher."""

import os
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from free_claude_code.config.settings import Settings
from free_claude_code.core.json_types import JsonObject


def _settings(*, token: str = "proxy-token") -> Settings:
    settings = Settings(
        host="0.0.0.0",
        port=9191,
        proxy_auth_enabled=False,
        proxy_auth_token=token or "placeholder-token",
        model="nvidia_nim/test-model",
    )
    return settings.model_copy(update={"proxy_auth_token": token})


def _models_payload() -> JsonObject:
    return {
        "object": "list",
        "data": [
            {
                "id": "nvidia_nim/vendor/model",
                "provider_model_ref": "nvidia_nim/vendor/model",
                "display_name": "Nested model",
            },
            {
                "id": "claude-3-freecc-no-thinking/open_router/plain-model",
                "provider_model_ref": "open_router/plain-model",
                "display_name": "No-thinking model",
            },
        ],
    }


@pytest.mark.parametrize(
    ("argv", "str_mode", "command_index"),
    [
        ([], "tui", None),
        (["fix the bug"], "tui", None),
        (["--workspace", "repo", "fix the bug"], "tui", None),
        (["exec", "--json", "fix the bug"], "exec", 0),
        (["resume"], "resume", 0),
        (["resume", "--last"], "resume", 0),
        (["--workspace", "repo", "resume", "session-id"], "resume", 2),
        (["-w", "resume", "--last"], "resume", 1),
        (["--worktree", "existing", "resume"], "resume", 2),
        (["--worktree=create", "resume"], "resume", 1),
        (["--", "resume"], "tui", None),
    ],
)
def test_muse_invocation_classification(
    argv: list[str], str_mode: str, command_index: int | None
) -> None:
    from free_claude_code.cli.launchers.muse import (
        MuseInvocationMode,
        classify_muse_invocation,
    )

    invocation = classify_muse_invocation(argv)

    assert invocation.mode is MuseInvocationMode(str_mode)
    assert invocation.command_index == command_index


@pytest.mark.parametrize(
    "argv",
    [
        ["--help"],
        ["exec", "--help"],
        ["resume", "--help"],
        ["--version"],
        ["auth", "status"],
        ["login"],
        ["logout"],
        ["config", "validate"],
        ["export", "session-id"],
        ["trace", "session-id"],
        ["skills", "list"],
        ["sandbox", "status"],
        ["session-message", "send"],
        ["init"],
    ],
)
def test_management_surfaces_are_native_passthrough(argv: list[str]) -> None:
    from free_claude_code.cli.launchers import muse

    with (
        patch.object(muse, "resolve_client_binary", return_value="resolved-muse"),
        patch.object(muse, "require_compatible_muse") as require_version,
        patch.object(muse, "get_settings") as get_settings,
        patch.object(muse, "run_client_process") as run_client_process,
    ):
        muse.launch(argv)

    assert run_client_process.call_args.kwargs["command"] == [
        "resolved-muse",
        *argv,
    ]
    assert run_client_process.call_args.kwargs["env"] is os.environ
    require_version.assert_not_called()
    get_settings.assert_not_called()


@pytest.mark.parametrize(
    "argv",
    [
        ["--provider", "meta", "prompt"],
        ["--provider=meta", "prompt"],
        ["--base-url", "https://example.test/v1", "prompt"],
        ["--base-url=https://example.test/v1", "prompt"],
        ["resume", "--provider", "meta"],
        ["exec", "--api-key-stdin", "prompt"],
    ],
)
def test_routed_invocations_reject_caller_owned_route(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    from free_claude_code.cli.launchers import muse

    with (
        patch.object(muse, "resolve_client_binary", return_value="resolved-muse"),
        patch.object(muse, "require_compatible_muse") as require_version,
        pytest.raises(SystemExit) as exc_info,
    ):
        muse.launch(argv)

    assert exc_info.value.code == 2
    assert "ordinary muse" in capsys.readouterr().err
    require_version.assert_not_called()


def test_route_like_prompt_content_after_separator_is_not_rejected() -> None:
    from free_claude_code.cli.launchers import muse

    with (
        patch.object(muse, "resolve_client_binary", return_value="resolved-muse"),
        patch.object(muse, "require_compatible_muse"),
        patch.object(muse, "get_settings", return_value=_settings()),
        patch.object(muse, "preflight_proxy", return_value=None),
        patch.object(
            muse, "fetch_proxy_models_response", return_value=_models_payload()
        ),
        patch.object(muse, "run_client_process") as run_client_process,
    ):
        muse.launch(["exec", "--", "--provider", "native", "--model", "words"])

    assert run_client_process.call_args.kwargs["command"] == [
        "resolved-muse",
        "exec",
        "--provider",
        "meta",
        "--base-url",
        "http://127.0.0.1:9191/v1",
        "--model",
        "nvidia_nim/vendor/model",
        "--",
        "--provider",
        "native",
        "--model",
        "words",
    ]


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (
            ["fix the bug"],
            [
                "--provider",
                "meta",
                "--base-url",
                "http://127.0.0.1:9191/v1",
                "--model",
                "nvidia_nim/vendor/model",
                "fix the bug",
            ],
        ),
        (
            ["exec", "--json", "fix the bug"],
            [
                "exec",
                "--provider",
                "meta",
                "--base-url",
                "http://127.0.0.1:9191/v1",
                "--model",
                "nvidia_nim/vendor/model",
                "--json",
                "fix the bug",
            ],
        ),
        (
            ["resume", "--last"],
            [
                "resume",
                "--provider",
                "meta",
                "--base-url",
                "http://127.0.0.1:9191/v1",
                "--model",
                "nvidia_nim/vendor/model",
                "--last",
            ],
        ),
        (
            ["--workspace", "repo", "resume", "session-id"],
            [
                "--workspace",
                "repo",
                "resume",
                "--provider",
                "meta",
                "--base-url",
                "http://127.0.0.1:9191/v1",
                "--model",
                "nvidia_nim/vendor/model",
                "session-id",
            ],
        ),
    ],
)
def test_routed_launch_places_flags_in_muse_grammar(
    argv: list[str], expected: list[str]
) -> None:
    from free_claude_code.cli.launchers import muse

    with (
        patch.object(muse, "resolve_client_binary", return_value="resolved-muse"),
        patch.object(muse, "require_compatible_muse"),
        patch.object(muse, "get_settings", return_value=_settings()),
        patch.object(muse, "preflight_proxy", return_value=None),
        patch.object(
            muse, "fetch_proxy_models_response", return_value=_models_payload()
        ) as fetch_models,
        patch.object(muse, "run_client_process") as run_client_process,
    ):
        muse.launch(argv)

    assert run_client_process.call_args.kwargs["command"] == [
        "resolved-muse",
        *expected,
    ]
    fetch_models.assert_called_once_with("http://127.0.0.1:9191", "proxy-token")


@pytest.mark.parametrize(
    "model_args",
    [
        ["--model", "claude-3-freecc-no-thinking/open_router/plain-model"],
        ["--model=claude-3-freecc-no-thinking/open_router/plain-model"],
    ],
)
def test_explicit_model_is_validated_and_reinserted_once(
    model_args: list[str],
) -> None:
    from free_claude_code.cli.launchers import muse

    with (
        patch.object(muse, "resolve_client_binary", return_value="resolved-muse"),
        patch.object(muse, "require_compatible_muse"),
        patch.object(muse, "get_settings", return_value=_settings()),
        patch.object(muse, "preflight_proxy", return_value=None),
        patch.object(
            muse, "fetch_proxy_models_response", return_value=_models_payload()
        ),
        patch.object(muse, "run_client_process") as run_client_process,
    ):
        muse.launch(["exec", *model_args, "prompt"])

    command = run_client_process.call_args.kwargs["command"]
    assert command == [
        "resolved-muse",
        "exec",
        "--provider",
        "meta",
        "--base-url",
        "http://127.0.0.1:9191/v1",
        "--model",
        "claude-3-freecc-no-thinking/open_router/plain-model",
        "prompt",
    ]
    assert command.count("--model") == 1


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["exec", "--model"], "requires a model value"),
        (["exec", "--model="], "requires a model value"),
        (["exec", "--model", "missing/model", "prompt"], "not in the current"),
        (
            [
                "exec",
                "--model",
                "nvidia_nim/vendor/model",
                "--model=nvidia_nim/vendor/model",
            ],
            "multiple --model",
        ),
    ],
)
def test_invalid_model_selection_fails_closed(
    argv: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    from free_claude_code.cli.launchers import muse

    with (
        patch.object(muse, "resolve_client_binary", return_value="resolved-muse"),
        patch.object(muse, "require_compatible_muse"),
        patch.object(muse, "get_settings", return_value=_settings()),
        patch.object(muse, "preflight_proxy", return_value=None),
        patch.object(
            muse, "fetch_proxy_models_response", return_value=_models_payload()
        ),
        patch.object(muse, "run_client_process") as run_client_process,
        pytest.raises(SystemExit) as exc_info,
    ):
        muse.launch(argv)

    assert exc_info.value.code == 2
    assert message in capsys.readouterr().err
    run_client_process.assert_not_called()


@pytest.mark.parametrize(
    ("output", "return_code", "expected"),
    [
        ("Muse Code 0.2.1 (0.2.1-R1215.1)\n", 0, (0, 2, 1)),
        ("Muse Code 1.4.2 (1.4.2-R99.3)\n", 0, (1, 4, 2)),
        ("Muse Code 0.2.1\n", 0, (0, 2, 1)),
        ("unexpected Muse Code 0.2.1\n", 0, None),
        ("Muse Code 0.2.1 (0.2.1-R1215.1)\n", 1, None),
    ],
)
def test_muse_version_probe(
    output: str,
    return_code: int,
    expected: tuple[int, int, int] | None,
) -> None:
    from free_claude_code.cli.launchers import muse

    with patch.object(
        muse.subprocess,
        "run",
        return_value=SimpleNamespace(stdout=output, returncode=return_code),
    ):
        assert muse.muse_binary_version("resolved-muse") == expected


def test_muse_version_probe_handles_timeout() -> None:
    from free_claude_code.cli.launchers import muse

    with patch.object(
        muse.subprocess,
        "run",
        side_effect=subprocess.TimeoutExpired("muse", 5),
    ):
        assert muse.muse_binary_version("resolved-muse") is None


@pytest.mark.parametrize("version", [(0, 2, 0), None])
def test_incompatible_muse_fails_before_settings(
    version: tuple[int, int, int] | None,
) -> None:
    from free_claude_code.cli.launchers import muse

    with (
        patch.object(muse, "resolve_client_binary", return_value="resolved-muse"),
        patch.object(muse, "muse_binary_version", return_value=version),
        patch.object(muse, "get_settings") as get_settings,
        pytest.raises(SystemExit) as exc_info,
    ):
        muse.launch([])

    assert exc_info.value.code == 1
    get_settings.assert_not_called()


def test_muse_child_environment_replaces_only_route_controls() -> None:
    from free_claude_code.cli.launchers.muse import build_muse_launcher_env

    env = build_muse_launcher_env(
        proxy_root_url="http://127.0.0.1:9191",
        auth_token="proxy-token",
        base_env={
            "META_API_KEY": "native-token",
            "MUSE_MODEL": "native-model",
            "MUSE_CUSTOM_HEADERS": "Authorization: native",
            "MUSE_WWW_ROUTING": "native",
            "FCC_MUSE_OLD": "stale",
            "MUSE_NO_AUTO_UPDATE": "1",
            "HOME": "native-home",
            "HTTPS_PROXY": "http://proxy.test:8080",
            "NO_PROXY": "existing.test",
            "no_proxy": "lower.test",
        },
    )

    assert env["META_API_KEY"] == "proxy-token"
    assert "MUSE_MODEL" not in env
    assert "MUSE_CUSTOM_HEADERS" not in env
    assert "MUSE_WWW_ROUTING" not in env
    assert "FCC_MUSE_OLD" not in env
    assert env["MUSE_NO_AUTO_UPDATE"] == "1"
    assert env["HOME"] == "native-home"
    assert env["HTTPS_PROXY"] == "http://proxy.test:8080"
    assert "127.0.0.1" in env["NO_PROXY"]
    assert "127.0.0.1" in env["no_proxy"]


@pytest.mark.parametrize(
    ("token", "preflight_error", "catalog", "expected"),
    [
        ("", None, _models_payload(), "authentication token is empty"),
        ("proxy-token", "connection refused", _models_payload(), "not reachable"),
        ("proxy-token", None, {"object": "list", "data": []}, "no routable"),
        ("proxy-token", None, {"unexpected": True}, "no routable"),
    ],
)
def test_preflight_failures_never_start_muse(
    token: str,
    preflight_error: str | None,
    catalog: JsonObject,
    expected: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from free_claude_code.cli.launchers import muse

    with (
        patch.object(muse, "resolve_client_binary", return_value="resolved-muse"),
        patch.object(muse, "require_compatible_muse"),
        patch.object(muse, "get_settings", return_value=_settings(token=token)),
        patch.object(muse, "preflight_proxy", return_value=preflight_error),
        patch.object(muse, "fetch_proxy_models_response", return_value=catalog),
        patch.object(muse, "run_client_process") as run_client_process,
        pytest.raises(SystemExit) as exc_info,
    ):
        muse.launch(["prompt"])

    assert exc_info.value.code == 1
    assert expected in capsys.readouterr().err
    run_client_process.assert_not_called()
