"""Contract tests for the installed `fcc-cline` launcher."""

import json
import os
import subprocess
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from free_claude_code.cli.launchers.cline_config import (
    CLINE_PROVIDER_ID,
    build_cline_config,
)
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


def _models_payload() -> JsonObject:
    return {
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
        ]
    }


def test_cline_config_uses_responses_and_only_known_metadata() -> None:
    config = build_cline_config(
        (
            ClientModel(
                wire_slug="nvidia_nim/vendor/model",
                provider_model_ref="nvidia_nim/vendor/model",
                display_name="Nested model",
                allows_reasoning=True,
            ),
            ClientModel(
                wire_slug="claude-3-freecc-no-thinking/open_router/plain-model",
                provider_model_ref="open_router/plain-model",
                display_name="No-thinking model",
                allows_reasoning=False,
            ),
        ),
        proxy_root_url="http://127.0.0.1:9191/",
        auth_token="proxy-token",
        now=datetime(2026, 8, 15, 12, 30, tzinfo=UTC),
    )

    assert config.providers == {
        "version": 1,
        "modes": {},
        "lastUsedProvider": "openai-native",
        "providers": {
            "openai-native": {
                "settings": {
                    "provider": "openai-native",
                    "apiKey": "proxy-token",
                    "model": "nvidia_nim/vendor/model",
                    "protocol": "openai-responses",
                    "baseUrl": "http://127.0.0.1:9191/v1",
                    "capabilities": ["streaming", "tools"],
                },
                "updatedAt": "2026-08-15T12:30:00Z",
                "tokenSource": "manual",
            }
        },
    }
    assert config.models == {
        "version": 1,
        "providers": {
            "openai-native": {
                "provider": {
                    "name": "Free Claude Code",
                    "baseUrl": "http://127.0.0.1:9191/v1",
                    "defaultModelId": "nvidia_nim/vendor/model",
                    "protocol": "openai-responses",
                    "client": "openai",
                    "capabilities": ["streaming", "tools"],
                },
                "models": {
                    "nvidia_nim/vendor/model": {
                        "name": "Nested model",
                        "capabilities": ["streaming", "tools", "reasoning"],
                        "supportsReasoning": True,
                        "apiFormat": "openai-responses",
                    },
                    "claude-3-freecc-no-thinking/open_router/plain-model": {
                        "name": "No-thinking model",
                        "capabilities": ["streaming", "tools"],
                        "supportsReasoning": False,
                        "apiFormat": "openai-responses",
                    },
                },
            }
        },
    }
    serialized = json.dumps(config.providers | config.models)
    assert "contextWindow" not in serialized
    assert "maxTokens" not in serialized
    assert "reasoningEffort" not in serialized
    assert "proxy-token" not in repr(config)


def test_cline_config_rejects_empty_model_catalog() -> None:
    with pytest.raises(ValueError, match="at least one"):
        build_cline_config(
            (),
            proxy_root_url="http://127.0.0.1:9191",
            auth_token="proxy-token",
        )


@pytest.mark.parametrize(
    "argv",
    [
        ["--help"],
        ["--verbose", "--version"],
        ["auth", "login"],
        ["--config", "elsewhere", "config"],
        ["history"],
        ["h"],
        ["plugin", "list"],
        ["skill", "list"],
        ["mcp", "list"],
        ["doctor"],
        ["update"],
        ["version"],
    ],
)
def test_cline_management_commands_are_native_passthrough(argv: list[str]) -> None:
    from free_claude_code.cli.launchers import cline

    assert cline.is_cline_passthrough(argv)
    with (
        patch.object(cline, "resolve_client_binary", return_value="resolved-cline"),
        patch.object(cline, "get_settings") as get_settings,
        patch.object(cline, "require_compatible_cline") as require_version,
        patch.object(cline, "run_client_process") as run_client_process,
    ):
        cline.launch(argv)

    assert run_client_process.call_args.kwargs["command"] == [
        "resolved-cline",
        *argv,
    ]
    get_settings.assert_not_called()
    require_version.assert_not_called()


@pytest.mark.parametrize(
    "argv",
    [
        ["connect", "discord"],
        ["--cwd", "workspace", "schedule", "list"],
        ["hub", "start"],
        ["dashboard"],
        ["hook"],
        ["kanban"],
        ["--zen", "prompt"],
        ["-z", "prompt"],
        ["--kanban"],
    ],
)
def test_cline_detached_surfaces_are_rejected_before_fcc_calls(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    from free_claude_code.cli.launchers import cline

    with (
        patch.object(cline, "resolve_client_binary", return_value="resolved-cline"),
        patch.object(cline, "get_settings") as get_settings,
        patch.object(cline, "require_compatible_cline") as require_version,
        patch.object(cline, "run_client_process") as run_client_process,
        pytest.raises(SystemExit) as exc_info,
    ):
        cline.launch(argv)

    assert exc_info.value.code == 2
    assert "attached terminal sessions only" in capsys.readouterr().err
    get_settings.assert_not_called()
    require_version.assert_not_called()
    run_client_process.assert_not_called()


@pytest.mark.parametrize(
    "argv",
    [
        ["-P", "openrouter", "prompt"],
        ["-Popenrouter", "prompt"],
        ["--provider", "openrouter", "prompt"],
        ["--provider=openrouter", "prompt"],
        ["-k", "secret", "prompt"],
        ["-ksecret", "prompt"],
        ["--key", "secret", "prompt"],
        ["--key=secret", "prompt"],
    ],
)
def test_cline_rejects_caller_owned_routing(argv: list[str]) -> None:
    from free_claude_code.cli.launchers import cline

    with (
        patch.object(cline, "resolve_client_binary", return_value="resolved-cline"),
        patch.object(cline, "require_compatible_cline") as require_version,
        pytest.raises(SystemExit) as exc_info,
    ):
        cline.launch(argv)

    assert exc_info.value.code == 2
    require_version.assert_not_called()


@pytest.mark.parametrize(
    ("argv", "sandbox_env"),
    [
        (["--data-dir", "isolated", "prompt"], None),
        (["--data-dir=isolated", "prompt"], None),
        (["prompt"], "1"),
    ],
)
def test_cline_rejects_sandbox_overrides_that_replace_ephemeral_credentials(
    argv: list[str],
    sandbox_env: str | None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from free_claude_code.cli.launchers import cline

    if sandbox_env is None:
        monkeypatch.delenv("CLINE_SANDBOX", raising=False)
    else:
        monkeypatch.setenv("CLINE_SANDBOX", sandbox_env)
    with (
        patch.object(cline, "resolve_client_binary", return_value="resolved-cline"),
        patch.object(cline, "require_compatible_cline") as require_version,
        patch.object(cline, "run_client_process") as run_client_process,
        pytest.raises(SystemExit) as exc_info,
    ):
        cline.launch(argv)

    assert exc_info.value.code == 2
    assert "replaces the ephemeral FCC provider file" in capsys.readouterr().err
    require_version.assert_not_called()
    run_client_process.assert_not_called()


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        ([], None),
        (["prompt"], "prompt"),
        (["--cwd", "workspace", "prompt"], "prompt"),
        (["--cwd=workspace", "prompt"], "prompt"),
        (["-cworkspace", "prompt"], "prompt"),
        (["--retries", "4", "prompt"], "prompt"),
        (["--retries=4", "prompt"], "prompt"),
        (["--verbose", "doctor"], "doctor"),
        (["--unknown", "hub"], None),
        (["--", "hub"], None),
    ],
)
def test_cline_root_positional_scanner(argv: list[str], expected: str | None) -> None:
    from free_claude_code.cli.launchers.cline import first_root_positional

    assert first_root_positional(argv) == expected


@pytest.mark.parametrize(
    ("output", "return_code", "expected"),
    [
        ("3.0.55\n", 0, (3, 0, 55)),
        ("cline version 3.1.0\n", 0, (3, 1, 0)),
        ("v3.1.0+release.1\n", 0, (3, 1, 0)),
        ("3.1.0-beta.1\n", 0, None),
        ("unexpected 3.1.0\n", 0, None),
        ("3.1.0\n", 1, None),
    ],
)
def test_cline_version_probe(
    output: str,
    return_code: int,
    expected: tuple[int, int, int] | None,
) -> None:
    from free_claude_code.cli.launchers import cline

    with patch.object(
        cline.subprocess,
        "run",
        return_value=SimpleNamespace(stdout=output, returncode=return_code),
    ):
        assert cline.cline_binary_version("resolved-cline") == expected


def test_cline_version_probe_handles_timeout() -> None:
    from free_claude_code.cli.launchers import cline

    with patch.object(
        cline.subprocess,
        "run",
        side_effect=subprocess.TimeoutExpired("cline", 5),
    ):
        assert cline.cline_binary_version("resolved-cline") is None


def test_cline_rejects_incompatible_version_with_update_hint(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from free_claude_code.cli.launchers import cline

    with (
        patch.object(cline, "cline_binary_version", return_value=(3, 0, 54)),
        pytest.raises(SystemExit) as exc_info,
    ):
        cline.require_compatible_cline("resolved-cline")

    assert exc_info.value.code == 126
    captured = capsys.readouterr()
    assert "at least version 3.0.55" in captured.err
    assert "cline update" in captured.err


def test_cline_child_env_is_process_local_and_preserves_native_state(
    tmp_path: Path,
) -> None:
    from free_claude_code.cli.launchers.cline import build_cline_launcher_env

    providers_path = tmp_path / "settings" / "providers.json"
    env = build_cline_launcher_env(
        proxy_root_url="http://127.0.0.1:9191",
        providers_path=providers_path,
        base_env={
            "PATH": "keep",
            "CLINE_PROVIDER_SETTINGS_PATH": "stale-provider-file",
            "CLINE_SESSION_BACKEND_MODE": "hub",
            "CLINE_DATA_DIR": "keep-user-state",
            "CLINE_CONFIG_DIR": "keep-user-config",
            "HTTP_PROXY": "http://proxy.example",
        },
    )

    assert env["PATH"] == "keep"
    assert env["CLINE_PROVIDER_SETTINGS_PATH"] == str(providers_path)
    assert env["CLINE_SESSION_BACKEND_MODE"] == "local"
    assert env["CLINE_DATA_DIR"] == "keep-user-state"
    assert env["CLINE_CONFIG_DIR"] == "keep-user-config"
    assert env["HTTP_PROXY"] == "http://proxy.example"
    assert env["NO_PROXY"] == "127.0.0.1,localhost,::1"
    assert env["no_proxy"] == env["NO_PROXY"]


def test_cline_launch_uses_private_ephemeral_files_and_preserves_model_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from free_claude_code.cli.launchers import cline

    monkeypatch.setenv("CLINE_DATA_DIR", "keep-native-state")
    observed_providers_path: Path | None = None
    observed_models_path: Path | None = None

    def observe_process(
        *,
        command: list[str],
        env: Mapping[str, str],
        binary_name: str,
        display_name: str,
        install_hint: str,
    ) -> None:
        nonlocal observed_providers_path, observed_models_path
        del binary_name, display_name, install_hint
        assert command == [
            "resolved-cline",
            "--provider",
            CLINE_PROVIDER_ID,
            "--model",
            "open_router/chosen/model",
            "hello",
        ]
        observed_providers_path = Path(env["CLINE_PROVIDER_SETTINGS_PATH"])
        observed_models_path = observed_providers_path.with_name("models.json")
        providers = json.loads(observed_providers_path.read_text(encoding="utf-8"))
        models = json.loads(observed_models_path.read_text(encoding="utf-8"))
        settings = providers["providers"][CLINE_PROVIDER_ID]["settings"]
        assert settings["apiKey"] == "proxy-token"
        assert settings["protocol"] == "openai-responses"
        assert settings["baseUrl"] == "http://127.0.0.1:9191/v1"
        assert (
            "nvidia_nim/vendor/model"
            in models["providers"][CLINE_PROVIDER_ID]["models"]
        )
        assert env["CLINE_SESSION_BACKEND_MODE"] == "local"
        assert env["CLINE_DATA_DIR"] == "keep-native-state"

    with (
        patch.object(cline, "resolve_client_binary", return_value="resolved-cline"),
        patch.object(cline, "require_compatible_cline"),
        patch.object(cline, "get_settings", return_value=_settings()),
        patch.object(cline, "preflight_proxy", return_value=None),
        patch.object(
            cline,
            "fetch_proxy_models_response",
            return_value=_models_payload(),
        ) as fetch_models,
        patch.object(cline, "run_client_process", side_effect=observe_process),
    ):
        cline.launch(["--model", "open_router/chosen/model", "hello"])

    fetch_models.assert_called_once_with("http://127.0.0.1:9191", "proxy-token")
    assert observed_providers_path is not None
    assert observed_models_path is not None
    assert not observed_providers_path.exists()
    assert not observed_models_path.exists()


def test_cline_fails_closed_when_proxy_is_unreachable(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from free_claude_code.cli.launchers import cline

    with (
        patch.object(cline, "resolve_client_binary", return_value="resolved-cline"),
        patch.object(cline, "require_compatible_cline"),
        patch.object(cline, "get_settings", return_value=_settings()),
        patch.object(cline, "preflight_proxy", return_value="connection refused"),
        patch.object(cline, "fetch_proxy_models_response") as fetch_models,
        patch.object(cline, "run_client_process") as run_client_process,
        pytest.raises(SystemExit) as exc_info,
    ):
        cline.launch(["hello"])

    assert exc_info.value.code == 1
    assert "fcc-server" in capsys.readouterr().err
    fetch_models.assert_not_called()
    run_client_process.assert_not_called()


def test_cline_fails_closed_when_catalog_has_no_routable_models(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from free_claude_code.cli.launchers import cline

    with (
        patch.object(cline, "resolve_client_binary", return_value="resolved-cline"),
        patch.object(cline, "require_compatible_cline"),
        patch.object(cline, "get_settings", return_value=_settings()),
        patch.object(cline, "preflight_proxy", return_value=None),
        patch.object(
            cline,
            "fetch_proxy_models_response",
            return_value={"data": [{"id": "claude-opus-4-20250514"}]},
        ),
        patch.object(cline, "run_client_process") as run_client_process,
        pytest.raises(SystemExit) as exc_info,
    ):
        cline.launch(["hello"])

    assert exc_info.value.code == 1
    assert "at least one routable FCC model" in capsys.readouterr().err
    run_client_process.assert_not_called()


def test_cline_rejects_empty_proxy_token_before_network_calls(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from free_claude_code.cli.launchers import cline

    with (
        patch.object(cline, "resolve_client_binary", return_value="resolved-cline"),
        patch.object(cline, "require_compatible_cline"),
        patch.object(
            cline,
            "get_settings",
            return_value=SimpleNamespace(
                host="0.0.0.0",
                port=9191,
                proxy_auth_token="   ",
            ),
        ),
        patch.object(cline, "preflight_proxy") as preflight_proxy,
        patch.object(cline, "run_client_process") as run_client_process,
        pytest.raises(SystemExit) as exc_info,
    ):
        cline.launch([])

    assert exc_info.value.code == 1
    assert "token is empty" in capsys.readouterr().err
    preflight_proxy.assert_not_called()
    run_client_process.assert_not_called()


def test_cline_reports_temporary_config_creation_failure(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from free_claude_code.cli.launchers import cline

    with (
        patch.object(cline, "resolve_client_binary", return_value="resolved-cline"),
        patch.object(cline, "require_compatible_cline"),
        patch.object(cline, "get_settings", return_value=_settings()),
        patch.object(cline, "preflight_proxy", return_value=None),
        patch.object(
            cline, "fetch_proxy_models_response", return_value=_models_payload()
        ),
        patch.object(
            cline.tempfile,
            "TemporaryDirectory",
            side_effect=OSError("temporary directory unavailable"),
        ),
        patch.object(cline, "run_client_process") as run_client_process,
        pytest.raises(SystemExit) as exc_info,
    ):
        cline.launch(["hello"])

    assert exc_info.value.code == 1
    assert "temporary directory unavailable" in capsys.readouterr().err
    run_client_process.assert_not_called()


@pytest.mark.skipif(
    os.name == "nt", reason="POSIX file modes are not enforced on Windows"
)
def test_cline_provider_file_is_owner_only(monkeypatch: pytest.MonkeyPatch) -> None:
    from free_claude_code.cli.launchers import cline

    observed_mode: int | None = None

    def observe_process(**kwargs: object) -> None:
        nonlocal observed_mode
        env = kwargs["env"]
        assert isinstance(env, Mapping)
        observed_mode = Path(env["CLINE_PROVIDER_SETTINGS_PATH"]).stat().st_mode & 0o777

    with (
        patch.object(cline, "resolve_client_binary", return_value="resolved-cline"),
        patch.object(cline, "require_compatible_cline"),
        patch.object(cline, "get_settings", return_value=_settings()),
        patch.object(cline, "preflight_proxy", return_value=None),
        patch.object(
            cline, "fetch_proxy_models_response", return_value=_models_payload()
        ),
        patch.object(cline, "run_client_process", side_effect=observe_process),
    ):
        cline.launch(["hello"])

    assert observed_mode == 0o600
