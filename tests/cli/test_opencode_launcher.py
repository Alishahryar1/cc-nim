"""OpenCode launcher contract tests."""

import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from free_claude_code.cli.launchers.model_catalog import ClientModel
from free_claude_code.cli.launchers.opencode_config import build_opencode_config
from free_claude_code.config.settings import Settings
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.model_capabilities import ModelInputModality


def _settings() -> Settings:
    return Settings(
        host="0.0.0.0",
        port=9191,
        proxy_auth_enabled=False,
        proxy_auth_token="proxy-token",
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


def test_opencode_config_uses_responses_sdk_and_only_known_metadata() -> None:
    config = build_opencode_config(
        (
            ClientModel(
                wire_slug="nvidia_nim/vendor/model",
                provider_model_ref="nvidia_nim/vendor/model",
                display_name="Nested model",
                supports_reasoning=True,
                input_modalities=frozenset(
                    {ModelInputModality.TEXT, ModelInputModality.IMAGE}
                ),
                context_window_tokens=131072,
                max_output_tokens=8192,
            ),
            ClientModel(
                wire_slug="claude-3-freecc-no-thinking/open_router/plain-model",
                provider_model_ref="open_router/plain-model",
                display_name="No-thinking model",
                supports_reasoning=False,
                input_modalities=frozenset({ModelInputModality.TEXT}),
                context_window_tokens=65536,
            ),
            ClientModel(
                wire_slug="future_provider/unknown-model",
                provider_model_ref="future_provider/unknown-model",
                display_name="Unknown model",
                supports_reasoning=None,
            ),
            ClientModel(
                wire_slug="future_provider/output-only",
                provider_model_ref="future_provider/output-only",
                display_name="Output-only model",
                supports_reasoning=None,
                max_output_tokens=4096,
            ),
        ),
        proxy_root_url="http://127.0.0.1:9191",
    )

    provider = config.file["provider"]
    assert isinstance(provider, dict)
    fcc = provider["free-claude-code"]
    assert isinstance(fcc, dict)
    assert fcc["npm"] == "@ai-sdk/openai"
    assert fcc["options"] == {
        "baseURL": "http://127.0.0.1:9191/v1",
        "apiKey": "{env:FCC_OPENCODE_API_KEY}",
    }
    assert fcc["models"] == {
        "nvidia_nim/vendor/model": {
            "name": "Nested model",
            "reasoning": True,
            "modalities": {"input": ["text", "image"]},
            "limit": {"context": 131072, "output": 8192},
        },
        "claude-3-freecc-no-thinking/open_router/plain-model": {
            "name": "No-thinking model",
            "reasoning": False,
            "modalities": {"input": ["text"]},
            "limit": {"context": 65536, "output": 0},
        },
        "future_provider/unknown-model": {
            "name": "Unknown model",
            "reasoning": True,
        },
        "future_provider/output-only": {
            "name": "Output-only model",
            "reasoning": True,
            "limit": {"context": 0, "output": 4096},
        },
    }
    assert config.overlay == {
        "provider": {
            "free-claude-code": {
                "name": "Free Claude Code",
                "npm": "@ai-sdk/openai",
                "options": {
                    "baseURL": "http://127.0.0.1:9191/v1",
                    "apiKey": "{env:FCC_OPENCODE_API_KEY}",
                },
            }
        },
        "enabled_providers": ["free-claude-code"],
        "disabled_providers": [],
        "model": "free-claude-code/nvidia_nim/vendor/model",
        "small_model": "free-claude-code/nvidia_nim/vendor/model",
    }
    serialized = json.dumps(config.file | config.overlay)
    assert "proxy-token" not in serialized
    assert "attachment" not in serialized


def test_opencode_config_rejects_empty_model_catalog() -> None:
    with pytest.raises(ValueError, match="at least one"):
        build_opencode_config((), proxy_root_url="http://127.0.0.1:9191")


def test_opencode_child_env_is_process_local_and_preserves_user_state(
    tmp_path: Path,
) -> None:
    from free_claude_code.cli.launchers.opencode import build_opencode_launcher_env

    config_path = tmp_path / "opencode.json"
    env = build_opencode_launcher_env(
        proxy_root_url="http://127.0.0.1:9191",
        auth_token="proxy-token",
        config_path=config_path,
        overlay={"model": "free-claude-code/nvidia_nim/vendor/model"},
        base_env={
            "PATH": "keep",
            "OPENCODE_CONFIG": "",
            "OPENCODE_CONFIG_CONTENT": "",
            "OPENCODE_CONFIG_DIR": "keep-user-state",
            "FCC_OPENCODE_API_KEY": "stale-token",
            "FCC_OPENCODE_OTHER": "stale-value",
            "HTTP_PROXY": "http://proxy.example",
        },
    )

    assert env["PATH"] == "keep"
    assert env["OPENCODE_CONFIG"] == str(config_path)
    assert json.loads(env["OPENCODE_CONFIG_CONTENT"]) == {
        "model": "free-claude-code/nvidia_nim/vendor/model"
    }
    assert env["OPENCODE_CONFIG_DIR"] == "keep-user-state"
    assert env["FCC_OPENCODE_API_KEY"] == "proxy-token"
    assert "FCC_OPENCODE_OTHER" not in env
    assert env["HTTP_PROXY"] == "http://proxy.example"
    assert env["NO_PROXY"] == "127.0.0.1,localhost,::1"
    assert env["no_proxy"] == env["NO_PROXY"]


@pytest.mark.parametrize(
    ("output", "return_code", "expected"),
    [
        ("1.18.18\n", 0, (1, 18, 18)),
        ("opencode version 1.19.0\n", 0, (1, 19, 0)),
        ("v1.19.0+release.1\n", 0, (1, 19, 0)),
        ("v2.0.0-beta.1\n", 0, None),
        ("unexpected text 1.19.0\n", 0, None),
        ("not-a-version\n", 0, None),
        ("1.18.18\n", 1, None),
    ],
)
def test_opencode_version_probe(
    output: str,
    return_code: int,
    expected: tuple[int, int, int] | None,
) -> None:
    from free_claude_code.cli.launchers.opencode import opencode_binary_version

    with patch(
        "free_claude_code.cli.launchers.opencode.subprocess.run",
        return_value=SimpleNamespace(stdout=output, returncode=return_code),
    ):
        assert opencode_binary_version("resolved-opencode") == expected


def test_opencode_version_probe_handles_timeout() -> None:
    from free_claude_code.cli.launchers.opencode import opencode_binary_version

    with patch(
        "free_claude_code.cli.launchers.opencode.subprocess.run",
        side_effect=subprocess.TimeoutExpired("opencode", 5),
    ):
        assert opencode_binary_version("resolved-opencode") is None


@pytest.mark.parametrize(
    "argv",
    [
        ["--help"],
        ["run", "--help"],
        ["--version"],
        ["auth", "login"],
        ["upgrade"],
        ["uninstall"],
        ["completion", "bash"],
    ],
)
def test_opencode_management_commands_pass_through_without_fcc(
    argv: list[str],
) -> None:
    from free_claude_code.cli.launchers import opencode

    with (
        patch.object(
            opencode, "resolve_client_binary", return_value="resolved-opencode"
        ),
        patch.object(opencode, "get_settings") as get_settings,
        patch.object(opencode, "preflight_proxy") as preflight_proxy,
        patch.object(opencode, "run_client_process") as run_client_process,
    ):
        opencode.launch(argv)

    run_client_process.assert_called_once()
    assert run_client_process.call_args.kwargs["command"] == [
        "resolved-opencode",
        *argv,
    ]
    get_settings.assert_not_called()
    preflight_proxy.assert_not_called()


@pytest.mark.parametrize(
    "argv",
    [[], ["models"], ["run", "hello"], ["serve"], ["web"], ["acp"]],
)
def test_opencode_inference_commands_are_never_raw_passthrough(
    argv: list[str],
) -> None:
    from free_claude_code.cli.launchers.opencode import is_opencode_passthrough

    assert not is_opencode_passthrough(argv)


def test_opencode_models_is_fcc_configured_and_temp_config_is_cleaned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from free_claude_code.cli.launchers import opencode

    monkeypatch.setenv("KEEP_ME", "yes")
    observed_path: Path | None = None
    observed_file: JsonObject | None = None
    observed_env: dict[str, str] | None = None

    def observe_process(
        *,
        command: list[str],
        env: Mapping[str, str],
        binary_name: str,
        display_name: str,
        install_hint: str,
    ) -> None:
        nonlocal observed_path, observed_file, observed_env
        del binary_name, display_name, install_hint
        observed_path = Path(env["OPENCODE_CONFIG"])
        payload = json.loads(observed_path.read_text(encoding="utf-8"))
        assert isinstance(payload, dict)
        observed_file = payload
        observed_env = dict(env)
        assert command == ["resolved-opencode", "models"]

    with (
        patch.object(
            opencode, "resolve_client_binary", return_value="resolved-opencode"
        ),
        patch.object(opencode, "require_compatible_opencode"),
        patch.object(opencode, "get_settings", return_value=_settings()),
        patch.object(opencode, "preflight_proxy", return_value=None),
        patch.object(
            opencode,
            "fetch_proxy_models_response",
            return_value=_models_payload(),
        ) as fetch_models,
        patch.object(opencode, "run_client_process", side_effect=observe_process),
    ):
        opencode.launch(["models"])

    fetch_models.assert_called_once_with("http://127.0.0.1:9191", "proxy-token")
    assert observed_path is not None
    assert not observed_path.exists()
    assert observed_file is not None
    providers = observed_file["provider"]
    assert isinstance(providers, dict)
    assert "free-claude-code" in providers
    assert observed_env is not None
    assert observed_env["FCC_OPENCODE_API_KEY"] == "proxy-token"
    assert observed_env["KEEP_ME"] == "yes"
    assert "proxy-token" not in json.dumps(observed_file)
    assert "proxy-token" not in observed_env["OPENCODE_CONFIG_CONTENT"]


@pytest.mark.parametrize("key", ["OPENCODE_CONFIG", "OPENCODE_CONFIG_CONTENT"])
def test_opencode_rejects_inherited_process_config_before_fcc_calls(
    key: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from free_claude_code.cli.launchers import opencode

    monkeypatch.setenv(key, "user-config")
    with (
        patch.object(
            opencode, "resolve_client_binary", return_value="resolved-opencode"
        ),
        patch.object(opencode, "require_compatible_opencode"),
        patch.object(opencode, "get_settings") as get_settings,
        patch.object(opencode, "preflight_proxy") as preflight_proxy,
        patch.object(opencode, "run_client_process") as run_client_process,
        pytest.raises(SystemExit) as exc_info,
    ):
        opencode.launch([])

    assert exc_info.value.code == 1
    assert key in capsys.readouterr().err
    get_settings.assert_not_called()
    preflight_proxy.assert_not_called()
    run_client_process.assert_not_called()


def test_opencode_fails_closed_when_catalog_has_no_routable_models(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from free_claude_code.cli.launchers import opencode

    with (
        patch.object(
            opencode, "resolve_client_binary", return_value="resolved-opencode"
        ),
        patch.object(opencode, "require_compatible_opencode"),
        patch.object(opencode, "get_settings", return_value=_settings()),
        patch.object(opencode, "preflight_proxy", return_value=None),
        patch.object(
            opencode,
            "fetch_proxy_models_response",
            return_value={"data": [{"id": "claude-opus-4-20250514"}]},
        ),
        patch.object(opencode, "run_client_process") as run_client_process,
        pytest.raises(SystemExit) as exc_info,
    ):
        opencode.launch([])

    assert exc_info.value.code == 1
    run_client_process.assert_not_called()
    assert "at least one routable FCC model" in capsys.readouterr().err


def test_opencode_fails_closed_when_proxy_is_unreachable(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from free_claude_code.cli.launchers import opencode

    with (
        patch.object(
            opencode, "resolve_client_binary", return_value="resolved-opencode"
        ),
        patch.object(opencode, "require_compatible_opencode"),
        patch.object(opencode, "get_settings", return_value=_settings()),
        patch.object(opencode, "preflight_proxy", return_value="connection refused"),
        patch.object(opencode, "fetch_proxy_models_response") as fetch_models,
        patch.object(opencode, "run_client_process") as run_client_process,
        pytest.raises(SystemExit) as exc_info,
    ):
        opencode.launch(["run", "hello"])

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "connection refused" in captured.err
    assert "fcc-server" in captured.err
    fetch_models.assert_not_called()
    run_client_process.assert_not_called()


def test_opencode_fails_closed_when_catalog_fetch_fails(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from free_claude_code.cli.launchers import opencode

    with (
        patch.object(
            opencode, "resolve_client_binary", return_value="resolved-opencode"
        ),
        patch.object(opencode, "require_compatible_opencode"),
        patch.object(opencode, "get_settings", return_value=_settings()),
        patch.object(opencode, "preflight_proxy", return_value=None),
        patch.object(
            opencode,
            "fetch_proxy_models_response",
            side_effect=OSError("catalog unavailable"),
        ),
        patch.object(opencode, "run_client_process") as run_client_process,
        pytest.raises(SystemExit) as exc_info,
    ):
        opencode.launch(["run", "hello"])

    assert exc_info.value.code == 1
    assert "catalog unavailable" in capsys.readouterr().err
    run_client_process.assert_not_called()


def test_opencode_reports_temporary_config_creation_failure(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from free_claude_code.cli.launchers import opencode

    with (
        patch.object(
            opencode, "resolve_client_binary", return_value="resolved-opencode"
        ),
        patch.object(opencode, "require_compatible_opencode"),
        patch.object(opencode, "get_settings", return_value=_settings()),
        patch.object(opencode, "preflight_proxy", return_value=None),
        patch.object(
            opencode,
            "fetch_proxy_models_response",
            return_value=_models_payload(),
        ),
        patch.object(
            opencode.tempfile,
            "TemporaryDirectory",
            side_effect=OSError("temporary directory unavailable"),
        ),
        patch.object(opencode, "run_client_process") as run_client_process,
        pytest.raises(SystemExit) as exc_info,
    ):
        opencode.launch(["run", "hello"])

    assert exc_info.value.code == 1
    assert "temporary directory unavailable" in capsys.readouterr().err
    run_client_process.assert_not_called()


def test_opencode_reports_temporary_config_write_failure(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from free_claude_code.cli.launchers import opencode

    with (
        patch.object(
            opencode, "resolve_client_binary", return_value="resolved-opencode"
        ),
        patch.object(opencode, "require_compatible_opencode"),
        patch.object(opencode, "get_settings", return_value=_settings()),
        patch.object(opencode, "preflight_proxy", return_value=None),
        patch.object(
            opencode,
            "fetch_proxy_models_response",
            return_value=_models_payload(),
        ),
        patch.object(
            opencode.Path,
            "write_text",
            side_effect=OSError("temporary file unavailable"),
        ),
        patch.object(opencode, "run_client_process") as run_client_process,
        pytest.raises(SystemExit) as exc_info,
    ):
        opencode.launch(["run", "hello"])

    assert exc_info.value.code == 1
    assert "temporary file unavailable" in capsys.readouterr().err
    run_client_process.assert_not_called()


def test_opencode_rejects_empty_proxy_token_before_network_calls(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from free_claude_code.cli.launchers import opencode

    with (
        patch.object(
            opencode, "resolve_client_binary", return_value="resolved-opencode"
        ),
        patch.object(opencode, "require_compatible_opencode"),
        patch.object(
            opencode,
            "get_settings",
            return_value=SimpleNamespace(
                host="0.0.0.0",
                port=9191,
                proxy_auth_token="   ",
            ),
        ),
        patch.object(opencode, "preflight_proxy") as preflight_proxy,
        patch.object(opencode, "run_client_process") as run_client_process,
        pytest.raises(SystemExit) as exc_info,
    ):
        opencode.launch([])

    assert exc_info.value.code == 1
    assert "token is empty" in capsys.readouterr().err
    preflight_proxy.assert_not_called()
    run_client_process.assert_not_called()


def test_opencode_rejects_incompatible_version_with_upgrade_hint(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from free_claude_code.cli.launchers import opencode

    with (
        patch.object(opencode, "opencode_binary_version", return_value=(1, 18, 17)),
        pytest.raises(SystemExit) as exc_info,
    ):
        opencode.require_compatible_opencode("resolved-opencode")

    assert exc_info.value.code == 126
    captured = capsys.readouterr()
    assert "stable release at least version 1.18.18" in captured.err
    assert "opencode upgrade" in captured.err


@pytest.mark.parametrize("version", [(1, 18, 18), (1, 19, 0), (2, 0, 0)])
def test_opencode_accepts_stable_minimum_and_newer_versions(
    version: tuple[int, int, int],
) -> None:
    from free_claude_code.cli.launchers import opencode

    with patch.object(opencode, "opencode_binary_version", return_value=version):
        opencode.require_compatible_opencode("resolved-opencode")
