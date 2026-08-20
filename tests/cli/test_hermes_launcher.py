"""Contract tests for the installed `fcc-hermes` launcher."""

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from free_claude_code.cli.launchers.hermes_config import (
    build_hermes_managed_config,
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


def _models() -> tuple[ClientModel, ...]:
    return (
        ClientModel(
            wire_slug="nvidia_nim/vendor/model",
            provider_model_ref="nvidia_nim/vendor/model",
            display_name="Nested model",
            allows_reasoning=True,
        ),
    )


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["chat"],
        ["-z", "hello"],
        ["--oneshot=hello"],
        ["--tui"],
        ["--resume", "session-id"],
        ["-c"],
        ["-c", "session name"],
        ["--profile", "coder", "chat", "--query", "hello"],
    ],
)
def test_hermes_attached_surfaces_are_routed(argv: list[str]) -> None:
    from free_claude_code.cli.launchers import hermes

    assert not hermes.is_hermes_passthrough(argv)
    assert not hermes.detached_hermes_surface(argv)


@pytest.mark.parametrize(
    "argv",
    [
        ["--help"],
        ["chat", "--help"],
        ["--version"],
        ["setup"],
        ["auth", "list"],
        ["update"],
        ["config", "show"],
        ["model"],
        ["profile", "list"],
        ["sessions", "list"],
        ["plugin-command"],
    ],
)
def test_hermes_management_surfaces_are_native_passthrough(argv: list[str]) -> None:
    from free_claude_code.cli.launchers import hermes

    assert hermes.is_hermes_passthrough(argv)


@pytest.mark.parametrize(
    "argv",
    [
        ["acp"],
        ["rl"],
        ["gateway"],
        ["cron"],
        ["serve"],
        ["dashboard"],
        ["gui"],
        ["desktop"],
        ["proxy"],
        ["mcp", "serve"],
    ],
)
def test_hermes_detached_surfaces_are_rejected(argv: list[str]) -> None:
    from free_claude_code.cli.launchers import hermes

    assert hermes.detached_hermes_surface(argv)
    assert not hermes.is_hermes_passthrough(argv)


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["--profile", "coder", "chat"], ("--profile", "coder")),
        (["chat", "-p", "coder"], ("-p", "coder")),
        (["--profile=coder", "-z", "hello"], ("--profile=coder",)),
        (["-m", "provider/model", "chat"], ()),
        (["--", "--profile", "not-hermes"], ()),
    ],
)
def test_hermes_profile_selector_matches_upstream_preparse(
    argv: list[str], expected: tuple[str, ...]
) -> None:
    from free_claude_code.cli.launchers.hermes import hermes_profile_args

    assert hermes_profile_args(argv) == expected


@pytest.mark.parametrize(
    ("output", "return_code", "expected"),
    [
        ("Hermes Agent v0.20.4 (2026-08-19)\n", 0, (0, 20, 4)),
        ("0.21.0\n", 0, (0, 21, 0)),
        ("Hermes Agent v0.20.3\n", 0, (0, 20, 3)),
        ("not a version\n", 0, None),
        ("Hermes Agent v0.20.4\n", 1, None),
    ],
)
def test_hermes_version_parsing(
    output: str, return_code: int, expected: tuple[int, int, int] | None
) -> None:
    from free_claude_code.cli.launchers import hermes

    with patch.object(
        hermes.subprocess,
        "run",
        return_value=SimpleNamespace(returncode=return_code, stdout=output),
    ):
        assert hermes.hermes_binary_version("resolved-hermes") == expected


def test_hermes_version_timeout_is_incompatible() -> None:
    from free_claude_code.cli.launchers import hermes

    with patch.object(
        hermes.subprocess, "run", side_effect=subprocess.TimeoutExpired("hermes", 5)
    ):
        assert hermes.hermes_binary_version("resolved-hermes") is None


def test_hermes_old_version_is_rejected_before_proxy(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from free_claude_code.cli.launchers import hermes

    with (
        patch.object(hermes, "resolve_client_binary", return_value="resolved-hermes"),
        patch.object(hermes, "hermes_binary_version", return_value=(0, 20, 3)),
        patch.object(hermes, "get_settings") as get_settings,
        pytest.raises(SystemExit) as exc_info,
    ):
        hermes.launch([])

    assert exc_info.value.code == 126
    assert "at least version 0.20.4" in capsys.readouterr().err
    get_settings.assert_not_called()


def test_hermes_management_passthrough_needs_no_proxy() -> None:
    from free_claude_code.cli.launchers import hermes

    with (
        patch.object(hermes, "resolve_client_binary", return_value="resolved-hermes"),
        patch.object(hermes, "get_settings") as get_settings,
        patch.object(hermes, "require_compatible_hermes") as require_version,
        patch.object(hermes, "run_client_process") as run_client_process,
    ):
        hermes.launch(["config", "show"])

    run_client_process.assert_called_once_with(
        command=["resolved-hermes", "config", "show"],
        env=os.environ,
        binary_name="hermes",
        display_name="Hermes Agent",
        install_hint="Install Hermes Agent from: "
        "https://hermes-agent.nousresearch.com/docs/installation",
    )
    get_settings.assert_not_called()
    require_version.assert_not_called()


def test_hermes_detached_surface_fails_before_proxy(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from free_claude_code.cli.launchers import hermes

    with (
        patch.object(hermes, "resolve_client_binary", return_value="resolved-hermes"),
        patch.object(hermes, "get_settings") as get_settings,
        patch.object(hermes, "run_client_process") as run_client_process,
        pytest.raises(SystemExit) as exc_info,
    ):
        hermes.launch(["gateway"])

    assert exc_info.value.code == 2
    assert "ordinary hermes" in capsys.readouterr().err
    get_settings.assert_not_called()
    run_client_process.assert_not_called()


@pytest.mark.parametrize(
    "argv",
    [
        ["--provider", "openrouter"],
        ["--provider=openrouter"],
        ["chat", "--provider", "openrouter"],
    ],
)
def test_hermes_provider_override_is_rejected(argv: list[str]) -> None:
    from free_claude_code.cli.launchers import hermes

    with (
        patch.object(hermes, "resolve_client_binary", return_value="resolved-hermes"),
        patch.object(hermes, "get_settings") as get_settings,
        pytest.raises(SystemExit) as exc_info,
    ):
        hermes.launch(argv)

    assert exc_info.value.code == 2
    get_settings.assert_not_called()


@pytest.mark.parametrize("argv", [["--model"], ["--model="], ["-m"]])
def test_hermes_missing_model_value_is_rejected(argv: list[str]) -> None:
    from free_claude_code.cli.launchers import hermes

    with (
        patch.object(hermes, "resolve_client_binary", return_value="resolved-hermes"),
        pytest.raises(SystemExit) as exc_info,
    ):
        hermes.launch(argv)

    assert exc_info.value.code == 2


def test_hermes_duplicate_model_override_is_rejected() -> None:
    from free_claude_code.cli.launchers import hermes

    with (
        patch.object(hermes, "resolve_client_binary", return_value="resolved-hermes"),
        pytest.raises(SystemExit) as exc_info,
    ):
        hermes.launch(["-m", "one/model", "chat", "--model=two/model"])

    assert exc_info.value.code == 2


def test_hermes_launch_prepares_validated_catalog_and_selected_model() -> None:
    from free_claude_code.cli.launchers import hermes

    settings = _settings()
    with (
        patch.object(hermes, "resolve_client_binary", return_value="resolved-hermes"),
        patch.object(hermes, "require_compatible_hermes"),
        patch.object(hermes, "_system_policy_exists", return_value=False),
        patch.object(hermes, "get_settings", return_value=settings),
        patch.object(hermes, "preflight_proxy", return_value=None),
        patch.object(
            hermes, "fetch_proxy_models_response", return_value=_models_payload()
        ) as fetch_models,
        patch.object(hermes.secrets, "token_hex", return_value="a1b2c3"),
        patch.object(hermes, "_run_with_managed_config") as run_managed,
    ):
        hermes.launch(
            [
                "--profile",
                "coder",
                "chat",
                "--model",
                "nvidia_nim/vendor/model",
                "--query",
                "hello",
            ]
        )

    fetch_models.assert_called_once_with("http://127.0.0.1:9191", "proxy-token")
    call = run_managed.call_args.kwargs
    assert call["binary_path"] == "resolved-hermes"
    assert call["args"] == ["--profile", "coder", "chat", "--query", "hello"]
    assert call["profile_args"] == ("--profile", "coder")
    assert call["managed"].provider_ref == "custom:fcc-a1b2c3"
    assert call["managed"].selected_model == "nvidia_nim/vendor/model"
    assert call["auth_token"] == "proxy-token"


def test_hermes_unknown_model_fails_before_child(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from free_claude_code.cli.launchers import hermes

    with (
        patch.object(hermes, "resolve_client_binary", return_value="resolved-hermes"),
        patch.object(hermes, "require_compatible_hermes"),
        patch.object(hermes, "_system_policy_exists", return_value=False),
        patch.object(hermes, "get_settings", return_value=_settings()),
        patch.object(hermes, "preflight_proxy", return_value=None),
        patch.object(
            hermes, "fetch_proxy_models_response", return_value=_models_payload()
        ),
        patch.object(hermes, "_run_with_managed_config") as run_managed,
        pytest.raises(SystemExit) as exc_info,
    ):
        hermes.launch(["--model", "missing/model"])

    assert exc_info.value.code == 1
    run_managed.assert_not_called()
    assert "not in the FCC catalog" in capsys.readouterr().err


def test_hermes_proxy_failure_does_not_fetch_catalog(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from free_claude_code.cli.launchers import hermes

    with (
        patch.object(hermes, "resolve_client_binary", return_value="resolved-hermes"),
        patch.object(hermes, "require_compatible_hermes"),
        patch.object(hermes, "_system_policy_exists", return_value=False),
        patch.object(hermes, "get_settings", return_value=_settings()),
        patch.object(hermes, "preflight_proxy", return_value="connection refused"),
        patch.object(hermes, "fetch_proxy_models_response") as fetch_models,
        pytest.raises(SystemExit) as exc_info,
    ):
        hermes.launch([])

    assert exc_info.value.code == 1
    fetch_models.assert_not_called()
    assert "fcc-server" in capsys.readouterr().err


def test_hermes_empty_proxy_token_fails_before_health_check(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from free_claude_code.cli.launchers import hermes

    with (
        patch.object(hermes, "resolve_client_binary", return_value="resolved-hermes"),
        patch.object(hermes, "require_compatible_hermes"),
        patch.object(hermes, "_system_policy_exists", return_value=False),
        patch.object(
            hermes,
            "get_settings",
            return_value=SimpleNamespace(proxy_auth_token="   "),
        ),
        patch.object(hermes, "preflight_proxy") as preflight,
        pytest.raises(SystemExit) as exc_info,
    ):
        hermes.launch([])

    assert exc_info.value.code == 1
    preflight.assert_not_called()
    assert "authentication token is empty" in capsys.readouterr().err


def test_hermes_catalog_failure_never_starts_child(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from free_claude_code.cli.launchers import hermes

    with (
        patch.object(hermes, "resolve_client_binary", return_value="resolved-hermes"),
        patch.object(hermes, "require_compatible_hermes"),
        patch.object(hermes, "_system_policy_exists", return_value=False),
        patch.object(hermes, "get_settings", return_value=_settings()),
        patch.object(hermes, "preflight_proxy", return_value=None),
        patch.object(
            hermes,
            "fetch_proxy_models_response",
            side_effect=RuntimeError("catalog unavailable"),
        ),
        patch.object(hermes, "_run_with_managed_config") as run_managed,
        pytest.raises(SystemExit) as exc_info,
    ):
        hermes.launch([])

    assert exc_info.value.code == 1
    run_managed.assert_not_called()
    assert "catalog unavailable" in capsys.readouterr().err


def test_hermes_existing_managed_policy_is_not_shadowed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from free_claude_code.cli.launchers import hermes

    monkeypatch.setenv("HERMES_MANAGED_DIR", "administrator-policy")
    with (
        patch.object(hermes, "resolve_client_binary", return_value="resolved-hermes"),
        patch.object(hermes, "get_settings") as get_settings,
        pytest.raises(SystemExit) as exc_info,
    ):
        hermes.launch([])

    assert exc_info.value.code == 2
    get_settings.assert_not_called()
    assert "managed policy" in capsys.readouterr().err


def test_hermes_child_environment_preserves_native_state_and_replaces_owned_keys(
    tmp_path: Path,
) -> None:
    from free_claude_code.cli.launchers.hermes import build_hermes_launcher_env

    env = build_hermes_launcher_env(
        managed_directory=tmp_path,
        key_env="FCC_HERMES_FRESH",
        auth_token="proxy-token",
        proxy_root_url="http://127.0.0.1:9191",
        base_env={
            "PATH": "keep",
            "HERMES_HOME": "native-home",
            "OPENROUTER_API_KEY": "native-key",
            "FCC_HERMES_STALE": "stale",
            "HERMES_MANAGED_DIR": "stale-policy",
            "HERMES_INFERENCE_MODEL": "stale-model",
            "HERMES_INFERENCE_PROVIDER": "stale-provider",
            "NO_PROXY": "example.com",
        },
    )

    assert env["PATH"] == "keep"
    assert env["HERMES_HOME"] == "native-home"
    assert env["OPENROUTER_API_KEY"] == "native-key"
    assert env["HERMES_MANAGED_DIR"] == str(tmp_path)
    assert env["FCC_HERMES_FRESH"] == "proxy-token"
    assert "FCC_HERMES_STALE" not in env
    assert "HERMES_INFERENCE_MODEL" not in env
    assert "HERMES_INFERENCE_PROVIDER" not in env
    assert env["NO_PROXY"] == "example.com,127.0.0.1,localhost,::1"
    assert env["no_proxy"] == env["NO_PROXY"]


def test_hermes_temporary_overlay_is_verified_and_removed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from free_claude_code.cli.launchers import hermes

    monkeypatch.setenv("HERMES_HOME", "native-home")
    managed = build_hermes_managed_config(
        _models(),
        proxy_root_url="http://127.0.0.1:9191",
        nonce="abc123",
    )
    observed_directory: Path | None = None

    def fake_run_process(**kwargs: object) -> None:
        nonlocal observed_directory
        command = kwargs["command"]
        env = kwargs["env"]
        assert isinstance(command, list)
        assert isinstance(env, dict)
        assert command == [
            "resolved-hermes",
            "--provider",
            "custom:fcc-abc123",
            "--model",
            "nvidia_nim/vendor/model",
            "--profile",
            "coder",
            "-z",
            "hello",
        ]
        observed_directory = Path(env["HERMES_MANAGED_DIR"])
        config_path = observed_directory / "config.yaml"
        assert config_path.exists()
        if os.name != "nt":
            assert observed_directory.stat().st_mode & 0o777 == 0o700
            assert config_path.stat().st_mode & 0o777 == 0o600
        assert json.loads(config_path.read_text(encoding="utf-8")) == managed.config
        assert "proxy-token" not in config_path.read_text(encoding="utf-8")
        assert env["FCC_HERMES_ABC123"] == "proxy-token"
        assert env["HERMES_HOME"] == "native-home"
        raise SystemExit(0)

    with (
        patch.object(
            hermes.subprocess,
            "run",
            return_value=SimpleNamespace(
                returncode=0,
                stdout='"custom:fcc-abc123"\n',
                stderr="",
            ),
        ) as probe,
        patch.object(
            hermes, "run_client_process", side_effect=fake_run_process
        ) as run_process,
        pytest.raises(SystemExit) as exc_info,
    ):
        hermes._run_with_managed_config(
            binary_path="resolved-hermes",
            args=["--profile", "coder", "-z", "hello"],
            profile_args=["--profile", "coder"],
            managed=managed,
            proxy_root_url="http://127.0.0.1:9191",
            auth_token="proxy-token",
        )

    assert exc_info.value.code == 0
    probe.assert_called_once()
    assert probe.call_args.args[0] == [
        "resolved-hermes",
        "--profile",
        "coder",
        "config",
        "get",
        "model.provider",
        "--json",
    ]
    run_process.assert_called_once()
    assert observed_directory is not None
    assert not observed_directory.exists()


@pytest.mark.parametrize(
    "probe_result",
    [
        SimpleNamespace(returncode=1, stdout="", stderr="failure"),
        SimpleNamespace(returncode=0, stdout="not-json\n", stderr=""),
        SimpleNamespace(returncode=0, stdout='"custom:other"\n', stderr=""),
    ],
)
def test_hermes_overlay_activation_failure_never_starts_child(
    probe_result: SimpleNamespace,
) -> None:
    from free_claude_code.cli.launchers import hermes

    managed = build_hermes_managed_config(
        _models(),
        proxy_root_url="http://127.0.0.1:9191",
        nonce="abc123",
    )
    with (
        patch.object(hermes.subprocess, "run", return_value=probe_result),
        patch.object(hermes, "run_client_process") as run_process,
        pytest.raises(SystemExit) as exc_info,
    ):
        hermes._run_with_managed_config(
            binary_path="resolved-hermes",
            args=[],
            profile_args=[],
            managed=managed,
            proxy_root_url="http://127.0.0.1:9191",
            auth_token="proxy-token",
        )

    assert exc_info.value.code == 1
    run_process.assert_not_called()


def test_hermes_overlay_activation_timeout_never_starts_child() -> None:
    from free_claude_code.cli.launchers import hermes

    managed = build_hermes_managed_config(
        _models(),
        proxy_root_url="http://127.0.0.1:9191",
        nonce="abc123",
    )
    with (
        patch.object(
            hermes.subprocess,
            "run",
            side_effect=subprocess.TimeoutExpired("hermes", 15),
        ),
        patch.object(hermes, "run_client_process") as run_process,
        pytest.raises(SystemExit) as exc_info,
    ):
        hermes._run_with_managed_config(
            binary_path="resolved-hermes",
            args=[],
            profile_args=[],
            managed=managed,
            proxy_root_url="http://127.0.0.1:9191",
            auth_token="proxy-token",
        )

    assert exc_info.value.code == 1
    run_process.assert_not_called()
