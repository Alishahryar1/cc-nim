"""Contract tests for the installed `fcc-dsh` launcher."""

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from free_claude_code.cli.launchers.dsh import DshInvocation
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
        provider_progress_timeout=600,
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
            supports_reasoning=True,
        ),
    )


@pytest.mark.parametrize(
    ("argv", "expected_args", "patch_index"),
    [
        ([], ("web",), 1),
        (["--no-open"], ("web", "--no-open"), 1),
        (["web"], ("web",), 1),
        (["web", "--port", "3199"], ("web", "--port", "3199"), 1),
        (
            ["--profile", "web", "--no-open"],
            ("--profile", "web", "--no-open"),
            2,
        ),
        (
            ["--profile=headless", "write", "--patch", "as prompt"],
            ("--profile=headless", "write", "--patch", "as prompt"),
            1,
        ),
        (
            ["--profile", "headless", "hello"],
            ("--profile", "headless", "hello"),
            2,
        ),
        (["--dump-config"], ("web", "--dump-config"), 1),
        (
            ["--profile", "web", "--dump-config"],
            ("--profile", "web", "--dump-config"),
            3,
        ),
    ],
)
def test_dsh_routed_surface_classification(
    argv: list[str], expected_args: tuple[str, ...], patch_index: int
) -> None:
    from free_claude_code.cli.launchers.dsh import classify_dsh_invocation

    invocation = classify_dsh_invocation(argv)

    assert invocation.is_routed
    assert invocation.args == expected_args
    assert invocation.patch_index == patch_index


@pytest.mark.parametrize(
    "argv",
    [
        ["--help"],
        ["-h"],
        ["--version"],
        ["-V"],
        ["plugin", "--profile", "web", "list"],
        ["web", "--help"],
        ["--profile", "headless", "--help"],
        ["--dump-default-config"],
        ["web", "--dump-default-config"],
        ["--profile", "web", "--dump-default-config"],
    ],
)
def test_dsh_management_surfaces_are_native_passthrough(argv: list[str]) -> None:
    from free_claude_code.cli.launchers.dsh import classify_dsh_invocation

    invocation = classify_dsh_invocation(argv)

    assert not invocation.is_routed
    assert invocation.args == tuple(argv)


@pytest.mark.parametrize(
    "argv",
    [
        ["--patch", "custom.yml"],
        ["--patch=custom.yml"],
        ["web", "--patch", "custom.yml"],
        ["--profile", "headless", "--patch=custom.yml", "task"],
    ],
)
def test_dsh_caller_patch_is_rejected_before_launch(argv: list[str]) -> None:
    from free_claude_code.cli.launchers.dsh import classify_dsh_invocation

    with pytest.raises(SystemExit) as exc_info:
        classify_dsh_invocation(argv)

    assert exc_info.value.code == 2


@pytest.mark.parametrize(
    "argv",
    [
        ["--profile", "tui"],
        ["--profile=custom", "task"],
        ["--profile", ""],
        ["--profile"],
    ],
)
def test_dsh_unsupported_or_missing_profile_is_rejected(argv: list[str]) -> None:
    from free_claude_code.cli.launchers.dsh import classify_dsh_invocation

    with pytest.raises(SystemExit) as exc_info:
        classify_dsh_invocation(argv)

    assert exc_info.value.code == 2


@pytest.mark.parametrize(
    ("output", "return_code", "expected"),
    [
        ("0.1.0-rc.8\n", 0, "0.1.0-rc.8"),
        ("dsh v0.1.0-rc.8\n", 0, "0.1.0-rc.8"),
        ("0.1.0-rc.7\n", 0, "0.1.0-rc.7"),
        ("0.1.0\n", 0, "0.1.0"),
        ("not a version\n", 0, None),
        ("0.1.0-rc.8\n", 1, None),
    ],
)
def test_dsh_version_parsing(
    output: str, return_code: int, expected: str | None
) -> None:
    from free_claude_code.cli.launchers import dsh

    with patch.object(
        dsh.subprocess,
        "run",
        return_value=SimpleNamespace(returncode=return_code, stdout=output),
    ):
        assert dsh.dsh_binary_version("resolved-dsh") == expected


def test_dsh_version_timeout_is_incompatible() -> None:
    from free_claude_code.cli.launchers import dsh

    with patch.object(
        dsh.subprocess,
        "run",
        side_effect=subprocess.TimeoutExpired("dsh", 5),
    ):
        assert dsh.dsh_binary_version("resolved-dsh") is None


@pytest.mark.parametrize("version", ["0.1.0-rc.7", "0.1.0", "0.1.0-rc.9", None])
def test_dsh_requires_exact_audited_version_before_proxy(
    version: str | None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from free_claude_code.cli.launchers import dsh

    with (
        patch.object(dsh, "resolve_client_binary", return_value="resolved-dsh"),
        patch.object(dsh, "dsh_binary_version", return_value=version),
        patch.object(dsh, "get_settings") as get_settings,
        pytest.raises(SystemExit) as exc_info,
    ):
        dsh.launch([])

    assert exc_info.value.code == 126
    assert "0.1.0-rc.8" in capsys.readouterr().err
    get_settings.assert_not_called()


def test_dsh_native_passthrough_needs_no_proxy() -> None:
    from free_claude_code.cli.launchers import dsh

    with (
        patch.object(dsh, "resolve_client_binary", return_value="resolved-dsh"),
        patch.object(dsh, "get_settings") as get_settings,
        patch.object(dsh, "require_compatible_dsh") as require_version,
        patch.object(dsh, "run_client_process") as run_client_process,
    ):
        dsh.launch(["plugin", "--profile", "web", "list"])

    run_client_process.assert_called_once_with(
        command=["resolved-dsh", "plugin", "--profile", "web", "list"],
        env=os.environ,
        binary_name="dsh",
        display_name="DeepSeek Harness",
        install_hint="Install the supported DeepSeek Harness release with: "
        "npm install -g @deepseek-ai/dsh@0.1.0-rc.8",
    )
    get_settings.assert_not_called()
    require_version.assert_not_called()


def test_dsh_launch_prepares_authenticated_catalog_and_timeout() -> None:
    from free_claude_code.cli.launchers import dsh

    with (
        patch.object(dsh, "resolve_client_binary", return_value="resolved-dsh"),
        patch.object(dsh, "require_compatible_dsh"),
        patch.object(dsh, "get_settings", return_value=_settings()),
        patch.object(dsh, "preflight_proxy", return_value=None),
        patch.object(
            dsh, "fetch_proxy_models_response", return_value=_models_payload()
        ) as fetch_models,
        patch.object(dsh, "_run_with_dsh_config") as run_configured,
    ):
        dsh.launch(["--profile", "headless", "hello", "world"])

    fetch_models.assert_called_once_with("http://127.0.0.1:9191", "proxy-token")
    call = run_configured.call_args.kwargs
    assert call["binary_path"] == "resolved-dsh"
    assert call["invocation"] == DshInvocation(
        args=("--profile", "headless", "hello", "world"), patch_index=2
    )
    assert [model.wire_slug for model in call["models"]] == [
        "nvidia_nim/vendor/model",
        "claude-3-freecc-no-thinking/open_router/plain-model",
    ]
    assert call["auth_token"] == "proxy-token"
    assert call["provider_progress_timeout"] == 600


def test_dsh_proxy_failure_does_not_fetch_catalog(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from free_claude_code.cli.launchers import dsh

    with (
        patch.object(dsh, "resolve_client_binary", return_value="resolved-dsh"),
        patch.object(dsh, "require_compatible_dsh"),
        patch.object(dsh, "get_settings", return_value=_settings()),
        patch.object(dsh, "preflight_proxy", return_value="connection refused"),
        patch.object(dsh, "fetch_proxy_models_response") as fetch_models,
        pytest.raises(SystemExit) as exc_info,
    ):
        dsh.launch([])

    assert exc_info.value.code == 1
    fetch_models.assert_not_called()
    assert "fcc-server" in capsys.readouterr().err


def test_dsh_empty_proxy_token_fails_before_health_check() -> None:
    from free_claude_code.cli.launchers import dsh

    with (
        patch.object(dsh, "resolve_client_binary", return_value="resolved-dsh"),
        patch.object(dsh, "require_compatible_dsh"),
        patch.object(
            dsh,
            "get_settings",
            return_value=SimpleNamespace(proxy_auth_token="   "),
        ),
        patch.object(dsh, "preflight_proxy") as preflight,
        pytest.raises(SystemExit) as exc_info,
    ):
        dsh.launch([])

    assert exc_info.value.code == 1
    preflight.assert_not_called()


def test_dsh_catalog_failure_never_starts_child(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from free_claude_code.cli.launchers import dsh

    with (
        patch.object(dsh, "resolve_client_binary", return_value="resolved-dsh"),
        patch.object(dsh, "require_compatible_dsh"),
        patch.object(dsh, "get_settings", return_value=_settings()),
        patch.object(dsh, "preflight_proxy", return_value=None),
        patch.object(
            dsh,
            "fetch_proxy_models_response",
            side_effect=RuntimeError("catalog unavailable"),
        ),
        patch.object(dsh, "_run_with_dsh_config") as run_configured,
        pytest.raises(SystemExit) as exc_info,
    ):
        dsh.launch([])

    assert exc_info.value.code == 1
    run_configured.assert_not_called()
    assert "catalog unavailable" in capsys.readouterr().err


def test_dsh_child_environment_preserves_native_state_and_replaces_owned_keys() -> None:
    from free_claude_code.cli.launchers.dsh import build_dsh_launcher_env

    env = build_dsh_launcher_env(
        auth_token="proxy-token",
        proxy_root_url="http://127.0.0.1:9191",
        base_env={
            "PATH": "keep",
            "DSH_HOME": "native-home",
            "DEEPSEEK_API_KEY": "native-key",
            "FCC_DSH_API_KEY": "stale",
            "FCC_DSH_OTHER": "stale",
            "DSH_TELEMETRY_DISABLED": "0",
            "NO_PROXY": "example.com",
        },
    )

    assert env["PATH"] == "keep"
    assert env["DSH_HOME"] == "native-home"
    assert env["DEEPSEEK_API_KEY"] == "native-key"
    assert env["FCC_DSH_API_KEY"] == "proxy-token"
    assert "FCC_DSH_OTHER" not in env
    assert env["DSH_TELEMETRY_DISABLED"] == "1"
    for key in ("NO_PROXY", "no_proxy"):
        assert "example.com" in env[key]
        assert "127.0.0.1" in env[key]


def test_dsh_private_config_lives_for_child_and_is_removed_after_exit(
    tmp_path: Path,
) -> None:
    from free_claude_code.cli.launchers import dsh

    observed_directory: Path | None = None

    def inspect_child(**kwargs: object) -> None:
        nonlocal observed_directory
        command = kwargs["command"]
        env = kwargs["env"]
        assert isinstance(command, list)
        assert isinstance(env, dict)
        patch_index = command.index("--patch") + 1
        patch_path = Path(command[patch_index])
        observed_directory = patch_path.parent
        assert patch_path.is_file()
        assert command[:4] == [
            "resolved-dsh",
            "--profile",
            "headless",
            "--patch",
        ]
        assert command[-1] == "hello"
        patch_payload = json.loads(patch_path.read_text(encoding="utf-8"))
        settings_path = patch_path.parent / "settings.yaml"
        credentials_path = patch_path.parent / ".credentials.yaml"
        assert json.loads(settings_path.read_text(encoding="utf-8")) == {}
        assert json.loads(credentials_path.read_text(encoding="utf-8")) == {}
        settings_row = next(row for row in patch_payload if row["id"] == "settings")
        credentials_row = next(
            row for row in patch_payload if row["id"] == "credentials"
        )
        assert settings_row["config"]["path"] == str(settings_path)
        assert credentials_row["config"]["path"] == str(credentials_path)
        assert "proxy-token" not in json.dumps(patch_payload)
        assert env["FCC_DSH_API_KEY"] == "proxy-token"
        raise SystemExit(23)

    invocation = DshInvocation(
        args=("--profile", "headless", "hello"),
        patch_index=2,
    )
    with (
        patch.object(dsh.tempfile, "tempdir", str(tmp_path), create=True),
        patch.object(dsh, "run_client_process", side_effect=inspect_child),
        pytest.raises(SystemExit) as exc_info,
    ):
        dsh._run_with_dsh_config(
            binary_path="resolved-dsh",
            invocation=invocation,
            models=_models(),
            proxy_root_url="http://127.0.0.1:9191",
            auth_token="proxy-token",
            provider_progress_timeout=600,
        )

    assert exc_info.value.code == 23
    assert observed_directory is not None
    assert not observed_directory.exists()


def test_dsh_empty_catalog_fails_before_child() -> None:
    from free_claude_code.cli.launchers import dsh

    with (
        patch.object(dsh, "run_client_process") as run_client_process,
        pytest.raises(SystemExit) as exc_info,
    ):
        dsh._run_with_dsh_config(
            binary_path="resolved-dsh",
            invocation=DshInvocation(args=("web",), patch_index=1),
            models=(),
            proxy_root_url="http://127.0.0.1:9191",
            auth_token="proxy-token",
            provider_progress_timeout=600,
        )

    assert exc_info.value.code == 1
    run_client_process.assert_not_called()
