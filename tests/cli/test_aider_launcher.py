"""Contracts for the installed `fcc-aider` launcher."""

import json
import os
from collections.abc import Mapping
from pathlib import Path
from unittest.mock import patch

import pytest

from free_claude_code.cli.launchers.aider_config import build_aider_config
from free_claude_code.cli.launchers.model_catalog import ClientModel
from free_claude_code.config.settings import Settings
from free_claude_code.core.json_types import JsonObject


def _settings(*, token: str = "proxy-token") -> Settings:
    return Settings.model_construct(
        host="0.0.0.0",
        port=9191,
        proxy_auth_enabled=False,
        proxy_auth_token=token,
        model="nvidia_nim/test-model",
    )


def _models() -> tuple[ClientModel, ...]:
    return (
        ClientModel(
            wire_slug="nvidia_nim/vendor/main-model",
            provider_model_ref="nvidia_nim/vendor/main-model",
            display_name="Main model",
            allows_reasoning=True,
        ),
        ClientModel(
            wire_slug="ollama_cloud/qwen3-coder:480b",
            provider_model_ref="ollama_cloud/qwen3-coder:480b",
            display_name="Colon model",
            allows_reasoning=False,
        ),
    )


def _models_payload() -> JsonObject:
    return {
        "data": [
            {
                "id": model.wire_slug,
                "provider_model_ref": model.provider_model_ref,
                "display_name": model.display_name,
            }
            for model in _models()
        ]
    }


@pytest.mark.parametrize(
    "argv",
    [
        ["-h"],
        ["--help"],
        ["--version"],
        ["--just-check-update"],
        ["--upgrade"],
        ["--update"],
        ["--install-main-branch"],
        ["--shell-completions", "bash"],
    ],
)
def test_aider_maintenance_surfaces_are_native_passthrough(
    argv: list[str],
) -> None:
    from free_claude_code.cli.launchers import aider

    with (
        patch.object(aider, "resolve_client_binary", return_value="resolved-aider"),
        patch.object(aider, "get_settings") as get_settings,
        patch.object(aider, "preflight_proxy") as preflight_proxy,
        patch.object(aider, "fetch_proxy_models_response") as fetch_models,
        patch.object(aider, "run_client_process") as run_client_process,
    ):
        aider.launch(argv)

    assert run_client_process.call_args.kwargs["command"] == [
        "resolved-aider",
        *argv,
    ]
    assert run_client_process.call_args.kwargs["env"] is os.environ
    get_settings.assert_not_called()
    preflight_proxy.assert_not_called()
    fetch_models.assert_not_called()


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["--list-models", "qwen"],
        ["--gui"],
        ["--browser"],
        ["--copy-paste"],
        ["--apply", "changes.diff"],
        ["-m", "one-shot prompt"],
        ["src/app.py"],
        ["--", "--version"],
    ],
)
def test_aider_attached_surfaces_are_not_passthrough(argv: list[str]) -> None:
    from free_claude_code.cli.launchers.aider import is_aider_passthrough

    assert not is_aider_passthrough(argv)


@pytest.mark.parametrize(
    "argv",
    [
        ["--model-settings-file", "custom.yml"],
        ["--model-settings-file=custom.yml"],
        ["--model-metadata-file", "custom.json"],
        ["--model-metadata-file=custom.json"],
    ],
)
def test_aider_rejects_caller_owned_route_files(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    from free_claude_code.cli.launchers import aider

    with pytest.raises(SystemExit) as exc_info:
        aider.reject_aider_route_file_overrides(argv)

    assert exc_info.value.code == 2
    assert "ordinary aider" in capsys.readouterr().err.lower()


def test_aider_route_file_flags_after_separator_remain_positional() -> None:
    from free_claude_code.cli.launchers.aider import reject_aider_route_file_overrides

    reject_aider_route_file_overrides(
        ["src/app.py", "--", "--model-settings-file", "fixture.yml"]
    )


def test_aider_model_options_normalize_and_preserve_other_argument_order(
    tmp_path: Path,
) -> None:
    from free_claude_code.cli.launchers.aider import normalized_aider_arguments

    result = normalized_aider_arguments(
        [
            "--verbose",
            "--weak-model=ollama_cloud/qwen3-coder:480b",
            "src/app.py",
            "--model",
            "anthropic/ollama_cloud/qwen3-coder:480b",
            "--editor-model",
            "nvidia_nim/vendor/main-model",
            "--",
            "--model",
            "after-separator",
        ],
        models=_models(),
        settings_path=tmp_path / "settings.yml",
        metadata_path=tmp_path / "metadata.json",
    )

    assert result == [
        "--verbose",
        "src/app.py",
        "--model",
        "anthropic/ollama_cloud/qwen3-coder:480b",
        "--weak-model",
        "anthropic/ollama_cloud/qwen3-coder:480b",
        "--editor-model",
        "anthropic/nvidia_nim/vendor/main-model",
        "--model-settings-file",
        str(tmp_path / "settings.yml"),
        "--model-metadata-file",
        str(tmp_path / "metadata.json"),
        "--set-env",
        "ANTHROPIC_API_KEY=fcc-local",
        "--",
        "--model",
        "after-separator",
    ]


def test_aider_default_weak_and_editor_follow_resolved_main(tmp_path: Path) -> None:
    from free_claude_code.cli.launchers.aider import normalized_aider_arguments

    result = normalized_aider_arguments(
        ["-m", "keep this one-shot", "--model", _models()[1].wire_slug],
        models=_models(),
        settings_path=tmp_path / "settings.yml",
        metadata_path=tmp_path / "metadata.json",
    )

    canonical = "anthropic/ollama_cloud/qwen3-coder:480b"
    assert result[:2] == ["-m", "keep this one-shot"]
    assert result[2:8] == [
        "--model",
        canonical,
        "--weak-model",
        canonical,
        "--editor-model",
        canonical,
    ]


def test_aider_default_main_uses_first_catalog_entry(tmp_path: Path) -> None:
    from free_claude_code.cli.launchers.aider import normalized_aider_arguments

    result = normalized_aider_arguments(
        [],
        models=_models(),
        settings_path=tmp_path / "settings.yml",
        metadata_path=tmp_path / "metadata.json",
    )

    assert result[:6] == [
        "--model",
        "anthropic/nvidia_nim/vendor/main-model",
        "--weak-model",
        "anthropic/nvidia_nim/vendor/main-model",
        "--editor-model",
        "anthropic/nvidia_nim/vendor/main-model",
    ]


@pytest.mark.parametrize(
    "argv",
    [
        ["--model"],
        ["--weak-model="],
        ["--editor-model", "--verbose"],
    ],
)
def test_aider_rejects_missing_model_values(
    argv: list[str], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from free_claude_code.cli.launchers.aider import normalized_aider_arguments

    with pytest.raises(SystemExit) as exc_info:
        normalized_aider_arguments(
            argv,
            models=_models(),
            settings_path=tmp_path / "settings.yml",
            metadata_path=tmp_path / "metadata.json",
        )

    assert exc_info.value.code == 2
    assert "requires one model" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    [
        ["--model", _models()[0].wire_slug, "--model=" + _models()[1].wire_slug],
        [
            "--weak-model",
            _models()[0].wire_slug,
            "--weak-model",
            _models()[1].wire_slug,
        ],
        [
            "--editor-model=" + _models()[0].wire_slug,
            "--editor-model=" + _models()[1].wire_slug,
        ],
    ],
)
def test_aider_rejects_duplicate_model_options(
    argv: list[str], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from free_claude_code.cli.launchers.aider import normalized_aider_arguments

    with pytest.raises(SystemExit) as exc_info:
        normalized_aider_arguments(
            argv,
            models=_models(),
            settings_path=tmp_path / "settings.yml",
            metadata_path=tmp_path / "metadata.json",
        )

    assert exc_info.value.code == 2
    assert "only once" in capsys.readouterr().err


def test_aider_rejects_unknown_initial_model(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from free_claude_code.cli.launchers.aider import normalized_aider_arguments

    with pytest.raises(SystemExit) as exc_info:
        normalized_aider_arguments(
            ["--model", "anthropic/not/in/catalog"],
            models=_models(),
            settings_path=tmp_path / "settings.yml",
            metadata_path=tmp_path / "metadata.json",
        )

    assert exc_info.value.code == 2
    assert "not in the current FCC model catalog" in capsys.readouterr().err


def test_aider_child_env_scrubs_stale_keys_and_preserves_native_state() -> None:
    from free_claude_code.cli.launchers.aider import build_aider_launcher_env

    env = build_aider_launcher_env(
        proxy_root_url="http://127.0.0.1:9191",
        api_key_env="FCC_AIDER_PROXY_AUTH_FRESH123",
        auth_token="real-secret",
        base_env={
            "PATH": "keep",
            "AIDER_CONFIG_FILE": "keep-user-config",
            "HTTP_PROXY": "http://proxy.example",
            "NO_PROXY": "example.com",
            "fcc_aider_proxy_auth_old": "remove",
            "FCC_AIDER_PROXY_AUTH_OTHER": "remove",
        },
    )

    assert env["PATH"] == "keep"
    assert env["AIDER_CONFIG_FILE"] == "keep-user-config"
    assert env["HTTP_PROXY"] == "http://proxy.example"
    assert env["NO_PROXY"] == "example.com,127.0.0.1,localhost,::1"
    assert env["no_proxy"] == env["NO_PROXY"]
    assert env["FCC_AIDER_PROXY_AUTH_FRESH123"] == "real-secret"
    assert "fcc_aider_proxy_auth_old" not in env
    assert "FCC_AIDER_PROXY_AUTH_OTHER" not in env


def test_aider_launch_uses_messages_catalog_and_private_ephemeral_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from free_claude_code.cli.launchers import aider

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("AIDER_CONFIG_FILE", "keep-native-state")
    observed_directory: Path | None = None
    observed_env_name: str | None = None

    def observe_process(
        *,
        command: list[str],
        env: Mapping[str, str],
        binary_name: str,
        display_name: str,
        install_hint: str,
    ) -> None:
        nonlocal observed_directory, observed_env_name
        del binary_name, display_name, install_hint
        settings_index = command.index("--model-settings-file") + 1
        metadata_index = command.index("--model-metadata-file") + 1
        settings_path = Path(command[settings_index])
        metadata_path = Path(command[metadata_index])
        observed_directory = settings_path.parent
        assert metadata_path.parent == observed_directory
        assert settings_path.name == "model-settings.yml"
        assert metadata_path.name == "model-metadata.json"
        settings_payload = json.loads(settings_path.read_text(encoding="utf-8"))
        metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        env_reference = settings_payload[0]["extra_params"]["api_key"]
        assert isinstance(env_reference, str)
        observed_env_name = env_reference.removeprefix("os.environ/")
        assert observed_env_name.startswith("FCC_AIDER_PROXY_AUTH_")
        assert env[observed_env_name] == "proxy-token"
        assert env["AIDER_CONFIG_FILE"] == "keep-native-state"
        rendered = (
            " ".join(command)
            + json.dumps(settings_payload)
            + json.dumps(metadata_payload)
        )
        assert "proxy-token" not in rendered
        assert command[:3] == ["resolved-aider", "--verbose", "src/app.py"]
        assert command[-2:] == ["--set-env", "ANTHROPIC_API_KEY=fcc-local"]

    with (
        patch.object(aider, "resolve_client_binary", return_value="resolved-aider"),
        patch.object(aider, "get_settings", return_value=_settings()),
        patch.object(aider, "preflight_proxy", return_value=None),
        patch.object(
            aider,
            "fetch_proxy_models_response",
            return_value=_models_payload(),
        ) as fetch_models,
        patch.object(aider, "run_client_process", side_effect=observe_process),
    ):
        aider.launch(["--verbose", "src/app.py"])

    fetch_models.assert_called_once_with(
        "http://127.0.0.1:9191",
        "proxy-token",
        view="messages",
    )
    assert observed_directory is not None
    assert observed_env_name is not None
    assert observed_directory.parent == tmp_path / ".fcc" / "tmp" / "aider"
    assert not observed_directory.exists()


@pytest.mark.parametrize("error", [SystemExit(7), KeyboardInterrupt()])
def test_aider_temp_files_cleanup_when_child_ends_abnormally(
    error: BaseException,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from free_claude_code.cli.launchers import aider

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    observed_directory: Path | None = None

    def fail_process(**kwargs: object) -> None:
        nonlocal observed_directory
        command = kwargs["command"]
        assert isinstance(command, list)
        settings_path = Path(command[command.index("--model-settings-file") + 1])
        observed_directory = settings_path.parent
        assert observed_directory.is_dir()
        raise error

    with (
        patch.object(aider, "resolve_client_binary", return_value="resolved-aider"),
        patch.object(aider, "get_settings", return_value=_settings()),
        patch.object(aider, "preflight_proxy", return_value=None),
        patch.object(
            aider,
            "fetch_proxy_models_response",
            return_value=_models_payload(),
        ),
        patch.object(aider, "run_client_process", side_effect=fail_process),
        pytest.raises(type(error)),
    ):
        aider.launch([])

    assert observed_directory is not None
    assert not observed_directory.exists()


def test_aider_temporary_preparations_are_unique(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from free_claude_code.cli.launchers.aider import temporary_aider_config_files

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    first_config = build_aider_config(
        _models(),
        messages_url="http://127.0.0.1:9191/v1/messages",
        api_key_env="FCC_AIDER_PROXY_AUTH_FIRST",
    )
    second_config = build_aider_config(
        _models(),
        messages_url="http://127.0.0.1:9191/v1/messages",
        api_key_env="FCC_AIDER_PROXY_AUTH_SECOND",
    )

    with (
        temporary_aider_config_files(first_config) as first,
        temporary_aider_config_files(second_config) as second,
    ):
        first_directory = first.directory
        second_directory = second.directory
        assert first_directory != second_directory
        assert first_directory.is_dir()
        assert second_directory.is_dir()

    assert not first_directory.exists()
    assert not second_directory.exists()


@pytest.mark.skipif(
    os.name == "nt", reason="POSIX file modes are not enforced on Windows"
)
def test_aider_managed_directories_and_files_are_owner_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from free_claude_code.cli.launchers.aider import temporary_aider_config_files

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    base = tmp_path / ".fcc" / "tmp" / "aider"
    base.mkdir(parents=True, mode=0o755)
    base.chmod(0o755)
    config = build_aider_config(
        _models(),
        messages_url="http://127.0.0.1:9191/v1/messages",
        api_key_env="FCC_AIDER_PROXY_AUTH_MODES",
    )

    with temporary_aider_config_files(config) as files:
        assert base.stat().st_mode & 0o777 == 0o700
        assert files.directory.stat().st_mode & 0o777 == 0o700
        assert files.settings_path.stat().st_mode & 0o777 == 0o600
        assert files.metadata_path.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize(
    ("token", "preflight_error", "catalog", "expected"),
    [
        ("   ", None, _models_payload(), "token is empty"),
        ("proxy-token", "connection refused", _models_payload(), "fcc-server"),
        ("proxy-token", None, {"data": []}, "at least one routable"),
    ],
)
def test_aider_launch_fails_before_child_for_invalid_preparation(
    token: str,
    preflight_error: str | None,
    catalog: JsonObject,
    expected: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from free_claude_code.cli.launchers import aider

    with (
        patch.object(aider, "resolve_client_binary", return_value="resolved-aider"),
        patch.object(aider, "get_settings", return_value=_settings(token=token)),
        patch.object(aider, "preflight_proxy", return_value=preflight_error),
        patch.object(
            aider,
            "fetch_proxy_models_response",
            return_value=catalog,
        ),
        patch.object(aider, "run_client_process") as run_client_process,
        pytest.raises(SystemExit) as exc_info,
    ):
        aider.launch([])

    assert exc_info.value.code == 1
    assert expected in capsys.readouterr().err
    run_client_process.assert_not_called()


def test_aider_catalog_failure_is_reported_once_without_secret(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from free_claude_code.cli.launchers import aider

    with (
        patch.object(aider, "resolve_client_binary", return_value="resolved-aider"),
        patch.object(aider, "get_settings", return_value=_settings()),
        patch.object(aider, "preflight_proxy", return_value=None),
        patch.object(
            aider,
            "fetch_proxy_models_response",
            side_effect=ValueError("bad catalog"),
        ),
        patch.object(aider, "run_client_process") as run_client_process,
        pytest.raises(SystemExit) as exc_info,
    ):
        aider.launch([])

    assert exc_info.value.code == 1
    error = capsys.readouterr().err
    assert error.count("bad catalog") == 1
    assert "proxy-token" not in error
    run_client_process.assert_not_called()


def test_aider_file_creation_error_does_not_launch_or_disclose_token(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from free_claude_code.cli.launchers import aider

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    with (
        patch.object(aider, "resolve_client_binary", return_value="resolved-aider"),
        patch.object(aider, "get_settings", return_value=_settings()),
        patch.object(aider, "preflight_proxy", return_value=None),
        patch.object(
            aider,
            "fetch_proxy_models_response",
            return_value=_models_payload(),
        ),
        patch.object(
            aider.tempfile,
            "TemporaryDirectory",
            side_effect=OSError("storage unavailable"),
        ),
        patch.object(aider, "run_client_process") as run_client_process,
        pytest.raises(SystemExit) as exc_info,
    ):
        aider.launch([])

    assert exc_info.value.code == 1
    error = capsys.readouterr().err
    assert "storage unavailable" in error
    assert "proxy-token" not in error
    run_client_process.assert_not_called()


def test_aider_missing_binary_uses_shared_resolution_contract() -> None:
    from free_claude_code.cli.launchers import aider

    with (
        patch.object(
            aider,
            "resolve_client_binary",
            side_effect=SystemExit(127),
        ) as resolve_client_binary,
        patch.object(aider, "get_settings") as get_settings,
        pytest.raises(SystemExit) as exc_info,
    ):
        aider.launch([])

    assert exc_info.value.code == 127
    resolve_client_binary.assert_called_once_with(
        binary_name="aider",
        display_name="Aider",
        install_hint="Install Aider from: https://aider.chat/docs/install.html",
    )
    get_settings.assert_not_called()


def test_aider_config_repr_never_contains_the_proxy_token() -> None:
    config = build_aider_config(
        _models(),
        messages_url="http://127.0.0.1:9191/v1/messages",
        api_key_env="FCC_AIDER_PROXY_AUTH_SAFE",
    )

    assert "proxy-token" not in repr(config)
    assert "127.0.0.1" not in repr(config)


def test_aider_random_credentials_are_uppercase_and_distinct() -> None:
    from free_claude_code.cli.launchers.aider import new_aider_api_key_env_name

    with patch(
        "free_claude_code.cli.launchers.aider.secrets.token_hex",
        side_effect=["a1b2", "c3d4"],
    ):
        first = new_aider_api_key_env_name()
        second = new_aider_api_key_env_name()

    assert first == "FCC_AIDER_PROXY_AUTH_A1B2"
    assert second == "FCC_AIDER_PROXY_AUTH_C3D4"
    assert first != second


def test_aider_temp_path_is_owned_by_fcc_home(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from free_claude_code.config.paths import aider_temp_dir_path

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    assert aider_temp_dir_path() == tmp_path / ".fcc" / "tmp" / "aider"
