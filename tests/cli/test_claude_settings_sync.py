"""Tests for cli/claude_settings_sync.py."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from cli.claude_settings_sync import (
    build_fcc_claude_env,
    should_defer_claude_settings_sync,
    sync_claude_settings,
)


def _settings_path(home: Path) -> Path:
    return home / ".claude" / "settings.json"


def _redirect_home(monkeypatch: pytest.MonkeyPatch, home: Path) -> None:
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setattr("pathlib.Path.home", lambda: home)


def test_sync_creates_file_when_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _redirect_home(monkeypatch, tmp_path)
    settings_path = _settings_path(tmp_path)

    changed = sync_claude_settings(
        proxy_root_url="http://127.0.0.1:8082",
        auth_token="proxy-token",
        settings_path=settings_path,
    )

    assert changed is True
    assert settings_path.is_file()
    payload = json.loads(settings_path.read_text(encoding="utf-8"))
    assert payload["env"] == build_fcc_claude_env(
        "http://127.0.0.1:8082",
        "proxy-token",
    )


def test_sync_merges_without_clobbering_theme_and_other_env_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _redirect_home(monkeypatch, tmp_path)
    settings_path = _settings_path(tmp_path)
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text(
        json.dumps(
            {
                "theme": "dark",
                "env": {
                    "CUSTOM_FLAG": "1",
                    "ANTHROPIC_BASE_URL": "http://127.0.0.1:8082",
                    "ANTHROPIC_AUTH_TOKEN": "old-token",
                },
            }
        ),
        encoding="utf-8",
    )

    changed = sync_claude_settings(
        proxy_root_url="http://127.0.0.1:8082",
        auth_token="new-token",
        settings_path=settings_path,
    )

    assert changed is True
    payload = json.loads(settings_path.read_text(encoding="utf-8"))
    assert payload["theme"] == "dark"
    assert payload["env"]["CUSTOM_FLAG"] == "1"
    assert payload["env"]["ANTHROPIC_AUTH_TOKEN"] == "new-token"
    assert payload["env"]["CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY"] == "1"
    assert payload["env"]["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] == "190000"


def test_sync_strips_stale_anthropic_env_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _redirect_home(monkeypatch, tmp_path)
    settings_path = _settings_path(tmp_path)
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text(
        json.dumps(
            {
                "env": {
                    "CUSTOM_FLAG": "1",
                    "ANTHROPIC_API_KEY": "official-key",
                    "ANTHROPIC_API_URL": "https://api.anthropic.com/v1",
                }
            }
        ),
        encoding="utf-8",
    )

    sync_claude_settings(
        proxy_root_url="http://127.0.0.1:8082",
        auth_token="proxy-token",
        settings_path=settings_path,
    )

    payload = json.loads(settings_path.read_text(encoding="utf-8"))
    assert payload["env"]["CUSTOM_FLAG"] == "1"
    assert payload["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8082"
    assert payload["env"]["ANTHROPIC_AUTH_TOKEN"] == "proxy-token"
    assert "ANTHROPIC_API_KEY" not in payload["env"]
    assert "ANTHROPIC_API_URL" not in payload["env"]


def test_sync_updates_when_port_or_token_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _redirect_home(monkeypatch, tmp_path)
    settings_path = _settings_path(tmp_path)
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text(
        json.dumps(
            {
                "env": build_fcc_claude_env(
                    "http://127.0.0.1:8082",
                    "old-token",
                )
            }
        ),
        encoding="utf-8",
    )

    changed = sync_claude_settings(
        proxy_root_url="http://127.0.0.1:9191",
        auth_token="new-token",
        settings_path=settings_path,
    )

    assert changed is True
    payload = json.loads(settings_path.read_text(encoding="utf-8"))
    assert payload["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:9191"
    assert payload["env"]["ANTHROPIC_AUTH_TOKEN"] == "new-token"


def test_sync_uses_fcc_no_auth_when_token_blank(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _redirect_home(monkeypatch, tmp_path)
    settings_path = _settings_path(tmp_path)

    sync_claude_settings(
        proxy_root_url="http://127.0.0.1:8082",
        auth_token="",
        settings_path=settings_path,
    )

    payload = json.loads(settings_path.read_text(encoding="utf-8"))
    assert payload["env"]["ANTHROPIC_AUTH_TOKEN"] == "fcc-no-auth"


def test_sync_returns_false_when_already_up_to_date(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _redirect_home(monkeypatch, tmp_path)
    settings_path = _settings_path(tmp_path)
    desired_env = build_fcc_claude_env("http://127.0.0.1:8082", "proxy-token")
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text(
        json.dumps({"env": desired_env}),
        encoding="utf-8",
    )

    changed = sync_claude_settings(
        proxy_root_url="http://127.0.0.1:8082",
        auth_token="proxy-token",
        settings_path=settings_path,
    )

    assert changed is False
    assert json.loads(settings_path.read_text(encoding="utf-8")) == {"env": desired_env}


def test_sync_skips_invalid_json_without_overwriting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _redirect_home(monkeypatch, tmp_path)
    settings_path = _settings_path(tmp_path)
    settings_path.parent.mkdir(parents=True)
    original = "{not-json"
    settings_path.write_text(original, encoding="utf-8")

    changed = sync_claude_settings(
        proxy_root_url="http://127.0.0.1:8082",
        auth_token="proxy-token",
        settings_path=settings_path,
    )

    assert changed is False
    assert settings_path.read_text(encoding="utf-8") == original
    assert "skipping FCC Claude settings sync" in capsys.readouterr().err


def test_sync_skips_when_settings_path_is_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _redirect_home(monkeypatch, tmp_path)
    settings_path = _settings_path(tmp_path)
    settings_path.mkdir(parents=True)

    changed = sync_claude_settings(
        proxy_root_url="http://127.0.0.1:8082",
        auth_token="proxy-token",
        settings_path=settings_path,
    )

    assert changed is False
    assert settings_path.is_dir()
    assert "is a directory" in capsys.readouterr().err


@pytest.mark.skipif(os.name == "nt", reason="Unix file modes are not enforced on Windows")
def test_sync_restricts_settings_file_permissions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _redirect_home(monkeypatch, tmp_path)
    settings_path = _settings_path(tmp_path)
    old_umask = os.umask(0)

    try:
        sync_claude_settings(
            proxy_root_url="http://127.0.0.1:8082",
            auth_token="proxy-token",
            settings_path=settings_path,
        )
    finally:
        os.umask(old_umask)

    mode = stat.S_IMODE(settings_path.stat().st_mode)
    assert mode == stat.S_IRUSR | stat.S_IWUSR


def test_should_defer_claude_settings_sync_when_port_changes() -> None:
    assert should_defer_claude_settings_sync(["PORT"]) is True
    assert should_defer_claude_settings_sync(["ANTHROPIC_AUTH_TOKEN"]) is False
