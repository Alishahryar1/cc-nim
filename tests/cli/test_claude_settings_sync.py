"""Tests for cli/claude_settings_sync.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cli.claude_settings_sync import (
    build_fcc_claude_env,
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


def test_sync_handles_invalid_json_gracefully(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _redirect_home(monkeypatch, tmp_path)
    settings_path = _settings_path(tmp_path)
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text("{not-json", encoding="utf-8")

    changed = sync_claude_settings(
        proxy_root_url="http://127.0.0.1:8082",
        auth_token="proxy-token",
        settings_path=settings_path,
    )

    assert changed is True
    assert "invalid JSON" in capsys.readouterr().err
    payload = json.loads(settings_path.read_text(encoding="utf-8"))
    assert payload["env"] == build_fcc_claude_env(
        "http://127.0.0.1:8082",
        "proxy-token",
    )
