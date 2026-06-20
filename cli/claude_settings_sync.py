"""Sync Free Claude Code proxy env into Claude Code settings.json."""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

from cli.claude_env import CLAUDE_CODE_AUTO_COMPACT_WINDOW, claude_auth_token
from config.paths import claude_settings_path

FCC_CLAUDE_ENV_KEYS = frozenset(
    {
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_AUTH_TOKEN",
        "CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY",
        "CLAUDE_CODE_AUTO_COMPACT_WINDOW",
    }
)

_SETTINGS_FILE_MODE = stat.S_IRUSR | stat.S_IWUSR


def build_fcc_claude_env(proxy_root_url: str, auth_token: str) -> dict[str, str]:
    """Return the FCC-managed Claude Code env keys for settings.json."""

    return {
        "ANTHROPIC_BASE_URL": proxy_root_url,
        "ANTHROPIC_AUTH_TOKEN": claude_auth_token(auth_token),
        "CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY": "1",
        "CLAUDE_CODE_AUTO_COMPACT_WINDOW": CLAUDE_CODE_AUTO_COMPACT_WINDOW,
    }


def should_defer_claude_settings_sync(pending_fields: list[str]) -> bool:
    """Return whether Claude settings sync should wait until proxy restart."""

    return "PORT" in pending_fields


def _warn(message: str) -> None:
    print(message, file=sys.stderr)


def _load_settings_object(path: Path) -> dict[str, object] | None:
    if path.is_dir():
        _warn(
            f"Warning: {path} is a directory; skipping FCC Claude settings sync."
        )
        return None

    if not path.is_file():
        return {}

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        _warn(
            f"Warning: could not read {path} ({exc}); skipping FCC Claude settings sync."
        )
        return None
    except UnicodeDecodeError as exc:
        _warn(
            f"Warning: {path} is not valid UTF-8 ({exc}); skipping FCC Claude settings sync."
        )
        return None

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        _warn(
            f"Warning: {path} contains invalid JSON ({exc}); skipping FCC Claude settings sync."
        )
        return None

    if not isinstance(parsed, dict):
        _warn(
            f"Warning: {path} is not a JSON object; skipping FCC Claude settings sync."
        )
        return None

    return parsed


def _merge_fcc_env(
    existing_env: dict[str, object], desired_fcc_env: dict[str, str]
) -> dict[str, str]:
    merged_env = {
        key: value
        for key, value in existing_env.items()
        if isinstance(key, str)
        and isinstance(value, str)
        and not key.startswith("ANTHROPIC_")
    }
    merged_env.update(desired_fcc_env)
    return merged_env


def _restrict_settings_permissions(path: Path) -> None:
    try:
        os.chmod(path, _SETTINGS_FILE_MODE)
    except OSError:
        return


def sync_claude_settings(
    *,
    proxy_root_url: str,
    auth_token: str,
    settings_path: Path | None = None,
) -> bool:
    """Merge FCC proxy env keys into Claude Code settings.json when needed.

    Best-effort only: logs warnings and returns False instead of raising.
    """

    path = settings_path or claude_settings_path()
    desired_fcc_env = build_fcc_claude_env(proxy_root_url, auth_token)

    settings = _load_settings_object(path)
    if settings is None:
        return False

    existing_env = settings.get("env")
    if not isinstance(existing_env, dict):
        existing_env = {}

    merged_env = _merge_fcc_env(existing_env, desired_fcc_env)
    updated_settings = dict(settings)
    updated_settings["env"] = merged_env

    if path.is_file() and settings == updated_settings:
        return False

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        temp_path.write_text(
            json.dumps(updated_settings, indent=2) + "\n",
            encoding="utf-8",
        )
        _restrict_settings_permissions(temp_path)
        os.replace(temp_path, path)
        _restrict_settings_permissions(path)
    except OSError as exc:
        _warn(
            f"Warning: could not write {path} ({exc}); skipping FCC Claude settings sync."
        )
        return False

    return True
