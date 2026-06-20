"""Sync Free Claude Code proxy env into Claude Code settings.json."""

from __future__ import annotations

import json
import os
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


def build_fcc_claude_env(proxy_root_url: str, auth_token: str) -> dict[str, str]:
    """Return the FCC-managed Claude Code env keys for settings.json."""

    return {
        "ANTHROPIC_BASE_URL": proxy_root_url,
        "ANTHROPIC_AUTH_TOKEN": claude_auth_token(auth_token),
        "CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY": "1",
        "CLAUDE_CODE_AUTO_COMPACT_WINDOW": CLAUDE_CODE_AUTO_COMPACT_WINDOW,
    }


def sync_claude_settings(
    *,
    proxy_root_url: str,
    auth_token: str,
    settings_path: Path | None = None,
) -> bool:
    """Merge FCC proxy env keys into Claude Code settings.json when needed."""

    path = settings_path or claude_settings_path()
    desired_fcc_env = build_fcc_claude_env(proxy_root_url, auth_token)

    settings: dict[str, object] = {}
    if path.is_file():
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(parsed, dict):
                settings = parsed
            else:
                print(
                    f"Warning: {path} is not a JSON object; replacing with FCC env sync.",
                    file=sys.stderr,
                )
        except json.JSONDecodeError as exc:
            print(
                f"Warning: {path} contains invalid JSON ({exc}); replacing with FCC env sync.",
                file=sys.stderr,
            )

    existing_env = settings.get("env")
    if not isinstance(existing_env, dict):
        existing_env = {}

    merged_env = dict(existing_env)
    for key, value in desired_fcc_env.items():
        merged_env[key] = value

    updated_settings = dict(settings)
    updated_settings["env"] = merged_env

    if path.is_file() and settings == updated_settings:
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(updated_settings, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temp_path, path)
    return True
