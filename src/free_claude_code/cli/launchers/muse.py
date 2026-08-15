"""Installed ``fcc-muse`` launcher for Meta Muse Code, including optional MCP."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

from free_claude_code.config.loader import get_settings
from free_claude_code.config.server_urls import local_proxy_root_url

from .common import preflight_proxy, resolve_client_binary, run_client_process
from .openai_compat import build_openai_compat_env, proxy_bearer_token

_DISPLAY_NAME = "Muse Code"
_BINARY_NAME = "muse"
_INSTALL_HINT = "Install Muse Code from https://dev.meta.ai/docs/muse-code/"
_MCP_COMMAND_ENV = "FCC_MUSE_MCP_COMMAND"
_MCP_SERVERS_ENV = "FCC_MUSE_MCP_SERVERS"


def launch(argv: Sequence[str] | None = None) -> None:
    """Launch Muse Code against the local FCC proxy, with optional MCP servers."""

    args = list(sys.argv[1:] if argv is None else argv)
    settings = get_settings()
    proxy_root_url = local_proxy_root_url(settings)
    if error := preflight_proxy(proxy_root_url):
        print(
            f"Free Claude Code proxy is not reachable at {proxy_root_url}: {error}",
            file=sys.stderr,
        )
        print("Start it in another terminal with: fcc-server", file=sys.stderr)
        raise SystemExit(1)

    binary_path = resolve_client_binary(
        binary_name=_BINARY_NAME,
        display_name=_DISPLAY_NAME,
        install_hint=_INSTALL_HINT,
    )
    settings_path = write_muse_settings_file(
        Path(tempfile.mkdtemp(prefix="fcc-muse-")),
        mcp_servers=build_muse_mcp_servers(os.environ),
    )
    env = build_muse_launcher_env(
        proxy_root_url=proxy_root_url,
        auth_token=settings.proxy_auth_token,
        model=settings.model,
        base_env=os.environ,
    )
    print(
        f"fcc-muse: routing Muse Code through {proxy_root_url} model {settings.model}",
        file=sys.stderr,
    )
    run_client_process(
        command=build_muse_launcher_command(
            binary_path=binary_path,
            argv=args,
            proxy_root_url=proxy_root_url,
            model=settings.model,
            settings_path=settings_path,
        ),
        env=env,
        binary_name=_BINARY_NAME,
        display_name=_DISPLAY_NAME,
        install_hint=_INSTALL_HINT,
    )


def build_muse_mcp_servers(env: Mapping[str, str]) -> dict[str, object]:
    """Return Muse ``mcp_servers`` from env. Never reads passwords from disk."""

    raw = env.get(_MCP_SERVERS_ENV, "").strip()
    if raw:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError(f"{_MCP_SERVERS_ENV} must be a JSON object.")
        return parsed

    command = env.get(_MCP_COMMAND_ENV, "").strip()
    if not command:
        return {}
    return {
        "fcc-tools": {
            "transport": "stdio",
            "command": command,
            "args": [],
            "enabled": True,
            "mode": "optional",
        }
    }


def build_muse_settings_document(*, mcp_servers: Mapping[str, object]) -> dict[str, object]:
    """Return a Muse settings object. Empty MCP map is omitted."""

    document: dict[str, object] = {}
    if mcp_servers:
        document["mcp_servers"] = dict(mcp_servers)
    return document


def write_muse_settings_file(
    directory: Path,
    *,
    mcp_servers: Mapping[str, object],
) -> Path | None:
    """Write a session settings file when MCP servers are configured."""

    document = build_muse_settings_document(mcp_servers=mcp_servers)
    if not document:
        return None
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "settings.json"
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return path


def build_muse_launcher_command(
    *,
    binary_path: str,
    argv: Sequence[str],
    proxy_root_url: str,
    model: str,
    settings_path: Path | None,
) -> list[str]:
    """Return the Muse CLI command pointed at FCC."""

    args = list(argv)
    extras: list[str] = []
    if not _has_flag(args, "--provider"):
        extras.extend(["--provider", "meta"])
    if not _has_flag(args, "--base-url"):
        extras.extend(["--base-url", proxy_root_url])
    if not _has_flag(args, "--model"):
        extras.extend(["--model", model])
    if settings_path is not None and not _has_flag(args, "--settings"):
        extras.extend(["--settings", str(settings_path)])
    return [binary_path, *extras, *args]


def build_muse_launcher_env(
    *,
    proxy_root_url: str,
    auth_token: str,
    model: str,
    base_env: Mapping[str, str],
) -> dict[str, str]:
    """Return Muse process env. The proxy token is not written to disk."""

    env = build_openai_compat_env(
        proxy_root_url=proxy_root_url,
        auth_token=auth_token,
        base_env=base_env,
        model=model,
    )
    token = proxy_bearer_token(auth_token)
    env["META_API_KEY"] = token
    env["META_BASE_URL"] = proxy_root_url
    return env


def _has_flag(args: Sequence[str], flag: str) -> bool:
    return any(item == flag or item.startswith(f"{flag}=") for item in args)
