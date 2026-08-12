"""Installed `fcc-codex-desktop` launcher."""

import json
import os
import shutil
import sys
import tomllib
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from free_claude_code.config.server_urls import local_proxy_root_url
from free_claude_code.config.settings import get_settings

from .codex import _ensure_v1_url, build_codex_launcher_env
from .common import (
    preflight_proxy,
    run_client_process,
)

_DISPLAY_NAME = "Codex Desktop"
_DEFAULT_BINARY = "codex-desktop"
_INSTALL_HINT = "Install Codex Desktop from https://openai.com/codex or add codex-desktop to your PATH."


def launch(argv: Sequence[str] | None = None) -> None:
    """Launch Codex Desktop with Free Claude Code proxy configuration."""

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

    binary_path = resolve_codex_desktop_binary()
    config_path = codex_config_path()

    env = build_codex_launcher_env(
        proxy_root_url=proxy_root_url,
        base_env=os.environ,
    )

    with ephemeral_codex_config(
        config_path,
        proxy_root_url=proxy_root_url,
        model=getattr(settings, "model", None),
    ):
        run_client_process(
            command=[binary_path, *args],
            env=env,
            binary_name=_DEFAULT_BINARY,
            display_name=_DISPLAY_NAME,
            install_hint=_INSTALL_HINT,
        )


def codex_config_path() -> Path:
    """Return the platform Codex config.toml path."""

    if codex_home := os.environ.get("CODEX_HOME"):
        return Path(codex_home) / "config.toml"
    return Path.home() / ".codex" / "config.toml"


def resolve_codex_desktop_binary() -> str:
    """Resolve the Codex Desktop executable path across Linux, Windows, and macOS."""

    for env_var in ("CODEX_DESKTOP_PATH", "CODEX_PATH"):
        if (override := os.environ.get(env_var)) and Path(override).is_file():
            return override

    candidates: list[str] = []
    if sys.platform == "darwin":
        candidates.extend(
            [
                "/Applications/Codex.app/Contents/MacOS/Codex",
                "/Applications/Codex Desktop.app/Contents/MacOS/Codex Desktop",
                str(
                    Path.home()
                    / "Applications"
                    / "Codex.app"
                    / "Contents"
                    / "MacOS"
                    / "Codex"
                ),
                str(
                    Path.home()
                    / "Applications"
                    / "Codex Desktop.app"
                    / "Contents"
                    / "MacOS"
                    / "Codex Desktop"
                ),
            ]
        )
    elif sys.platform == "win32":
        local_app_data = os.environ.get("LOCALAPPDATA", "")
        app_data = os.environ.get("APPDATA", "")
        program_files = os.environ.get("PROGRAMFILES", "")
        program_files_x86 = os.environ.get("PROGRAMFILES(X86)", "")
        if local_app_data:
            candidates.extend(
                [
                    str(Path(local_app_data) / "Programs" / "Codex" / "Codex.exe"),
                    str(
                        Path(local_app_data)
                        / "Programs"
                        / "Codex Desktop"
                        / "Codex Desktop.exe"
                    ),
                ]
            )
        if app_data:
            candidates.append(str(Path(app_data) / "Codex" / "Codex.exe"))
        if program_files:
            candidates.append(str(Path(program_files) / "Codex" / "Codex.exe"))
        if program_files_x86:
            candidates.append(str(Path(program_files_x86) / "Codex" / "Codex.exe"))
    else:
        # Linux and other Unix platforms
        home = Path.home()
        candidates.extend(
            [
                "/usr/bin/codex-desktop",
                "/usr/local/bin/codex-desktop",
                "/snap/bin/codex-desktop",
                "/snap/bin/codex",
                str(home / ".local" / "bin" / "codex-desktop"),
                str(home / ".local" / "bin" / "codex"),
                "/opt/Codex/codex-desktop",
                "/opt/codex/codex",
            ]
        )

    for candidate in candidates:
        if Path(candidate).is_file():
            return candidate

    binary_names = (
        ["codex-desktop.exe", "codex-desktop", "codex.exe", "codex"]
        if sys.platform == "win32"
        else ["codex-desktop", "codex"]
    )
    for name in binary_names:
        if found := shutil.which(name):
            return found

    print(
        f"Could not find {_DISPLAY_NAME} command: {_DEFAULT_BINARY}",
        file=sys.stderr,
    )
    print(_INSTALL_HINT, file=sys.stderr)
    raise SystemExit(127)


def prepare_codex_config_content(
    original_content: str | None,
    *,
    api_url: str,
    model: str | None = None,
) -> str:
    """Prepare injected config.toml content with model_provider = "fcc"."""

    data: dict[str, Any] = {}
    if original_content:
        try:
            data = tomllib.loads(original_content)
        except Exception:
            data = {}

    data["model_provider"] = "fcc"
    if model:
        data["model"] = model

    model_providers = data.setdefault("model_providers", {})
    if not isinstance(model_providers, dict):
        model_providers = {}
        data["model_providers"] = model_providers

    model_providers["fcc"] = {
        "name": "Free Claude Code",
        "base_url": api_url,
        "wire_api": "responses",
        "auth": {
            "command": "fcc-codex",
            "args": ["--print-proxy-auth-token"],
        },
    }

    return dump_toml(data)


def dump_toml(data: dict[str, Any]) -> str:
    """Serialize a dictionary of basic primitive types and sub-dicts into TOML format."""

    lines: list[str] = []

    for k, v in data.items():
        if not isinstance(v, dict):
            lines.append(f"{k} = {json.dumps(v)}")

    def _write_table(prefix: str, table: dict[str, Any]) -> None:
        lines.append("")
        lines.append(f"[{prefix}]")
        for k, v in table.items():
            if not isinstance(v, dict):
                lines.append(f"{k} = {json.dumps(v)}")
        for k, v in table.items():
            if isinstance(v, dict):
                _write_table(f"{prefix}.{k}", v)

    for k, v in data.items():
        if isinstance(v, dict):
            _write_table(k, v)

    return "\n".join(lines).strip() + "\n"


@contextmanager
def ephemeral_codex_config(
    config_path: Path,
    *,
    proxy_root_url: str,
    model: str | None = None,
) -> Generator[Path]:
    """Context manager for ephemeral injection and cleanup of Codex config.toml."""

    existed = config_path.exists()
    original_content: str | None = None
    if existed:
        try:
            original_content = config_path.read_text(encoding="utf-8")
        except OSError:
            original_content = None

    api_url = _ensure_v1_url(proxy_root_url)
    new_content = prepare_codex_config_content(
        original_content,
        api_url=api_url,
        model=model,
    )

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(new_content, encoding="utf-8")
    try:
        yield config_path
    finally:
        try:
            if existed and original_content is not None:
                config_path.write_text(original_content, encoding="utf-8")
            elif config_path.exists():
                config_path.unlink()
        except OSError as exc:
            print(
                f"Free Claude Code warning: failed to restore {config_path}: {exc}",
                file=sys.stderr,
            )
