"""Installed `fcc-codex-desktop` launcher."""

import json
import os
import re
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


def _display_path(p: Path) -> str:
    """Format path relative to home directory with ~ if possible."""
    try:
        return f"~/{p.relative_to(Path.home())}"
    except ValueError:
        return str(p)


def launch(argv: Sequence[str] | None = None) -> None:
    """Launch Codex Desktop with Free Claude Code proxy configuration."""

    args = list(sys.argv[1:] if argv is None else argv)

    if "--setup" in args:
        setup_persistent_config(is_fallback=False)
        raise SystemExit(0)

    if "--reset" in args or "--restore" in args:
        reset_persistent_config()
        raise SystemExit(0)

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
    if binary_path is None:
        setup_persistent_config(is_fallback=True)
        raise SystemExit(0)

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


def resolve_codex_desktop_binary() -> str | None:
    """Resolve the Codex Desktop executable path across Linux, Windows, and macOS."""

    if (override := os.environ.get("CODEX_DESKTOP_PATH")) and Path(override).is_file():
        return override

    candidates: list[str] = []
    if sys.platform == "darwin":
        home = Path.home()
        candidates.extend(
            [
                "/Applications/ChatGPT.app/Contents/MacOS/ChatGPT",
                "/Applications/Codex.app/Contents/MacOS/Codex",
                "/Applications/Codex Desktop.app/Contents/MacOS/Codex Desktop",
                str(
                    home
                    / "Applications"
                    / "ChatGPT.app"
                    / "Contents"
                    / "MacOS"
                    / "ChatGPT"
                ),
                str(
                    home / "Applications" / "Codex.app" / "Contents" / "MacOS" / "Codex"
                ),
                str(
                    home
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
                    str(Path(local_app_data) / "Programs" / "ChatGPT" / "ChatGPT.exe"),
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
            candidates.extend(
                [
                    str(Path(app_data) / "ChatGPT" / "ChatGPT.exe"),
                    str(Path(app_data) / "Codex" / "Codex.exe"),
                ]
            )
        if program_files:
            candidates.extend(
                [
                    str(Path(program_files) / "ChatGPT" / "ChatGPT.exe"),
                    str(Path(program_files) / "Codex" / "Codex.exe"),
                ]
            )
        if program_files_x86:
            candidates.extend(
                [
                    str(Path(program_files_x86) / "ChatGPT" / "ChatGPT.exe"),
                    str(Path(program_files_x86) / "Codex" / "Codex.exe"),
                ]
            )
    else:
        # Linux and other Unix platforms
        home = Path.home()
        candidates.extend(
            [
                "/usr/bin/chatgpt",
                "/usr/lib/chatgpt/codex-launcher",
                "/usr/lib/chatgpt/ChatGPT",
                "/opt/chatgpt/chatgpt",
                "/usr/bin/codex-desktop",
                "/usr/local/bin/codex-desktop",
                "/snap/bin/codex-desktop",
                str(home / ".local" / "bin" / "chatgpt"),
                str(home / ".local" / "bin" / "codex-desktop"),
                "/opt/Codex/codex-desktop",
                "/opt/codex-desktop/codex-desktop",
            ]
        )

    for candidate in candidates:
        if Path(candidate).is_file():
            return candidate

    binary_names = (
        ["chatgpt.exe", "chatgpt", "codex-desktop.exe", "codex-desktop"]
        if sys.platform == "win32"
        else ["chatgpt", "codex-desktop"]
    )
    for name in binary_names:
        if found := shutil.which(name):
            return found

    return None


def setup_persistent_config(*, is_fallback: bool = False) -> None:
    """Apply persistent Free Claude Code configuration to Codex config.toml."""

    config_path = codex_config_path()
    backup_path = config_path.parent / f"{config_path.name}.fccbak"

    original_content: str | None = None
    if config_path.exists():
        try:
            original_content = config_path.read_text(encoding="utf-8")
        except OSError:
            original_content = None

        if not backup_path.exists() and original_content is not None:
            try:
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                backup_path.write_text(original_content, encoding="utf-8")
            except OSError:
                pass

    settings = get_settings()
    proxy_root_url = local_proxy_root_url(settings)
    api_url = _ensure_v1_url(proxy_root_url)
    model = getattr(settings, "model", None)

    new_content = prepare_codex_config_content(
        original_content,
        api_url=api_url,
        model=model,
    )

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(new_content, encoding="utf-8")

    disp_path = _display_path(config_path)
    if is_fallback:
        print(
            "[Free Claude Code] Codex Desktop / ChatGPT GUI binary was not found in standard PATH."
        )
    print(f"[Free Claude Code] Persistent configuration applied to {disp_path}.\n")
    print(
        "Setup completed! Please launch ChatGPT / Codex Desktop from your application menu or shortcut.\n"
    )
    print("To restore your original configuration at any time, run:")
    print("  fcc-codex-desktop --reset")


def reset_persistent_config() -> None:
    """Restore Codex config.toml to its original state or remove injected FCC settings."""

    config_path = codex_config_path()
    backup_path = config_path.parent / f"{config_path.name}.fccbak"

    if backup_path.exists():
        try:
            backup_content = backup_path.read_text(encoding="utf-8")
            config_path.write_text(backup_content, encoding="utf-8")
            backup_path.unlink()
        except OSError:
            pass
    elif config_path.exists():
        try:
            content = config_path.read_text(encoding="utf-8")
            data = tomllib.loads(content)
            modified = False
            if data.get("model_provider") == "fcc":
                del data["model_provider"]
                modified = True
            if "model_providers" in data and isinstance(data["model_providers"], dict):
                if "fcc" in data["model_providers"]:
                    del data["model_providers"]["fcc"]
                    modified = True
                if not data["model_providers"]:
                    del data["model_providers"]
                    modified = True
            if modified:
                if not data:
                    config_path.unlink()
                else:
                    config_path.write_text(dump_toml(data), encoding="utf-8")
        except Exception:
            pass

    print("[Free Claude Code] Configuration reset successfully!")
    print("Codex Desktop configuration restored to original settings.")


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


_BARE_KEY_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _format_toml_key(key: str) -> str:
    """Format key as bare TOML key or quoted TOML key if it contains special characters."""

    if _BARE_KEY_RE.match(key):
        return key
    return json.dumps(key)


def dump_toml(data: dict[str, Any]) -> str:
    """Serialize a dictionary of basic primitive types and sub-dicts into valid TOML format."""

    lines: list[str] = []

    for k, v in data.items():
        if not isinstance(v, dict):
            lines.append(f"{_format_toml_key(k)} = {json.dumps(v)}")

    def _write_table(prefix: str, table: dict[str, Any]) -> None:
        lines.append("")
        lines.append(f"[{prefix}]")
        for k, v in table.items():
            if not isinstance(v, dict):
                lines.append(f"{_format_toml_key(k)} = {json.dumps(v)}")
        for k, v in table.items():
            if isinstance(v, dict):
                _write_table(f"{prefix}.{_format_toml_key(k)}", v)

    for k, v in data.items():
        if isinstance(v, dict):
            _write_table(_format_toml_key(k), v)

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
