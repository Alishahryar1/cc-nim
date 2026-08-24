"""Claude Desktop routing support shared by the FCC desktop shell.

Two responsibilities:

1. Merge the Free Claude Code 3P-gateway routing block into the user's
   ``claude_desktop_config.json`` so Claude Desktop's model picker runs
   ``/v1/models`` discovery against the local FCC server. The gateway URL
   and auth key are derived from server settings, never hardcoded.
   ``unconfigure`` reverses the merge, preserving every other key.
2. Locate the Claude Desktop binary and spawn it with
   ``--ignore-certificate-errors`` (harmless on plain HTTP, required when
   the user fronts FCC with a self-signed TLS proxy).

The desktop controller applies the merge automatically at startup, and the
tray exposes a "Launch Claude Desktop" item built on the same helpers.
"""

import argparse
import json
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from loguru import logger

from free_claude_code.cli.tls_proxy import resolve_gateway_base_url
from free_claude_code.config.loader import get_settings
from free_claude_code.config.settings import Settings


class MalformedConfigError(ValueError):
    """Raised when the config file exists but cannot be parsed as a JSON object."""

    def __init__(self, path: Path, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"Claude Desktop config at {path} is malformed: {reason}")


CLAUDE_DESKTOP_BINARY = "claude-desktop"
CONFIG_FILENAME = "claude_desktop_config.json"

# Top-level key flipping Claude Desktop out of first-party Anthropic mode.
_DISCOVERY_KEY = "modelDiscoveryEnabled"
_INFERENCE_KEY = "inference"

# Per-platform candidate locations beyond PATH. First existing hit wins.
_BINARY_CANDIDATES: dict[str, tuple[str, ...]] = {
    "darwin": ("/Applications/Claude.app/Contents/MacOS/Claude",),
    "win32": (),
}


def _config_path() -> Path:
    """Return the platform-specific Claude Desktop config path."""

    if sys.platform.startswith("win"):
        appdata = sys.modules["os"].environ.get("APPDATA")
        if appdata:
            return Path(appdata) / "Claude" / CONFIG_FILENAME
        return Path.home() / "AppData" / "Roaming" / "Claude" / CONFIG_FILENAME
    if sys.platform == "darwin":
        return (
            Path.home() / "Library" / "Application Support" / "Claude" / CONFIG_FILENAME
        )
    return Path.home() / ".config" / "Claude" / CONFIG_FILENAME


def fcc_managed_block(
    settings: Settings,
    gateway_base_url: str | None = None,
) -> dict[str, object]:
    """Inference keys FCC owns in Claude Desktop's config, from live settings.

    ``gateway_base_url`` overrides the resolved URL (HTTPS when a TLS front
    answers, plain HTTP otherwise).
    """

    return {
        "provider": "gateway",
        "credentialKind": "static",
        "inferenceProvider": "gateway",
        "inferenceCredentialKind": "static",
        "inferenceGatewayBaseUrl": gateway_base_url
        or resolve_gateway_base_url(settings),
        "inferenceGatewayAuthScheme": "x-api-key",
        "inferenceAnthropicApiKey": settings.proxy_auth_token,
    }


def load_existing_config(path: Path) -> dict[str, object]:
    """Read an existing JSON config as a dict.

    Raises:
        FileNotFoundError: If the config file does not exist.
        MalformedConfigError: If the file exists but is not valid JSON or is not a JSON object.
    """

    if not path.exists():
        raise FileNotFoundError(f"Claude Desktop config not found: {path}")

    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MalformedConfigError(path, f"invalid JSON: {exc}") from exc

    if not isinstance(loaded, dict):
        raise MalformedConfigError(path, "root element is not a JSON object")

    return loaded


def _save_config(path: Path, data: dict[str, object]) -> None:
    """Atomically write the merged config; create parent directories as needed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def configure_claude_desktop_config(
    path: Path | None = None,
    settings: Settings | None = None,
    gateway_base_url: str | None = None,
) -> bool:
    """Merge the FCC routing block into ``path``. Returns whether anything changed.

    Returns ``False`` without writing when the existing config is malformed
    (invalid JSON or non-object root). Creates a new config when absent.
    """

    resolved_settings = settings or get_settings()
    managed = fcc_managed_block(resolved_settings, gateway_base_url)
    config_path = path or _config_path()
    try:
        data: dict[str, object] = load_existing_config(config_path)
    except FileNotFoundError:
        data = {}
    except MalformedConfigError as exc:
        logger.warning(
            "Skipping FCC config merge: malformed config at {}: {}",
            exc.path,
            exc.reason,
        )
        return False

    changed = False

    if data.get(_DISCOVERY_KEY) is not True:
        data[_DISCOVERY_KEY] = True
        changed = True

    inference_raw = data.get(_INFERENCE_KEY)
    inference_dict: dict[str, object] = {}
    if isinstance(inference_raw, dict):
        for key, value in inference_raw.items():
            inference_dict[str(key)] = value

    for key, value in managed.items():
        if inference_dict.get(key) != value:
            inference_dict[key] = value
            changed = True

    if data.get(_INFERENCE_KEY) != inference_dict:
        data[_INFERENCE_KEY] = inference_dict
        changed = True

    if changed:
        _save_config(config_path, data)
    return changed


def unconfigure_claude_desktop_config(
    path: Path | None = None,
    settings: Settings | None = None,
    gateway_base_url: str | None = None,
) -> bool:
    """Reverse the merge. Preserves every key outside the FCC-managed surface.

    ``gateway_base_url`` must match the value a previous ``configure`` wrote.
    Returns ``False`` without writing when the existing config is malformed,
    or silently when the file does not exist.
    """

    resolved_settings = settings or get_settings()
    managed = fcc_managed_block(resolved_settings, gateway_base_url)
    config_path = path or _config_path()
    if not config_path.exists():
        return False
    try:
        data: dict[str, object] = load_existing_config(config_path)
    except MalformedConfigError as exc:
        logger.warning(
            "Skipping FCC config unmerge: malformed config at {}: {}",
            exc.path,
            exc.reason,
        )
        return False

    changed = False

    if _DISCOVERY_KEY in data:
        del data[_DISCOVERY_KEY]
        changed = True

    if _INFERENCE_KEY in data:
        inference = data[_INFERENCE_KEY]
        if isinstance(inference, dict):
            inference_dict: dict[str, object] = {}
            for key, value in inference.items():
                inference_dict[str(key)] = value
            for key, value in managed.items():
                if key in inference_dict and inference_dict[key] == value:
                    del inference_dict[key]
                    changed = True
            if inference_dict:
                data[_INFERENCE_KEY] = inference_dict
            else:
                del data[_INFERENCE_KEY]
        else:
            del data[_INFERENCE_KEY]
            changed = True

    if changed:
        _save_config(config_path, data)
    return changed


def find_binary() -> str | None:
    """Locate the Claude Desktop binary via PATH, then platform defaults."""

    which_hit = shutil.which(CLAUDE_DESKTOP_BINARY)
    if which_hit is not None:
        return which_hit
    for candidate in _BINARY_CANDIDATES.get(sys.platform, ()):
        if Path(candidate).exists():
            return candidate
    return None


def launch_binary(
    extra_args: Sequence[str] = (),
) -> subprocess.Popen[bytes]:
    """Spawn Claude Desktop with the certificate-tolerance flag."""

    binary_path = find_binary()
    if binary_path is None:
        raise FileNotFoundError(CLAUDE_DESKTOP_BINARY)
    command: list[str] = [
        binary_path,
        "--ignore-certificate-errors",
        *extra_args,
    ]
    return subprocess.Popen(command)


def ensure_configured_and_launch(
    settings: Settings | None = None,
) -> subprocess.Popen[bytes]:
    """Merge the routing block, then spawn Claude Desktop.

    Raises:
        FileNotFoundError: If no Claude Desktop binary can be located.
    """

    configure_claude_desktop_config(settings=settings)
    return launch_binary()


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m free_claude_code.cli.claude_desktop",
        description=(
            "Merge or reverse the Free Claude Code routing block in "
            "claude_desktop_config.json."
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--configure",
        action="store_true",
        help="Merge the FCC 3P-gateway block into Claude Desktop's config.",
    )
    mode.add_argument(
        "--unconfigure",
        action="store_true",
        help="Reverse the merge applied by --configure.",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=None,
        help="Override the Claude Desktop config path (primarily for tests).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point for explicit configure/unconfigure runs."""

    args = _build_argparser().parse_args(list(sys.argv[1:] if argv is None else argv))
    target = args.config_path or _config_path()

    if args.unconfigure:
        changed = unconfigure_claude_desktop_config(args.config_path)
        print(f"{'Removed FCC block' if changed else 'Nothing to remove'}: {target}")
        return

    changed = configure_claude_desktop_config(args.config_path)
    print(f"{'Updated' if changed else 'Already merged'}: {target}")


if __name__ == "__main__":
    main()
