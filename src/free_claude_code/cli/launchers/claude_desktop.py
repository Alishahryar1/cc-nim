"""Installed ``fcc-claude-desktop`` launcher and config helper.

Two responsibilities:

1. Launch the Claude Desktop binary with the
   ``--ignore-certificate-errors`` flag so Caddy's self-signed cert is
   accepted without per-host certificate trust work.
2. Merge the Free Claude Code 3P-gateway routing block into the user's
   ``claude_desktop_config.json`` (``--configure``) so the picker
   triggers ``/v1/models`` discovery at the FCC endpoint. ``--unconfigure``
   reverses the merge, preserving every other key byte-for-byte.

Both modes are invoked through the same console script so install-time
auto-configuration can call ``python3 -m
free_claude_code.cli.launchers.claude_desktop --configure`` and avoid
``jq``/``sed`` JSON parsing on platforms without those tools.
"""

import argparse
import json
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from loguru import logger


class MalformedConfigError(ValueError):
    """Raised when the Claude Desktop config file exists but cannot be parsed as a JSON object."""

    def __init__(self, path: Path, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"Claude Desktop config at {path} is malformed: {reason}")


CLAUDE_DESKTOP_BINARY = "claude-desktop"
_CONFIG_FILENAME = "claude_desktop_config.json"

# Top-level key flipping Claude Desktop out of first-party Anthropic mode.
_DISCOVERY_KEY = "modelDiscoveryEnabled"

# Inference block routed to the local FCC endpoint. Values mirror the
# working user config documented in
# ``docs/claude-desktop-picker-aliasing.md`` §5.
_INFERENCE_KEY = "inference"
INFERENCE_BLOCK: dict[str, object] = {
    "provider": "gateway",
    "credentialKind": "static",
    "inferenceProvider": "gateway",
    "inferenceCredentialKind": "static",
    "inferenceGatewayBaseUrl": "https://localhost:8443",
    "inferenceGatewayAuthScheme": "x-api-key",
    "inferenceAnthropicApiKey": "freecc",
}


def _config_path() -> Path:
    """Return the platform-specific Claude Desktop config path."""

    if sys.platform.startswith("win"):
        appdata = sys.modules["os"].environ.get("APPDATA")
        if appdata:
            return Path(appdata) / "Claude" / _CONFIG_FILENAME
        return Path.home() / "AppData" / "Roaming" / "Claude" / _CONFIG_FILENAME
    if sys.platform == "darwin":
        return (
            Path.home()
            / "Library"
            / "Application Support"
            / "Claude"
            / _CONFIG_FILENAME
        )
    return Path.home() / ".config" / "Claude" / _CONFIG_FILENAME


def load_existing_config(path: Path) -> dict[str, object]:
    """Read existing JSON config as a dict.

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

    # json.loads returns dict[str, Any] at runtime; narrow to dict[str, object]
    return loaded


def _save_config(path: Path, data: dict[str, object]) -> None:
    """Atomically write the merged config; create parent directories as needed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _inference_matches_fcc_block(block: dict[str, object]) -> bool:
    """Whether every FCC-managed key in the inference block matches our default."""

    return all(block.get(key) == value for key, value in INFERENCE_BLOCK.items())


def configure_claude_desktop_config(path: Path | None = None) -> bool:
    """Merge the FCC routing block into ``path``. Returns whether anything changed.

    Returns ``False`` and aborts without writing when the existing config is
    malformed (invalid JSON or non-object root). Creates a new config when
    the file does not exist.
    """

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

    # Get inference block with proper type narrowing
    inference_raw = data.get(_INFERENCE_KEY)
    inference_dict: dict[str, object] = {}
    if isinstance(inference_raw, dict):
        for k, v in inference_raw.items():
            inference_dict[str(k)] = v

    for key, value in INFERENCE_BLOCK.items():
        if inference_dict.get(key) != value:
            inference_dict[key] = value
            changed = True

    if _INFERENCE_KEY not in data or data[_INFERENCE_KEY] != inference_dict:
        data[_INFERENCE_KEY] = inference_dict
        changed = True

    if changed:
        _save_config(config_path, data)
    return changed


def unconfigure_claude_desktop_config(path: Path | None = None) -> bool:
    """Reverse the merge. Preserves every key outside the FCC-managed surface.

    Returns ``False`` and aborts without writing when the existing config is
    malformed (invalid JSON or non-object root). Returns ``False`` silently
    when the file does not exist.
    """

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
            for k, v in inference.items():
                inference_dict[str(k)] = v
            for key in INFERENCE_BLOCK:
                if (
                    key in inference_dict
                    and inference_dict[key] == INFERENCE_BLOCK[key]
                ):
                    del inference_dict[key]
                    changed = True
            if not inference_dict:
                del data[_INFERENCE_KEY]
            else:
                data[_INFERENCE_KEY] = inference_dict
        else:
            del data[_INFERENCE_KEY]
            changed = True

    if changed:
        _save_config(config_path, data)
    return changed


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fcc-claude-desktop",
        description=(
            "Launch Claude Desktop with the Free Claude Code routing layer, "
            "or merge/reverse the FCC routing block in claude_desktop_config.json."
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
    parser.add_argument(
        "binary_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the claude-desktop binary.",
    )
    return parser


def _launch_subprocess(
    binary_path: str, extra_args: Sequence[str]
) -> subprocess.Popen[bytes]:
    command: list[str] = [binary_path, "--ignore-certificate-errors", *extra_args]
    return subprocess.Popen(command)


def launch(argv: Sequence[str] | None = None) -> None:
    """Entry point for the ``fcc-claude-desktop`` console script."""

    args = list(sys.argv[1:] if argv is None else argv)
    parsed = _build_argparser().parse_args(args)

    if parsed.configure:
        changed = configure_claude_desktop_config(parsed.config_path)
        target = parsed.config_path or _config_path()
        print(f"{'Updated' if changed else 'Already merged'}: {target}")
        return

    if parsed.unconfigure:
        changed = unconfigure_claude_desktop_config(parsed.config_path)
        target = parsed.config_path or _config_path()
        print(f"{'Removed FCC block' if changed else 'Nothing to remove'}: {target}")
        return

    binary_path = shutil.which(CLAUDE_DESKTOP_BINARY)
    if binary_path is None:
        print(
            f"Could not find '{CLAUDE_DESKTOP_BINARY}' on PATH.",
            file=sys.stderr,
        )
        print(
            "Install Claude Desktop from https://claude.ai/download.",
            file=sys.stderr,
        )
        raise SystemExit(127)

    try:
        process = _launch_subprocess(binary_path, list(parsed.binary_args))
    except FileNotFoundError:
        print(
            f"Could not find '{CLAUDE_DESKTOP_BINARY}' binary at {binary_path}.",
            file=sys.stderr,
        )
        raise SystemExit(127) from None
    try:
        return_code = process.wait()
    except KeyboardInterrupt:
        process.terminate()
        process.wait()
        raise
    raise SystemExit(return_code)
