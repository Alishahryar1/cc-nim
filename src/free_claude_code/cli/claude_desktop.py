"""Claude Desktop routing support shared by the FCC desktop shell.

Two responsibilities:

1. Merge the Free Claude Code 3P-gateway routing block into the user's
   ``claude_desktop_config.json`` so Claude Desktop's model picker runs
   ``/v1/models`` discovery against the local FCC server. The gateway URL
   and auth key are derived from server settings, never hardcoded.
   ``unconfigure`` reverses the merge, preserving every other key.
2. Locate the Claude Desktop binary and spawn it. TLS trust for the local
   FCC front comes from the certificate the install scripts add to the
   per-user NSS store; no process-wide certificate bypass is used.

The desktop controller applies the merge automatically at startup, and the
tray exposes a "Launch Claude Desktop" item built on the same helpers.
"""

import argparse
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from loguru import logger

from free_claude_code.cli.tls_proxy import (
    desktop_gateway_base_url,
    verified_https_gateway_url,
)
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

# Snapshot of the managed surface taken before the first merge, so
# ``unconfigure`` restores exactly what the user had instead of deleting it.
_BACKUP_KEY = "fccPriorConfig"

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

    ``gateway_base_url`` overrides the resolved URL (desktop-scoped: HTTPS
    root when a TLS front answers plus the desktop path prefix).
    """

    return {
        "provider": "gateway",
        "credentialKind": "static",
        "inferenceProvider": "gateway",
        "inferenceCredentialKind": "static",
        "inferenceGatewayBaseUrl": gateway_base_url
        or desktop_gateway_base_url(settings),
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
    """Atomically write the merged config; create private directories.

    The block embeds the gateway auth token, so the file is written with
    owner-only permissions and the config directories stay owner-traversable
    regardless of the process umask. The temporary file is newly allocated
    with ``O_EXCL`` (never following or reusing a pre-existing path), so an
    attacker-placed permissive file or symlink at a predictable name can
    neither read the token nor redirect the write to another file; its mode
    is forced to owner-only before the atomic replace.
    """

    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    _restrict_permissions(path.parent)
    descriptor, tmp_name = tempfile.mkstemp(
        prefix=f"{path.name}.", suffix=".tmp", dir=path.parent
    )
    tmp_path = Path(tmp_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(data, indent=2))
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    tmp_path.replace(path)


def _restrict_permissions(directory: Path) -> None:
    """Force an existing directory to owner-only modes.

    ``mkdir(mode=...)`` only applies to directories it creates; a directory
    that already exists keeps its modes, so tighten it explicitly.
    """

    current = directory.stat().st_mode
    target = stat.S_IRWXU
    if current & 0o777 != target:
        directory.chmod(target)


def configure_claude_desktop_config(
    path: Path | None = None,
    settings: Settings | None = None,
    gateway_base_url: str | None = None,
) -> bool:
    """Merge the FCC routing block into ``path``. Returns whether anything changed.

    Before the first merge the prior presence and values of the managed
    surface (the discovery flag and the inference keys FCC overwrites) are
    snapshotted under ``fccPriorConfig``; ``unconfigure`` restores that
    snapshot so temporarily enabling FCC never destroys the user's previous
    provider, gateway, credential, or discovery settings. Re-merges keep the
    original snapshot untouched.

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

    # The snapshot must observe the config BEFORE the merge touches it, so
    # capture the discovery flag's prior presence and value first: presence
    # alone is not enough, since a user who explicitly disabled discovery
    # must get ``false`` back, not ``true``.
    had_discovery = _DISCOVERY_KEY in data
    prior_discovery = data.get(_DISCOVERY_KEY)

    if data.get(_DISCOVERY_KEY) is not True:
        data[_DISCOVERY_KEY] = True
        changed = True

    inference_raw = data.get(_INFERENCE_KEY)
    inference_dict: dict[str, object] = {}
    if isinstance(inference_raw, dict):
        for key, value in inference_raw.items():
            inference_dict[str(key)] = value

    if _BACKUP_KEY not in data:
        backup: dict[str, object] = {}
        # Omit the discovery entry when the key was absent so the restore
        # can tell "was present with this value" from "was not present".
        if had_discovery:
            backup["discovery"] = prior_discovery
        if _INFERENCE_KEY in data:
            if isinstance(inference_raw, dict):
                backup["inference"] = {
                    key: value
                    for key, value in inference_dict.items()
                    if key in managed
                }
            else:
                # The original value is not a JSON object; the merge
                # replaces it wholesale, so record it verbatim for an
                # exact restore.
                backup["inferenceRaw"] = inference_raw
        data[_BACKUP_KEY] = backup
        changed = True
    elif not isinstance(inference_raw, dict) and _INFERENCE_KEY in data:
        # A non-object ``inference`` with a snapshot already recorded is
        # the user's wholesale replacement of the FCC block (a scalar or
        # ``null`` in place of the mapping configure wrote). The merge
        # still writes its managed mapping — configure's contract is to
        # ensure the block exists — but the snapshot must now restore the
        # REPLACEMENT on unconfigure; keeping the first-merge value would
        # silently discard the user's later change.
        backup = data[_BACKUP_KEY]
        if isinstance(backup, dict):
            backup.pop("inference", None)
            backup["inferenceRaw"] = inference_raw
            changed = True

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

    When the first merge recorded a ``fccPriorConfig`` snapshot, removal is
    lossless: every managed key is deleted, then the snapshot's prior
    presence and values are restored, and only keys FCC originally added
    stay gone. A non-object ``inference`` value recorded verbatim by the
    snapshot is restored exactly. Without a snapshot (a config written
    before backups existed or by hand) ownership is decided by the
    recorded auth token: it is settings-derived and stable across front
    state, so when it matches the FCC token the whole block is FCC's and
    every managed key is removed — even a gateway URL that no longer
    matches current resolution because the HTTPS front is gone and
    resolution fell back to plain HTTP — along with the discovery flag.
    Without that token match only keys whose recorded values exactly
    match current resolution are removed, so user-owned values that share
    FCC key names survive, and the discovery flag is preserved because
    ownership is ambiguous. Returns ``False`` without writing when the
    existing config is malformed, or silently when the file does not
    exist.
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

    backup_raw = data.get(_BACKUP_KEY)
    # ``None`` means "no snapshot recorded" — an EMPTY snapshot dict is a
    # real one (nothing prior to restore) and must still take the restore
    # path so it gets deleted.
    backup: dict[str, object] | None = None
    if isinstance(backup_raw, dict):
        backup = backup_raw

    changed = False

    inference_raw = data.get(_INFERENCE_KEY)
    inference_dict: dict[str, object] = {}
    if isinstance(inference_raw, dict):
        for key, value in inference_raw.items():
            inference_dict[str(key)] = value

    if backup is not None:
        # Snapshot path: the merge recorded what FCC replaced, so the
        # discovery flag and every managed key are FCC's — remove them
        # all, then restore the snapshot below.
        if _DISCOVERY_KEY in data:
            del data[_DISCOVERY_KEY]
            changed = True
        for key in managed:
            if key in inference_dict:
                del inference_dict[key]
                changed = True
        prior_raw = backup.get("inference")
        if isinstance(prior_raw, dict):
            for key, value in prior_raw.items():
                inference_dict[str(key)] = value
                changed = True
    else:
        # Legacy path (no snapshot): ownership is decided by the recorded
        # auth token. The discovery flag is removed only once FCC
        # ownership is established — deleting it first would erase a
        # user-owned preference from a hand-authored or legacy config.
        managed_token = managed["inferenceAnthropicApiKey"]
        fcc_owned = inference_dict.get("inferenceAnthropicApiKey") == managed_token
        if fcc_owned and _DISCOVERY_KEY in data:
            del data[_DISCOVERY_KEY]
            changed = True
        for key, value in managed.items():
            recorded = inference_dict.get(key)
            if recorded is not None and (fcc_owned or recorded == value):
                del inference_dict[key]
                changed = True

    if _INFERENCE_KEY in data:
        if inference_dict:
            if data[_INFERENCE_KEY] != inference_dict:
                data[_INFERENCE_KEY] = inference_dict
                changed = True
        else:
            del data[_INFERENCE_KEY]
            changed = True

    if backup is not None:
        if "inferenceRaw" in backup:
            # The original value was not a JSON object; restore it verbatim.
            data[_INFERENCE_KEY] = backup["inferenceRaw"]
            changed = True
        # Restore the original discovery value when the key was present
        # before the first merge; leave it deleted when it was absent.
        if "discovery" in backup:
            data[_DISCOVERY_KEY] = backup["discovery"]
            changed = True
        del data[_BACKUP_KEY]
        changed = True

    if changed:
        _save_config(config_path, data)
    return changed


def report_configure_result(target: Path, changed: bool) -> None:
    """Report a configure outcome, refusing a malformed-config skip.

    ``configure_claude_desktop_config`` returns ``False`` for both an
    idempotent no-op and a malformed-config skip; only the skip leaves
    Claude Desktop unrouted, so reading the config back distinguishes
    them. "Already merged" is reserved for a valid config that already
    carries the routing block; a malformed one exits nonzero.
    """

    if not changed:
        try:
            load_existing_config(target)
        except MalformedConfigError as exc:
            print(
                f"Refusing to configure Claude Desktop: malformed config at "
                f"{exc.path}: {exc.reason}. Fix or remove the file and retry.",
                file=sys.stderr,
            )
            raise SystemExit(1) from None
    print(f"{'Updated' if changed else 'Already merged'}: {target}")


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
    """Spawn Claude Desktop.

    TLS trust for the local FCC front is provided by the certificate the
    install scripts add to the per-user NSS store, so no process-wide
    certificate bypass flag is passed.
    """

    binary_path = find_binary()
    if binary_path is None:
        raise FileNotFoundError(CLAUDE_DESKTOP_BINARY)
    command: list[str] = [binary_path, *extra_args]
    return subprocess.Popen(command)


def ensure_configured_and_launch(
    settings: Settings | None = None,
) -> subprocess.Popen[bytes]:
    """Merge the routing block, then spawn Claude Desktop.

    The routing block carries the proxy auth token, so it is only written
    behind a verified HTTPS front: ``verified_https_gateway_url`` returns
    ``None`` — never the plain-HTTP fallback — when no front proves it
    belongs to this FCC install, and the launch is refused instead of
    pointing the credential at an unverified or cleartext listener.

    The merge result is also verified before spawning: ``configure`` returns
    ``False`` both when it skips a malformed config and when an earlier merge
    already matches, so the block is read back and the launch refused unless
    the routing actually landed. Launching on a skipped (malformed) config
    would start Claude Desktop without FCC routing.

    Raises:
        RuntimeError: If no verified HTTPS front answers, or the routing
            block did not land in the config.
        FileNotFoundError: If no Claude Desktop binary can be located.
    """

    resolved = settings or get_settings()
    gateway_url = verified_https_gateway_url(resolved)
    if gateway_url is None:
        raise RuntimeError(
            "Refusing to launch Claude Desktop: no verified FCC HTTPS front "
            "is available. Start the FCC desktop host (which manages the "
            "front) or enable the caddy TLS proxy, then retry."
        )
    configure_claude_desktop_config(settings=resolved, gateway_base_url=gateway_url)
    if not _routing_block_present(gateway_url):
        raise RuntimeError(
            "Refusing to launch Claude Desktop: the FCC routing block could "
            "not be written to the config (it may be malformed). Fix or "
            "remove the config file and retry."
        )
    return launch_binary()


def _routing_block_present(gateway_url: str) -> bool:
    """Whether the managed routing block actually landed in the config.

    Distinguishes a successful (possibly idempotent) merge from a skipped
    malformed config, which ``configure_claude_desktop_config`` reports with
    the same ``False`` return.
    """

    try:
        data = load_existing_config(_config_path())
    except FileNotFoundError:
        return False
    except MalformedConfigError:
        return False
    inference = data.get(_INFERENCE_KEY)
    if not isinstance(inference, dict):
        return False
    return (
        data.get(_DISCOVERY_KEY) is True
        and inference.get("inferenceGatewayBaseUrl") == gateway_url
    )


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

    # --configure persists the gateway credential, so it only writes
    # against a verified HTTPS front: without one the URL would fall back
    # to plain HTTP and persist the reusable proxy token against a
    # cleartext endpoint. Adoption is probe-only — this command exits
    # right after writing, so a front it spawned itself would die with it
    # and leave the config it just wrote pointing at a dead gateway.
    gateway_url = verified_https_gateway_url(get_settings())
    if gateway_url is None:
        print(
            "Refusing to configure Claude Desktop: no verified FCC HTTPS "
            "front is available. Enable the caddy TLS proxy (or start the "
            "FCC desktop host, which manages it) and retry.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    changed = configure_claude_desktop_config(
        args.config_path, gateway_base_url=gateway_url
    )
    report_configure_result(target, changed)


if __name__ == "__main__":
    main()
