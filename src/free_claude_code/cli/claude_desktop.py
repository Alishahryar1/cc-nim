"""Claude Desktop routing support shared by the FCC desktop shell.

Two responsibilities:

1. Merge the Free Claude Code 3P-gateway routing block into the user's
   ``claude_desktop_config.json`` so Claude Desktop's model picker runs
   ``/v1/models`` discovery against the local FCC server. The gateway URL
   and auth key are derived from server settings, never hardcoded.
   ``unconfigure`` reverses the merge by restoring exactly what configure
   inserted or overwrote, tracked in an ownership record beside the config.
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
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TypedDict

from loguru import logger

from free_claude_code.cli.tls_proxy import desktop_gateway_base_url
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
    """Atomically write the merged config; create parent directories as needed.

    The block embeds the gateway auth token, so the file is written with
    owner-only permissions regardless of the process umask.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    descriptor = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(data, indent=2))
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    tmp_path.replace(path)


class _OwnershipEntry(TypedDict):
    """Pre-merge state of one managed config key."""

    present: bool
    value: object


@dataclass(frozen=True, slots=True)
class _OwnershipRecord:
    """What ``configure`` inserted or overwrote, for exact restoration.

    The managed inference keys are always tracked individually so values
    added to ``inference`` after a merge survive unconfiguration.
    ``inference_written`` snapshots the mapping configure produced; a key
    whose current value still equals its written value is FCC's to restore,
    while anything the user changed since the merge is left alone. When the
    pre-merge ``inference`` entry was not an object, configure replaced it
    wholesale and ``inference_scalar`` snapshots what it clobbered.
    """

    discovery: _OwnershipEntry
    inference_existed: bool
    inference_was_object: bool
    inference_scalar: object | None
    inference_keys: dict[str, _OwnershipEntry]
    inference_written: dict[str, object] = field(default_factory=dict)


_RECORD_SUFFIX = ".fcc-merge.json"
_RECORD_INFERENCE_KEYS = "inferenceKeys"
_RECORD_INFERENCE_EXISTED = "inferenceExisted"
_RECORD_INFERENCE_WAS_OBJECT = "inferenceWasObject"
_RECORD_INFERENCE_SCALAR = "inferenceScalar"
_RECORD_INFERENCE_WRITTEN = "inferenceWritten"


def _record_payload(record: _OwnershipRecord) -> dict[str, object]:
    """JSON-able sidecar representation of ``record``."""

    return {
        _DISCOVERY_KEY: record.discovery,
        _RECORD_INFERENCE_EXISTED: record.inference_existed,
        _RECORD_INFERENCE_WAS_OBJECT: record.inference_was_object,
        _RECORD_INFERENCE_KEYS: record.inference_keys,
        _RECORD_INFERENCE_WRITTEN: record.inference_written,
        **(
            {_RECORD_INFERENCE_SCALAR: record.inference_scalar}
            if record.inference_existed and not record.inference_was_object
            else {}
        ),
    }


def _record_path(config_path: Path) -> Path:
    """Sidecar recording what configure inserted or overwrote."""

    return config_path.with_name(config_path.name + _RECORD_SUFFIX)


def _snapshot_entry(mapping: Mapping[str, object], key: str) -> _OwnershipEntry:
    """Capture the pre-merge state of one key inside ``mapping``."""

    if key in mapping:
        return {"present": True, "value": mapping[key]}
    return {"present": False, "value": None}


def _ownership_record(
    data: dict[str, object],
    managed_keys: Sequence[str],
) -> _OwnershipRecord:
    """Snapshot every managed key so unconfigure can restore exactly it."""

    discovery = _snapshot_entry(data, _DISCOVERY_KEY)
    existed = _INFERENCE_KEY in data
    raw = data.get(_INFERENCE_KEY)
    keys = {
        key: _snapshot_entry(raw if isinstance(raw, dict) else {}, key)
        for key in managed_keys
    }
    if isinstance(raw, dict):
        return _OwnershipRecord(
            discovery=discovery,
            inference_existed=True,
            inference_was_object=True,
            inference_scalar=None,
            inference_keys=keys,
        )
    return _OwnershipRecord(
        discovery=discovery,
        inference_existed=existed,
        inference_was_object=False,
        inference_scalar=raw if existed else None,
        inference_keys=keys,
    )


def _parse_entry(raw: object) -> _OwnershipEntry | None:
    """Parse one ownership snapshot; ``None`` when malformed."""

    if not isinstance(raw, dict):
        return None
    present = raw.get("present")
    if not isinstance(present, bool) or (present and "value" not in raw):
        return None
    return {"present": present, "value": raw.get("value")}


def _load_ownership_record(path: Path) -> _OwnershipRecord | None:
    """Read the ownership sidecar; ``None`` when missing or malformed."""

    if not path.exists():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(loaded, dict):
        return None

    discovery = _parse_entry(loaded.get(_DISCOVERY_KEY))
    if discovery is None:
        return None

    existed = loaded.get(_RECORD_INFERENCE_EXISTED)
    was_object = loaded.get(_RECORD_INFERENCE_WAS_OBJECT)
    if not isinstance(existed, bool) or not isinstance(was_object, bool):
        return None

    keys_raw = loaded.get(_RECORD_INFERENCE_KEYS)
    if not isinstance(keys_raw, dict):
        return None
    inference_keys: dict[str, _OwnershipEntry] = {}
    for key, raw in keys_raw.items():
        parsed = _parse_entry(raw)
        if not isinstance(key, str) or parsed is None:
            return None
        inference_keys[key] = parsed

    written = loaded.get(_RECORD_INFERENCE_WRITTEN)
    if not isinstance(written, dict):
        return None
    normalized_written: dict[str, object] = {
        str(key): value for key, value in written.items()
    }

    scalar: object | None = None
    if existed and not was_object:
        if _RECORD_INFERENCE_SCALAR not in loaded:
            return None
        scalar = loaded[_RECORD_INFERENCE_SCALAR]

    return _OwnershipRecord(
        discovery=discovery,
        inference_existed=existed,
        inference_was_object=was_object,
        inference_scalar=scalar,
        inference_keys=inference_keys,
        inference_written=normalized_written,
    )


def configure_claude_desktop_config(
    path: Path | None = None,
    settings: Settings | None = None,
    gateway_base_url: str | None = None,
) -> bool:
    """Merge the FCC routing block into ``path``. Returns whether anything changed.

    Records the pre-merge values of the managed keys in a sidecar so a later
    ``unconfigure`` restores exactly them. Returns ``False`` without writing
    when the existing config is malformed (invalid JSON or non-object root).
    Creates a new config when absent.
    """

    resolved_settings = settings or get_settings()
    managed = fcc_managed_block(resolved_settings, gateway_base_url)
    config_path = path or _config_path()
    try:
        data: dict[str, object] = load_existing_config(config_path)
        existed = True
    except FileNotFoundError:
        data = {}
        existed = False
    except MalformedConfigError as exc:
        logger.warning(
            "Skipping FCC config merge: malformed config at {}: {}",
            exc.path,
            exc.reason,
        )
        return False

    prior = _ownership_record(data, tuple(managed))
    record_path = _record_path(config_path)
    previous = (
        _load_ownership_record(record_path)
        if existed and record_path.exists()
        else None
    )

    changed = False

    if data.get(_DISCOVERY_KEY) is not True:
        data[_DISCOVERY_KEY] = True
        changed = True

    inference_raw = data.get(_INFERENCE_KEY)
    inference_dict: dict[str, object] = {}
    if isinstance(inference_raw, dict):
        for key, value in inference_raw.items():
            inference_dict[str(key)] = value

    # A non-object ``inference`` after a recorded merge is the user's
    # wholesale replacement of the FCC block: they swapped the whole
    # entry for a scalar or null. The merge still writes its managed
    # mapping (configure's contract is to ensure the block exists), but
    # ownership must now restore the REPLACEMENT on unconfigure — not the
    # value from before the first merge, which would discard the user's
    # later change.
    replaced_wholesale = (
        previous is not None
        and _INFERENCE_KEY in data
        and not isinstance(inference_raw, dict)
    )

    for key, value in managed.items():
        if inference_dict.get(key) != value:
            inference_dict[key] = value
            changed = True

    if data.get(_INFERENCE_KEY) != inference_dict:
        data[_INFERENCE_KEY] = inference_dict
        changed = True

    if changed:
        prior = replace(
            prior,
            discovery=(
                previous.discovery
                if previous is not None and data.get(_DISCOVERY_KEY) is True
                else prior.discovery
            ),
            inference_written=inference_dict,
            inference_keys=(
                _absorb_user_edits(previous, tuple(managed), inference_raw)
                if previous is not None
                else prior.inference_keys
            ),
            # The container provenance (absent / object / scalar origin) is a
            # fact about the FIRST merge: this merge sees an ``inference``
            # that configure itself wrote, which would otherwise overwrite
            # the original origin with ``existed=True, was_object=True`` and
            # leave unconfigure restoring into an FCC-shaped container the
            # user never had. A wholesale user replacement is the one
            # exception: its scalar value is the new origin, so unconfigure
            # restores the replacement instead of the pre-FCC state.
            inference_existed=(
                True
                if replaced_wholesale
                else (
                    previous.inference_existed
                    if previous is not None
                    else prior.inference_existed
                )
            ),
            inference_was_object=(
                False
                if replaced_wholesale
                else (
                    previous.inference_was_object
                    if previous is not None
                    else prior.inference_was_object
                )
            ),
            inference_scalar=(
                inference_raw
                if replaced_wholesale
                else (
                    previous.inference_scalar
                    if previous is not None
                    else prior.inference_scalar
                )
            ),
        )
        _save_config(record_path, _record_payload(prior))
        _save_config(config_path, data)
    return changed


def _absorb_user_edits(
    previous: _OwnershipRecord,
    managed_keys: Sequence[str],
    current_raw: object,
) -> dict[str, _OwnershipEntry]:
    """Re-target restore values at keys the user edited between merges.

    Starts from the recorded restore targets and updates a key only when its
    live value differs from both what the last merge wrote and what the
    record would restore — that combination means a post-merge user edit,
    and reverting it on unconfigure would discard the user's change. Keys
    still at their written values keep their original restore targets.
    """

    current = current_raw if isinstance(current_raw, dict) else {}
    absorbed = dict(previous.inference_keys)
    for key in managed_keys:
        written_value = previous.inference_written.get(key)
        current_value = current.get(key)
        if current_value == written_value:
            continue  # Still FCC's value; nothing user-owned to absorb.
        restores_absent = key not in absorbed or not absorbed[key]["present"]
        if restores_absent and key not in current:
            continue  # Record already removes this key.
        if (
            not restores_absent
            and key in current
            and absorbed[key]["value"] == current_value
        ):
            continue  # Record already restores exactly this user value.
        absorbed[key] = {
            "present": key in current,
            "value": current_value,
        }
    return absorbed


def unconfigure_claude_desktop_config(path: Path | None = None) -> bool:
    """Reverse the merge recorded by ``configure`` for ``path``.

    Only the managed surface FCC actually touched is restored: values that
    existed before configure come back verbatim from the ownership record,
    keys FCC inserted are removed, and every other key is preserved. With no
    valid record nothing is owned by FCC, so the config is left untouched
    and ``False`` is returned. Also returns ``False`` without writing when
    the existing config or record is malformed, or silently when the file
    does not exist.
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

    record_path = _record_path(config_path)
    record = _load_ownership_record(record_path)
    if record is None:
        logger.warning(
            "Skipping FCC config unmerge: no FCC ownership record at {}",
            record_path,
        )
        return False

    changed = False

    # Configure always writes ``True`` here. A current value of ``True`` is
    # still FCC's to manage; anything else means the user edited it after the
    # merge and their choice wins.
    discovery = record.discovery
    if discovery["present"]:
        if (
            data.get(_DISCOVERY_KEY) is True
            and data[_DISCOVERY_KEY] != (discovery["value"])
        ):
            data[_DISCOVERY_KEY] = discovery["value"]
            changed = True
    elif data.get(_DISCOVERY_KEY) is True:
        del data[_DISCOVERY_KEY]
        changed = True

    written = record.inference_written
    if not record.inference_existed:
        changed = (
            _remove_inference_keys(data, record.inference_keys, written) or changed
        )
    elif record.inference_was_object:
        changed = (
            _restore_inference_keys(data, record.inference_keys, written) or changed
        )
    elif data.get(_INFERENCE_KEY) == written:
        # Configure clobbered a non-object entry and the entry is still
        # exactly what configure wrote; restore what it replaced.
        data[_INFERENCE_KEY] = record.inference_scalar
        changed = True
    # Any other current value is a post-merge user replacement or removal;
    # it is no longer FCC's to touch, so it stays as-is.

    if changed:
        _save_config(config_path, data)
        record_path.unlink()
    return changed


def _current_inference_object(
    data: dict[str, object],
) -> dict[str, object] | None:
    """The live ``inference`` mapping FCC wrote, or ``None`` once user-owned.

    Configure leaves ``inference`` an object; a missing entry or any other
    JSON value means the user replaced or removed it wholesale and it is no
    longer FCC's to touch.
    """

    raw = data.get(_INFERENCE_KEY)
    if not isinstance(raw, dict):
        return None
    return {str(key): value for key, value in raw.items()}


def _restore_inference_keys(
    data: dict[str, object],
    owned: dict[str, _OwnershipEntry],
    written: dict[str, object],
) -> bool:
    """Restore or remove only FCC-managed inference keys, keeping the rest.

    A key still holding exactly what configure wrote is FCC's to restore;
    any other value is a post-merge user edit and wins over the snapshot.
    """

    current = _current_inference_object(data)
    if current is None:
        return False

    changed = False
    for key, entry in owned.items():
        if current.get(key) != written.get(key):
            continue  # user-edited since the merge; leave it alone.
        if entry["present"]:
            if key not in current or current[key] != entry["value"]:
                current[key] = entry["value"]
                changed = True
        elif key in current:
            del current[key]
            changed = True

    if changed:
        data[_INFERENCE_KEY] = current
    return changed


def _remove_inference_keys(
    data: dict[str, object],
    owned: dict[str, _OwnershipEntry],
    written: dict[str, object],
) -> bool:
    """Remove FCC-inserted inference keys; drop the entry when it empties.

    Used when FCC created ``inference`` itself, so any fields added to it
    after the merge survive unconfiguration. Keys whose values diverged from
    what configure wrote are user edits and are kept.
    """

    current = _current_inference_object(data)
    if current is None:
        return False

    changed = False
    for key in owned:
        if key in current and current[key] == written.get(key):
            del current[key]
            changed = True

    if not changed:
        return False
    if current:
        data[_INFERENCE_KEY] = current
    else:
        del data[_INFERENCE_KEY]
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
