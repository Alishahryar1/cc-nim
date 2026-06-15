"""Per-provider health-check history with optional on-disk persistence.

Bounded ring buffer (50 entries per provider) of past test_provider /
local_provider_status outcomes. Persisted to ~/.fcc-cache/health_history.json
on every record; serverless-safe fallback skips persistence when the FS is
read-only (same pattern as configure_logging).
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

_MAX_ENTRIES_PER_PROVIDER = 50
_LOCK = threading.Lock()
_STORE: dict[str, deque[dict[str, Any]]] = defaultdict(
    lambda: deque(maxlen=_MAX_ENTRIES_PER_PROVIDER)
)
_PERSIST_PATH: Path | None = None
_PERSIST_DISABLED = False


def _cache_path() -> Path:
    """Resolve the persistence path, honoring FCC_CACHE_DIR override."""
    base = os.environ.get("FCC_CACHE_DIR")
    if base:
        return Path(base) / "health_history.json"
    return Path.home() / ".fcc-cache" / "health_history.json"


def _load_from_disk() -> None:
    """Best-effort restore from JSON. Silent on missing/corrupt files."""
    global _PERSIST_PATH
    _PERSIST_PATH = _cache_path()
    try:
        raw = _PERSIST_PATH.read_text(encoding="utf-8")
    except FileNotFoundError, OSError, PermissionError:
        return
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return
    if not isinstance(data, dict):
        return
    for provider_id, entries in data.items():
        if not isinstance(entries, list):
            continue
        bucket = _STORE[provider_id]
        for entry in entries[-_MAX_ENTRIES_PER_PROVIDER:]:
            if isinstance(entry, dict) and "ts" in entry and "status" in entry:
                bucket.append(entry)


def _persist_to_disk_locked() -> None:
    """Write the full store as JSON. Disables itself on read-only FS."""
    global _PERSIST_DISABLED
    if _PERSIST_DISABLED or _PERSIST_PATH is None:
        return
    payload = {pid: list(entries) for pid, entries in _STORE.items()}
    try:
        _PERSIST_PATH.parent.mkdir(parents=True, exist_ok=True)
        _PERSIST_PATH.write_text(json.dumps(payload), encoding="utf-8")
    except OSError, PermissionError:
        # Vercel/Lambda read-only FS — give up and never retry this run.
        _PERSIST_DISABLED = True


_load_from_disk()


def record(
    *,
    provider_id: str,
    status: str,
    latency_ms: float,
    error_type: str | None = None,
) -> None:
    """Record one health-check outcome for *provider_id*."""
    entry: dict[str, Any] = {
        "ts": time.time(),
        "status": status,
        "latency_ms": round(latency_ms, 1),
    }
    if error_type is not None:
        entry["error_type"] = error_type
    with _LOCK:
        _STORE[provider_id].append(entry)
        _persist_to_disk_locked()


def snapshot() -> dict[str, list[dict[str, Any]]]:
    """Return the full per-provider history (oldest first)."""
    with _LOCK:
        return {pid: list(entries) for pid, entries in _STORE.items()}


def clear() -> None:
    """Reset store (for tests)."""
    global _PERSIST_DISABLED
    with _LOCK:
        _STORE.clear()
        _PERSIST_DISABLED = False
