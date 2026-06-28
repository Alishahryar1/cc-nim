"""Atomic JSON persistence for messaging session state."""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
import threading
from collections.abc import Callable
from typing import Any

from loguru import logger


class DebouncedJsonPersistence:
    """Thread-safe debounced JSON writer with atomic replace semantics."""

    def __init__(
        self,
        storage_path: str,
        *,
        snapshot: Callable[[], dict[str, Any]],
        on_dirty: Callable[[bool], None],
        debounce_secs: float = 0.5,
    ) -> None:
        self.storage_path = storage_path
        self._snapshot = snapshot
        self._on_dirty = on_dirty
        self._debounce_secs = debounce_secs
        self._save_timer: threading.Timer | None = None

    def load_json(self) -> dict[str, Any]:
        if not os.path.exists(self.storage_path):
            return {}
        with open(self.storage_path, encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, dict) else {}

    def schedule_save(self) -> None:
        self._on_dirty(True)
        if self._save_timer is not None:
            self._save_timer.cancel()
            self._save_timer = None
        self._save_timer = threading.Timer(self._debounce_secs, self._save_from_timer)
        self._save_timer.daemon = True
        self._save_timer.start()

    def flush(self) -> None:
        snapshot = self._snapshot_for_write()
        try:
            self.write_data(snapshot)
        except Exception as e:
            logger.error("Failed to save sessions: {}", e)
            self._on_dirty(True)

    def _save_from_timer(self) -> None:
        snapshot = self._snapshot_for_write()
        try:
            self.write_data(snapshot)
        except Exception as e:
            logger.error("Failed to save sessions: {}", e)
            self._on_dirty(True)

    def _snapshot_for_write(self) -> dict[str, Any]:
        if self._save_timer is not None:
            self._save_timer.cancel()
            self._save_timer = None
        snapshot = self._snapshot()
        self._on_dirty(False)
        return snapshot

    def write_data(self, data: dict[str, Any]) -> None:
        abs_target = os.path.abspath(self.storage_path)
        dir_name = os.path.dirname(abs_target) or "."
        fd, tmp_path = tempfile.mkstemp(
            dir=dir_name,
            prefix=".sessions.",
            suffix=".tmp.json",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                json.dump(data, file, indent=2)
                file.flush()
                os.fsync(file.fileno())
            os.replace(tmp_path, abs_target)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise
