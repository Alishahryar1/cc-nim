"""Per-chat message ID log used by messaging clear commands."""

from datetime import UTC, datetime
from typing import Any


class MessageLog:
    """Track inbound/outbound platform message IDs in insertion order."""

    def __init__(self, *, cap: int | None = None) -> None:
        self._items: dict[str, list[dict[str, Any]]] = {}
        self._ids: dict[str, set[str]] = {}
        self._cap = cap

    @property
    def cap(self) -> int | None:
        return self._cap

    @classmethod
    def from_json(cls, raw_log: Any, *, cap: int | None = None) -> MessageLog:
        log = cls(cap=cap)
        if not isinstance(raw_log, dict):
            return log
        for chat_key, items in raw_log.items():
            if not isinstance(chat_key, str) or not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                message_id = item.get("message_id")
                if message_id is None:
                    continue
                log._append(
                    chat_key,
                    str(message_id),
                    ts=str(item.get("ts") or ""),
                    direction=str(item.get("direction") or ""),
                    kind=str(item.get("kind") or ""),
                )
        return log

    def to_json(self) -> dict[str, list[dict[str, Any]]]:
        return {chat_key: list(items) for chat_key, items in self._items.items()}

    def record(
        self,
        *,
        platform: str,
        chat_id: str,
        message_id: str,
        direction: str,
        kind: str,
    ) -> bool:
        chat_key = make_chat_key(platform, chat_id)
        return self._append(
            chat_key,
            str(message_id),
            ts=datetime.now(UTC).isoformat(),
            direction=str(direction),
            kind=str(kind),
        )

    def get_message_ids_for_chat(self, platform: str, chat_id: str) -> list[str]:
        chat_key = make_chat_key(platform, chat_id)
        return [
            str(item.get("message_id"))
            for item in self._items.get(chat_key, [])
            if item.get("message_id") is not None
        ]

    def clear(self) -> None:
        self._items.clear()
        self._ids.clear()

    def _append(
        self,
        chat_key: str,
        message_id: str,
        *,
        ts: str,
        direction: str,
        kind: str,
    ) -> bool:
        seen = self._ids.setdefault(chat_key, set())
        if message_id in seen:
            return False
        self._items.setdefault(chat_key, []).append(
            {
                "message_id": message_id,
                "ts": ts,
                "direction": direction,
                "kind": kind,
            }
        )
        seen.add(message_id)
        self._trim(chat_key)
        return True

    def _trim(self, chat_key: str) -> None:
        if self._cap is None or self._cap <= 0:
            return
        items = self._items.get(chat_key, [])
        if len(items) <= self._cap:
            return
        self._items[chat_key] = items[-self._cap :]
        self._ids[chat_key] = {
            str(item.get("message_id"))
            for item in self._items[chat_key]
            if item.get("message_id") is not None
        }


def make_chat_key(platform: str, chat_id: str) -> str:
    return f"{platform}:{chat_id}"
