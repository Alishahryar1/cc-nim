"""Shared queued delivery helper for messaging platforms."""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, cast

SendOperation = Callable[
    [str, str, str | None, str | None, str | None],
    Awaitable[str],
]
EditOperation = Callable[[str, str, str, str | None], Awaitable[None]]
DeleteOperation = Callable[[str, str], Awaitable[None]]
DeleteManyOperation = Callable[[str, list[str]], Awaitable[None]]
LimiterGetter = Callable[[], Any | None]


class PlatformOutbox:
    """Own queueing, deduplication, and fire-and-forget delivery policy."""

    def __init__(
        self,
        *,
        get_limiter: LimiterGetter,
        send: SendOperation,
        edit: EditOperation,
        delete: DeleteOperation,
        delete_many: DeleteManyOperation,
    ) -> None:
        self._get_limiter = get_limiter
        self._send = send
        self._edit = edit
        self._delete = delete
        self._delete_many = delete_many

    async def queue_send_message(
        self,
        chat_id: str,
        text: str,
        reply_to: str | None = None,
        parse_mode: str | None = None,
        fire_and_forget: bool = True,
        message_thread_id: str | None = None,
    ) -> str | None:
        """Queue or immediately send a platform message."""
        limiter = self._get_limiter()
        if limiter is None:
            return await self._send(
                chat_id,
                text,
                reply_to,
                parse_mode,
                message_thread_id,
            )

        async def _send() -> str:
            return await self._send(
                chat_id,
                text,
                reply_to,
                parse_mode,
                message_thread_id,
            )

        if fire_and_forget:
            limiter.fire_and_forget(_send)
            return None
        return cast(str | None, await limiter.enqueue(_send))

    async def queue_edit_message(
        self,
        chat_id: str,
        message_id: str,
        text: str,
        parse_mode: str | None = None,
        fire_and_forget: bool = True,
    ) -> None:
        """Queue or immediately edit a platform message."""
        limiter = self._get_limiter()
        if limiter is None:
            await self._edit(chat_id, message_id, text, parse_mode)
            return

        async def _edit() -> None:
            await self._edit(chat_id, message_id, text, parse_mode)

        dedup_key = f"edit:{chat_id}:{message_id}"
        if fire_and_forget:
            limiter.fire_and_forget(_edit, dedup_key=dedup_key)
        else:
            await limiter.enqueue(_edit, dedup_key=dedup_key)

    async def queue_delete_message(
        self,
        chat_id: str,
        message_id: str,
        fire_and_forget: bool = True,
    ) -> None:
        """Queue or immediately delete a platform message."""
        limiter = self._get_limiter()
        if limiter is None:
            await self._delete(chat_id, message_id)
            return

        async def _delete() -> None:
            await self._delete(chat_id, message_id)

        dedup_key = f"del:{chat_id}:{message_id}"
        if fire_and_forget:
            limiter.fire_and_forget(_delete, dedup_key=dedup_key)
        else:
            await limiter.enqueue(_delete, dedup_key=dedup_key)

    async def queue_delete_messages(
        self,
        chat_id: str,
        message_ids: list[str],
        fire_and_forget: bool = True,
    ) -> None:
        """Queue or immediately bulk-delete platform messages."""
        if not message_ids:
            return

        limiter = self._get_limiter()
        if limiter is None:
            await self._delete_many(chat_id, message_ids)
            return

        async def _delete_many() -> None:
            await self._delete_many(chat_id, message_ids)

        dedup_key = f"del_bulk:{chat_id}:{hash(tuple(message_ids))}"
        if fire_and_forget:
            limiter.fire_and_forget(_delete_many, dedup_key=dedup_key)
        else:
            await limiter.enqueue(_delete_many, dedup_key=dedup_key)

    def fire_and_forget(self, task: Awaitable[Any]) -> None:
        """Execute a coroutine or future without awaiting it."""
        if asyncio.iscoroutine(task):
            _ = asyncio.create_task(task)
        else:
            _ = asyncio.ensure_future(task)
