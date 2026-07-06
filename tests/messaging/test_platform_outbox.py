from unittest.mock import AsyncMock, MagicMock

import pytest

from messaging.platforms.outbox import PlatformOutbox


def _noop_outbox(*, limiter=None, delete_many=None) -> PlatformOutbox:
    async def send(
        chat_id: str,
        text: str,
        reply_to: str | None,
        parse_mode: str | None,
        message_thread_id: str | None,
    ) -> str:
        return f"{chat_id}:{text}:{reply_to}:{parse_mode}:{message_thread_id}"

    async def edit(
        chat_id: str,
        message_id: str,
        text: str,
        parse_mode: str | None,
    ) -> None:
        return None

    async def default_delete_many(chat_id: str, message_ids: list[str]) -> None:
        return None

    return PlatformOutbox(
        get_limiter=lambda: limiter,
        send=send,
        edit=edit,
        delete_many=delete_many or default_delete_many,
    )


@pytest.mark.asyncio
async def test_queue_send_without_limiter_calls_raw_send() -> None:
    outbox = _noop_outbox()

    result = await outbox.queue_send_message(
        "chat",
        "hello",
        reply_to="reply",
        parse_mode="MarkdownV2",
        fire_and_forget=False,
        message_thread_id="thread",
    )

    assert result == "chat:hello:reply:MarkdownV2:thread"


@pytest.mark.asyncio
async def test_queue_edit_awaits_limiter_with_dedup_key() -> None:
    limiter = MagicMock()
    limiter.enqueue = AsyncMock()
    outbox = _noop_outbox(limiter=limiter)

    await outbox.queue_edit_message(
        "chat",
        "message",
        "updated",
        parse_mode="MarkdownV2",
        fire_and_forget=False,
    )

    limiter.enqueue.assert_awaited_once()
    operation = limiter.enqueue.call_args.args[0]
    assert limiter.enqueue.call_args.kwargs["dedup_key"] == "edit:chat:message"
    await operation()


@pytest.mark.asyncio
async def test_queue_delete_many_skips_empty_batches() -> None:
    limiter = MagicMock()
    outbox = _noop_outbox(limiter=limiter)

    await outbox.queue_delete_messages("chat", [], fire_and_forget=True)

    limiter.fire_and_forget.assert_not_called()


@pytest.mark.asyncio
async def test_queue_delete_many_dedups_by_batch() -> None:
    limiter = MagicMock()
    outbox = _noop_outbox(limiter=limiter)

    await outbox.queue_delete_messages("chat", ["1", "2"], fire_and_forget=True)

    limiter.fire_and_forget.assert_called_once()
    assert (
        limiter.fire_and_forget.call_args.kwargs["dedup_key"]
        == "del_bulk:chat:11f0530a8259fffb"
    )


@pytest.mark.asyncio
async def test_queue_delete_many_snapshots_ids_before_queueing() -> None:
    limiter = MagicMock()
    deleted: list[list[str]] = []

    async def delete_many(_chat_id: str, message_ids: list[str]) -> None:
        deleted.append(message_ids)

    outbox = _noop_outbox(limiter=limiter, delete_many=delete_many)
    message_ids = ["1", "2"]

    await outbox.queue_delete_messages("chat", message_ids, fire_and_forget=True)
    message_ids.append("3")
    operation = limiter.fire_and_forget.call_args.args[0]
    await operation()

    assert deleted == [["1", "2"]]
