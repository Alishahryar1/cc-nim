"""Hold each upstream ``tool_use`` content block until it is complete.

DeepSeek (and other Anthropic-compatible upstreams) occasionally drop the
connection in the middle of a large ``tool_use`` block — typically a big
``Edit``/``Write`` whose ``input_json_delta`` stream is long. Relaying those
partial bytes hands the client a truncated tool call and, when the upstream
closes without a terminal ``message_stop``, the client SDK aborts the turn with
``API Error: stream closed before completion``.

This buffer makes a ``tool_use`` block atomic from the client's point of view:
its ``content_block_start`` + ``input_json_delta`` chunks are held until the
matching ``content_block_stop`` arrives, then flushed as a unit. If the upstream
stream ends (cleanly OR via an exception) while a ``tool_use`` block is still
open, the partial block is discarded and :class:`IncompleteUpstreamStreamError`
is raised so the orchestration layer can either

* retry the turn cleanly (when the ``tool_use`` was the first content and
  nothing reached the client yet — the transport surfaces this as a
  :class:`providers.exceptions.PreStreamProviderError`), or
* emit a well-formed error tail (when earlier text already reached the client).

Either way the client never receives a truncated tool call.

Pings are forwarded untouched during buffering so the connection stays alive
while a long tool call is generated upstream.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from core.anthropic.native_sse_block_policy import parse_native_sse_event

__all__ = ["IncompleteUpstreamStreamError", "buffer_incomplete_tool_use"]


class IncompleteUpstreamStreamError(Exception):
    """Upstream SSE stream ended with a ``tool_use`` block left open/incomplete.

    Carries ``emitted_visible_content`` purely for tracing/diagnostics; the
    transport decides retry-vs-error-tail from its own ``sent_any_event`` state.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


def _frame_event_and_block_type(frame: str) -> tuple[str | None, str | None]:
    """Return ``(event_name, content_block_type)`` for one SSE frame.

    ``content_block_type`` is only populated for ``content_block_start`` frames.
    Any parsing failure degrades to ``(event_name_or_None, None)`` so unknown or
    malformed frames are always forwarded rather than dropped or mis-held.
    """
    event_name, data_text = parse_native_sse_event(frame)
    if not event_name:
        return None, None
    if event_name != "content_block_start" or not data_text:
        return event_name, None
    try:
        payload = json.loads(data_text)
    except (json.JSONDecodeError, ValueError):
        return event_name, None
    block = payload.get("content_block") if isinstance(payload, dict) else None
    if isinstance(block, dict):
        block_type = block.get("type")
        return event_name, block_type if isinstance(block_type, str) else None
    return event_name, None


async def buffer_incomplete_tool_use(
    chunks: AsyncIterator[str],
) -> AsyncIterator[str]:
    """Re-yield SSE chunks, holding each ``tool_use`` block until it is complete.

    Forwarded frames are replayed using their original chunk boundaries whenever
    a frame ends exactly on a chunk boundary (the common line-mode case), so the
    output chunking is unchanged for everything except a held ``tool_use`` block
    (which is necessarily regrouped — but transparent to any SSE consumer). Raises
    :class:`IncompleteUpstreamStreamError` if the source is exhausted — or emits a
    message terminator — while a ``tool_use`` block is still open; the partial
    block's bytes are discarded.
    """
    buf = ""
    frame_chunks: list[str] = []  # raw chunks composing the in-progress frame
    held: list[str] = []  # raw chunks of the held (incomplete) tool_use block
    holding = False

    async for chunk in chunks:
        buf += chunk
        frame_chunks.append(chunk)
        while True:
            sep = buf.find("\n\n")
            if sep < 0:
                break
            frame = buf[: sep + 2]
            buf = buf[sep + 2 :]
            if buf == "":
                # Frame ended exactly at a chunk boundary: replay raw chunks.
                raw = frame_chunks
                frame_chunks = []
            else:
                # A chunk carried bytes past this frame boundary: emit the
                # regrouped frame and seed the next frame with the remainder.
                raw = [frame]
                frame_chunks = [buf]

            event_name, block_type = _frame_event_and_block_type(frame)

            if holding:
                if event_name == "ping":
                    # Keep-alive: forward immediately without disturbing the
                    # held block (pings are positionally irrelevant to clients).
                    for piece in raw:
                        yield piece
                elif event_name == "content_block_stop":
                    holding = False
                    for piece in held:
                        yield piece
                    held = []
                    for piece in raw:
                        yield piece
                elif event_name in ("message_delta", "message_stop"):
                    # Message terminated without closing the tool_use block: the
                    # held block is incomplete. Discard it and signal so the
                    # transport retries or emits a clean error tail.
                    held = []
                    raise IncompleteUpstreamStreamError(
                        "upstream ended the message without closing a tool_use block"
                    )
                else:
                    held.extend(raw)
                continue

            if event_name == "content_block_start" and block_type == "tool_use":
                holding = True
                held = list(raw)
                continue

            for piece in raw:
                yield piece

    # Source exhausted. Any leftover partial frame in ``buf`` (no terminating
    # blank line) is an incomplete event and is dropped. A still-open tool_use
    # block means the upstream cut out mid tool call.
    if holding:
        raise IncompleteUpstreamStreamError(
            "upstream stream ended mid tool_use block (no content_block_stop)"
        )
