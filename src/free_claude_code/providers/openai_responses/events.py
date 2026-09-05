"""Owned Responses SSE decoding before lifecycle policy or identity normalization."""

import json
import re
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from contextlib import aclosing
from typing import cast

import httpx
import httpx2

from free_claude_code.core.json_types import JsonObject
from free_claude_code.providers.stream_recovery import TruncatedProviderStreamError

type ResponsesEventAdapter = Callable[[str, JsonObject], JsonObject]
_LINE_END = re.compile(rb"\r\n|\r|\n")


class ResponsesEventSource:
    """Own one decoder; the attempt resource scope owns its HTTP response too."""

    def __init__(
        self,
        response: httpx.Response | httpx2.Response,
        *,
        adapter: ResponsesEventAdapter | None = None,
    ) -> None:
        self._events = _decode_sse(response)
        self._adapter = adapter

    def __aiter__(self) -> AsyncIterator[tuple[str, JsonObject]]:
        return self

    async def __anext__(self) -> tuple[str, JsonObject]:
        return await anext(self._events)

    def normalize(self, event_type: str, payload: JsonObject) -> JsonObject:
        return (
            self._adapter(event_type, payload) if self._adapter is not None else payload
        )

    async def aclose(self) -> None:
        await self._events.aclose()


async def _decode_sse(
    response: httpx.Response | httpx2.Response,
) -> AsyncGenerator[tuple[str, JsonObject]]:
    event_type = ""
    data_lines: list[str] = []
    async with aclosing(_sse_lines(response)) as lines:
        async for line in lines:
            if not line:
                if not data_lines:
                    event_type = ""
                    continue
                raw_data = "\n".join(data_lines)
                data_lines = []
                if raw_data == "[DONE]":
                    return
                try:
                    payload = json.loads(raw_data)
                except json.JSONDecodeError as exc:
                    raise TruncatedProviderStreamError(
                        "Provider returned malformed Responses SSE."
                    ) from exc
                if not isinstance(payload, dict):
                    raise TruncatedProviderStreamError(
                        "Provider returned a non-object Responses event."
                    )
                # Framing carries the SSE label unchanged. The Responses
                # interpreter owns payload types and structured error evidence.
                resolved_type = event_type
                event_type = ""
                yield resolved_type, payload
                continue
            field, _, value = line.partition(":")
            value = value.removeprefix(" ")
            if field == "event":
                event_type = value
            elif field == "data":
                data_lines.append(value)

    if data_lines:
        raise TruncatedProviderStreamError(
            "Provider Responses stream ended during an SSE event."
        )


async def _sse_lines(
    response: httpx.Response | httpx2.Response,
) -> AsyncGenerator[str]:
    """Frame CR/LF bytes before UTF-8 decoding; Unicode separators are data."""
    parts: list[bytes] = []
    skip_lf = False
    encoding = "utf-8-sig"
    async with aclosing(cast(AsyncGenerator[bytes], response.aiter_bytes())) as chunks:
        async for chunk in chunks:
            if not chunk:
                continue
            if skip_lf:
                chunk = chunk.removeprefix(b"\n")
                skip_lf = False
            start = 0
            for boundary in _LINE_END.finditer(chunk):
                parts.append(chunk[start : boundary.start()])
                line = b"".join(parts)
                parts.clear()
                start = boundary.end()
                skip_lf = boundary.group() == b"\r" and start == len(chunk)
                yield line.decode(encoding, errors="replace")
                encoding = "utf-8"
            if start < len(chunk):
                parts.append(chunk[start:])
    if parts:
        yield b"".join(parts).decode(encoding, errors="replace")
