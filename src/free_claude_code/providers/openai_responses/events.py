"""Owned Responses SSE decoding before lifecycle policy or identity normalization."""

import json
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from contextlib import aclosing
from typing import cast

import httpx
import httpx2

from free_claude_code.core.json_types import JsonObject
from free_claude_code.providers.stream_recovery import TruncatedProviderStreamError

type ResponsesEventAdapter = Callable[[str, JsonObject], JsonObject]


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
    async with aclosing(cast(AsyncGenerator[str], response.aiter_lines())) as lines:
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
                resolved_type = event_type or payload.get("type")
                if not resolved_type and payload.get("error") is not None:
                    resolved_type = "error"
                event_type = ""
                if isinstance(resolved_type, str) and resolved_type:
                    yield resolved_type, payload
                continue
            if line.startswith("event:"):
                event_type = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].lstrip())

    if data_lines:
        raise TruncatedProviderStreamError(
            "Provider Responses stream ended during an SSE event."
        )
