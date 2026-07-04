"""SSE streaming for local web_search / web_fetch server tool results."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

from api.models.anthropic import MessagesRequest
from core.anthropic.server_tool_sse import (
    SERVER_TOOL_USE,
    WEB_FETCH_TOOL_ERROR,
    WEB_FETCH_TOOL_RESULT,
    WEB_SEARCH_TOOL_RESULT,
    WEB_SEARCH_TOOL_RESULT_ERROR,
)
from core.anthropic.streaming import format_sse_event

from . import outbound
from .constants import _MAX_FETCH_CHARS
from .egress import WebFetchEgressPolicy
from .parsers import extract_query, extract_url
from .request import (
    forced_server_tool_name,
    forced_tool_turn_text,
    has_tool_named,
)


def _search_summary(query: str, results: list[dict[str, str]]) -> str:
    if not results:
        return f"No web search results found for: {query}"
    lines = [f"Search results for: {query}"]
    for index, result in enumerate(results, start=1):
        lines.append(f"{index}. {result['title']}\n{result['url']}")
    return "\n\n".join(lines)


async def stream_web_server_tool_response(
    request: MessagesRequest,
    input_tokens: int,
    *,
    web_fetch_egress: WebFetchEgressPolicy,
    verbose_client_errors: bool = False,
) -> AsyncIterator[str]:
    """Stream a minimal Anthropic-shaped turn for forced `web_search` / `web_fetch` (local fallback).

    When `ENABLE_WEB_SERVER_TOOLS` is on, this is a proxy-side execution path — not a full
    hosted Anthropic citation or encrypted-content pipeline.
    """
    tool_name = forced_server_tool_name(request)
    if tool_name is None or not has_tool_named(request, tool_name):
        return

    text = forced_tool_turn_text(request)
    message_id = f"msg_{uuid.uuid4()}"
    tool_id = f"srvtoolu_{uuid.uuid4().hex}"
    usage_key = (
        "web_search_requests" if tool_name == "web_search" else "web_fetch_requests"
    )
    tool_input = (
        {"query": extract_query(text)}
        if tool_name == "web_search"
        else {"url": extract_url(text)}
    )
    _result_block_for_tool = {
        "web_search": WEB_SEARCH_TOOL_RESULT,
        "web_fetch": WEB_FETCH_TOOL_RESULT,
    }
    _error_payload_type_for_tool = {
        "web_search": WEB_SEARCH_TOOL_RESULT_ERROR,
        "web_fetch": WEB_FETCH_TOOL_ERROR,
    }

    yield format_sse_event(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": request.model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": input_tokens, "output_tokens": 1},
            },
        },
    )
    yield format_sse_event(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": SERVER_TOOL_USE,
                "id": tool_id,
                "name": tool_name,
                "input": tool_input,
            },
        },
    )
    yield format_sse_event(
        "content_block_stop", {"type": "content_block_stop", "index": 0}
    )

    try:
        if tool_name == "web_search":
            query = str(tool_input["query"])
            results = await outbound._run_web_search(query)
            result_content: Any = [
                {
                    "type": "web_search_result",
                    "title": result["title"],
                    "url": result["url"],
                }
                for result in results
            ]
            summary = _search_summary(query, results)
            result_block_type = WEB_SEARCH_TOOL_RESULT
        else:
            fetched = await outbound._run_web_fetch(
                str(tool_input["url"]), web_fetch_egress
            )
            result_content = {
                "type": "web_fetch_result",
                "url": fetched["url"],
                "content": {
                    "type": "document",
                    "source": {
                        "type": "text",
                        "media_type": fetched["media_type"],
                        "data": fetched["data"],
                    },
                    "title": fetched["title"],
                    "citations": {"enabled": True},
                },
                "retrieved_at": datetime.now(UTC).isoformat(),
            }
            summary = fetched["data"][:_MAX_FETCH_CHARS]
            result_block_type = WEB_FETCH_TOOL_RESULT
    except Exception as error:
        fetch_url = str(tool_input["url"]) if tool_name == "web_fetch" else None
        outbound._log_web_tool_failure(tool_name, error, fetch_url=fetch_url)
        result_block_type = _result_block_for_tool[tool_name]
        result_content = {
            "type": _error_payload_type_for_tool[tool_name],
            "error_code": "unavailable",
        }
        summary = outbound._web_tool_client_error_summary(
            tool_name, error, verbose=verbose_client_errors
        )

    output_tokens = max(1, len(summary) // 4)

    yield format_sse_event(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": result_block_type,
                "tool_use_id": tool_id,
                "content": result_content,
            },
        },
    )
    yield format_sse_event(
        "content_block_stop", {"type": "content_block_stop", "index": 1}
    )
    # Model-facing summary: stream as normal text deltas (CLI/transcript code reads `text_delta`,
    # not eager `text` on `content_block_start`).
    yield format_sse_event(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 2,
            "content_block": {"type": "text", "text": ""},
        },
    )
    yield format_sse_event(
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": 2,
            "delta": {"type": "text_delta", "text": summary},
        },
    )
    yield format_sse_event(
        "content_block_stop", {"type": "content_block_stop", "index": 2}
    )
    yield format_sse_event(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "server_tool_use": {usage_key: 1},
            },
        },
    )
    yield format_sse_event("message_stop", {"type": "message_stop"})
