"""SSE streaming for local web_search / web_fetch server-tool results."""

import uuid
from collections.abc import AsyncIterator, Callable, Mapping
from datetime import UTC, datetime

from free_claude_code.core.anthropic import MessagesRequest
from free_claude_code.core.anthropic.server_tool_sse import (
    SERVER_TOOL_USE,
    WEB_FETCH_TOOL_ERROR,
    WEB_FETCH_TOOL_RESULT,
    WEB_SEARCH_TOOL_RESULT,
    WEB_SEARCH_TOOL_RESULT_ERROR,
)
from free_claude_code.core.anthropic.streaming import format_sse_event
from free_claude_code.core.json_types import JsonValue

from . import outbound
from .constants import _MAX_FETCH_CHARS
from .egress import WebFetchEgressPolicy
from .parsers import extract_query, extract_url
from .request import forced_server_tool_name, forced_tool_turn_text, has_tool_named

type SearchResultFilter = Callable[
    [list[dict[str, str]]],
    list[dict[str, str]],
]


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
    response_model: str | None = None,
    verbose_client_errors: bool = False,
) -> AsyncIterator[str]:
    """Stream the existing forced `web_search` / `web_fetch` local fallback."""
    tool_name = forced_server_tool_name(request)
    if tool_name is None or not has_tool_named(request, tool_name):
        return

    text = forced_tool_turn_text(request)
    tool_input = (
        {"query": extract_query(text)}
        if tool_name == "web_search"
        else {"url": extract_url(text)}
    )
    async for frame in _stream_local_web_tool_response(
        tool_name=tool_name,
        tool_input=tool_input,
        input_tokens=input_tokens,
        web_fetch_egress=web_fetch_egress,
        response_model=request.model if response_model is None else response_model,
        verbose_client_errors=verbose_client_errors,
    ):
        yield frame


async def stream_selected_web_search_response(
    query: str,
    *,
    input_tokens: int,
    response_model: str,
    provider_usage: Mapping[str, object],
    result_filter: SearchResultFilter,
    verbose_client_errors: bool = False,
) -> AsyncIterator[str]:
    """Stream the existing local result shape for one provider-selected query."""
    async for frame in _stream_local_web_tool_response(
        tool_name="web_search",
        tool_input={"query": query},
        input_tokens=input_tokens,
        web_fetch_egress=None,
        response_model=response_model,
        verbose_client_errors=verbose_client_errors,
        provider_usage=provider_usage,
        search_result_filter=result_filter,
    ):
        yield frame


async def _stream_local_web_tool_response(
    *,
    tool_name: str,
    tool_input: dict[str, str],
    input_tokens: int,
    web_fetch_egress: WebFetchEgressPolicy | None,
    response_model: str,
    verbose_client_errors: bool,
    provider_usage: Mapping[str, object] | None = None,
    search_result_filter: SearchResultFilter | None = None,
) -> AsyncIterator[str]:
    message_id = f"msg_{uuid.uuid4()}"
    tool_id = f"srvtoolu_{uuid.uuid4().hex}"
    usage_key = (
        "web_search_requests" if tool_name == "web_search" else "web_fetch_requests"
    )
    result_block_for_tool = {
        "web_search": WEB_SEARCH_TOOL_RESULT,
        "web_fetch": WEB_FETCH_TOOL_RESULT,
    }
    error_payload_type_for_tool = {
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
                "model": response_model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": _message_start_usage(input_tokens, provider_usage),
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
            query = tool_input["query"]
            results = await outbound._run_web_search(query)
            if search_result_filter is not None:
                results = search_result_filter(results)
            result_content: JsonValue = [
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
            if web_fetch_egress is None:
                raise RuntimeError("web_fetch requires an egress policy")
            fetched = await outbound._run_web_fetch(tool_input["url"], web_fetch_egress)
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
        fetch_url = tool_input.get("url") if tool_name == "web_fetch" else None
        outbound._log_web_tool_failure(tool_name, error, fetch_url=fetch_url)
        result_block_type = result_block_for_tool[tool_name]
        result_content = {
            "type": error_payload_type_for_tool[tool_name],
            "error_code": "unavailable",
        }
        summary = outbound._web_tool_client_error_summary(
            tool_name, error, verbose=verbose_client_errors
        )

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
            "usage": _completion_usage(
                input_tokens,
                summary,
                provider_usage,
                usage_key=usage_key,
            ),
        },
    )
    yield format_sse_event("message_stop", {"type": "message_stop"})


def _message_start_usage(
    input_tokens: int,
    provider_usage: Mapping[str, object] | None,
) -> dict[str, object]:
    if provider_usage is None:
        return {"input_tokens": input_tokens, "output_tokens": 1}
    usage = _integer_usage(provider_usage)
    usage.setdefault("input_tokens", input_tokens)
    usage["output_tokens"] = 1
    return usage


def _completion_usage(
    input_tokens: int,
    summary: str,
    provider_usage: Mapping[str, object] | None,
    *,
    usage_key: str,
) -> dict[str, object]:
    if provider_usage is None:
        usage: dict[str, object] = {
            "input_tokens": input_tokens,
            "output_tokens": max(1, len(summary) // 4),
        }
    else:
        usage = _integer_usage(provider_usage)
        usage.setdefault("input_tokens", input_tokens)
        usage.setdefault("output_tokens", max(1, len(summary) // 4))
    usage["server_tool_use"] = {usage_key: 1}
    return usage


def _integer_usage(usage: Mapping[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in usage.items()
        if isinstance(value, int) and not isinstance(value, bool)
    }
