"""Translate Anthropic-style responses into OpenAI Responses streams."""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator, Mapping
from typing import Any

from .anthropic_sse import iter_sse_events
from .events import format_response_sse_event
from .ids import new_response_id
from .items import (
    base_response,
    in_progress_item,
    message_item_text,
    reasoning_item_text,
)
from .output import convert_message_to_response
from .stream_state import ResponsesStreamAssembler


async def iter_responses_sse_from_anthropic(
    chunks: AsyncIterable[Any],
    request: Mapping[str, Any],
) -> AsyncIterator[str]:
    """Yield Responses SSE events translated from an Anthropic SSE stream."""

    assembler = ResponsesStreamAssembler(request)
    async for event in iter_sse_events(chunks):
        for chunk in assembler.process_anthropic_event(event):
            yield chunk
        if assembler.terminal:
            return
    for chunk in assembler.finish_if_needed():
        yield chunk


async def collect_response_from_anthropic_sse(
    chunks: AsyncIterable[Any],
    request: Mapping[str, Any],
) -> dict[str, Any]:
    """Collect a translated Anthropic SSE stream into one Responses object."""

    assembler = ResponsesStreamAssembler(request)
    async for event in iter_sse_events(chunks):
        assembler.process_anthropic_event(event)
        if assembler.terminal:
            break
    if assembler.final_response is not None:
        return assembler.final_response
    assembler.finish_if_needed()
    return assembler.final_response or assembler.response_payload(status="completed")


def iter_responses_sse_from_message(
    message: Mapping[str, Any],
    request: Mapping[str, Any],
) -> list[str]:
    """Return Responses SSE chunks for a non-stream Anthropic message response."""

    response_id = new_response_id()
    response = base_response(request, response_id=response_id, status="in_progress")
    chunks = [
        format_response_sse_event(
            "response.created",
            {"type": "response.created", "response": response},
        )
    ]
    completed = convert_message_to_response(message, request, response_id=response_id)
    for output_index, item in enumerate(completed["output"]):
        chunks.append(
            format_response_sse_event(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "output_index": output_index,
                    "item": in_progress_item(item),
                },
            )
        )
        if item.get("type") == "message":
            text = message_item_text(item)
            chunks.extend(_message_text_events(item, output_index, text))
        elif item.get("type") == "reasoning":
            text = reasoning_item_text(item)
            if text:
                chunks.extend(_reasoning_text_events(item, output_index, text))
        elif item.get("type") == "function_call":
            chunks.extend(_function_call_events(item, output_index))
        elif item.get("type") == "custom_tool_call":
            chunks.extend(_custom_tool_call_events(item, output_index))
        chunks.append(
            format_response_sse_event(
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "output_index": output_index,
                    "item": item,
                },
            )
        )
    chunks.append(
        format_response_sse_event(
            "response.completed",
            {"type": "response.completed", "response": completed},
        )
    )
    return chunks


def _function_call_events(item: Mapping[str, Any], output_index: int) -> list[str]:
    arguments = str(item.get("arguments", ""))
    chunks: list[str] = []
    if arguments:
        chunks.append(
            format_response_sse_event(
                "response.function_call_arguments.delta",
                {
                    "type": "response.function_call_arguments.delta",
                    "item_id": item.get("id"),
                    "output_index": output_index,
                    "delta": arguments,
                },
            )
        )
    chunks.append(
        format_response_sse_event(
            "response.function_call_arguments.done",
            {
                "type": "response.function_call_arguments.done",
                "item_id": item.get("id"),
                "output_index": output_index,
                "arguments": arguments,
            },
        )
    )
    return chunks


def _custom_tool_call_events(item: Mapping[str, Any], output_index: int) -> list[str]:
    input_text = str(item.get("input", ""))
    chunks: list[str] = []
    if input_text:
        chunks.append(
            format_response_sse_event(
                "response.custom_tool_call_input.delta",
                {
                    "type": "response.custom_tool_call_input.delta",
                    "item_id": item.get("id"),
                    "output_index": output_index,
                    "delta": input_text,
                },
            )
        )
    chunks.append(
        format_response_sse_event(
            "response.custom_tool_call_input.done",
            {
                "type": "response.custom_tool_call_input.done",
                "item_id": item.get("id"),
                "output_index": output_index,
                "input": input_text,
            },
        )
    )
    return chunks


def _message_text_events(
    item: Mapping[str, Any], output_index: int, text: str
) -> list[str]:
    item_id = str(item.get("id", ""))
    return [
        format_response_sse_event(
            "response.content_part.added",
            {
                "type": "response.content_part.added",
                "item_id": item_id,
                "output_index": output_index,
                "content_index": 0,
                "part": {"type": "output_text", "text": "", "annotations": []},
            },
        ),
        format_response_sse_event(
            "response.output_text.delta",
            {
                "type": "response.output_text.delta",
                "item_id": item_id,
                "output_index": output_index,
                "content_index": 0,
                "delta": text,
            },
        ),
        format_response_sse_event(
            "response.output_text.done",
            {
                "type": "response.output_text.done",
                "item_id": item_id,
                "output_index": output_index,
                "content_index": 0,
                "text": text,
            },
        ),
        format_response_sse_event(
            "response.content_part.done",
            {
                "type": "response.content_part.done",
                "item_id": item_id,
                "output_index": output_index,
                "content_index": 0,
                "part": {"type": "output_text", "text": text, "annotations": []},
            },
        ),
    ]


def _reasoning_text_events(
    item: Mapping[str, Any], output_index: int, text: str
) -> list[str]:
    item_id = str(item.get("id", ""))
    return [
        format_response_sse_event(
            "response.reasoning_text.delta",
            {
                "type": "response.reasoning_text.delta",
                "item_id": item_id,
                "output_index": output_index,
                "content_index": 0,
                "delta": text,
            },
        ),
        format_response_sse_event(
            "response.reasoning_text.done",
            {
                "type": "response.reasoning_text.done",
                "item_id": item_id,
                "output_index": output_index,
                "content_index": 0,
                "text": text,
            },
        ),
    ]
