"""Translate Anthropic SSE streams into OpenAI Chat Completions SSE streams."""

import asyncio
import sys
import time
from collections.abc import AsyncIterable, AsyncIterator, Callable, Mapping
from typing import Any

from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.diagnostics import safe_exception_message
from free_claude_code.core.failures import ExecutionFailure, find_execution_failure
from free_claude_code.core.openai_responses import (
    openai_error_payload,
    openai_error_type_for_failure,
)
from free_claude_code.core.trace import close_stream_input

from .events import CHAT_COMPLETION_SSE_DONE, format_chat_sse_chunk
from .ids import new_chat_completion_id, new_tool_call_id
from .models import OpenAIChatCompletionsRequest
from .stop_reason import finish_reason_from_stop_reason

PostStartTerminalFailureObserver = Callable[[BaseException], None]


class ChatCompletionStreamAssembler:
    """Assemble ``chat.completion.chunk`` SSE frames from Anthropic content blocks."""

    def __init__(self, request: OpenAIChatCompletionsRequest) -> None:
        self._request = request
        self._id = new_chat_completion_id()
        self._created = int(time.time())
        self._model = request.model
        self._include_usage = request.wants_usage()
        self._role_sent = False
        self._emitted_content = False
        self._tool_index_by_block: dict[int, int] = {}
        self._next_tool_index = 0
        self._stop_reason: str | None = None
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self.terminal = False

    def process_anthropic_event(self, payload: Mapping[str, Any]) -> list[str]:
        if self.terminal:
            return []
        event_type = payload.get("type")
        if event_type == "message_start":
            self._record_message_start(payload)
            return []
        if event_type == "content_block_start":
            return self._handle_block_start(payload)
        if event_type == "content_block_delta":
            return self._handle_block_delta(payload)
        if event_type == "message_delta":
            self._record_message_delta(payload)
            return []
        if event_type == "message_stop":
            return self.finish()
        if event_type == "error":
            return self.fail_error(payload.get("error"))
        return []

    def finish(self) -> list[str]:
        if self.terminal:
            return []
        self.terminal = True
        chunks = self._ensure_role()
        finish_reason = finish_reason_from_stop_reason(
            self._stop_reason, has_tool_calls=bool(self._tool_index_by_block)
        )
        chunks.append(
            format_chat_sse_chunk(self._chunk({}, finish_reason=finish_reason))
        )
        if self._include_usage:
            chunks.append(format_chat_sse_chunk(self._usage_chunk()))
        chunks.append(CHAT_COMPLETION_SSE_DONE)
        return chunks

    def finish_incomplete_stream(self) -> list[str]:
        """Terminal handling when the provider stream ended with no message_stop.

        If the model already signaled completion through
        ``message_delta.stop_reason``, only the SSE terminator was missing, so
        finish normally. If content was streamed with no completion signal at
        all, the stream was cut off, so emit an error frame instead of a
        fabricated successful completion. An empty stream (nothing emitted)
        finishes normally, matching the Responses dialect.
        """
        if self.terminal:
            return []
        if self._stop_reason is not None or not self._emitted_content:
            return self.finish()
        self.terminal = True
        payload = openai_error_payload(
            message="Provider stream ended before completion.",
            error_type="api_error",
        )
        return [format_chat_sse_chunk(payload), CHAT_COMPLETION_SSE_DONE]

    def fail_error(self, error: Any) -> list[str]:
        if self.terminal:
            return []
        self.terminal = True
        error_type = "api_error"
        message = "Provider request failed unexpectedly."
        if isinstance(error, Mapping):
            error_type = _string_value(error.get("type")) or error_type
            message = _string_value(error.get("message")) or message
        payload = openai_error_payload(message=message, error_type=error_type)
        return [format_chat_sse_chunk(payload), CHAT_COMPLETION_SSE_DONE]

    def fail_execution(self, failure: ExecutionFailure) -> list[str]:
        if self.terminal:
            return []
        self.terminal = True
        payload = openai_error_payload(
            message=failure.message,
            error_type=openai_error_type_for_failure(failure),
        )
        return [format_chat_sse_chunk(payload), CHAT_COMPLETION_SSE_DONE]

    def fail_unexpected(self, exc: BaseException) -> list[str]:
        if self.terminal:
            return []
        self.terminal = True
        payload = openai_error_payload(
            message=safe_exception_message(exc), error_type="api_error"
        )
        return [format_chat_sse_chunk(payload), CHAT_COMPLETION_SSE_DONE]

    def _handle_block_start(self, payload: Mapping[str, Any]) -> list[str]:
        block = payload.get("content_block")
        if not isinstance(block, Mapping):
            return []
        if block.get("type") != "tool_use":
            # Text blocks stream via deltas; thinking blocks are dropped for chat.
            if block.get("type") == "text":
                return self._ensure_role()
            return []
        index = _event_index(payload)
        if index is None:
            return []
        tool_index = self._next_tool_index
        self._next_tool_index += 1
        self._tool_index_by_block[index] = tool_index
        self._emitted_content = True
        chunks = self._ensure_role()
        chunks.append(
            format_chat_sse_chunk(
                self._chunk(
                    {
                        "tool_calls": [
                            {
                                "index": tool_index,
                                "id": _string_value(block.get("id"))
                                or new_tool_call_id(),
                                "type": "function",
                                "function": {
                                    "name": _string_value(block.get("name")),
                                    "arguments": "",
                                },
                            }
                        ]
                    }
                )
            )
        )
        return chunks

    def _handle_block_delta(self, payload: Mapping[str, Any]) -> list[str]:
        delta = payload.get("delta")
        if not isinstance(delta, Mapping):
            return []
        delta_type = delta.get("type")
        if delta_type == "text_delta":
            text = _string_value(delta.get("text"))
            if not text:
                return []
            self._emitted_content = True
            chunks = self._ensure_role()
            chunks.append(format_chat_sse_chunk(self._chunk({"content": text})))
            return chunks
        if delta_type == "input_json_delta":
            index = _event_index(payload)
            tool_index = (
                self._tool_index_by_block.get(index) if index is not None else None
            )
            if tool_index is None:
                return []
            self._emitted_content = True
            partial = _string_value(delta.get("partial_json"))
            return [
                format_chat_sse_chunk(
                    self._chunk(
                        {
                            "tool_calls": [
                                {
                                    "index": tool_index,
                                    "function": {"arguments": partial},
                                }
                            ]
                        }
                    )
                )
            ]
        return []

    def _record_message_start(self, payload: Mapping[str, Any]) -> None:
        message = payload.get("message")
        if not isinstance(message, Mapping):
            return
        if model := _string_value(message.get("model")):
            self._model = model
        usage = message.get("usage")
        if isinstance(usage, Mapping):
            self._prompt_tokens = _int(usage.get("input_tokens"), self._prompt_tokens)
            self._completion_tokens = _int(
                usage.get("output_tokens"), self._completion_tokens
            )

    def _record_message_delta(self, payload: Mapping[str, Any]) -> None:
        delta = payload.get("delta")
        if isinstance(delta, Mapping):
            stop_reason = delta.get("stop_reason")
            if isinstance(stop_reason, str):
                self._stop_reason = stop_reason
        usage = payload.get("usage")
        if isinstance(usage, Mapping):
            self._completion_tokens = _int(
                usage.get("output_tokens"), self._completion_tokens
            )
            self._prompt_tokens = _int(usage.get("input_tokens"), self._prompt_tokens)

    def _ensure_role(self) -> list[str]:
        if self._role_sent:
            return []
        self._role_sent = True
        return [format_chat_sse_chunk(self._chunk({"role": "assistant"}))]

    def _chunk(
        self, delta: dict[str, Any], *, finish_reason: str | None = None
    ) -> dict[str, Any]:
        chunk: dict[str, Any] = {
            "id": self._id,
            "object": "chat.completion.chunk",
            "created": self._created,
            "model": self._model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        if self._include_usage:
            chunk["usage"] = None
        return chunk

    def _usage_chunk(self) -> dict[str, Any]:
        return {
            "id": self._id,
            "object": "chat.completion.chunk",
            "created": self._created,
            "model": self._model,
            "choices": [],
            "usage": {
                "prompt_tokens": self._prompt_tokens,
                "completion_tokens": self._completion_tokens,
                "total_tokens": self._prompt_tokens + self._completion_tokens,
            },
        }


async def iter_chat_completions_sse_from_anthropic(
    chunks: AsyncIterable[Any],
    request: OpenAIChatCompletionsRequest,
    *,
    on_post_start_terminal_failure: PostStartTerminalFailureObserver | None = None,
) -> AsyncIterator[str]:
    """Yield Chat Completions SSE frames translated from an Anthropic SSE stream."""
    assembler = ChatCompletionStreamAssembler(request)
    emitted_any_chunk = False
    buffer = ""
    iterator = aiter(chunks)
    try:
        async for chunk in iterator:
            text = (
                chunk.decode("utf-8", "replace")
                if isinstance(chunk, bytes)
                else str(chunk)
            )
            # Normalize CRLF framing/line-endings so events split on "\n\n"
            # regardless of whether the provider uses LF or CRLF.
            buffer += text.replace("\r\n", "\n")
            while "\n\n" in buffer:
                raw_event, buffer = buffer.split("\n\n", 1)
                for event in parse_sse_text(raw_event + "\n\n"):
                    for frame in assembler.process_anthropic_event(event.data):
                        yield frame
                        emitted_any_chunk = True
                    if assembler.terminal:
                        return
        for frame in assembler.finish_incomplete_stream():
            yield frame
            emitted_any_chunk = True
    except GeneratorExit, asyncio.CancelledError:
        raise
    except ExecutionFailure as exc:
        if not emitted_any_chunk:
            raise
        _observe(on_post_start_terminal_failure, exc)
        for frame in assembler.fail_execution(exc):
            yield frame
    except BaseExceptionGroup as exc:
        if not emitted_any_chunk:
            raise
        failure = find_execution_failure(exc)
        _observe(on_post_start_terminal_failure, failure or exc)
        frames = (
            assembler.fail_execution(failure)
            if failure is not None
            else assembler.fail_unexpected(exc)
        )
        for frame in frames:
            yield frame
    except Exception as exc:
        if not emitted_any_chunk:
            raise
        _observe(on_post_start_terminal_failure, exc)
        for frame in assembler.fail_unexpected(exc):
            yield frame
    finally:
        await close_stream_input(
            iterator,
            owner="openai_chat.streaming",
            source="core",
            preserved_error=sys.exception(),
        )


def _observe(
    observer: PostStartTerminalFailureObserver | None, exc: BaseException
) -> None:
    if observer is not None:
        observer(exc)


def _event_index(payload: Mapping[str, Any]) -> int | None:
    value = payload.get("index")
    return value if isinstance(value, int) else None


def _string_value(value: Any) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def _int(value: Any, default: int) -> int:
    return value if isinstance(value, int) else default
