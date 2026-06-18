"""OpenAI-chat stream recovery event construction."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterator
from typing import Any

from core.anthropic import SSEBuilder
from core.anthropic.stream_recovery import (
    MIDSTREAM_RECOVERY_ATTEMPTS,
    accept_tool_json_repair,
    continuation_suffix,
    is_retryable_stream_error,
    make_openai_text_recovery_body,
    make_openai_tool_repair_body,
    parse_complete_tool_input,
    tool_schemas_by_name,
)
from core.trace import trace_event

from .tool_calls import all_started_tools_complete, started_tool_states

CreateStream = Callable[[dict[str, Any]], Awaitable[tuple[Any, dict[str, Any]]]]


class OpenAIChatRecovery:
    """Construct recovery events for interrupted OpenAI-chat streams."""

    def __init__(self, *, provider_name: str, create_stream: CreateStream) -> None:
        self._provider_name = provider_name
        self._create_stream = create_stream

    async def collect_text(self, body: dict[str, Any]) -> tuple[str, str]:
        """Collect text/reasoning from an internal recovery request."""
        last_error: Exception | None = None
        for attempt in range(MIDSTREAM_RECOVERY_ATTEMPTS):
            try:
                stream, _ = await self._create_stream(body)
                text_parts: list[str] = []
                thinking_parts: list[str] = []
                async for chunk in stream:
                    if not getattr(chunk, "choices", None):
                        continue
                    choice = chunk.choices[0]
                    delta = choice.delta
                    if delta is None:
                        continue
                    reasoning = getattr(delta, "reasoning_content", None)
                    if isinstance(reasoning, str) and reasoning:
                        thinking_parts.append(reasoning)
                    content = getattr(delta, "content", None)
                    if isinstance(content, str) and content:
                        text_parts.append(content)
                return "".join(text_parts), "".join(thinking_parts)
            except Exception as error:
                last_error = error
                if not is_retryable_stream_error(error):
                    raise
                trace_event(
                    stage="provider",
                    event="provider.recovery.retry",
                    source="provider",
                    provider=self._provider_name,
                    recovery_kind="openai_text",
                    attempt=attempt + 1,
                    max_attempts=MIDSTREAM_RECOVERY_ATTEMPTS,
                    exc_type=type(error).__name__,
                )
        if last_error is not None:
            raise last_error
        return "", ""

    async def events(
        self,
        *,
        body: dict[str, Any],
        sse: SSEBuilder,
        request: Any,
        request_id: str | None,
        error: Exception,
        tool_argument_alias_buffers: dict[int, str],
    ) -> list[str] | None:
        """Build recovery events, or return None when recovery is impossible."""
        if not is_retryable_stream_error(error):
            return None

        if sse.blocks.has_emitted_tool_block():
            if not all_started_tools_complete(sse, request):
                repair_events = await self._repair_tool_args(
                    body=body,
                    sse=sse,
                    request=request,
                    tool_argument_alias_buffers=tool_argument_alias_buffers,
                )
                if repair_events is None:
                    return None
            else:
                repair_events = []
            events = list(repair_events)
            events.extend(sse.close_all_blocks())
            events.append(sse.message_delta("tool_use", sse.estimate_output_tokens()))
            events.append(sse.message_stop())
            trace_event(
                stage="provider",
                event="provider.recovery.tool_salvaged",
                source="provider",
                provider=self._provider_name,
                request_id=request_id,
            )
            return events

        partial_text = sse.accumulated_text
        partial_thinking = sse.accumulated_reasoning
        if not partial_text and not partial_thinking:
            return None

        recovery_body = make_openai_text_recovery_body(body, partial_text)
        text, thinking = await self.collect_text(recovery_body)
        text_suffix = continuation_suffix(partial_text, text)
        thinking_suffix = continuation_suffix(partial_thinking, thinking)
        events: list[str] = []
        if thinking_suffix:
            for event in sse.ensure_thinking_block():
                events.append(event)
            events.append(sse.emit_thinking_delta(thinking_suffix))
        if text_suffix:
            for event in sse.ensure_text_block():
                events.append(event)
            events.append(sse.emit_text_delta(text_suffix))
        if not events:
            return None
        events.extend(sse.close_all_blocks())
        events.append(sse.message_delta("end_turn", sse.estimate_output_tokens()))
        events.append(sse.message_stop())
        trace_event(
            stage="provider",
            event="provider.recovery.continued",
            source="provider",
            provider=self._provider_name,
            request_id=request_id,
        )
        return events

    def emit_error_tail(self, sse: SSEBuilder, error_message: str) -> Iterator[str]:
        """Emit the canonical OpenAI-chat final error tail."""
        yield from sse.close_all_blocks()
        if sse.blocks.has_emitted_tool_block():
            yield sse.emit_top_level_error(error_message)
        else:
            yield from sse.emit_error(error_message)
        yield sse.message_delta("end_turn", 1)
        yield sse.message_stop()

    async def _repair_tool_args(
        self,
        *,
        body: dict[str, Any],
        sse: SSEBuilder,
        request: Any,
        tool_argument_alias_buffers: dict[int, str],
    ) -> list[str] | None:
        schemas = tool_schemas_by_name(request)
        events: list[str] = []
        for tool_index, state in started_tool_states(sse):
            emitted_prefix = "".join(state.contents)
            repair_prefix = emitted_prefix
            if not repair_prefix and state.name == "Task" and state.task_arg_buffer:
                repair_prefix = state.task_arg_buffer
            if not repair_prefix and tool_index in tool_argument_alias_buffers:
                repair_prefix = tool_argument_alias_buffers[tool_index]
            if (
                parse_complete_tool_input(repair_prefix, state.name, schemas)
                is not None
            ):
                if not emitted_prefix:
                    yield_text = repair_prefix
                    if yield_text:
                        events.append(sse.emit_tool_delta(tool_index, yield_text))
                continue

            schema = schemas.get(state.name)
            recovery_body = make_openai_tool_repair_body(
                body,
                tool_name=state.name,
                prefix=repair_prefix,
                input_schema=schema.input_schema if schema is not None else None,
            )
            accepted_suffix: str | None = None
            for attempt in range(MIDSTREAM_RECOVERY_ATTEMPTS):
                text, _ = await self.collect_text(recovery_body)
                repair = accept_tool_json_repair(
                    repair_prefix,
                    text,
                    tool_name=state.name,
                    schemas=schemas,
                )
                if repair is not None:
                    accepted_suffix = repair.suffix
                    trace_event(
                        stage="provider",
                        event="provider.recovery.tool_repaired",
                        source="provider",
                        provider=self._provider_name,
                        tool_name=state.name,
                        attempt=attempt + 1,
                    )
                    break
            if accepted_suffix is None:
                return None
            to_emit = (
                accepted_suffix if emitted_prefix else repair_prefix + accepted_suffix
            )
            if to_emit:
                events.append(sse.emit_tool_delta(tool_index, to_emit))
        if not all_started_tools_complete(sse, request):
            return None
        return events
