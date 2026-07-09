"""OpenAI-chat upstream adapter."""

import asyncio
import uuid
from collections.abc import AsyncIterator, Iterator
from typing import Any

from loguru import logger

from core.anthropic import (
    ContentType,
    HeuristicToolParser,
    ThinkTagParser,
)
from core.anthropic.streaming import (
    AnthropicStreamLedger,
    RecoveryController,
    RecoveryFailureAction,
    TruncatedProviderStreamError,
    map_stop_reason,
)
from core.trace import provider_chat_body_snapshot, trace_event
from providers.error_mapping import map_error, map_stream_start_error
from providers.transports.http import maybe_await_aclose

from .recovery import OpenAIChatRecovery
from .tool_calls import (
    OpenAIToolCallAssembler,
    all_emitted_tools_complete,
    has_committed_sse_output,
    iter_heuristic_tool_use_sse,
    tool_call_extra_content,
)
from .usage import request_stream_usage, usage_int


class OpenAIChatStreamAdapter:
    """Convert one OpenAI-chat upstream stream into Anthropic SSE."""

    def __init__(
        self,
        transport: Any,
        *,
        request: Any,
        input_tokens: int,
        request_id: str | None,
        thinking_enabled: bool | None,
    ) -> None:
        self._transport = transport
        self._request = request
        self._input_tokens = input_tokens
        self._request_id = request_id
        self._thinking_enabled = thinking_enabled
        self._message_id = f"msg_{uuid.uuid4()}"
        self._tool_calls = OpenAIToolCallAssembler(
            record_extra_content=transport._record_tool_call_extra_content
        )
        self._recovery = OpenAIChatRecovery(
            provider_name=transport._provider_name,
            create_stream=transport._create_stream,
        )

    async def run(self) -> AsyncIterator[str]:
        """Stream response in Anthropic SSE format."""
        tag = self._transport._provider_name
        req_tag = f" request_id={self._request_id}" if self._request_id else ""
        ledger = self._new_ledger()
        recovery = RecoveryController(
            provider_name=tag,
            request_id=self._request_id,
        )

        def hold_event(event: str) -> Iterator[str]:
            yield from recovery.push(event)

        def hold_events(events: Iterator[str]) -> Iterator[str]:
            for event in events:
                yield from hold_event(event)

        body = self._transport._build_request_body(
            self._request, thinking_enabled=self._thinking_enabled
        )
        request_stream_usage(body)
        thinking_enabled = self._transport._is_thinking_enabled(
            self._request, self._thinking_enabled
        )
        trace_event(
            stage="provider",
            event="provider.request.sent",
            source="provider",
            provider=tag,
            gateway_model=self._request.model,
            downstream_model=body.get("model"),
            message_count=len(body.get("messages", [])),
            tool_count=len(body.get("tools", [])),
            body=provider_chat_body_snapshot(body),
        )

        think_parser = ThinkTagParser()
        heuristic_parser = HeuristicToolParser()
        finish_reason = None
        usage_info = None
        tool_argument_aliases: dict[str, dict[str, str]] = {}
        tool_argument_alias_buffers: dict[int, str] = {}

        async with self._transport._global_rate_limiter.concurrency_slot():
            while True:
                if not ledger.message_started:
                    for event in hold_event(ledger.message_start()):
                        yield event
                stream: Any | None = None
                stream_opened = False
                try:
                    stream, body = await self._transport._create_stream(body)
                    stream_opened = True
                    tool_argument_aliases = self._transport._tool_argument_aliases(body)
                    async for chunk in stream:
                        chunk_usage = getattr(chunk, "usage", None)
                        if chunk_usage is not None:
                            usage_info = chunk_usage

                        if not chunk.choices:
                            continue

                        choice = chunk.choices[0]
                        delta = choice.delta
                        if delta is None:
                            continue

                        if choice.finish_reason:
                            finish_reason = choice.finish_reason
                            logger.debug("{} finish_reason: {}", tag, finish_reason)

                        reasoning = getattr(delta, "reasoning_content", None)
                        if thinking_enabled and isinstance(reasoning, str):
                            for event in hold_events(ledger.ensure_thinking_block()):
                                yield event
                            if reasoning:
                                for event in hold_event(
                                    ledger.emit_thinking_delta(reasoning)
                                ):
                                    yield event

                        for event in self._transport._handle_extra_reasoning(
                            delta,
                            ledger,
                            thinking_enabled=thinking_enabled,
                        ):
                            for out_event in hold_event(event):
                                yield out_event

                        if delta.content:
                            for part in think_parser.feed(delta.content):
                                if part.type == ContentType.THINKING:
                                    if not thinking_enabled:
                                        continue
                                    for event in hold_events(
                                        ledger.ensure_thinking_block()
                                    ):
                                        yield event
                                    for event in hold_event(
                                        ledger.emit_thinking_delta(part.content)
                                    ):
                                        yield event
                                else:
                                    (
                                        filtered_text,
                                        detected_tools,
                                    ) = heuristic_parser.feed(part.content)

                                    if filtered_text:
                                        for event in hold_events(
                                            ledger.ensure_text_block()
                                        ):
                                            yield event
                                        for event in hold_event(
                                            ledger.emit_text_delta(filtered_text)
                                        ):
                                            yield event

                                    for tool_use in detected_tools:
                                        for event in iter_heuristic_tool_use_sse(
                                            ledger, tool_use
                                        ):
                                            for out_event in hold_event(event):
                                                yield out_event

                        if delta.tool_calls:
                            for event in hold_events(ledger.close_content_blocks()):
                                yield event
                            for tc in delta.tool_calls:
                                extra_content = tool_call_extra_content(tc)
                                tc_info = {
                                    "index": tc.index,
                                    "id": tc.id,
                                    "function": {
                                        "name": tc.function.name,
                                        "arguments": tc.function.arguments,
                                    },
                                }
                                if extra_content:
                                    tc_info["extra_content"] = extra_content
                                for event in self._tool_calls.process_tool_call(
                                    tc_info,
                                    ledger,
                                    tool_argument_aliases=tool_argument_aliases,
                                    tool_argument_alias_buffers=tool_argument_alias_buffers,
                                ):
                                    for out_event in hold_event(event):
                                        yield out_event

                    if finish_reason is None:
                        raise TruncatedProviderStreamError(
                            "Provider stream ended without finish_reason."
                        )
                    break

                except asyncio.CancelledError, GeneratorExit:
                    raise
                except Exception as error:
                    generated_output = has_committed_sse_output(ledger)
                    complete_tool_salvageable = (
                        generated_output
                        and ledger.has_emitted_tool_block()
                        and all_emitted_tools_complete(ledger, self._request)
                    )
                    decision = recovery.advance_failure(
                        error,
                        stream_opened=stream_opened,
                        generated_output=generated_output,
                        complete_tool_salvageable=complete_tool_salvageable,
                    )
                    if decision.action == RecoveryFailureAction.EARLY_RETRY:
                        ledger = self._new_ledger()
                        think_parser = ThinkTagParser()
                        heuristic_parser = HeuristicToolParser()
                        finish_reason = None
                        usage_info = None
                        tool_argument_aliases = {}
                        tool_argument_alias_buffers = {}
                        continue

                    if decision.action == RecoveryFailureAction.MIDSTREAM_RECOVERY:
                        try:
                            recovery_events = await self._recovery.events(
                                body=body,
                                ledger=ledger,
                                request=self._request,
                                request_id=self._request_id,
                                error=error,
                                tool_argument_alias_buffers=tool_argument_alias_buffers,
                            )
                        except Exception as recovery_error:
                            trace_event(
                                stage="provider",
                                event="provider.recovery.failed",
                                source="provider",
                                provider=tag,
                                request_id=self._request_id,
                                exc_type=type(recovery_error).__name__,
                            )
                            recovery_events = None
                        if recovery_events is not None:
                            for event in recovery.flush_uncommitted(decision):
                                yield event
                            for event in recovery_events:
                                yield event
                            return

                    self._transport._log_stream_transport_error(
                        tag, req_tag, error, request_id=self._request_id
                    )
                    error_message = self._transport._openai_error_message(
                        error, self._request_id
                    )
                    trace_event(
                        stage="provider",
                        event="provider.response.error",
                        source="provider",
                        provider=tag,
                        request_id=self._request_id,
                        exc_type=type(error).__name__,
                        error_message=error_message,
                        mapped_error_type=type(
                            map_error(
                                error,
                                rate_limiter=self._transport._global_rate_limiter,
                            )
                        ).__name__,
                    )
                    if (
                        not decision.committed
                        and decision.has_buffered
                        and complete_tool_salvageable
                    ):
                        for event in recovery.flush():
                            yield event
                    elif not decision.committed:
                        recovery.discard()
                        raise map_stream_start_error(
                            error,
                            provider_name=tag,
                            read_timeout_s=self._transport._config.http_read_timeout,
                            request_id=self._request_id,
                            rate_limiter=self._transport._global_rate_limiter,
                        ) from error
                    for event in self._recovery.emit_error_tail(ledger, error_message):
                        yield event
                    return
                finally:
                    if stream is not None:
                        await maybe_await_aclose(stream)

        remaining = think_parser.flush()
        if remaining:
            if remaining.type == ContentType.THINKING:
                if not thinking_enabled:
                    remaining = None
                else:
                    for event in hold_events(ledger.ensure_thinking_block()):
                        yield event
                    for event in hold_event(
                        ledger.emit_thinking_delta(remaining.content)
                    ):
                        yield event
            if remaining and remaining.type == ContentType.TEXT:
                for event in hold_events(ledger.ensure_text_block()):
                    yield event
                for event in hold_event(ledger.emit_text_delta(remaining.content)):
                    yield event

        for tool_use in heuristic_parser.flush():
            for event in iter_heuristic_tool_use_sse(ledger, tool_use):
                for out_event in hold_event(event):
                    yield out_event

        has_emitted_tool = ledger.has_emitted_tool_block()
        has_content_blocks = (
            ledger.blocks.text_index != -1
            or ledger.blocks.thinking_index != -1
            or has_emitted_tool
        )
        if not has_content_blocks or (
            not has_emitted_tool
            and not ledger.accumulated_text.strip()
            and ledger.accumulated_reasoning.strip()
        ):
            for event in hold_events(ledger.ensure_text_block()):
                yield event
            for event in hold_event(ledger.emit_text_delta(" ")):
                yield event

        for event in self._tool_calls.flush_tool_argument_alias_buffers(
            ledger, tool_argument_aliases, tool_argument_alias_buffers
        ):
            for out_event in hold_event(event):
                yield out_event

        for event in self._tool_calls.flush_task_arg_buffers(ledger):
            for out_event in hold_event(event):
                yield out_event

        for event in hold_events(ledger.close_all_blocks()):
            yield event

        completion = usage_int(usage_info, "completion_tokens")
        if isinstance(completion, int):
            output_tokens = completion
        else:
            output_tokens = ledger.estimate_output_tokens()
        provider_input = usage_int(usage_info, "prompt_tokens")
        if provider_input is not None:
            logger.debug(
                "TOKEN_ESTIMATE: our={} provider={} diff={:+d}",
                self._input_tokens,
                provider_input,
                provider_input - self._input_tokens,
            )
        input_tokens = (
            provider_input if provider_input is not None else self._input_tokens
        )
        trace_event(
            stage="provider",
            event="provider.response.completed",
            source="provider",
            provider=tag,
            finish_reason=(None if finish_reason is None else str(finish_reason)),
            output_tokens=output_tokens,
            prompt_tokens=input_tokens,
            prompt_tokens_estimate=self._input_tokens,
        )
        for event in hold_event(
            ledger.message_delta(
                ledger.final_stop_reason(map_stop_reason(finish_reason)),
                output_tokens,
                input_tokens=input_tokens,
                usage_fields=self._transport._anthropic_usage_fields(usage_info),
            )
        ):
            yield event
        for event in hold_event(ledger.message_stop()):
            yield event
        for event in recovery.flush():
            yield event

    def _new_ledger(self) -> AnthropicStreamLedger:
        return AnthropicStreamLedger(
            self._message_id,
            self._request.model,
            self._input_tokens,
            log_raw_events=self._transport._config.log_raw_sse_events,
        )
