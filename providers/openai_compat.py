"""OpenAI-style chat base for :class:`OpenAIChatTransport` (NIM, etc.).

``AnthropicMessagesTransport``-based providers (OpenRouter, LM Studio, DeepSeek, …) live
in separate modules; do not list them as subclasses of this class.
"""

import asyncio
import json
import uuid
from abc import abstractmethod
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx
from loguru import logger
from openai import AsyncOpenAI

from core.anthropic import (
    ContentType,
    HeuristicToolParser,
    SSEBuilder,
    ThinkTagParser,
    map_stop_reason,
)
from core.anthropic.stream_recovery import (
    MIDSTREAM_RECOVERY_ATTEMPTS,
    TruncatedProviderStreamError,
    accept_tool_json_repair,
    continuation_suffix,
    is_retryable_stream_error,
    make_openai_text_recovery_body,
    make_openai_tool_repair_body,
    parse_complete_tool_input,
    tool_schemas_by_name,
)
from core.anthropic.stream_recovery_session import (
    StreamFailureAction,
    StreamRecoverySession,
)
from core.trace import provider_chat_body_snapshot, trace_event
from providers.base import BaseProvider, ProviderConfig
from providers.error_mapping import (
    extract_provider_error_detail,
    map_error,
    user_visible_message_for_mapped_provider_error,
)
from providers.model_listing import extract_openai_model_ids
from providers.rate_limit import GlobalRateLimiter


def _iter_heuristic_tool_use_sse(
    sse: SSEBuilder, tool_use: dict[str, Any]
) -> Iterator[str]:
    """Emit SSE for one heuristic tool_use block (closes open text/thinking first)."""
    if tool_use.get("name") == "Task" and isinstance(tool_use.get("input"), dict):
        task_input = tool_use["input"]
        if task_input.get("run_in_background") is not False:
            task_input["run_in_background"] = False
    yield from sse.close_content_blocks()
    block_idx = sse.blocks.allocate_index()
    yield sse.content_block_start(
        block_idx,
        "tool_use",
        id=tool_use["id"],
        name=tool_use["name"],
    )
    yield sse.content_block_delta(
        block_idx,
        "input_json_delta",
        json.dumps(tool_use["input"]),
    )
    yield sse.content_block_stop(block_idx)


def _tool_call_extra_content(tool_call: Any) -> dict[str, Any] | None:
    if isinstance(tool_call, dict):
        value = tool_call.get("extra_content")
        return value if isinstance(value, dict) else None

    value = getattr(tool_call, "extra_content", None)
    if isinstance(value, dict):
        return value

    model_extra = getattr(tool_call, "model_extra", None)
    if isinstance(model_extra, dict):
        value = model_extra.get("extra_content")
        if isinstance(value, dict):
            return value

    pydantic_extra = getattr(tool_call, "__pydantic_extra__", None)
    if isinstance(pydantic_extra, dict):
        value = pydantic_extra.get("extra_content")
        if isinstance(value, dict):
            return value

    return None


class OpenAIChatTransport(BaseProvider):
    """Base for OpenAI-compatible ``/chat/completions`` adapters (NIM, …)."""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        provider_name: str,
        base_url: str,
        api_key: str,
    ):
        super().__init__(config)
        self._provider_name = provider_name
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._global_rate_limiter = GlobalRateLimiter.get_scoped_instance(
            provider_name.lower(),
            rate_limit=config.rate_limit,
            rate_window=config.rate_window,
            max_concurrency=config.max_concurrency,
        )
        http_client = None
        if config.proxy:
            http_client = httpx.AsyncClient(
                proxy=config.proxy,
                timeout=httpx.Timeout(
                    config.http_read_timeout,
                    connect=config.http_connect_timeout,
                    read=config.http_read_timeout,
                    write=config.http_write_timeout,
                ),
            )
        self._client = AsyncOpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            max_retries=0,
            timeout=httpx.Timeout(
                config.http_read_timeout,
                connect=config.http_connect_timeout,
                read=config.http_read_timeout,
                write=config.http_write_timeout,
            ),
            http_client=http_client,
        )

    async def cleanup(self) -> None:
        """Release HTTP client resources."""
        client = getattr(self, "_client", None)
        if client is not None:
            await client.close()

    async def list_model_ids(self) -> frozenset[str]:
        """Return model ids from the provider's OpenAI-compatible models endpoint."""
        payload = await self._client.models.list()
        return extract_openai_model_ids(payload, provider_name=self._provider_name)

    @abstractmethod
    def _build_request_body(
        self, request: Any, thinking_enabled: bool | None = None
    ) -> dict:
        """Build request body. Must be implemented by subclasses."""

    def _handle_extra_reasoning(
        self, delta: Any, sse: SSEBuilder, *, thinking_enabled: bool
    ) -> Iterator[str]:
        """Hook for provider-specific reasoning (e.g. OpenRouter reasoning_details)."""
        return iter(())

    def _get_retry_request_body(self, error: Exception, body: dict) -> dict | None:
        """Return a modified request body for one retry, or None."""
        return None

    def _prepare_create_body(self, body: dict[str, Any]) -> dict[str, Any]:
        """Return the body passed to the upstream OpenAI-compatible client."""
        return body

    def _record_tool_call_extra_content(
        self, tool_call_id: str, extra_content: dict[str, Any]
    ) -> None:
        """Hook for providers that must replay OpenAI tool-call metadata later."""

    def _tool_argument_aliases(self, body: dict[str, Any]) -> dict[str, dict[str, str]]:
        """Return provider-specific per-tool argument aliases for this request."""
        return {}

    async def _create_stream(self, body: dict) -> tuple[Any, dict]:
        """Create a streaming chat completion, optionally retrying once."""
        try:
            create_body = self._prepare_create_body(body)
            stream = await self._global_rate_limiter.execute_with_retry(
                self._client.chat.completions.create, **create_body, stream=True
            )
            return stream, body
        except Exception as error:
            retry_body = self._get_retry_request_body(error, body)
            if retry_body is None:
                raise

            create_retry_body = self._prepare_create_body(retry_body)
            stream = await self._global_rate_limiter.execute_with_retry(
                self._client.chat.completions.create, **create_retry_body, stream=True
            )
            return stream, retry_body

    def _restore_aliased_tool_arguments(
        self, argument_json: str, aliases: dict[str, str]
    ) -> str | None:
        try:
            parsed = json.loads(argument_json)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict):
            return argument_json
        restored = self._restore_aliased_tool_argument_value(parsed, aliases)
        return json.dumps(restored)

    def _restore_aliased_tool_argument_value(
        self, value: Any, aliases: dict[str, str]
    ) -> Any:
        if isinstance(value, dict):
            return {
                aliases.get(key, key): self._restore_aliased_tool_argument_value(
                    item, aliases
                )
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [
                self._restore_aliased_tool_argument_value(item, aliases)
                for item in value
            ]
        return value

    def _emit_tool_arg_delta(
        self,
        sse: SSEBuilder,
        tc_index: int,
        args: str,
        *,
        tool_argument_aliases: dict[str, dict[str, str]] | None = None,
        tool_argument_alias_buffers: dict[int, str] | None = None,
    ) -> Iterator[str]:
        """Emit one argument fragment for a started tool block (Task buffer or raw JSON)."""
        if not args:
            return
        state = sse.blocks.tool_states.get(tc_index)
        if state is None:
            return
        if state.name == "Task":
            parsed = sse.blocks.buffer_task_args(tc_index, args)
            if parsed is not None:
                yield sse.emit_tool_delta(tc_index, json.dumps(parsed))
            return
        aliases = (
            tool_argument_aliases.get(state.name, {}) if tool_argument_aliases else {}
        )
        if aliases:
            if tool_argument_alias_buffers is None:
                restored = self._restore_aliased_tool_arguments(args, aliases)
                if restored is not None:
                    yield sse.emit_tool_delta(tc_index, restored)
                return

            buffered_args = tool_argument_alias_buffers.get(tc_index, "") + args
            restored = self._restore_aliased_tool_arguments(buffered_args, aliases)
            if restored is None:
                tool_argument_alias_buffers[tc_index] = buffered_args
                return
            tool_argument_alias_buffers.pop(tc_index, None)
            yield sse.emit_tool_delta(tc_index, restored)
            return
        yield sse.emit_tool_delta(tc_index, args)

    def _process_tool_call(
        self,
        tc: dict,
        sse: SSEBuilder,
        *,
        tool_argument_aliases: dict[str, dict[str, str]] | None = None,
        tool_argument_alias_buffers: dict[int, str] | None = None,
    ) -> Iterator[str]:
        """Process a single tool call delta and yield SSE events."""
        raw_index = tc.get("index", 0)
        tc_index = raw_index if isinstance(raw_index, int) else 0
        if tc_index < 0:
            tc_index = len(sse.blocks.tool_states)

        fn_delta = tc.get("function", {})
        incoming_name = fn_delta.get("name")
        arguments = fn_delta.get("arguments", "") or ""

        if tc.get("id") is not None:
            sse.blocks.set_stream_tool_id(tc_index, tc.get("id"))

        raw_extra_content = tc.get("extra_content")
        extra_content = (
            raw_extra_content
            if isinstance(raw_extra_content, dict) and raw_extra_content
            else None
        )
        if extra_content:
            sse.blocks.set_tool_extra_content(tc_index, extra_content)

        if incoming_name is not None:
            sse.blocks.register_tool_name(tc_index, incoming_name)

        state = sse.blocks.tool_states.get(tc_index)
        resolved_id = (state.tool_id if state and state.tool_id else None) or tc.get(
            "id"
        )
        resolved_name = (state.name if state else "") or ""

        if not state or not state.started:
            name_ok = bool((resolved_name or "").strip())
            if name_ok:
                tool_id = str(resolved_id) if resolved_id else f"tool_{uuid.uuid4()}"
                display_name = (resolved_name or "").strip() or "tool_call"
                start_extra_content = state.extra_content if state else extra_content
                if start_extra_content:
                    self._record_tool_call_extra_content(tool_id, start_extra_content)
                yield sse.start_tool_block(
                    tc_index,
                    tool_id,
                    display_name,
                    extra_content=start_extra_content,
                )
                state = sse.blocks.tool_states[tc_index]
                if state.pre_start_args:
                    pre = state.pre_start_args
                    state.pre_start_args = ""
                    yield from self._emit_tool_arg_delta(
                        sse,
                        tc_index,
                        pre,
                        tool_argument_aliases=tool_argument_aliases,
                        tool_argument_alias_buffers=tool_argument_alias_buffers,
                    )

        state = sse.blocks.tool_states.get(tc_index)
        if state is not None and state.tool_id and extra_content:
            self._record_tool_call_extra_content(state.tool_id, extra_content)
        if not arguments:
            return
        if state is None or not state.started:
            state = sse.blocks.ensure_tool_state(tc_index)
            if not (resolved_name or "").strip():
                state.pre_start_args += arguments
                return

        yield from self._emit_tool_arg_delta(
            sse,
            tc_index,
            arguments,
            tool_argument_aliases=tool_argument_aliases,
            tool_argument_alias_buffers=tool_argument_alias_buffers,
        )

    def _flush_task_arg_buffers(self, sse: SSEBuilder) -> Iterator[str]:
        """Emit buffered Task args as a single JSON delta (best-effort)."""
        for tool_index, out in sse.blocks.flush_task_arg_buffers():
            yield sse.emit_tool_delta(tool_index, out)

    def _flush_tool_argument_alias_buffers(
        self,
        sse: SSEBuilder,
        tool_argument_aliases: dict[str, dict[str, str]],
        tool_argument_alias_buffers: dict[int, str],
    ) -> Iterator[str]:
        """Emit remaining aliased tool args without losing data on malformed JSON."""
        for tool_index, buffered_args in list(tool_argument_alias_buffers.items()):
            if not buffered_args:
                tool_argument_alias_buffers.pop(tool_index, None)
                continue
            state = sse.blocks.tool_states.get(tool_index)
            if state is None or state.name == "Task":
                continue
            aliases = tool_argument_aliases.get(state.name, {})
            if not aliases:
                continue
            restored = self._restore_aliased_tool_arguments(buffered_args, aliases)
            yield sse.emit_tool_delta(
                tool_index,
                restored if restored is not None else buffered_args,
            )
            tool_argument_alias_buffers.pop(tool_index, None)

    def _has_committed_sse_output(self, sse: SSEBuilder) -> bool:
        return (
            sse.blocks.text_index != -1
            or sse.blocks.thinking_index != -1
            or sse.blocks.has_emitted_tool_block()
        )

    def _openai_error_message(self, error: Exception, request_id: str | None) -> str:
        mapped_error = map_error(error, rate_limiter=self._global_rate_limiter)
        return user_visible_message_for_mapped_provider_error(
            mapped_error,
            provider_name=self._provider_name,
            read_timeout_s=self._config.http_read_timeout,
            detail=extract_provider_error_detail(error),
            request_id=request_id,
        )

    async def _collect_recovery_text(self, body: dict[str, Any]) -> tuple[str, str]:
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

    def _started_tool_states(self, sse: SSEBuilder) -> list[tuple[int, Any]]:
        return [
            (tool_index, state)
            for tool_index, state in sse.blocks.tool_states.items()
            if state.started
        ]

    def _all_started_tools_complete(self, sse: SSEBuilder, request: Any) -> bool:
        schemas = tool_schemas_by_name(request)
        started = self._started_tool_states(sse)
        if not started:
            return False
        for _, state in started:
            raw = "".join(state.contents)
            if parse_complete_tool_input(raw, state.name, schemas) is None:
                return False
        return True

    async def _repair_openai_tool_args(
        self,
        *,
        body: dict[str, Any],
        sse: SSEBuilder,
        request: Any,
        tool_argument_alias_buffers: dict[int, str],
    ) -> list[str] | None:
        schemas = tool_schemas_by_name(request)
        events: list[str] = []
        for tool_index, state in self._started_tool_states(sse):
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
                text, _ = await self._collect_recovery_text(recovery_body)
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
        if not self._all_started_tools_complete(sse, request):
            return None
        return events

    async def _openai_recovery_events(
        self,
        *,
        body: dict[str, Any],
        sse: SSEBuilder,
        request: Any,
        request_id: str | None,
        error: Exception,
        tool_argument_alias_buffers: dict[int, str],
    ) -> list[str] | None:
        if not is_retryable_stream_error(error):
            return None

        if sse.blocks.has_emitted_tool_block():
            if not self._all_started_tools_complete(sse, request):
                repair_events = await self._repair_openai_tool_args(
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
        text, thinking = await self._collect_recovery_text(recovery_body)
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

    def _emit_openai_error_tail(
        self, sse: SSEBuilder, error_message: str
    ) -> Iterator[str]:
        yield from sse.close_all_blocks()
        if sse.blocks.has_emitted_tool_block():
            yield sse.emit_top_level_error(error_message)
        else:
            yield from sse.emit_error(error_message)
        yield sse.message_delta("end_turn", 1)
        yield sse.message_stop()

    async def stream_response(
        self,
        request: Any,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        thinking_enabled: bool | None = None,
    ) -> AsyncIterator[str]:
        """Stream response in Anthropic SSE format."""
        with logger.contextualize(request_id=request_id):
            async for event in self._stream_response_impl(
                request, input_tokens, request_id, thinking_enabled=thinking_enabled
            ):
                yield event

    async def _stream_response_impl(
        self,
        request: Any,
        input_tokens: int,
        request_id: str | None,
        *,
        thinking_enabled: bool | None,
    ) -> AsyncIterator[str]:
        """Shared streaming implementation."""
        tag = self._provider_name
        message_id = f"msg_{uuid.uuid4()}"

        def new_sse_builder() -> SSEBuilder:
            return SSEBuilder(
                message_id,
                request.model,
                input_tokens,
                log_raw_events=self._config.log_raw_sse_events,
            )

        sse = new_sse_builder()
        recovery_session = StreamRecoverySession(
            provider_name=tag,
            request_id=request_id,
        )

        def hold_event(event: str) -> Iterator[str]:
            yield from recovery_session.push(event)

        def hold_events(events: Iterator[str]) -> Iterator[str]:
            for event in events:
                yield from hold_event(event)

        body = self._build_request_body(request, thinking_enabled=thinking_enabled)
        thinking_enabled = self._is_thinking_enabled(request, thinking_enabled)
        req_tag = f" request_id={request_id}" if request_id else ""
        trace_event(
            stage="provider",
            event="provider.request.sent",
            source="provider",
            provider=self._provider_name,
            gateway_model=request.model,
            downstream_model=body.get("model"),
            message_count=len(body.get("messages", [])),
            tool_count=len(body.get("tools", [])),
            body=provider_chat_body_snapshot(body),
        )

        yield sse.message_start()

        think_parser = ThinkTagParser()
        heuristic_parser = HeuristicToolParser()
        finish_reason = None
        usage_info = None
        tool_argument_aliases: dict[str, dict[str, str]] = {}
        tool_argument_alias_buffers: dict[int, str] = {}

        async with self._global_rate_limiter.concurrency_slot():
            while True:
                stream_opened = False
                try:
                    stream, body = await self._create_stream(body)
                    stream_opened = True
                    tool_argument_aliases = self._tool_argument_aliases(body)
                    async for chunk in stream:
                        if getattr(chunk, "usage", None):
                            usage_info = chunk.usage

                        if not chunk.choices:
                            continue

                        choice = chunk.choices[0]
                        delta = choice.delta
                        if delta is None:
                            continue

                        if choice.finish_reason:
                            finish_reason = choice.finish_reason
                            logger.debug("{} finish_reason: {}", tag, finish_reason)

                        # Handle reasoning_content (OpenAI extended format)
                        reasoning = getattr(delta, "reasoning_content", None)
                        if thinking_enabled and reasoning:
                            for event in hold_events(sse.ensure_thinking_block()):
                                yield event
                            for event in hold_event(sse.emit_thinking_delta(reasoning)):
                                yield event

                        # Provider-specific extra reasoning (e.g. OpenRouter reasoning_details)
                        for event in self._handle_extra_reasoning(
                            delta,
                            sse,
                            thinking_enabled=thinking_enabled,
                        ):
                            for out_event in hold_event(event):
                                yield out_event

                        # Handle text content
                        if delta.content:
                            for part in think_parser.feed(delta.content):
                                if part.type == ContentType.THINKING:
                                    if not thinking_enabled:
                                        continue
                                    for event in hold_events(
                                        sse.ensure_thinking_block()
                                    ):
                                        yield event
                                    for event in hold_event(
                                        sse.emit_thinking_delta(part.content)
                                    ):
                                        yield event
                                else:
                                    (
                                        filtered_text,
                                        detected_tools,
                                    ) = heuristic_parser.feed(part.content)

                                    if filtered_text:
                                        for event in hold_events(
                                            sse.ensure_text_block()
                                        ):
                                            yield event
                                        for event in hold_event(
                                            sse.emit_text_delta(filtered_text)
                                        ):
                                            yield event

                                    for tool_use in detected_tools:
                                        for event in _iter_heuristic_tool_use_sse(
                                            sse, tool_use
                                        ):
                                            for out_event in hold_event(event):
                                                yield out_event

                        # Handle native tool calls
                        if delta.tool_calls:
                            for event in hold_events(sse.close_content_blocks()):
                                yield event
                            for tc in delta.tool_calls:
                                extra_content = _tool_call_extra_content(tc)
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
                                for event in self._process_tool_call(
                                    tc_info,
                                    sse,
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
                except Exception as e:
                    generated_output = self._has_committed_sse_output(sse)
                    complete_tool_salvageable = (
                        generated_output
                        and sse.blocks.has_emitted_tool_block()
                        and self._all_started_tools_complete(sse, request)
                    )
                    decision = recovery_session.advance_failure(
                        e,
                        stream_opened=stream_opened,
                        generated_output=generated_output,
                        complete_tool_salvageable=complete_tool_salvageable,
                    )
                    if decision.action == StreamFailureAction.EARLY_RETRY:
                        sse = new_sse_builder()
                        think_parser = ThinkTagParser()
                        heuristic_parser = HeuristicToolParser()
                        finish_reason = None
                        usage_info = None
                        tool_argument_aliases = {}
                        tool_argument_alias_buffers = {}
                        continue

                    if decision.action == StreamFailureAction.MIDSTREAM_RECOVERY:
                        try:
                            recovery_events = await self._openai_recovery_events(
                                body=body,
                                sse=sse,
                                request=request,
                                request_id=request_id,
                                error=e,
                                tool_argument_alias_buffers=tool_argument_alias_buffers,
                            )
                        except Exception as recovery_error:
                            trace_event(
                                stage="provider",
                                event="provider.recovery.failed",
                                source="provider",
                                provider=tag,
                                request_id=request_id,
                                exc_type=type(recovery_error).__name__,
                            )
                            recovery_events = None
                        if recovery_events is not None:
                            for event in recovery_session.flush_uncommitted(decision):
                                yield event
                            for event in recovery_events:
                                yield event
                            return

                    self._log_stream_transport_error(
                        tag, req_tag, e, request_id=request_id
                    )
                    error_message = self._openai_error_message(e, request_id)
                    trace_event(
                        stage="provider",
                        event="provider.response.error",
                        source="provider",
                        provider=tag,
                        error_message=error_message,
                        mapped_error_type=type(
                            map_error(e, rate_limiter=self._global_rate_limiter)
                        ).__name__,
                    )
                    if not decision.committed and decision.has_buffered:
                        for event in recovery_session.flush():
                            yield event
                    elif not decision.committed:
                        recovery_session.discard()
                        sse = new_sse_builder()
                    for event in self._emit_openai_error_tail(sse, error_message):
                        yield event
                    return

        # Flush remaining content
        remaining = think_parser.flush()
        if remaining:
            if remaining.type == ContentType.THINKING:
                if not thinking_enabled:
                    remaining = None
                else:
                    for event in hold_events(sse.ensure_thinking_block()):
                        yield event
                    for event in hold_event(sse.emit_thinking_delta(remaining.content)):
                        yield event
            if remaining and remaining.type == ContentType.TEXT:
                for event in hold_events(sse.ensure_text_block()):
                    yield event
                for event in hold_event(sse.emit_text_delta(remaining.content)):
                    yield event

        for tool_use in heuristic_parser.flush():
            for event in _iter_heuristic_tool_use_sse(sse, tool_use):
                for out_event in hold_event(event):
                    yield out_event

        has_started_tool = any(s.started for s in sse.blocks.tool_states.values())
        has_content_blocks = (
            sse.blocks.text_index != -1
            or sse.blocks.thinking_index != -1
            or has_started_tool
        )
        if not has_content_blocks:
            for event in hold_events(sse.ensure_text_block()):
                yield event
            for event in hold_event(sse.emit_text_delta(" ")):
                yield event
        elif (
            not has_started_tool
            and not sse.accumulated_text.strip()
            and sse.accumulated_reasoning.strip()
        ):
            # Some OpenAI-compatible models (e.g. NIM reasoning templates) stream only
            # ``reasoning_content`` with no ``content``; emit a minimal text block so
            # clients and smoke ``text_content()`` see a completed assistant message.
            for event in hold_events(sse.ensure_text_block()):
                yield event
            for event in hold_event(sse.emit_text_delta(" ")):
                yield event

        for event in self._flush_tool_argument_alias_buffers(
            sse, tool_argument_aliases, tool_argument_alias_buffers
        ):
            for out_event in hold_event(event):
                yield out_event

        for event in self._flush_task_arg_buffers(sse):
            for out_event in hold_event(event):
                yield out_event

        for event in hold_events(sse.close_all_blocks()):
            yield event

        completion = (
            getattr(usage_info, "completion_tokens", None)
            if usage_info is not None
            else None
        )
        if isinstance(completion, int):
            output_tokens = completion
        else:
            output_tokens = sse.estimate_output_tokens()
        if usage_info and hasattr(usage_info, "prompt_tokens"):
            provider_input = usage_info.prompt_tokens
            if isinstance(provider_input, int):
                logger.debug(
                    "TOKEN_ESTIMATE: our={} provider={} diff={:+d}",
                    input_tokens,
                    provider_input,
                    provider_input - input_tokens,
                )
        trace_event(
            stage="provider",
            event="provider.response.completed",
            source="provider",
            provider=self._provider_name,
            finish_reason=(None if finish_reason is None else str(finish_reason)),
            output_tokens=output_tokens,
            prompt_tokens_estimate=input_tokens,
        )
        for event in hold_event(
            sse.message_delta(map_stop_reason(finish_reason), output_tokens)
        ):
            yield event
        for event in hold_event(sse.message_stop()):
            yield event
        for event in recovery_session.flush():
            yield event
