"""Concrete OpenAI-compatible provider and per-request stream execution."""

import asyncio
import sys
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping
from typing import Any

import httpx
from loguru import logger
from openai import AsyncOpenAI

from free_claude_code.core.anthropic import (
    ContentType,
    HeuristicToolParser,
    ThinkTagParser,
)
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.anthropic.streaming import (
    AnthropicStreamLedger,
    accept_tool_json_repair,
    continuation_suffix,
    make_text_recovery_body,
    make_tool_repair_body,
    map_stop_reason,
    parse_complete_tool_input,
    tool_schemas_by_name,
)
from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.core.reasoning import DEFAULT_REASONING_POLICY, ReasoningPolicy
from free_claude_code.core.trace import provider_chat_body_snapshot, trace_event
from free_claude_code.providers.base import BaseProvider, ProviderConfig
from free_claude_code.providers.failure_policy import classify_provider_failure
from free_claude_code.providers.http import (
    close_provider_stream,
    maybe_await_aclose,
)
from free_claude_code.providers.model_listing import extract_openai_model_ids
from free_claude_code.providers.rate_limit import ProviderRateLimiter
from free_claude_code.providers.stream_recovery import (
    MIDSTREAM_RECOVERY_ATTEMPTS,
    RecoveryController,
    RecoveryFailureAction,
    TruncatedProviderStreamError,
    is_retryable_stream_error,
)

from .output_cap import clamp_output_tokens, parse_output_token_cap
from .profiles import OpenAIChatProfile
from .request_policy import build_openai_chat_request_body
from .tool_calls import (
    OpenAIToolCallAssembler,
    all_emitted_tools_complete,
    has_committed_sse_output,
    iter_heuristic_tool_use_sse,
    started_tool_states,
    tool_call_extra_content,
)
from .usage import (
    clone_without_stream_usage,
    is_stream_usage_rejection,
    request_stream_usage,
    usage_int,
)

OpenAIAsyncCredentialProvider = Callable[[], Awaitable[str]]


class OpenAIChatProvider(BaseProvider):
    """OpenAI-compatible ``/chat/completions`` provider configured by a profile."""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        profile: OpenAIChatProfile,
        rate_limiter: ProviderRateLimiter,
        default_headers: Mapping[str, str] | None = None,
        api_key_provider: OpenAIAsyncCredentialProvider | None = None,
    ):
        super().__init__(config)
        self._profile = profile
        self._provider_name = profile.provider_name
        self._api_key = config.api_key
        self._base_url = profile.base_url(config.base_url).rstrip("/")
        # Learned per-model output-token caps from upstream 400 rejections, so
        # later requests clamp proactively instead of paying the 400 each time.
        self._model_output_caps: dict[str, int] = {}
        self._rate_limiter = rate_limiter
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
            api_key=api_key_provider or self._api_key,
            base_url=self._base_url,
            max_retries=0,
            default_headers=default_headers,
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

    def _build_request_body(
        self,
        request: MessagesRequest,
        *,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
    ) -> dict[str, Any]:
        """Build a provider request from the immutable profile."""
        return build_openai_chat_request_body(
            request,
            reasoning=reasoning,
            policy=self._profile.request_policy,
            postprocessors=self._profile.request_postprocessors,
        )

    def preflight_stream(
        self,
        request: MessagesRequest,
        *,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
    ) -> None:
        """Validate OpenAI-chat request conversion before streaming."""
        self._build_request_body(request, reasoning=reasoning)

    def _handle_extra_reasoning(
        self, delta: Any, ledger: AnthropicStreamLedger, *, output_reasoning: bool
    ) -> Iterator[str]:
        """Hook for provider-specific reasoning."""
        return iter(())

    def _get_retry_request_body(self, error: Exception, body: dict) -> dict | None:
        """Return a modified request body for one retry, or None."""
        return None

    def _provider_failure_override(self, error: Exception) -> ExecutionFailure | None:
        """Return provider-specific failure semantics, or defer to shared policy."""
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

    def _anthropic_usage_fields(self, usage_info: Any) -> dict[str, int]:
        """Return provider-specific Anthropic usage fields for final SSE usage."""
        return {}

    async def _create_stream(self, body: dict) -> tuple[Any, dict]:
        """Create a streaming chat completion with bounded request fallbacks."""
        body = self._apply_learned_output_cap(body)
        used_retry_kinds: set[str] = set()

        while True:
            try:
                create_body = self._prepare_create_body(body)
                stream = await self._rate_limiter.execute_with_retry(
                    self._client.chat.completions.create,
                    provider_failure_override=self._provider_failure_override,
                    **create_body,
                    stream=True,
                )
                return stream, body
            except Exception as error:
                retry_body = self._next_create_retry_body(error, body, used_retry_kinds)
                if retry_body is None:
                    raise
                body = retry_body

    def _next_create_retry_body(
        self,
        error: Exception,
        body: dict,
        used_retry_kinds: set[str],
    ) -> dict | None:
        retry_body = self._retry_body_for_output_cap(error, body)
        if retry_body is not None:
            return retry_body

        if "stream_usage" not in used_retry_kinds and is_stream_usage_rejection(error):
            retry_body = clone_without_stream_usage(body)
            if retry_body is not None:
                used_retry_kinds.add("stream_usage")
                logger.warning(
                    "{}_STREAM: retrying without stream_options.include_usage "
                    "after upstream rejection",
                    self._provider_name,
                )
                return retry_body

        if "provider_specific" not in used_retry_kinds:
            retry_body = self._get_retry_request_body(error, body)
            if retry_body is not None:
                used_retry_kinds.add("provider_specific")
                return retry_body

        return None

    def _apply_learned_output_cap(self, body: dict) -> dict:
        """Clamp output tokens to a previously learned cap for this model."""
        model = body.get("model")
        if not isinstance(model, str):
            return body
        cap = self._model_output_caps.get(model)
        if cap is None:
            return body
        clamped = clamp_output_tokens(body, cap)
        return clamped if clamped is not None else body

    def _retry_body_for_output_cap(self, error: Exception, body: dict) -> dict | None:
        """Learn an upstream output-token cap from a 400 and clamp for one retry."""
        cap = parse_output_token_cap(error)
        if cap is None:
            return None
        model = body.get("model")
        if isinstance(model, str):
            previous = self._model_output_caps.get(model)
            cap = cap if previous is None else min(previous, cap)
            self._model_output_caps[model] = cap
        clamped = clamp_output_tokens(body, cap)
        if clamped is None:
            return None
        logger.warning(
            "{}_STREAM: clamping output tokens to {} after upstream cap rejection",
            self._provider_name,
            cap,
        )
        return clamped

    def stream_response(
        self,
        request: MessagesRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
    ) -> AsyncIterator[str]:
        """Stream response in Anthropic SSE format."""
        runner = _OpenAIChatStreamRunner(
            self,
            request=request,
            input_tokens=input_tokens,
            request_id=request_id,
            reasoning=reasoning,
        )
        return runner.run()


class _OpenAIChatStreamRunner:
    """Own one OpenAI-chat request's stream, parsing, and recovery state."""

    def __init__(
        self,
        provider: OpenAIChatProvider,
        *,
        request: MessagesRequest,
        input_tokens: int,
        request_id: str | None,
        reasoning: ReasoningPolicy,
    ) -> None:
        self._provider = provider
        self._request = request
        self._input_tokens = input_tokens
        self._request_id = request_id
        self._reasoning = reasoning
        self._message_id = f"msg_{uuid.uuid4()}"
        self._tool_calls = OpenAIToolCallAssembler(
            record_extra_content=provider._record_tool_call_extra_content
        )

    async def run(self) -> AsyncIterator[str]:
        """Convert the upstream OpenAI-chat stream into Anthropic SSE."""
        tag = self._provider._provider_name
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

        body = self._provider._build_request_body(
            self._request,
            reasoning=self._reasoning,
        )
        request_stream_usage(body)
        output_reasoning = self._reasoning.output_enabled
        trace_event(
            stage="provider",
            event="provider.request.sent",
            source="provider",
            provider=tag,
            request_id=self._request_id,
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

        async with self._provider._rate_limiter.concurrency_slot():
            while True:
                if not ledger.message_started:
                    for event in hold_event(ledger.message_start()):
                        yield event
                stream: Any | None = None
                stream_opened = False
                try:
                    stream, body = await self._provider._create_stream(body)
                    stream_opened = True
                    tool_argument_aliases = self._provider._tool_argument_aliases(body)
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

                        reasoning = self._provider._profile.reasoning_delta(delta)
                        if output_reasoning and reasoning is not None:
                            for event in hold_events(ledger.ensure_thinking_block()):
                                yield event
                            if reasoning:
                                for event in hold_event(
                                    ledger.emit_thinking_delta(reasoning)
                                ):
                                    yield event

                        for event in self._provider._handle_extra_reasoning(
                            delta,
                            ledger,
                            output_reasoning=output_reasoning,
                        ):
                            for out_event in hold_event(event):
                                yield out_event

                        if delta.content:
                            for part in think_parser.feed(delta.content):
                                if part.type == ContentType.THINKING:
                                    if not output_reasoning:
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
                            for tool_call in delta.tool_calls:
                                extra_content = tool_call_extra_content(tool_call)
                                tool_call_info = {
                                    "index": tool_call.index,
                                    "id": tool_call.id,
                                    "function": {
                                        "name": tool_call.function.name,
                                        "arguments": tool_call.function.arguments,
                                    },
                                }
                                if extra_content:
                                    tool_call_info["extra_content"] = extra_content
                                for event in self._tool_calls.process_tool_call(
                                    tool_call_info,
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
                            recovery_events = await self._recovery_events(
                                body=body,
                                ledger=ledger,
                                error=error,
                                tool_argument_alias_buffers=tool_argument_alias_buffers,
                                output_reasoning=output_reasoning,
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

                    self._provider._log_stream_transport_error(
                        tag, req_tag, error, request_id=self._request_id
                    )
                    failure = classify_provider_failure(
                        error,
                        provider_name=tag,
                        read_timeout_s=self._provider._config.http_read_timeout,
                        request_id=self._request_id,
                        mark_rate_limited=(
                            self._provider._rate_limiter.extend_reactive_block
                        ),
                        provider_failure_override=(
                            self._provider._provider_failure_override
                        ),
                    )
                    error_trace: dict[str, Any] = {
                        "stage": "provider",
                        "event": "provider.response.error",
                        "source": "provider",
                        "provider": tag,
                        "request_id": self._request_id,
                        "exc_type": type(error).__name__,
                        "failure_kind": failure.kind.value,
                        "status_code": failure.status_code,
                        "provider_retryable": failure.retryable,
                    }
                    if self._provider._config.log_api_error_tracebacks:
                        error_trace["error_message"] = failure.message
                    trace_event(**error_trace)
                    if (
                        not decision.committed
                        and decision.has_buffered
                        and complete_tool_salvageable
                    ):
                        for event in recovery.flush():
                            yield event
                    elif not decision.committed:
                        recovery.discard()
                        raise failure from error
                    for event in ledger.close_unclosed_blocks():
                        yield event
                    raise failure from error
                finally:
                    if stream is not None:
                        await close_provider_stream(
                            stream,
                            active_error=sys.exception(),
                            provider_name=tag,
                            request_id=self._request_id,
                        )

        remaining = think_parser.flush()
        if remaining:
            if remaining.type == ContentType.THINKING:
                if not output_reasoning:
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
            request_id=self._request_id,
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
                usage_fields=self._provider._anthropic_usage_fields(usage_info),
            )
        ):
            yield event
        for event in hold_event(ledger.message_stop()):
            yield event
        for event in recovery.flush():
            yield event

    async def _collect_recovery_text(
        self, body: dict[str, Any], *, include_reasoning: bool
    ) -> tuple[str, str]:
        """Collect a complete text/reasoning continuation stream."""
        last_error: Exception | None = None
        for attempt in range(MIDSTREAM_RECOVERY_ATTEMPTS):
            stream: Any | None = None
            try:
                stream, _ = await self._provider._create_stream(body)
                text_parts: list[str] = []
                thinking_parts: list[str] = []
                terminal_seen = False
                async for chunk in stream:
                    if not getattr(chunk, "choices", None):
                        continue
                    choice = chunk.choices[0]
                    if choice.finish_reason is not None:
                        terminal_seen = True
                    delta = choice.delta
                    if delta is None:
                        continue
                    if include_reasoning:
                        reasoning = self._provider._profile.reasoning_delta(delta)
                        if reasoning:
                            thinking_parts.append(reasoning)
                    content = getattr(delta, "content", None)
                    if isinstance(content, str) and content:
                        text_parts.append(content)
                if not terminal_seen:
                    raise TruncatedProviderStreamError(
                        "Recovery stream ended without finish_reason."
                    )
                return "".join(text_parts), "".join(thinking_parts)
            except Exception as error:
                last_error = error
                if not is_retryable_stream_error(error):
                    raise
                trace_event(
                    stage="provider",
                    event="provider.recovery.retry",
                    source="provider",
                    provider=self._provider._provider_name,
                    recovery_kind="openai_text",
                    attempt=attempt + 1,
                    max_attempts=MIDSTREAM_RECOVERY_ATTEMPTS,
                    exc_type=type(error).__name__,
                )
            finally:
                if stream is not None:
                    await maybe_await_aclose(stream)
        if last_error is not None:
            raise last_error
        return "", ""

    async def _recovery_events(
        self,
        *,
        body: dict[str, Any],
        ledger: AnthropicStreamLedger,
        error: Exception,
        tool_argument_alias_buffers: dict[int, str],
        output_reasoning: bool,
    ) -> list[str] | None:
        """Build terminal recovery events when the interrupted stream permits it."""
        if not is_retryable_stream_error(error):
            return None

        if ledger.has_emitted_tool_block():
            if not all_emitted_tools_complete(ledger, self._request):
                repair_events = await self._repair_tool_args(
                    body=body,
                    ledger=ledger,
                    tool_argument_alias_buffers=tool_argument_alias_buffers,
                )
                if repair_events is None:
                    return None
            else:
                repair_events = []
            events = list(repair_events)
            events.extend(ledger.close_all_blocks())
            events.append(
                ledger.message_delta(
                    ledger.final_stop_reason("end_turn"),
                    ledger.estimate_output_tokens(),
                )
            )
            events.append(ledger.message_stop())
            trace_event(
                stage="provider",
                event="provider.recovery.tool_salvaged",
                source="provider",
                provider=self._provider._provider_name,
                request_id=self._request_id,
            )
            return events

        partial_text = ledger.accumulated_text
        partial_thinking = ledger.accumulated_reasoning
        if not partial_text and not partial_thinking:
            return None

        recovery_body = make_text_recovery_body(body, partial_text, partial_thinking)
        text, thinking = await self._collect_recovery_text(
            recovery_body, include_reasoning=output_reasoning
        )
        text_suffix = continuation_suffix(partial_text, text)
        thinking_suffix = continuation_suffix(partial_thinking, thinking)
        events: list[str] = []
        if thinking_suffix:
            events.extend(ledger.ensure_thinking_block())
            events.append(ledger.emit_thinking_delta(thinking_suffix))
        if text_suffix:
            events.extend(ledger.ensure_text_block())
            events.append(ledger.emit_text_delta(text_suffix))
        if not events:
            return None
        events.extend(ledger.close_all_blocks())
        events.append(
            ledger.message_delta(
                ledger.final_stop_reason("end_turn"), ledger.estimate_output_tokens()
            )
        )
        events.append(ledger.message_stop())
        trace_event(
            stage="provider",
            event="provider.recovery.continued",
            source="provider",
            provider=self._provider._provider_name,
            request_id=self._request_id,
        )
        return events

    async def _repair_tool_args(
        self,
        *,
        body: dict[str, Any],
        ledger: AnthropicStreamLedger,
        tool_argument_alias_buffers: dict[int, str],
    ) -> list[str] | None:
        schemas = tool_schemas_by_name(self._request)
        events: list[str] = []
        for tool_index, state in started_tool_states(ledger):
            block = ledger.tool_block_for_tool_index(tool_index)
            emitted_prefix = block.content if block is not None else ""
            repair_prefix = emitted_prefix
            if not repair_prefix and state.name == "Task" and state.task_arg_buffer:
                repair_prefix = state.task_arg_buffer
            if not repair_prefix and tool_index in tool_argument_alias_buffers:
                repair_prefix = tool_argument_alias_buffers[tool_index]
            if (
                parse_complete_tool_input(repair_prefix, state.name, schemas)
                is not None
            ):
                if not emitted_prefix and repair_prefix:
                    events.append(ledger.emit_tool_delta(tool_index, repair_prefix))
                continue

            schema = schemas.get(state.name)
            recovery_body = make_tool_repair_body(
                body,
                tool_name=state.name,
                prefix=repair_prefix,
                input_schema=schema.input_schema if schema is not None else None,
            )
            accepted_suffix: str | None = None
            for attempt in range(MIDSTREAM_RECOVERY_ATTEMPTS):
                text, _ = await self._collect_recovery_text(
                    recovery_body, include_reasoning=False
                )
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
                        provider=self._provider._provider_name,
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
                events.append(ledger.emit_tool_delta(tool_index, to_emit))
        if not all_emitted_tools_complete(ledger, self._request):
            return None
        return events

    def _new_ledger(self) -> AnthropicStreamLedger:
        return AnthropicStreamLedger(
            self._message_id,
            self._request.model,
            self._input_tokens,
            log_raw_events=self._provider._config.log_raw_sse_events,
        )
