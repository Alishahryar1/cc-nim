"""Concrete OpenAI-compatible provider and per-request stream execution."""

import asyncio
import sys
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import httpx2
from loguru import logger
from openai import AsyncOpenAI, DefaultAsyncHttpx2Client

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.core.anthropic import (
    ContentType,
    FunctionTagToolParser,
    HeuristicToolParser,
    OpenAIToolNameCodec,
    ThinkTagParser,
)
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.anthropic.streaming import (
    AnthropicStreamLedger,
    accept_tool_json_repair,
    continuation_suffix,
    make_response_recovery_body,
    make_text_recovery_body,
    make_tool_repair_body,
    map_stop_reason,
    parse_complete_tool_input,
    tool_schemas_by_name,
)
from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.core.reasoning import DEFAULT_REASONING_POLICY, ReasoningPolicy
from free_claude_code.core.trace import provider_chat_body_snapshot, trace_event
from free_claude_code.providers.admission import (
    ProviderAdmissionController,
    ProviderAttempt,
    ProviderCorrectionAction,
    ProviderExecution,
    ProviderOperationKind,
)
from free_claude_code.providers.base import BaseProvider, ProviderConfig
from free_claude_code.providers.failure_policy import (
    RetryableToolProtocolError,
    classify_provider_failure,
    is_retryable_stream_error,
    underlying_provider_error,
)
from free_claude_code.providers.http import (
    close_provider_stream,
    maybe_await_aclose,
)
from free_claude_code.providers.model_listing import (
    extract_openai_model_infos,
    merge_model_list_pages,
    validate_model_list_page,
)
from free_claude_code.providers.stream_recovery import (
    RecoveryController,
    RecoveryFailureAction,
    TruncatedProviderStreamError,
)

from .output_cap import clamp_output_tokens, parse_output_token_cap
from .profiles import OpenAIChatProfile
from .reasoning_details import StructuredReasoningStream
from .request_policy import build_openai_chat_request_body
from .tool_calls import (
    OpenAIToolCallAssembler,
    OpenAIToolCallCollector,
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
_ExtraReasoningEvents = Callable[[Any, AnthropicStreamLedger], Iterator[str]]


@dataclass(frozen=True, slots=True)
class _CollectedRecoveryOutput:
    text: str
    thinking: str
    tool_calls: tuple[dict[str, Any], ...]


def _iter_visible_text_events(
    ledger: AnthropicStreamLedger,
    text: str,
) -> Iterator[str]:
    yield from ledger.ensure_text_block()
    yield ledger.emit_text_delta(text)


def _iter_text_parser_events(
    ledger: AnthropicStreamLedger,
    parser: HeuristicToolParser,
    text: str,
    *,
    tool_names: OpenAIToolNameCodec,
) -> Iterator[str]:
    """Route visible text through the established heuristic tool parser."""
    filtered_text, detected_tools = parser.feed(text)
    if filtered_text:
        yield from _iter_visible_text_events(ledger, filtered_text)
    for tool_use in detected_tools:
        yield from iter_heuristic_tool_use_sse(
            ledger,
            tool_use,
            tool_names=tool_names,
        )


def _iter_text_tool_use_events(
    ledger: AnthropicStreamLedger,
    tool_uses: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    *,
    tool_names: OpenAIToolNameCodec,
) -> Iterator[str]:
    for tool_use in tool_uses:
        yield from iter_heuristic_tool_use_sse(
            ledger,
            tool_use,
            tool_names=tool_names,
        )


@dataclass(frozen=True, slots=True)
class _OpenAIChatCompletion:
    finish_reason: Any
    output_tokens: int
    input_tokens: int
    provider_input_tokens: int | None


class _OpenAIChatFailureOutcome(StrEnum):
    RETRY = "retry"
    COMPLETE = "complete"
    RAISE = "raise"


@dataclass(frozen=True, slots=True)
class _OpenAIChatFailureResolution:
    outcome: _OpenAIChatFailureOutcome
    events: tuple[str, ...] = ()
    failure: ExecutionFailure | None = None


class _OpenAIChatAttemptScope:
    """Own one returned provider stream and its admitted attempt."""

    def __init__(
        self,
        stream: Any,
        attempt: ProviderAttempt,
        *,
        provider_name: str,
        request_id: str | None,
    ) -> None:
        self.stream = stream
        self.attempt = attempt
        self._provider_name = provider_name
        self._request_id = request_id
        self._closed = False

    async def aclose(self, *, active_error: BaseException | None) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await close_provider_stream(
                self.stream,
                active_error=active_error,
                provider_name=self._provider_name,
                request_id=self._request_id,
            )
        finally:
            await self.attempt.aclose()


class _OpenAIChatStreamAssembler:
    """Own one discardable OpenAI-chat replay epoch."""

    def __init__(
        self,
        *,
        request: MessagesRequest,
        ledger: AnthropicStreamLedger,
        profile: OpenAIChatProfile,
        provider_name: str,
        output_reasoning: bool,
        tool_names: OpenAIToolNameCodec,
        tool_calls: OpenAIToolCallAssembler,
        extra_reasoning_events: _ExtraReasoningEvents,
    ) -> None:
        self._request = request
        self._ledger = ledger
        self._profile = profile
        self._provider_name = provider_name
        self._output_reasoning = output_reasoning
        self._tool_names = tool_names
        self._tool_calls = tool_calls
        self._extra_reasoning_events = extra_reasoning_events
        self._think_parser = ThinkTagParser()
        self._function_tag_parser = FunctionTagToolParser(request)
        self._heuristic_parser = HeuristicToolParser()
        self._structured_reasoning = (
            StructuredReasoningStream()
            if profile.structured_reasoning_details
            else None
        )
        self._finish_reason: Any = None
        self._usage_info: Any = None
        self._tool_argument_aliases: dict[str, dict[str, str]] = {}
        self._tool_argument_alias_buffers: dict[int, str] = {}
        self._tool_name_buffers: dict[int, str] = {}
        self._started = False
        self._aliases_bound = False
        self._upstream_finished = False
        self._completion: _OpenAIChatCompletion | None = None
        self._completed = False

    @property
    def ledger(self) -> AnthropicStreamLedger:
        return self._ledger

    @property
    def usage_info(self) -> Any:
        return self._usage_info

    @property
    def completion(self) -> _OpenAIChatCompletion:
        if self._completion is None:
            raise RuntimeError("stream completion has not been prepared")
        return self._completion

    @property
    def generated_output(self) -> bool:
        return has_committed_sse_output(self._ledger)

    @property
    def complete_tool_salvageable(self) -> bool:
        return (
            self.generated_output
            and self._ledger.has_emitted_tool_block()
            and all_emitted_tools_complete(self._ledger, self._request)
        )

    @property
    def tool_argument_alias_buffers(self) -> Mapping[int, str]:
        return self._tool_argument_alias_buffers

    def start_events(self) -> Iterator[str]:
        if self._started:
            return
        self._started = True
        yield self._ledger.message_start()

    def bind_tool_argument_aliases(self, aliases: dict[str, dict[str, str]]) -> None:
        if self._aliases_bound:
            raise RuntimeError("tool argument aliases already bound")
        self._aliases_bound = True
        self._tool_argument_aliases = aliases

    def feed(self, chunk: Any) -> Iterator[str]:
        if not self._started or self._upstream_finished:
            raise RuntimeError("stream assembler is not accepting chunks")

        chunk_usage = getattr(chunk, "usage", None)
        if chunk_usage is not None:
            self._usage_info = chunk_usage

        if not chunk.choices:
            return

        choice = chunk.choices[0]
        delta = choice.delta
        if delta is None:
            return

        if choice.finish_reason:
            self._finish_reason = choice.finish_reason
            logger.debug(
                "{} finish_reason: {}",
                self._provider_name,
                self._finish_reason,
            )

        reasoning = self._profile.reasoning_delta(delta)
        if self._output_reasoning:
            if self._structured_reasoning is not None:
                yield from self._structured_reasoning.events(
                    delta,
                    self._ledger,
                    native_reasoning=reasoning,
                )
            elif reasoning is not None:
                yield from self._ledger.ensure_thinking_block()
                if reasoning:
                    yield self._ledger.emit_thinking_delta(reasoning)

        yield from self._extra_reasoning_events(delta, self._ledger)

        native_tool_calls = delta.tool_calls
        if native_tool_calls:
            released_text = self._function_tag_parser.disable()
            if released_text:
                yield from _iter_visible_text_events(self._ledger, released_text)

        if delta.content:
            for part in self._think_parser.feed(delta.content):
                if part.type == ContentType.THINKING:
                    if not self._output_reasoning:
                        continue
                    yield from self._ledger.ensure_thinking_block()
                    yield self._ledger.emit_thinking_delta(part.content)
                else:
                    safe_text = self._function_tag_parser.feed(part.content)
                    if safe_text:
                        yield from _iter_text_parser_events(
                            self._ledger,
                            self._heuristic_parser,
                            safe_text,
                            tool_names=self._tool_names,
                        )

        if native_tool_calls:
            yield from self._ledger.close_content_blocks()
            for tool_call in native_tool_calls:
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
                yield from self._tool_calls.process_tool_call(
                    tool_call_info,
                    self._ledger,
                    tool_names=self._tool_names,
                    tool_name_buffers=self._tool_name_buffers,
                    tool_argument_aliases=self._tool_argument_aliases,
                    tool_argument_alias_buffers=self._tool_argument_alias_buffers,
                )

    def finish_upstream(self) -> Iterator[str]:
        if self._upstream_finished:
            return
        if self._finish_reason is None:
            raise TruncatedProviderStreamError(
                "Provider stream ended without finish_reason."
            )
        if any(
            not self._tool_names.is_unchanged_name(name)
            for name in self._tool_name_buffers.values()
        ):
            raise TruncatedProviderStreamError(
                "Provider stream ended with an incomplete tool name."
            )

        remaining = self._think_parser.flush()
        if remaining:
            if remaining.type == ContentType.THINKING:
                if self._output_reasoning:
                    yield from self._ledger.ensure_thinking_block()
                    yield self._ledger.emit_thinking_delta(remaining.content)
            else:
                safe_text = self._function_tag_parser.feed(remaining.content)
                if safe_text:
                    yield from _iter_text_parser_events(
                        self._ledger,
                        self._heuristic_parser,
                        safe_text,
                        tool_names=self._tool_names,
                    )

        fallback_text, function_tag_tools = self._function_tag_parser.finish()
        if fallback_text:
            yield from _iter_visible_text_events(self._ledger, fallback_text)
        yield from _iter_text_tool_use_events(
            self._ledger,
            function_tag_tools,
            tool_names=self._tool_names,
        )
        yield from _iter_text_tool_use_events(
            self._ledger,
            self._heuristic_parser.flush(),
            tool_names=self._tool_names,
        )
        self._upstream_finished = True

    def prepare_completion(self) -> Iterator[str]:
        if not self._upstream_finished or self._completion is not None:
            raise RuntimeError("stream completion cannot be prepared")

        yield from self._tool_calls.flush_tool_name_buffers(
            self._ledger,
            tool_names=self._tool_names,
            tool_name_buffers=self._tool_name_buffers,
            tool_argument_aliases=self._tool_argument_aliases,
            tool_argument_alias_buffers=self._tool_argument_alias_buffers,
        )

        has_emitted_tool = self._ledger.has_emitted_tool_block()
        has_content_blocks = (
            self._ledger.blocks.text_index != -1
            or self._ledger.blocks.thinking_index != -1
            or has_emitted_tool
        )
        if not has_content_blocks or (
            not has_emitted_tool
            and not self._ledger.accumulated_text.strip()
            and self._ledger.accumulated_reasoning.strip()
        ):
            yield from self._ledger.ensure_text_block()
            yield self._ledger.emit_text_delta(" ")

        yield from self._tool_calls.flush_tool_argument_alias_buffers(
            self._ledger,
            self._tool_argument_aliases,
            self._tool_argument_alias_buffers,
        )
        yield from self._tool_calls.flush_task_arg_buffers(self._ledger)
        yield from self._ledger.close_all_blocks()

        completion = usage_int(self._usage_info, "completion_tokens")
        output_tokens = (
            completion
            if isinstance(completion, int)
            else self._ledger.estimate_output_tokens()
        )
        provider_input = usage_int(self._usage_info, "prompt_tokens")
        input_tokens = (
            provider_input if provider_input is not None else self._ledger.input_tokens
        )
        self._completion = _OpenAIChatCompletion(
            finish_reason=self._finish_reason,
            output_tokens=output_tokens,
            input_tokens=input_tokens,
            provider_input_tokens=provider_input,
        )

    def terminal_events(self, *, usage_fields: dict[str, int]) -> Iterator[str]:
        if self._completed:
            return
        completion = self.completion
        yield self._ledger.message_delta(
            self._ledger.final_stop_reason(map_stop_reason(completion.finish_reason)),
            completion.output_tokens,
            input_tokens=completion.input_tokens,
            usage_fields=usage_fields,
        )
        yield self._ledger.message_stop()
        self._completed = True


class OpenAIChatProvider(BaseProvider):
    """OpenAI-compatible ``/chat/completions`` provider configured by a profile."""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        profile: OpenAIChatProfile,
        admission: ProviderAdmissionController,
        default_headers: Mapping[str, str] | None = None,
        api_key_provider: OpenAIAsyncCredentialProvider | None = None,
    ):
        super().__init__(config)
        self._profile = profile
        self._provider_name = profile.provider_name
        if config.api_key is None and api_key_provider is None:
            raise ValueError(
                f"{profile.provider_name} requires an API key or credential provider"
            )
        self._api_key = config.api_key
        self._base_url = profile.base_url(config.base_url).rstrip("/")
        # Learned per-model output-token caps from upstream 400 rejections, so
        # later requests clamp proactively instead of paying the 400 each time.
        self._model_output_caps: dict[str, int] = {}
        self._admission = admission
        timeout = httpx2.Timeout(
            config.http_read_timeout,
            connect=config.http_connect_timeout,
            read=config.http_read_timeout,
            write=config.http_write_timeout,
        )
        http_client = None
        if config.proxy:
            http_client = DefaultAsyncHttpx2Client(
                proxy=config.proxy,
                timeout=timeout,
            )
        self._client = AsyncOpenAI(
            api_key=api_key_provider or self._api_key,
            base_url=self._base_url,
            max_retries=0,
            default_headers=default_headers,
            timeout=timeout,
            http_client=http_client,
        )

    async def cleanup(self) -> None:
        """Release HTTP client resources."""
        client = getattr(self, "_client", None)
        if client is not None:
            await client.close()

    async def list_model_infos(self) -> frozenset[ProviderModelInfo]:
        """Return model metadata from the OpenAI-compatible models endpoint."""
        payload = await self._list_models_payload()
        if not self._profile.model_ids_are_routable:
            return frozenset()
        listing = self._profile.model_listing
        return extract_openai_model_infos(
            payload,
            provider_name=self._provider_name,
            collection_field=listing.collection_field,
            id_field=listing.id_field,
            aliases_field=listing.aliases_field,
            required_path_values=listing.required_path_values,
            required_null_field=listing.required_null_field,
            required_sequence_items=listing.required_sequence_items,
            exclude_missing_sequence_fields=listing.exclude_missing_sequence_fields,
            tags_field=listing.tags_field,
            thinking_tag=listing.thinking_tag,
            non_thinking_tag=listing.non_thinking_tag,
            thinking_boolean_path=listing.thinking_boolean_path,
        )

    async def _list_models_payload(self) -> Any:
        """Fetch one OpenAI-compatible model-list payload with shared retries."""
        return await self._fetch_models_payload()

    async def _fetch_models_payload(self) -> Any:
        """Fetch the complete profile-selected model-list payload."""
        listing = self._profile.model_listing
        if listing.path is not None and listing.pagination is not None:
            return await self._fetch_paginated_models_payload(listing.path)
        execution = self._admission.start_execution()
        return await execution.run_call(
            self._fetch_models_payload_once,
            operation_kind=ProviderOperationKind.MODEL_DISCOVERY,
            provider_failure_override=self._provider_failure_override,
        )

    async def _fetch_models_payload_once(self) -> Any:
        """Fetch the profile-selected model-list endpoint once."""
        listing = self._profile.model_listing
        path = listing.path
        if path is None:
            return await self._client.models.list()
        if listing.query_params:
            return await self._client.get(
                path,
                cast_to=object,
                options={"params": dict(listing.query_params)},
            )
        return await self._client.get(path, cast_to=object)

    async def _fetch_paginated_models_payload(self, path: str) -> Any:
        """Fetch a bounded model catalog with one execution per physical page."""
        listing = self._profile.model_listing
        pagination = listing.pagination
        if pagination is None:
            raise RuntimeError("paginated model fetch requires a pagination policy")

        payloads: list[Any] = []
        total_pages: int | None = None
        page = pagination.first_page
        while total_pages is None or page < pagination.first_page + total_pages:
            params = dict(listing.query_params)
            params[pagination.page_param] = str(page)
            execution = self._admission.start_execution()
            payload = await execution.run_call(
                lambda params=params: self._client.get(
                    path,
                    cast_to=object,
                    options={"params": params},
                ),
                operation_kind=ProviderOperationKind.MODEL_DISCOVERY,
                provider_failure_override=self._provider_failure_override,
            )
            total_pages = validate_model_list_page(
                payload,
                provider_name=self._provider_name,
                expected_page=page,
                current_page_path=pagination.current_page_path,
                total_pages_path=pagination.total_pages_path,
                max_pages=pagination.max_pages,
                expected_total_pages=total_pages,
            )
            payloads.append(payload)
            page += 1

        return merge_model_list_pages(
            payloads,
            provider_name=self._provider_name,
            collection_field=listing.collection_field,
        )

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

    async def _create_stream(
        self,
        body: dict,
        execution: ProviderExecution,
        operation_kind: ProviderOperationKind,
    ) -> tuple[Any, dict, ProviderAttempt]:
        """Create a streaming chat completion with bounded request fallbacks."""
        body = self._apply_learned_output_cap(body)
        used_retry_kinds: set[str] = set()

        while execution.can_attempt:
            attempt = await execution.open_attempt(operation_kind)
            stream: Any | None = None
            retain_attempt = False
            try:
                create_body = self._prepare_create_body(body)
                stream = await self._client.chat.completions.create(
                    **create_body,
                    stream=True,
                )
                stream = self._normalize_stream(stream, body)
                retain_attempt = True
                return stream, body, attempt
            except asyncio.CancelledError:
                raise
            except Exception as error:
                retry_body = self._next_create_retry_body(error, body, used_retry_kinds)
                if retry_body is not None:
                    correction = await attempt.correct(error)
                    if correction is ProviderCorrectionAction.RETRY:
                        body = retry_body
                        continue
                    raise
                decision = await attempt.fail(
                    error,
                    provider_failure_override=self._provider_failure_override,
                )
                if not decision.retry_allowed:
                    raise
            finally:
                if not retain_attempt:
                    if stream is not None:
                        await close_provider_stream(
                            stream,
                            active_error=sys.exception(),
                            provider_name=self._provider_name,
                            request_id=execution.request_id,
                        )
                    await attempt.aclose()

        if execution.last_failure is not None:
            raise execution.last_failure
        raise RuntimeError("provider execution ended without a final error")

    def _normalize_stream(self, stream: Any, _body: Mapping[str, Any]) -> Any:
        """Return the provider-specific stream view consumed by the base runner."""
        return stream

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
        response_model: str | None = None,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
    ) -> AsyncIterator[str]:
        """Stream response in Anthropic SSE format."""
        runner = _OpenAIChatStreamRunner(
            self,
            request=request,
            input_tokens=input_tokens,
            request_id=request_id,
            response_model=response_model,
            reasoning=reasoning,
        )
        return runner.run()


class _OpenAIChatStreamRunner:
    """Orchestrate one OpenAI-chat request and its recovery lifecycle."""

    def __init__(
        self,
        provider: OpenAIChatProvider,
        *,
        request: MessagesRequest,
        input_tokens: int,
        request_id: str | None,
        response_model: str | None,
        reasoning: ReasoningPolicy,
    ) -> None:
        self._provider = provider
        self._request = request
        self._input_tokens = input_tokens
        self._request_id = request_id
        self._response_model = (
            request.model if response_model is None else response_model
        )
        self._reasoning = reasoning
        self._tool_names = OpenAIToolNameCodec.from_request(request)
        self._message_id = f"msg_{uuid.uuid4()}"
        self._tool_calls = OpenAIToolCallAssembler(
            record_extra_content=provider._record_tool_call_extra_content
        )

    async def run(self) -> AsyncIterator[str]:
        """Convert the upstream OpenAI-chat stream into Anthropic SSE."""
        execution = self._provider._admission.start_execution(
            request_id=self._request_id
        )
        provider_stream = self._run_execution(execution)
        try:
            async for event in provider_stream:
                yield event
        except asyncio.CancelledError:
            raise
        except Exception as error:
            execution.fail(error)
            raise
        else:
            execution.succeed()
        finally:
            await maybe_await_aclose(provider_stream)
            execution.abandon()

    async def _run_execution(
        self,
        execution: ProviderExecution,
    ) -> AsyncIterator[str]:
        """Run one provider execution while retaining transport-owned state."""
        tag = self._provider._provider_name
        req_tag = f" request_id={self._request_id}" if self._request_id else ""
        recovery = RecoveryController()

        def hold_event(event: str) -> Iterator[str]:
            yield from recovery.push(event)

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
            execution_id=execution.execution_id,
            gateway_model=self._response_model,
            downstream_model=body.get("model"),
            message_count=len(body.get("messages", [])),
            tool_count=len(body.get("tools", [])),
            body=provider_chat_body_snapshot(body),
        )

        while True:
            assembler = self._new_stream_assembler(output_reasoning=output_reasoning)
            for event in assembler.start_events():
                for out_event in hold_event(event):
                    yield out_event
            scope: _OpenAIChatAttemptScope | None = None
            try:
                stream, body, attempt = await self._provider._create_stream(
                    body,
                    execution,
                    ProviderOperationKind.GENERATION,
                )
                scope = _OpenAIChatAttemptScope(
                    stream,
                    attempt,
                    provider_name=tag,
                    request_id=self._request_id,
                )
                assembler.bind_tool_argument_aliases(
                    self._provider._tool_argument_aliases(body)
                )
                async for chunk in scope.stream:
                    if not scope.attempt.accepted:
                        await scope.attempt.accept()
                    for event in assembler.feed(chunk):
                        for out_event in hold_event(event):
                            yield out_event

                for event in assembler.finish_upstream():
                    for out_event in hold_event(event):
                        yield out_event
                break

            except asyncio.CancelledError, GeneratorExit:
                raise
            except Exception as error:
                resolution = await self._resolve_attempt_failure(
                    error=error,
                    scope=scope,
                    assembler=assembler,
                    body=body,
                    execution=execution,
                    recovery=recovery,
                    req_tag=req_tag,
                )
                if resolution.outcome is _OpenAIChatFailureOutcome.RETRY:
                    continue
                for event in resolution.events:
                    yield event
                if resolution.outcome is _OpenAIChatFailureOutcome.COMPLETE:
                    return
                if resolution.failure is None:
                    raise AssertionError(
                        "raise resolution requires a failure"
                    ) from error
                raise resolution.failure from error
            finally:
                if scope is not None:
                    await scope.aclose(active_error=sys.exception())

        for event in assembler.prepare_completion():
            for out_event in hold_event(event):
                yield out_event
        completion = assembler.completion
        if completion.provider_input_tokens is not None:
            logger.debug(
                "TOKEN_ESTIMATE: our={} provider={} diff={:+d}",
                self._input_tokens,
                completion.provider_input_tokens,
                completion.provider_input_tokens - self._input_tokens,
            )
        trace_event(
            stage="provider",
            event="provider.response.completed",
            source="provider",
            provider=tag,
            request_id=self._request_id,
            finish_reason=(
                None
                if completion.finish_reason is None
                else str(completion.finish_reason)
            ),
            output_tokens=completion.output_tokens,
            prompt_tokens=completion.input_tokens,
            prompt_tokens_estimate=self._input_tokens,
        )
        for event in assembler.terminal_events(
            usage_fields=self._provider._anthropic_usage_fields(assembler.usage_info)
        ):
            for out_event in hold_event(event):
                yield out_event
        for event in recovery.flush():
            yield event

    async def _resolve_attempt_failure(
        self,
        *,
        error: Exception,
        scope: _OpenAIChatAttemptScope | None,
        assembler: _OpenAIChatStreamAssembler,
        body: dict[str, Any],
        execution: ProviderExecution,
        recovery: RecoveryController,
        req_tag: str,
    ) -> _OpenAIChatFailureResolution:
        """Resolve one failed generation attempt without owning retry policy."""
        attempt_failure = None
        if scope is not None and not scope.attempt.accepted:
            attempt_failure = await scope.attempt.fail(
                error,
                provider_failure_override=self._provider._provider_failure_override,
            )

        retryable = (
            attempt_failure.retryable
            if attempt_failure is not None
            else is_retryable_stream_error(error)
        )
        generated_output = assembler.generated_output
        complete_tool_salvageable = assembler.complete_tool_salvageable
        decision = recovery.advance_failure(
            retryable=retryable,
            stream_opened=scope is not None,
            generated_output=generated_output,
            complete_tool_salvageable=complete_tool_salvageable,
            attempts_remaining=execution.attempts_remaining,
        )
        tag = self._provider._provider_name
        if decision.action == RecoveryFailureAction.EARLY_RETRY:
            trace_event(
                stage="provider",
                event="provider.recovery.early_retry",
                source="provider",
                provider=tag,
                request_id=self._request_id,
                attempts_started=execution.attempts_started,
                max_attempts=execution.max_attempts,
                retryable=True,
            )
            return _OpenAIChatFailureResolution(outcome=_OpenAIChatFailureOutcome.RETRY)

        if decision.action == RecoveryFailureAction.MIDSTREAM_RECOVERY:
            if scope is not None:
                await scope.aclose(active_error=error)
            try:
                recovery_events = await self._recovery_events(
                    body=body,
                    ledger=assembler.ledger,
                    error=error,
                    tool_argument_alias_buffers=(assembler.tool_argument_alias_buffers),
                    output_reasoning=self._reasoning.output_enabled,
                    execution=execution,
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
                return _OpenAIChatFailureResolution(
                    outcome=_OpenAIChatFailureOutcome.COMPLETE,
                    events=(
                        *recovery.flush_uncommitted(decision),
                        *recovery_events,
                    ),
                )

        reported_error = underlying_provider_error(error)
        self._provider._log_stream_transport_error(
            tag,
            req_tag,
            reported_error,
            request_id=self._request_id,
        )
        failure = classify_provider_failure(
            reported_error,
            provider_name=tag,
            read_timeout_s=self._provider._config.http_read_timeout,
            request_id=self._request_id,
            provider_failure_override=self._provider._provider_failure_override,
        )
        error_trace: dict[str, Any] = {
            "stage": "provider",
            "event": "provider.response.error",
            "source": "provider",
            "provider": tag,
            "request_id": self._request_id,
            "exc_type": type(reported_error).__name__,
            "failure_kind": failure.kind.value,
            "status_code": failure.status_code,
            "provider_retryable": failure.retryable,
        }
        if self._provider._config.log_api_error_tracebacks:
            error_trace["error_message"] = failure.message
        trace_event(**error_trace)

        failure_events: list[str] = []
        if (
            not decision.committed
            and decision.has_buffered
            and complete_tool_salvageable
        ):
            failure_events.extend(recovery.flush())
        elif not decision.committed:
            recovery.discard()
            return _OpenAIChatFailureResolution(
                outcome=_OpenAIChatFailureOutcome.RAISE,
                failure=failure,
            )
        failure_events.extend(assembler.ledger.close_unclosed_blocks())
        return _OpenAIChatFailureResolution(
            outcome=_OpenAIChatFailureOutcome.RAISE,
            events=tuple(failure_events),
            failure=failure,
        )

    async def _collect_recovery_output(
        self,
        body: dict[str, Any],
        *,
        include_reasoning: bool,
        execution: ProviderExecution,
        operation_kind: ProviderOperationKind,
    ) -> _CollectedRecoveryOutput:
        """Collect one complete buffered continuation response."""
        last_error: Exception | None = None
        while execution.can_attempt:
            stream: Any | None = None
            attempt: ProviderAttempt | None = None
            try:
                stream, accepted_body, attempt = await self._provider._create_stream(
                    body,
                    execution,
                    operation_kind,
                )
                text_parts: list[str] = []
                thinking_parts: list[str] = []
                tool_calls = OpenAIToolCallCollector()
                terminal_seen = False
                async for chunk in stream:
                    if not attempt.accepted:
                        await attempt.accept()
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
                    native_tool_calls = getattr(delta, "tool_calls", None)
                    if isinstance(native_tool_calls, list | tuple):
                        for tool_call in native_tool_calls:
                            tool_calls.add(tool_call)

                completed_tool_calls = tool_calls.completed_calls(
                    self._request,
                    tool_names=self._tool_names,
                    tool_argument_aliases=self._provider._tool_argument_aliases(
                        accepted_body
                    ),
                )
                if tool_calls.has_calls and completed_tool_calls is None:
                    raise TruncatedProviderStreamError(
                        "Recovery stream ended with an incomplete tool call."
                    )
                if not terminal_seen and not completed_tool_calls:
                    raise TruncatedProviderStreamError(
                        "Recovery stream ended without finish_reason."
                    )
                return _CollectedRecoveryOutput(
                    text="".join(text_parts),
                    thinking="".join(thinking_parts),
                    tool_calls=completed_tool_calls or (),
                )
            except Exception as error:
                last_error = error
                retryable = is_retryable_stream_error(error)
                if attempt is not None and not attempt.accepted:
                    failure = await attempt.fail(
                        error,
                        provider_failure_override=(
                            self._provider._provider_failure_override
                        ),
                    )
                    retryable = failure.retryable
                if not retryable or not execution.can_attempt:
                    raise
                trace_event(
                    stage="provider",
                    event="provider.recovery.retry",
                    source="provider",
                    provider=self._provider._provider_name,
                    recovery_kind="openai_text",
                    attempts_started=execution.attempts_started,
                    max_attempts=execution.max_attempts,
                    exc_type=type(error).__name__,
                )
            finally:
                if stream is not None:
                    await maybe_await_aclose(stream)
                if attempt is not None:
                    await attempt.aclose()
        if last_error is not None:
            raise last_error
        return _CollectedRecoveryOutput(text="", thinking="", tool_calls=())

    async def _recovery_events(
        self,
        *,
        body: dict[str, Any],
        ledger: AnthropicStreamLedger,
        error: Exception,
        tool_argument_alias_buffers: Mapping[int, str],
        output_reasoning: bool,
        execution: ProviderExecution,
    ) -> list[str] | None:
        """Build terminal recovery events when the interrupted stream permits it."""
        if ledger.has_emitted_tool_block():
            if not all_emitted_tools_complete(ledger, self._request):
                repair_events = await self._repair_tool_args(
                    body=body,
                    ledger=ledger,
                    tool_argument_alias_buffers=tool_argument_alias_buffers,
                    execution=execution,
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

        if isinstance(error, RetryableToolProtocolError):
            recovery_body = make_response_recovery_body(
                body,
                partial_text,
                partial_thinking,
            )
        else:
            recovery_body = make_text_recovery_body(
                body,
                partial_text,
                partial_thinking,
            )
        recovered = await self._collect_recovery_output(
            recovery_body,
            include_reasoning=output_reasoning,
            execution=execution,
            operation_kind=ProviderOperationKind.CONTINUATION,
        )
        text_suffix = continuation_suffix(partial_text, recovered.text)
        thinking_suffix = continuation_suffix(partial_thinking, recovered.thinking)
        events: list[str] = []
        if thinking_suffix:
            events.extend(ledger.ensure_thinking_block())
            events.append(ledger.emit_thinking_delta(thinking_suffix))
        if text_suffix:
            events.extend(ledger.ensure_text_block())
            events.append(ledger.emit_text_delta(text_suffix))
        if recovered.tool_calls:
            events.extend(ledger.close_content_blocks())
            for tool_call in recovered.tool_calls:
                events.extend(self._tool_calls.process_tool_call(tool_call, ledger))
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
        tool_argument_alias_buffers: Mapping[int, str],
        execution: ProviderExecution,
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
            repair_attempt = 0
            while execution.can_attempt:
                repair_attempt += 1
                recovered = await self._collect_recovery_output(
                    recovery_body,
                    include_reasoning=False,
                    execution=execution,
                    operation_kind=ProviderOperationKind.TOOL_REPAIR,
                )
                repair = accept_tool_json_repair(
                    repair_prefix,
                    recovered.text,
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
                        attempt=repair_attempt,
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

    def _new_stream_assembler(
        self, *, output_reasoning: bool
    ) -> _OpenAIChatStreamAssembler:
        def extra_reasoning_events(
            delta: Any, ledger: AnthropicStreamLedger
        ) -> Iterator[str]:
            yield from self._provider._handle_extra_reasoning(
                delta,
                ledger,
                output_reasoning=output_reasoning,
            )

        return _OpenAIChatStreamAssembler(
            request=self._request,
            ledger=AnthropicStreamLedger(
                self._message_id,
                self._response_model,
                self._input_tokens,
                log_raw_events=self._provider._config.log_raw_sse_events,
            ),
            profile=self._provider._profile,
            provider_name=self._provider._provider_name,
            output_reasoning=output_reasoning,
            tool_names=self._tool_names,
            tool_calls=self._tool_calls,
            extra_reasoning_events=extra_reasoning_events,
        )
