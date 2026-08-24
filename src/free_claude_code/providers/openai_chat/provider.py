"""Concrete OpenAI-compatible provider and per-request stream execution."""

import asyncio
import sys
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import httpx2
from loguru import logger
from openai import AsyncOpenAI, DefaultAsyncHttpx2Client

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.core.anthropic import (
    ContentType,
    HeuristicToolParser,
    ThinkTagParser,
)
from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.core.inference import (
    FinishReason,
    InferenceEvent,
    InferenceRequest,
    InferenceStreamLedger,
    InferenceUsage,
    ReplayCompatibilityScope,
    TokenMeasurement,
    UsageSource,
)
from free_claude_code.core.reasoning import DEFAULT_REASONING_POLICY, ReasoningPolicy
from free_claude_code.core.trace import provider_chat_body_snapshot, trace_event
from free_claude_code.providers.admission import (
    ProviderAdmissionController,
    ProviderExecution,
    ProviderOperationKind,
)
from free_claude_code.providers.base import BaseProvider, ProviderConfig
from free_claude_code.providers.failure_policy import (
    ProviderFailureOverride,
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
    model_infos_from_ids,
    validate_model_list_page,
)
from free_claude_code.providers.openai_chat.function_tags import FunctionTagToolParser
from free_claude_code.providers.openai_chat.recovery import (
    accept_tool_json_repair,
    continuation_suffix,
    make_response_recovery_body,
    make_text_recovery_body,
    make_tool_repair_body,
    parse_complete_tool_input,
    tool_schemas_by_name,
)
from free_claude_code.providers.openai_compat import (
    OpenAIToolNameCodec,
    openai_replay_scope,
)
from free_claude_code.providers.stream_recovery import TruncatedProviderStreamError
from free_claude_code.providers.streaming import (
    BoundAttemptOperations,
    PublicationBuffer,
    RecoveryContext,
    RecoveryOutcome,
    StreamExecutionSupervisor,
    StreamFeed,
    StreamTraceContext,
)

from .output_cap import clamp_output_tokens, parse_output_token_cap
from .profiles import OpenAIChatProfile
from .reasoning_details import StructuredReasoningStream
from .request_policy import build_openai_chat_request_body
from .tool_calls import (
    OpenAIToolCallAssembler,
    OpenAIToolCallCollector,
    all_emitted_tools_complete,
    has_generated_output,
    iter_heuristic_tool_use_events,
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
_ExtraReasoningEvents = Callable[[Any, InferenceStreamLedger], Iterator[InferenceEvent]]


@dataclass(frozen=True, slots=True)
class _CollectedRecoveryOutput:
    text: str
    thinking: str
    tool_calls: tuple[dict[str, Any], ...]


def _iter_visible_text_events(
    ledger: InferenceStreamLedger,
    text: str,
) -> Iterator[InferenceEvent]:
    yield from ledger.ensure_text_block()
    yield ledger.emit_text_delta(text)


def _iter_text_parser_events(
    ledger: InferenceStreamLedger,
    parser: HeuristicToolParser,
    text: str,
    *,
    tool_names: OpenAIToolNameCodec,
) -> Iterator[InferenceEvent]:
    """Route visible text through the established heuristic tool parser."""
    filtered_text, detected_tools = parser.feed(text)
    if filtered_text:
        yield from _iter_visible_text_events(ledger, filtered_text)
    for tool_use in detected_tools:
        yield from iter_heuristic_tool_use_events(
            ledger,
            tool_use,
            tool_names=tool_names,
        )


def _iter_text_tool_use_events(
    ledger: InferenceStreamLedger,
    tool_uses: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    *,
    tool_names: OpenAIToolNameCodec,
) -> Iterator[InferenceEvent]:
    for tool_use in tool_uses:
        yield from iter_heuristic_tool_use_events(
            ledger,
            tool_use,
            tool_names=tool_names,
        )


@dataclass(frozen=True, slots=True)
class _OpenAIChatCompletion:
    finish_reason: Any
    output_tokens: int
    provider_output_tokens: int | None
    input_tokens: int
    provider_input_tokens: int | None


def _canonical_finish_reason(value: object) -> FinishReason:
    raw = str(value).lower() if value is not None else ""
    return {
        "stop": FinishReason.END_TURN,
        "length": FinishReason.OUTPUT_LIMIT,
        "max_tokens": FinishReason.OUTPUT_LIMIT,
        "tool_calls": FinishReason.TOOL_CALLS,
        "function_call": FinishReason.TOOL_CALLS,
        "content_filter": FinishReason.CONTENT_FILTER,
        "stop_sequence": FinishReason.STOP_SEQUENCE,
    }.get(raw, FinishReason.PROVIDER_UNKNOWN if raw else FinishReason.END_TURN)


def _chat_completion_usage(
    completion: _OpenAIChatCompletion,
    usage_fields: Mapping[str, int],
) -> InferenceUsage:
    input_override = usage_fields.get("input_tokens")
    input_tokens = (
        input_override
        if isinstance(input_override, int) and not isinstance(input_override, bool)
        else completion.input_tokens
    )
    input_reported = (
        completion.provider_input_tokens is not None or input_override is not None
    )
    return InferenceUsage(
        input_tokens=TokenMeasurement(
            max(input_tokens, 0),
            UsageSource.REPORTED if input_reported else UsageSource.ESTIMATED,
        ),
        cache_read_input_tokens=_reported_usage_field(
            usage_fields, "cache_read_input_tokens"
        ),
        cache_creation_input_tokens=_reported_usage_field(
            usage_fields, "cache_creation_input_tokens"
        ),
        output_tokens=TokenMeasurement(
            max(completion.output_tokens, 0),
            (
                UsageSource.REPORTED
                if completion.provider_output_tokens is not None
                else UsageSource.ESTIMATED
            ),
        ),
        reasoning_output_tokens=_reported_usage_field(
            usage_fields, "reasoning_output_tokens"
        ),
    )


def _reported_usage_field(
    usage_fields: Mapping[str, int], key: str
) -> TokenMeasurement | None:
    value = usage_fields.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        return None
    return TokenMeasurement(value, UsageSource.REPORTED)


def _estimated_recovery_usage(
    *, input_tokens: int, output_tokens: int
) -> InferenceUsage:
    return InferenceUsage(
        input_tokens=TokenMeasurement(max(input_tokens, 0), UsageSource.ESTIMATED),
        output_tokens=TokenMeasurement(max(output_tokens, 0), UsageSource.ESTIMATED),
    )


class _OpenAIChatStreamAssembler:
    """Own one discardable OpenAI-chat replay epoch."""

    def __init__(
        self,
        *,
        request: InferenceRequest,
        ledger: InferenceStreamLedger,
        profile: OpenAIChatProfile,
        provider_name: str,
        output_reasoning: bool,
        tool_names: OpenAIToolNameCodec,
        tool_calls: OpenAIToolCallAssembler,
        extra_reasoning_events: _ExtraReasoningEvents,
        replay_scope: ReplayCompatibilityScope,
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
            StructuredReasoningStream(replay_scope)
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
    def ledger(self) -> InferenceStreamLedger:
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
        return has_generated_output(self._ledger)

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

    @property
    def tool_calls(self) -> OpenAIToolCallAssembler:
        return self._tool_calls

    def start_events(self) -> Iterator[InferenceEvent]:
        if self._started:
            return
        self._started = True
        yield self._ledger.start_response()

    def bind_tool_argument_aliases(self, aliases: dict[str, dict[str, str]]) -> None:
        if self._aliases_bound:
            raise RuntimeError("tool argument aliases already bound")
        self._aliases_bound = True
        self._tool_argument_aliases = aliases

    def feed(self, chunk: Any) -> Iterator[InferenceEvent]:
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
                yield from self._ledger.ensure_reasoning_block()
                if reasoning:
                    yield self._ledger.emit_reasoning_delta(reasoning)

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
                    yield from self._ledger.ensure_reasoning_block()
                    yield self._ledger.emit_reasoning_delta(part.content)
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

    def finish_upstream(self) -> Iterator[InferenceEvent]:
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
                    yield from self._ledger.ensure_reasoning_block()
                    yield self._ledger.emit_reasoning_delta(remaining.content)
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

    def prepare_completion(self) -> Iterator[InferenceEvent]:
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
        has_content_blocks = self._ledger.has_content_block()
        if not has_content_blocks or (
            not has_emitted_tool
            and not self._ledger.accumulated_text.strip()
            and self._ledger.accumulated_reasoning.strip()
        ):
            yield from self._ledger.ensure_text_block()
            yield self._ledger.emit_text_delta(" ")

        yield from self._tool_calls.flush_tool_argument_alias_buffers(
            self._ledger,
            self._tool_names,
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
            provider_output_tokens=completion,
            input_tokens=input_tokens,
            provider_input_tokens=provider_input,
        )

    def terminal_events(
        self, *, usage_fields: dict[str, int]
    ) -> Iterator[InferenceEvent]:
        if self._completed:
            return
        completion = self.completion
        yield from self._ledger.finish_events(
            _canonical_finish_reason(completion.finish_reason),
            _chat_completion_usage(completion, usage_fields),
        )
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
        self._supervisor = StreamExecutionSupervisor(admission)
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
        live_model_infos = extract_openai_model_infos(
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
        model_infos_by_id = {
            model_info.model_id: model_info for model_info in live_model_infos
        }
        for model_info in model_infos_from_ids(listing.additional_model_ids):
            model_infos_by_id.setdefault(model_info.model_id, model_info)
        return frozenset(model_infos_by_id.values())

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
        request: InferenceRequest,
        *,
        provider_model: str,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
    ) -> dict[str, Any]:
        """Build a provider request from the immutable profile."""
        return build_openai_chat_request_body(
            request,
            provider_model=provider_model,
            reasoning=reasoning,
            policy=self._profile.request_policy,
            tool_names=OpenAIToolNameCodec.from_request(request),
            replay_scope=openai_replay_scope(
                self._provider_name,
                provider_model,
                replay_format="chat-completions",
            ),
            postprocessors=self._profile.request_postprocessors,
        )

    def preflight_stream(
        self,
        request: InferenceRequest,
        *,
        provider_model: str,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
    ) -> None:
        """Validate OpenAI-chat request conversion before streaming."""
        self._build_request_body(
            request,
            provider_model=provider_model,
            reasoning=reasoning,
        )

    def _handle_extra_reasoning(
        self, delta: Any, ledger: InferenceStreamLedger, *, output_reasoning: bool
    ) -> Iterator[InferenceEvent]:
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

    def _usage_fields(self, usage_info: Any) -> dict[str, int]:
        """Return provider-specific cumulative usage fields."""
        return {}

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
        request: InferenceRequest,
        input_tokens: int = 0,
        *,
        provider_model: str,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
    ) -> AsyncIterator[InferenceEvent]:
        """Stream provider-neutral inference events."""
        source = _OpenAIChatAttemptSource(
            self,
            request=request,
            provider_model=provider_model,
            input_tokens=input_tokens,
            request_id=request_id,
            response_model=response_model or request.model,
            reasoning=reasoning,
        )
        return self._supervisor.stream(
            source,
            publication=PublicationBuffer(),
            recovery=_OpenAIChatRecoveryStrategy(source),
        )


class _OpenAIChatAttemptState:
    """Persist one corrected Chat request across physical attempts."""

    def __init__(
        self,
        provider: OpenAIChatProvider,
        body: dict[str, Any],
        *,
        request_id: str | None,
    ) -> None:
        self._provider = provider
        self._body = provider._apply_learned_output_cap(body)
        self._request_id = request_id
        self._used_retry_kinds: set[str] = set()

    @property
    def body(self) -> dict[str, Any]:
        return self._body

    async def open_stream(self) -> Any:
        """Perform exactly one upstream call and normalize its stream."""
        stream: Any | None = None
        try:
            create_body = self._provider._prepare_create_body(self._body)
            stream = await self._provider._client.chat.completions.create(
                **create_body,
                stream=True,
            )
            return self._provider._normalize_stream(stream, self._body)
        except asyncio.CancelledError:
            raise
        except Exception:
            if stream is not None:
                await close_provider_stream(
                    stream,
                    active_error=sys.exception(),
                    provider_name=self._provider._provider_name,
                    request_id=self._request_id,
                )
            raise

    def apply_correction(self, error: Exception) -> bool:
        retry_body = self._provider._next_create_retry_body(
            error,
            self._body,
            self._used_retry_kinds,
        )
        if retry_body is None:
            return False
        self._body = retry_body
        return True


class _OpenAIChatStreamEpoch(AsyncIterator[object]):
    """One Chat raw stream and its fresh semantic decoder state."""

    def __init__(
        self,
        stream: Any,
        *,
        assembler: _OpenAIChatStreamAssembler,
        provider: OpenAIChatProvider,
        input_tokens: int,
        request_id: str | None,
    ) -> None:
        self._stream = stream
        self._iterator = aiter(stream)
        self._assembler = assembler
        self._provider = provider
        self._input_tokens = input_tokens
        self._request_id = request_id

    def __aiter__(self) -> AsyncIterator[object]:
        return self

    async def __anext__(self) -> object:
        return await anext(self._iterator)

    async def aclose(self) -> None:
        await maybe_await_aclose(self._stream)

    @property
    def recovery_snapshot(self) -> _OpenAIChatStreamAssembler:
        return self._assembler

    def start(self) -> StreamFeed:
        return StreamFeed(tuple(self._assembler.start_events()))

    def feed(self, raw: object) -> StreamFeed:
        return StreamFeed(tuple(self._assembler.feed(raw)))

    def finish(self) -> StreamFeed:
        events = list(self._assembler.finish_upstream())
        events.extend(self._assembler.prepare_completion())
        events.extend(
            self._assembler.terminal_events(
                usage_fields=self._provider._usage_fields(self._assembler.usage_info)
            )
        )
        return StreamFeed(tuple(events), terminal=True)

    def failure_events(self) -> tuple[InferenceEvent, ...]:
        return tuple(self._assembler.ledger.close_unclosed_blocks())

    def trace_completed(self) -> None:
        completion = self._assembler.completion
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
            provider=self._provider._provider_name,
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


class _OpenAIChatRecoveryEpoch(AsyncIterator[object]):
    """Collect one private Chat continuation or tool-repair response."""

    def __init__(
        self,
        stream: Any,
        *,
        provider: OpenAIChatProvider,
        request: InferenceRequest,
        tool_names: OpenAIToolNameCodec,
        accepted_body: Mapping[str, Any],
        include_reasoning: bool,
    ) -> None:
        self._stream = stream
        self._iterator = aiter(stream)
        self._provider = provider
        self._request = request
        self._tool_names = tool_names
        self._accepted_body = accepted_body
        self._include_reasoning = include_reasoning
        self._text_parts: list[str] = []
        self._thinking_parts: list[str] = []
        self._tool_calls = OpenAIToolCallCollector()
        self._terminal_seen = False

    def __aiter__(self) -> AsyncIterator[object]:
        return self

    async def __anext__(self) -> object:
        return await anext(self._iterator)

    async def aclose(self) -> None:
        await maybe_await_aclose(self._stream)

    def feed(self, raw: object) -> None:
        choices = getattr(raw, "choices", None)
        if not choices:
            return
        choice = choices[0]
        if choice.finish_reason is not None:
            self._terminal_seen = True
        delta = choice.delta
        if delta is None:
            return
        if self._include_reasoning:
            reasoning = self._provider._profile.reasoning_delta(delta)
            if reasoning:
                self._thinking_parts.append(reasoning)
        content = getattr(delta, "content", None)
        if isinstance(content, str) and content:
            self._text_parts.append(content)
        native_tool_calls = getattr(delta, "tool_calls", None)
        if isinstance(native_tool_calls, list | tuple):
            for tool_call in native_tool_calls:
                self._tool_calls.add(tool_call)

    def finish(self) -> _CollectedRecoveryOutput:
        completed_tool_calls = self._tool_calls.completed_calls(
            self._request,
            tool_names=self._tool_names,
            tool_argument_aliases=self._provider._tool_argument_aliases(
                dict(self._accepted_body)
            ),
        )
        if self._tool_calls.has_calls and completed_tool_calls is None:
            raise TruncatedProviderStreamError(
                "Recovery stream ended with an incomplete tool call."
            )
        if not self._terminal_seen and not completed_tool_calls:
            raise TruncatedProviderStreamError(
                "Recovery stream ended without finish_reason."
            )
        return _CollectedRecoveryOutput(
            text="".join(self._text_parts),
            thinking="".join(self._thinking_parts),
            tool_calls=completed_tool_calls or (),
        )


class _OpenAIChatRecoverySource:
    """One corrected private Chat recovery request on the shared attempt budget."""

    def __init__(
        self,
        primary: _OpenAIChatAttemptSource,
        body: dict[str, Any],
        *,
        include_reasoning: bool,
    ) -> None:
        self._primary = primary
        self._state = _OpenAIChatAttemptState(
            primary.provider,
            body,
            request_id=primary.request_id,
        )
        self._include_reasoning = include_reasoning

    @property
    def trace_context(self) -> StreamTraceContext:
        return StreamTraceContext(
            provider_name=self._primary.provider._provider_name,
            request_id=self._primary.request_id,
            recovery_kind="openai_text",
        )

    @property
    def failure_override(self) -> ProviderFailureOverride:
        return self._primary.provider._provider_failure_override

    def trace_started(self, execution: ProviderExecution) -> None:
        del execution

    async def open(self) -> _OpenAIChatRecoveryEpoch:
        stream = await self._state.open_stream()
        try:
            return _OpenAIChatRecoveryEpoch(
                stream,
                provider=self._primary.provider,
                request=self._primary.request,
                tool_names=self._primary.tool_names,
                accepted_body=self._state.body,
                include_reasoning=self._include_reasoning,
            )
        except Exception:
            await close_provider_stream(
                stream,
                active_error=sys.exception(),
                provider_name=self._primary.provider._provider_name,
                request_id=self._primary.request_id,
            )
            raise

    def apply_correction(self, error: Exception) -> bool:
        return self._state.apply_correction(error)

    def attempt_error(self, error: Exception) -> Exception:
        return error

    def is_retryable(self, error: Exception) -> bool:
        return is_retryable_stream_error(error)

    def classify_failure(self, error: Exception) -> ExecutionFailure:
        return classify_provider_failure(
            underlying_provider_error(error),
            provider_name=self._primary.provider._provider_name,
            read_timeout_s=self._primary.provider._config.http_read_timeout,
            request_id=self._primary.request_id,
            provider_failure_override=(
                self._primary.provider._provider_failure_override
            ),
        )


class _OpenAIChatAttemptSource:
    """Request-scoped Chat connector, decoder factory, and failure policy."""

    def __init__(
        self,
        provider: OpenAIChatProvider,
        *,
        request: InferenceRequest,
        provider_model: str,
        input_tokens: int,
        request_id: str | None,
        response_model: str,
        reasoning: ReasoningPolicy,
    ) -> None:
        self._provider = provider
        self._request = request
        self._provider_model = provider_model
        self._input_tokens = input_tokens
        self._request_id = request_id
        self._response_model = response_model
        self._reasoning = reasoning
        self._tool_names = OpenAIToolNameCodec.from_request(request)
        self._replay_scope = openai_replay_scope(
            provider._provider_name,
            provider_model,
            replay_format="chat-completions",
        )
        self._response_id = f"response_{uuid.uuid4().hex}"
        self._state: _OpenAIChatAttemptState | None = None

    @property
    def provider(self) -> OpenAIChatProvider:
        return self._provider

    @property
    def request(self) -> InferenceRequest:
        return self._request

    @property
    def request_id(self) -> str | None:
        return self._request_id

    @property
    def tool_names(self) -> OpenAIToolNameCodec:
        return self._tool_names

    @property
    def body(self) -> dict[str, Any]:
        return self._ensure_state().body

    @property
    def trace_context(self) -> StreamTraceContext:
        return StreamTraceContext(
            provider_name=self._provider._provider_name,
            request_id=self._request_id,
        )

    @property
    def failure_override(self) -> ProviderFailureOverride:
        return self._provider._provider_failure_override

    def _ensure_state(self) -> _OpenAIChatAttemptState:
        if self._state is None:
            body = self._provider._build_request_body(
                self._request,
                provider_model=self._provider_model,
                reasoning=self._reasoning,
            )
            request_stream_usage(body)
            self._state = _OpenAIChatAttemptState(
                self._provider,
                body,
                request_id=self._request_id,
            )
        return self._state

    def trace_started(self, execution: ProviderExecution) -> None:
        body = self.body
        trace_event(
            stage="provider",
            event="provider.request.sent",
            source="provider",
            provider=self._provider._provider_name,
            request_id=self._request_id,
            execution_id=execution.execution_id,
            gateway_model=self._response_model,
            downstream_model=body.get("model"),
            message_count=len(body.get("messages", [])),
            tool_count=len(body.get("tools", [])),
            body=provider_chat_body_snapshot(body),
        )

    async def open(self) -> _OpenAIChatStreamEpoch:
        state = self._ensure_state()
        stream = await state.open_stream()
        try:
            assembler = self._new_stream_assembler(
                output_reasoning=self._reasoning.output_enabled
            )
            assembler.bind_tool_argument_aliases(
                self._provider._tool_argument_aliases(state.body)
            )
            return _OpenAIChatStreamEpoch(
                stream,
                assembler=assembler,
                provider=self._provider,
                input_tokens=self._input_tokens,
                request_id=self._request_id,
            )
        except Exception:
            await close_provider_stream(
                stream,
                active_error=sys.exception(),
                provider_name=self._provider._provider_name,
                request_id=self._request_id,
            )
            raise

    def apply_correction(self, error: Exception) -> bool:
        return self._ensure_state().apply_correction(error)

    def attempt_error(self, error: Exception) -> Exception:
        return error

    def is_retryable(self, error: Exception) -> bool:
        return is_retryable_stream_error(error)

    def classify_failure(self, error: Exception) -> ExecutionFailure:
        reported_error = underlying_provider_error(error)
        req_tag = f" request_id={self._request_id}" if self._request_id else ""
        self._provider._log_stream_transport_error(
            self._provider._provider_name,
            req_tag,
            reported_error,
            request_id=self._request_id,
        )
        failure = classify_provider_failure(
            reported_error,
            provider_name=self._provider._provider_name,
            read_timeout_s=self._provider._config.http_read_timeout,
            request_id=self._request_id,
            provider_failure_override=self._provider._provider_failure_override,
        )
        error_trace: dict[str, Any] = {
            "stage": "provider",
            "event": "provider.response.error",
            "source": "provider",
            "provider": self._provider._provider_name,
            "request_id": self._request_id,
            "exc_type": type(reported_error).__name__,
            "failure_kind": failure.kind.value,
            "status_code": failure.status_code,
            "provider_retryable": failure.retryable,
        }
        if self._provider._config.log_api_error_tracebacks:
            error_trace["error_message"] = failure.message
        trace_event(**error_trace)
        return failure

    def _new_stream_assembler(
        self, *, output_reasoning: bool
    ) -> _OpenAIChatStreamAssembler:
        def extra_reasoning_events(
            delta: Any, ledger: InferenceStreamLedger
        ) -> Iterator[InferenceEvent]:
            yield from self._provider._handle_extra_reasoning(
                delta,
                ledger,
                output_reasoning=output_reasoning,
            )

        return _OpenAIChatStreamAssembler(
            request=self._request,
            ledger=InferenceStreamLedger(
                self._response_id,
                self._response_model,
                self._input_tokens,
            ),
            profile=self._provider._profile,
            provider_name=self._provider._provider_name,
            output_reasoning=output_reasoning,
            tool_names=self._tool_names,
            tool_calls=OpenAIToolCallAssembler(
                record_extra_content=self._provider._record_tool_call_extra_content,
                replay_scope=self._replay_scope,
            ),
            extra_reasoning_events=extra_reasoning_events,
            replay_scope=self._replay_scope,
        )


class _OpenAIChatRecoveryStrategy:
    """Chat-only continuation, tool salvage, and repair policy."""

    def __init__(self, primary: _OpenAIChatAttemptSource) -> None:
        self._primary = primary

    def prefers_recovery(
        self, context: RecoveryContext[_OpenAIChatStreamAssembler]
    ) -> bool:
        assembler = context.snapshot
        return (
            context.retryable
            and assembler.generated_output
            and (context.attempts_remaining > 0 or assembler.complete_tool_salvageable)
            and (
                context.published
                or assembler.complete_tool_salvageable
                or context.attempts_remaining == 1
            )
        )

    async def resolve(
        self,
        context: RecoveryContext[_OpenAIChatStreamAssembler],
        attempts: BoundAttemptOperations,
    ) -> RecoveryOutcome | None:
        assembler = context.snapshot
        if self.prefers_recovery(context):
            try:
                recovery_events = await self._recovery_events(
                    assembler=assembler,
                    error=context.error,
                    attempts=attempts,
                )
            except asyncio.CancelledError, GeneratorExit:
                raise
            except Exception as recovery_error:
                trace_event(
                    stage="provider",
                    event="provider.recovery.failed",
                    source="provider",
                    provider=self._primary.provider._provider_name,
                    request_id=self._primary.request_id,
                    exc_type=type(recovery_error).__name__,
                )
            else:
                if recovery_events is not None:
                    return RecoveryOutcome(
                        events=tuple(recovery_events),
                        publish_buffer=(not context.published and context.has_buffered),
                        completed=True,
                    )

        if (
            not context.published
            and context.has_buffered
            and assembler.complete_tool_salvageable
        ):
            return RecoveryOutcome(
                events=tuple(assembler.ledger.close_unclosed_blocks()),
                publish_buffer=True,
                completed=False,
            )
        return None

    async def _recovery_events(
        self,
        *,
        assembler: _OpenAIChatStreamAssembler,
        error: Exception,
        attempts: BoundAttemptOperations,
    ) -> list[InferenceEvent] | None:
        """Build terminal recovery events when the interrupted stream permits it."""
        ledger = assembler.ledger
        body = self._primary.body
        if ledger.has_emitted_tool_block():
            if not all_emitted_tools_complete(ledger, self._primary.request):
                repair_events = await self._repair_tool_args(
                    body=body,
                    assembler=assembler,
                    attempts=attempts,
                )
                if repair_events is None:
                    return None
            else:
                repair_events = []
            events: list[InferenceEvent] = list(repair_events)
            events.extend(ledger.close_all_blocks())
            events.extend(
                ledger.finish_events(
                    FinishReason.END_TURN,
                    _estimated_recovery_usage(
                        input_tokens=self._primary._input_tokens,
                        output_tokens=ledger.estimate_output_tokens(),
                    ),
                )
            )
            trace_event(
                stage="provider",
                event="provider.recovery.tool_salvaged",
                source="provider",
                provider=self._primary.provider._provider_name,
                request_id=self._primary.request_id,
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
        recovered = await attempts.collect(
            _OpenAIChatRecoverySource(
                self._primary,
                recovery_body,
                include_reasoning=self._primary._reasoning.output_enabled,
            ),
            operation_kind=ProviderOperationKind.CONTINUATION,
        )
        text_suffix = continuation_suffix(partial_text, recovered.text)
        thinking_suffix = continuation_suffix(partial_thinking, recovered.thinking)
        events: list[InferenceEvent] = []
        if thinking_suffix:
            events.extend(ledger.ensure_reasoning_block())
            events.append(ledger.emit_reasoning_delta(thinking_suffix))
        if text_suffix:
            events.extend(ledger.ensure_text_block())
            events.append(ledger.emit_text_delta(text_suffix))
        if recovered.tool_calls:
            events.extend(ledger.close_content_blocks())
            for tool_call in recovered.tool_calls:
                events.extend(
                    assembler.tool_calls.process_tool_call(
                        tool_call,
                        ledger,
                        tool_names=self._primary.tool_names,
                    )
                )
        if not events:
            return None
        events.extend(ledger.close_all_blocks())
        events.extend(
            ledger.finish_events(
                FinishReason.END_TURN,
                _estimated_recovery_usage(
                    input_tokens=self._primary._input_tokens,
                    output_tokens=ledger.estimate_output_tokens(),
                ),
            )
        )
        trace_event(
            stage="provider",
            event="provider.recovery.continued",
            source="provider",
            provider=self._primary.provider._provider_name,
            request_id=self._primary.request_id,
        )
        return events

    async def _repair_tool_args(
        self,
        *,
        body: dict[str, Any],
        assembler: _OpenAIChatStreamAssembler,
        attempts: BoundAttemptOperations,
    ) -> list[InferenceEvent] | None:
        ledger = assembler.ledger
        schemas = tool_schemas_by_name(self._primary.request)
        events: list[InferenceEvent] = []
        for tool_index, state in started_tool_states(ledger):
            block = ledger.tool_block_for_tool_index(tool_index)
            emitted_prefix = block.content if block is not None else ""
            repair_prefix = emitted_prefix
            if not repair_prefix and state.name == "Task":
                repair_prefix = assembler.tool_calls.buffered_task_args(tool_index)
            if (
                not repair_prefix
                and tool_index in assembler.tool_argument_alias_buffers
            ):
                repair_prefix = assembler.tool_argument_alias_buffers[tool_index]
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
            while attempts.can_attempt:
                repair_attempt += 1
                recovered = await attempts.collect(
                    _OpenAIChatRecoverySource(
                        self._primary,
                        recovery_body,
                        include_reasoning=False,
                    ),
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
                        provider=self._primary.provider._provider_name,
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
        if not all_emitted_tools_complete(ledger, self._primary.request):
            return None
        return events
