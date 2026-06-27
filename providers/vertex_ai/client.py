"""Vertex AI Generative Language API provider (streamGenerateContent)."""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx

from core.anthropic import ContentType, HeuristicToolParser, SSEBuilder, ThinkTagParser
from core.anthropic.sse import map_stop_reason
from core.trace import provider_vertex_body_snapshot, trace_event
from providers.base import BaseProvider, ProviderConfig
from providers.error_mapping import (
    attach_provider_error_body,
    extract_provider_error_detail,
    map_error,
    user_visible_message_for_mapped_provider_error,
)
from providers.exceptions import AuthenticationError
from providers.model_listing import ProviderModelInfo, model_infos_from_ids
from providers.rate_limit import GlobalRateLimiter
from providers.transports.openai_chat.tool_calls import iter_heuristic_tool_use_sse

from .request import build_request_body, save_thought_signature

_DEFAULT_BASE_URL = "https://aiplatform.googleapis.com/v1"

_VERTEX_MODEL_FALLBACK: frozenset[str] = frozenset(
    (
        "google/gemini-3.5-flash",
        "google/gemini-3.1-flash-lite",
        "google/gemini-2.5-pro",
        "google/gemini-2.5-flash",
        "google/gemini-2.5-flash-lite",
        "google/gemini-2.0-flash-001",
        "google/gemini-2.0-flash-lite-001",
    )
)


class VertexAIProvider(BaseProvider):
    """Vertex AI Generative Language API via streamGenerateContent."""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        project_id: str = "",
        location: str = "",
    ):
        super().__init__(config)
        self._project_id = project_id.strip()
        self._location = location.strip()
        base_url = (config.base_url or "").strip()
        explicit_base_url = bool(config.base_url and config.base_url.strip())
        if not base_url:
            base_url = _build_generativelanguage_base_url(self._location)
        if not base_url or (
            base_url == _DEFAULT_BASE_URL
            and not self._location
            and not explicit_base_url
        ):
            raise AuthenticationError(
                "VERTEX_AI_BASE_URL or VERTEX_AI_LOCATION must be set. "
                "Provide an explicit endpoint URL or a Google Cloud region (e.g. us-central1)."
            )
        if "endpoints/openapi" in base_url:
            raise AuthenticationError(
                "VERTEX_AI_BASE_URL cannot be an OpenAI-compatible endpoint (ending in '/endpoints/openapi'). "
                "This provider uses the native Vertex AI REST API. "
                "Use the 'gemini' provider for OpenAI-compatible Vertex AI endpoints."
            )
        if not config.api_key.strip():
            raise AuthenticationError(
                "VERTEX_AI_API_KEY is required for Generative Language API requests."
            )
        self._provider_name = "VERTEX_AI"
        self._api_key = config.api_key
        self._base_url = base_url.rstrip("/")
        self._global_rate_limiter = GlobalRateLimiter.get_scoped_instance(
            "vertex_ai",
            rate_limit=config.rate_limit,
            rate_window=config.rate_window,
            max_concurrency=config.max_concurrency,
        )
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            proxy=config.proxy or None,
            timeout=httpx.Timeout(
                config.http_read_timeout,
                connect=config.http_connect_timeout,
                read=config.http_read_timeout,
                write=config.http_write_timeout,
            ),
        )

    async def cleanup(self) -> None:
        from .request import flush_thought_signatures

        await flush_thought_signatures()
        await self._client.aclose()

    async def list_model_ids(self) -> frozenset[str]:
        return frozenset(_VERTEX_MODEL_FALLBACK)

    async def list_model_infos(self) -> frozenset[ProviderModelInfo]:
        return model_infos_from_ids(_VERTEX_MODEL_FALLBACK)

    def _build_request_body(
        self, request: Any, thinking_enabled: bool | None = None
    ) -> dict[str, Any]:
        return build_request_body(
            request,
            thinking_enabled=self._is_thinking_enabled(request, thinking_enabled),
        )

    async def _send_stream_request(
        self, body: dict[str, Any], model_ref: str
    ) -> httpx.Response:
        publisher, model = _split_publisher_model(model_ref)
        if self._project_id:
            location = self._location or "us-central1"
            path = f"projects/{self._project_id}/locations/{location}/publishers/{publisher}/models/{model}:streamGenerateContent"
        else:
            path = f"publishers/{publisher}/models/{model}:streamGenerateContent"
        headers = {"Content-Type": "application/json"}
        if self._api_key.startswith("ya29."):
            headers["Authorization"] = f"Bearer {self._api_key}"
        else:
            headers["x-goog-api-key"] = self._api_key

        request = self._client.build_request(
            "POST",
            path,
            json=body,
            params={"alt": "sse"},
            headers=headers,
        )
        return await self._client.send(request, stream=True)

    async def _iter_genai_stream(
        self, response: httpx.Response
    ) -> AsyncIterator[dict[str, Any]]:
        try:
            if response.status_code >= 400:
                await _log_response_error(response)
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as error:
                    body_bytes = getattr(response, "content", b"")
                    attach_provider_error_body(error, body_bytes, truncated=False)
                    raise error
            async for line in response.aiter_lines():
                if not line:
                    continue
                payload = line
                if line.startswith("data:"):
                    payload = line.split(":", 1)[1].strip()
                if payload == "[DONE]":
                    break
                try:
                    data = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if self._config.log_raw_sse_events:
                    from loguru import logger

                    logger.debug("VERTEX_AI_RAW_STREAM_DATA: {}", data)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            yield item
                    continue
                if isinstance(data, dict):
                    yield data
        finally:
            await response.aclose()

    async def stream_response(
        self,
        request: Any,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        thinking_enabled: bool | None = None,
    ) -> AsyncIterator[str]:
        message_id = f"msg_{request_id or 'vertex'}"
        sse = SSEBuilder(
            message_id,
            request.model,
            input_tokens,
            log_raw_events=self._config.log_raw_sse_events,
        )

        resolved_thinking_enabled = self._is_thinking_enabled(request, thinking_enabled)
        body = self._build_request_body(
            request, thinking_enabled=resolved_thinking_enabled
        )
        trace_event(
            stage="provider",
            event="provider.request.sent",
            source="provider",
            provider=self._provider_name,
            gateway_model=request.model,
            downstream_model=_strip_provider_prefix(request.model),
            message_count=len(body.get("contents", [])),
            tool_count=len(body.get("tools", [])),
            body=provider_vertex_body_snapshot(body),
        )

        yield sse.message_start()

        think_parser = ThinkTagParser()
        heuristic_parser = HeuristicToolParser()
        finish_reason = None
        usage_info: dict[str, Any] | None = None
        current_signature: str | None = None

        async with self._global_rate_limiter.concurrency_slot():
            try:
                response = await self._send_stream_request(
                    body,
                    _strip_provider_prefix(request.model),
                )
                async for data in self._iter_genai_stream(response):
                    usage_info = _update_usage(usage_info, data)
                    candidate = _first_candidate(data)
                    if not candidate:
                        continue
                    finish_reason = candidate.get("finishReason") or finish_reason
                    content = candidate.get("content") or {}
                    parts = content.get("parts") or []
                    for part in parts:
                        if "thoughtSignature" in part:
                            current_signature = part["thoughtSignature"]
                        if "text" in part:
                            is_thought = part.get("thought") is True
                            if is_thought:
                                if resolved_thinking_enabled:
                                    for event in sse.ensure_thinking_block():
                                        yield event
                                    yield sse.emit_thinking_delta(
                                        part.get("text") or ""
                                    )
                            else:
                                for chunk in think_parser.feed(part.get("text") or ""):
                                    if chunk.type == ContentType.THINKING:
                                        if resolved_thinking_enabled:
                                            for event in sse.ensure_thinking_block():
                                                yield event
                                            yield sse.emit_thinking_delta(chunk.content)
                                    else:
                                        filtered_text, detected_tools = (
                                            heuristic_parser.feed(chunk.content)
                                        )
                                        if filtered_text:
                                            for event in sse.ensure_text_block():
                                                yield event
                                            yield sse.emit_text_delta(filtered_text)
                                        for tool_use in detected_tools:
                                            for event in iter_heuristic_tool_use_sse(
                                                sse, tool_use
                                            ):
                                                yield event
                        if "functionCall" in part:
                            call = part["functionCall"]
                            name = call.get("name") or "tool"
                            args = call.get("args") or {}
                            if current_signature:
                                save_thought_signature(name, args, current_signature)

                            for event in sse.close_content_blocks():
                                yield event
                            for event in _emit_function_call(sse, call):
                                yield event

            except Exception as e:
                mapped_e = map_error(e, rate_limiter=self._global_rate_limiter)
                error_message = user_visible_message_for_mapped_provider_error(
                    mapped_e,
                    provider_name=self._provider_name,
                    read_timeout_s=self._config.http_read_timeout,
                    detail=extract_provider_error_detail(e),
                    request_id=request_id,
                )
                for event in sse.close_all_blocks():
                    yield event
                for event in sse.emit_error(error_message):
                    yield event
                yield sse.message_delta("end_turn", 1)
                yield sse.message_stop()
                return

        remaining = think_parser.flush()
        if remaining and remaining.type == ContentType.TEXT:
            for event in sse.ensure_text_block():
                yield event
            yield sse.emit_text_delta(remaining.content)

        for tool_use in heuristic_parser.flush():
            for event in iter_heuristic_tool_use_sse(sse, tool_use):
                yield event

        for event in sse.close_all_blocks():
            yield event

        output_tokens = 0
        if usage_info:
            output_tokens = int(
                usage_info.get("candidatesTokenCount")
                or usage_info.get("completionTokenCount")
                or 0
            )
        final_stop_reason = map_stop_reason(_map_finish_reason(finish_reason))
        if sse.blocks.tool_states and final_stop_reason == "end_turn":
            final_stop_reason = "tool_use"

        yield sse.message_delta(final_stop_reason, output_tokens)
        yield sse.message_stop()


def _build_generativelanguage_base_url(location: str) -> str:
    if location.strip():
        return f"https://{location.strip()}-aiplatform.googleapis.com/v1"
    return _DEFAULT_BASE_URL


def _strip_provider_prefix(model_ref: str) -> str:
    if "/" not in model_ref:
        return model_ref
    return model_ref.split("/", 1)[1]


def _split_publisher_model(model_ref: str) -> tuple[str, str]:
    model_name = _strip_provider_prefix(model_ref)
    if "/" in model_name:
        publisher, model = model_name.split("/", 1)
        return publisher, model
    return "google", model_name


def _first_candidate(data: dict[str, Any]) -> dict[str, Any] | None:
    candidates = data.get("candidates")
    if isinstance(candidates, list) and candidates and isinstance(candidates[0], dict):
        return candidates[0]
    return None


def _update_usage(
    usage: dict[str, Any] | None, data: dict[str, Any]
) -> dict[str, Any] | None:
    meta = data.get("usageMetadata")
    if isinstance(meta, dict):
        usage = dict(usage or {})
        usage.update(meta)
    return usage


def _map_finish_reason(reason: str | None) -> str | None:
    if reason is None:
        return None
    mapping = {
        "STOP": "end_turn",
        "MAX_TOKENS": "max_tokens",
        "SAFETY": "content_filter",
        "RECITATION": "content_filter",
        "OTHER": "end_turn",
    }
    return mapping.get(reason, "end_turn")


def _emit_function_call(sse: SSEBuilder, call: dict[str, Any]) -> list[str]:
    name = call.get("name") or "tool"
    args = call.get("args") or {}
    tool_index = len(sse.blocks.tool_states)
    tool_id = f"toolu_vertex_{uuid.uuid4().hex[:8]}"
    events: list[str] = []
    events.append(sse.start_tool_block(tool_index, tool_id, name))
    events.append(sse.emit_tool_delta(tool_index, json.dumps(args)))
    events.append(sse.stop_tool_block(tool_index))
    if tool_index in sse.blocks.tool_states:
        sse.blocks.tool_states[tool_index].started = False
    return events


async def _log_response_error(response: httpx.Response) -> None:
    text = ""
    try:
        body = await response.aread()
        text = body.decode("utf-8", errors="replace")
    except httpx.HTTPError:
        text = "<failed to read response body>"
    if len(text) > 2000:
        text = f"{text[:2000]}...<truncated>"
    trace_event(
        stage="provider",
        event="provider.response.error",
        source="provider",
        provider="VERTEX_AI",
        status_code=response.status_code,
        body=text,
    )
