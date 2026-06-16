"""Application services for the Claude-compatible API."""

from __future__ import annotations

import traceback
import uuid
import asyncio
import time
import random
import re
import json
from collections.abc import AsyncIterator, Callable
from typing import Any

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger

from config.settings import Settings
from core.anthropic import (
    SSEBuilder,
    get_token_count,
    get_user_facing_error_message,
)
from core.anthropic.sse import ANTHROPIC_SSE_RESPONSE_HEADERS
from core.trace import api_messages_request_snapshot, trace_event, traced_async_stream
from providers.base import BaseProvider
from providers.exceptions import InvalidRequestError, ProviderError

from .agent import AgentEngine
from .cache import ResponseCache
from .fusion import FusionEngine
from .orchestrator import Orchestrator
from .vision_bridge import VisionBridge
from .model_router import ModelRouter, ResolvedModel
from .models.anthropic import (
    ContentBlockText,
    Message,
    MessagesRequest,
    TokenCountRequest,
)
from .models.responses import MessagesResponse, TokenCountResponse
from .optimization_handlers import try_optimizations
from .web_tools.egress import WebFetchEgressPolicy
from .web_tools.request import (
    is_web_server_tool_request,
    openai_chat_upstream_server_tool_error,
)
from .web_tools.streaming import stream_web_server_tool_response
from .performance import performance_tracker

TokenCounter = Callable[[list[Any], str | list[Any] | None, list[Any] | None], int]

ProviderGetter = Callable[[str], BaseProvider]

# Providers that use ``/chat/completions`` + Anthropic-to-OpenAI conversion (not native Messages).
_OPENAI_CHAT_UPSTREAM_IDS = frozenset({"nvidia_nim", "opencode", "opencode_go"})


def anthropic_sse_streaming_response(
    body: AsyncIterator[str],
) -> StreamingResponse:
    """Return a :class:`StreamingResponse` for Anthropic-style SSE streams."""
    return StreamingResponse(
        body,
        media_type="text/event-stream",
        headers=ANTHROPIC_SSE_RESPONSE_HEADERS,
    )


def _http_status_for_unexpected_service_exception(_exc: BaseException) -> int:
    """HTTP status for uncaught non-provider failures (stable client contract)."""
    return 500


def _log_unexpected_service_exception(
    settings: Settings,
    exc: BaseException,
    *,
    context: str,
    request_id: str | None = None,
) -> None:
    """Log service-layer failures without echoing exception text unless opted in."""
    if settings.log_api_error_tracebacks:
        if request_id is not None:
            logger.error("{} request_id={}: {}", context, request_id, exc)
        else:
            logger.error("{}: {}", context, exc)
        logger.error(traceback.format_exc())
        return
    if request_id is not None:
        logger.error(
            "{} request_id={} exc_type={}",
            context,
            request_id,
            type(exc).__name__,
        )
    else:
        logger.error("{} exc_type={}", context, type(exc).__name__)


def _require_non_empty_messages(messages: list[Any]) -> None:
    if not messages:
        raise InvalidRequestError("messages cannot be empty")


class ClaudeProxyService:
    """Coordinate request optimization, model routing, token count, and providers."""

    def __init__(
        self,
        settings: Settings,
        provider_getter: ProviderGetter,
        model_router: ModelRouter | None = None,
        token_counter: TokenCounter = get_token_count,
        cache: ResponseCache | None = None,
    ):
        self._settings = settings
        self._provider_getter = provider_getter
        self._model_router = model_router or ModelRouter(settings)
        self._token_counter = token_counter
        self._cache = cache or ResponseCache(enabled=settings.enable_response_caching)
        self._fusion = FusionEngine(self)
        self._orchestrator = Orchestrator(self)
        self._vision = VisionBridge(self)
        self._agent = AgentEngine(self)

    def _resolve_request(self, request_data: MessagesRequest) -> tuple[list[ResolvedModel], MessagesRequest]:
        # Support historical test double (FixedProviderModelRouter)
        if hasattr(self._model_router, "resolve_messages_request"):
            # If it's the old FixedProviderModelRouter, use it
            if not hasattr(self._model_router, "resolve_candidates") or type(self._model_router).__name__ == "FixedProviderModelRouter":
                routed = self._model_router.resolve_messages_request(request_data)
                return [routed.resolved], routed.request

        try:
            return self._model_router.resolve_candidates(request_data.model), request_data
        except (AttributeError, NotImplementedError, ValueError):
            return [], request_data

    async def create_message(self, request_data: MessagesRequest) -> object:
        """Create a message response or streaming response with automatic failover."""
        try:
            _require_non_empty_messages(request_data.messages)

            if self._settings.log_raw_api_payloads:
                logger.debug("FULL_PAYLOAD [{}]: {}", request_data.model, api_messages_request_snapshot(request_data))

            # 1. Cache check (Global Priority)
            cached_response = self._cache.get(request_data)
            if cached_response:
                if request_data.stream:
                    async def _stream_cached() -> AsyncIterator[str]:
                        for chunk in cached_response: yield chunk
                    return anthropic_sse_streaming_response(_stream_cached())
                else:
                    # In a real scenario, we'd parse the cached SSE back to JSON if needed.
                    pass

            # 2. Routing and Specialized Logic
            res = await self._resolve_create_message(request_data)

            # 3. Handle Full Response (Optimizations/Mocks)
            if isinstance(res, MessagesResponse):
                return res

            # 4. Handle Stream
            it = res
            try:
                # Prime the stream to catch immediate errors (Auth, etc.) before returning 200 OK
                first_chunk = await it.__anext__()
            except StopAsyncIteration:
                return anthropic_sse_streaming_response(self._empty_gen())

            async def _combined_gen(first, rest):
                yield first
                async for chunk in rest:
                    yield chunk

            # Wrap in caching and return SSE
            return anthropic_sse_streaming_response(self._cap_and_cache(request_data, _combined_gen(first_chunk, it)))

        except (ProviderError, HTTPException):
            raise
        except Exception as e:
            _log_unexpected_service_exception(self._settings, e, context="CREATE_MESSAGE_ERROR")
            status_code = _http_status_for_unexpected_service_exception(e)
            detail = get_user_facing_error_message(e)
            if self._settings.log_api_error_tracebacks:
                detail = str(e)
            raise HTTPException(
                status_code=status_code,
                detail=detail,
            ) from e

    async def _resolve_create_message(self, request_data: MessagesRequest) -> AsyncIterator[str] | MessagesResponse:
        """Determine which routing mode or specialized handler to use."""
        model_name = request_data.model

        # Specialized Routing Handlers
        if model_name.startswith("fusion/"):
            panel_name = model_name[7:]
            rounds = 1
            if panel_name.startswith("v2/"):
                panel_name = panel_name[3:]
                rounds = 2
            resolved_panel = self._settings.resolve_fusion_panel(panel_name)
            if resolved_panel:
                judge, panel = resolved_panel
                return self._fusion.run_fusion(request_data, panel, judge, rounds=rounds)
            parts = panel_name.split(":", 1)
            if len(parts) == 2:
                judge, panel = parts[0], parts[1].split(",")
                return self._fusion.run_fusion(request_data, panel, judge, rounds=rounds)

        if model_name.startswith("race/"):
            models = model_name[5:].split(",")
            if len(models) >= 2: return self._run_race_mode(request_data, models)

        if model_name.startswith("cot/"):
            parts = model_name[4:].split(":", 1)
            if len(parts) == 2: return self._orchestrator.run_cot(request_data, parts[0], parts[1])

        if model_name.startswith("probe/"): return self._run_probe_routing(request_data)
        if model_name.startswith("verify/"): return self._run_verified_routing(request_data, model_name[7:])
        if model_name.startswith("research/"): return self._run_research_routing(request_data, model_name[9:])
        if model_name.startswith("agent/"):
            parts = model_name[6:].split(":", 1)
            if len(parts) == 2: return self._agent.run_agent(request_data, parts[0], parts[1])
        if model_name.startswith("cluster/"): return self._run_clustered_routing(request_data, model_name[8:].split(","))
        if model_name.startswith("vote/"): return self._run_ensemble_voting(request_data, model_name[5:].split(","))
        if model_name.startswith("rag/"): return self._run_rag_routing(request_data, model_name[4:])
        if model_name.startswith("hybrid/"): return self._run_hybrid_routing(request_data, model_name[7:])
        if model_name.startswith("shadow/"):
            parts = model_name[7:].split(":", 1)
            if len(parts) == 2: return self._run_shadow_routing(request_data, parts[0], parts[1])
        if model_name.startswith("repair/"): return self._run_auto_repair_routing(request_data, model_name[7:])

        # Context Compression
        if self._settings.context_compression_threshold > 0:
            current_tokens = self._token_counter(request_data.messages, request_data.system, request_data.tools)
            if current_tokens > self._settings.context_compression_threshold:
                request_data = await self._compress_context(request_data)

        # Standard Routing
        candidates, routed_request_data = self._resolve_request(request_data)
        if not candidates:
            raise HTTPException(status_code=500, detail="Failed to resolve any provider")

        # Vision Bridge
        if self._vision.has_images(request_data):
            primary = candidates[0]
            registry = getattr(self, "provider_registry", None)
            supports_vision = False
            if registry:
                supports_vision = registry.cached_model_supports_vision(primary.provider_id, primary.provider_model)
            if not supports_vision:
                model_low = primary.provider_model.lower()
                if "vision" not in model_low and "opus" not in model_low and "sonnet" not in model_low:
                    return self._vision.bridge_vision(request_data, request_data.model)

        # Optimizations (e.g. prefix detection, quota check mock)
        primary_resolved = candidates[0]
        primary_routed = routed_request_data.model_copy(deep=True)
        primary_routed.model = primary_resolved.provider_model
        optimized = try_optimizations(primary_routed, self._settings)
        if optimized is not None:
            return optimized

        # Local Web Server Tools
        if is_web_server_tool_request(request_data) and self._settings.enable_web_server_tools:
            input_tokens = self._token_counter(request_data.messages, request_data.system, request_data.tools)
            return stream_web_server_tool_response(
                request_data,
                input_tokens=input_tokens,
                web_fetch_egress=WebFetchEgressPolicy(
                    allow_private_network_targets=self._settings.allow_private_network_targets,
                    allowed_schemes=self._settings.allowed_schemes,
                ),
                verbose_client_errors=self._settings.log_api_error_tracebacks
            )

        # Final Fallback: Failover stream
        return self._stream_with_failover(request_data, candidates)

    async def _stream_optimized_json(self, response: MessagesResponse) -> AsyncIterator[str]:
        """Convert a static MessagesResponse into a standard Anthropic SSE stream."""
        message_id = response.id or f"msg_{uuid.uuid4().hex[:12]}"
        sse = SSEBuilder(message_id, response.model)
        yield sse.message_start()

        for block in response.content:
            if isinstance(block, ContentBlockText):
                yield sse.start_text_block()
                yield sse.emit_text_delta(block.text)
                yield sse.stop_text_block()
            elif isinstance(block, dict) and block.get("type") == "text":
                yield sse.start_text_block()
                yield sse.emit_text_delta(block.get("text", ""))
                yield sse.stop_text_block()

        yield sse.message_delta("end_turn", response.usage.output_tokens if response.usage else 1)
        yield sse.message_stop()

    async def _cap_and_cache(self, request_data: MessagesRequest, it: AsyncIterator[str]) -> AsyncIterator[str]:
        """Capture chunks to cache them if the stream completes successfully."""
        chunks = []
        async for chunk in it:
            chunks.append(chunk)
            yield chunk
        if chunks:
            self._cache.set(request_data, chunks)

    async def _empty_gen(self) -> AsyncIterator[str]:
        if False: yield ""

    async def _stream_with_failover(self, original_request: MessagesRequest, candidates: list[ResolvedModel]) -> AsyncIterator[str]:
        last_exception: Exception | None = None
        for i, resolved in enumerate(candidates):
            try:
                routed = original_request.model_copy(deep=True)
                routed.model = resolved.provider_model

                if resolved.provider_id in _OPENAI_CHAT_UPSTREAM_IDS:
                    tool_err = openai_chat_upstream_server_tool_error(routed, web_tools_enabled=self._settings.enable_web_server_tools)
                    if tool_err is not None:
                        if i == len(candidates) - 1: raise InvalidRequestError(tool_err)
                        continue

                provider = self._provider_getter(resolved.provider_id)
                provider.preflight_stream(routed, thinking_enabled=resolved.thinking_enabled)
                request_id = f"req_{uuid.uuid4().hex[:12]}"
                input_tokens = self._token_counter(routed.messages, routed.system, routed.tools)

                stream = traced_async_stream(
                    provider.stream_response(routed, input_tokens=input_tokens, request_id=request_id, thinking_enabled=resolved.thinking_enabled),
                    stage="egress", source="api", complete_event="api.response.stream_completed",
                    interrupted_event="api.response.stream_interrupted", chunk_event=None,
                    extra={"request_id": request_id, "provider_id": resolved.provider_id, "gateway_model": original_request.model},
                )

                chunk_received = False
                async for chunk in stream:
                    chunk_received = True
                    yield chunk
                if chunk_received: return

            except (ProviderError, HTTPException) as e:
                status_code = getattr(e, "status_code", 500)
                if status_code in (429, 500, 502, 503, 504) and i < len(candidates) - 1:
                    logger.warning("Provider {} failed ({}), trying fallback", resolved.provider_id, status_code)
                    last_exception = e
                    continue
                raise
            except Exception as e:
                if i < len(candidates) - 1:
                    logger.warning("Provider {} failed ({}), trying fallback", resolved.provider_id, type(e).__name__)
                    last_exception = e
                    continue
                raise
        if last_exception: raise last_exception

    async def _run_race_mode(self, request_data: MessagesRequest, models: list[str]) -> AsyncIterator[str]:
        logger.info("Starting Race Mode: models={}", models)
        start_time = time.time()
        async def _attempt(model_ref: str):
            candidates = self._model_router.resolve_candidates(model_ref)
            provider = self._provider_getter(candidates[0].provider_id)
            routed = request_data.model_copy(update={"model": candidates[0].provider_model})
            stream = provider.stream_response(routed, input_tokens=self._token_counter(routed.messages, routed.system, routed.tools))
            it = stream.__aiter__()
            first_chunk = await it.__anext__()
            ttft = time.time() - start_time
            async def _rest():
                yield first_chunk
                async for c in it: yield c
            return model_ref, _rest(), ttft

        tasks = [asyncio.create_task(_attempt(m)) for m in models]
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        winner_ref, winner_stream, winner_ttft = None, None, 0.0
        for t in done:
            try: winner_ref, winner_stream, winner_ttft = t.result(); break
            except: pass
        for t in pending: t.cancel()
        if winner_stream:
            logger.info("Race won by: {} in {:.2f}s", winner_ref, winner_ttft)
            performance_tracker.record_request(Settings.parse_provider_type(winner_ref), 0.0, 200, ttft=winner_ttft)
            async for chunk in winner_stream: yield chunk
        else: raise HTTPException(status_code=500, detail="Race failed")

    async def _run_clustered_routing(self, request_data: MessagesRequest, models: list[str]) -> AsyncIterator[str]:
        all_c = []
        for m in models: all_c.extend(self._model_router.resolve_candidates(m))
        def _score(c):
            metrics = performance_tracker.get_metrics(c.provider_id)
            if metrics.success_count == 0: return 1000.0
            return (1.0 - metrics.error_rate) / max(metrics.avg_latency + (metrics.avg_ttft * 2), 0.001)
        all_c.sort(key=_score, reverse=True)
        selected = random.choice(all_c[:3])
        return self._stream_with_failover(request_data, [selected] + [c for c in all_c if c != selected])

    async def _run_probe_routing(self, request_data: MessagesRequest) -> AsyncIterator[str]:
        probe_model = self._settings.routing_probe_model or self._settings.model_haiku or self._settings.model
        intent_raw = await self._orchestrator._get_full_response(probe_model, request_data.model_copy(update={
            "messages": [Message(role="user", content=f"Classify intent (OPUS, SONNET, or HAIKU) for: {request_data.messages[-1].content}")],
            "model": probe_model
        }))
        intent = intent_raw.strip().upper()
        fallback = self._settings.model
        if "OPUS" in intent: fallback = self._settings.model_opus or fallback
        elif "SONNET" in intent: fallback = self._settings.model_sonnet or fallback
        elif "HAIKU" in intent: fallback = self._settings.model_haiku or fallback
        target = performance_tracker.get_best_model_for_category(intent, fallback)
        if "CODING" in intent or "LOGIC" in intent:
            request_data.temperature = 0.0; request_data.top_p = 0.9
        elif "CREATIVE" in intent:
            request_data.temperature = 0.9; request_data.top_p = 1.0
        start = time.time()
        candidates = self._model_router.resolve_candidates(target)
        async def _track(it):
            success = False
            async for chunk in it:
                if "message_start" in chunk: success = True
                yield chunk
            performance_tracker.record_category_performance(intent, target, time.time() - start, 200 if success else 500)
        return _track(self._stream_with_failover(request_data, candidates))

    async def _run_verified_routing(self, request_data: MessagesRequest, target: str) -> AsyncIterator[str]:
        draft = await self._orchestrator._get_full_response(target, request_data)
        critique = await self._orchestrator._get_full_response(target, request_data.model_copy(update={
            "messages": [Message(role="user", content=f"Critique this draft for the request '{request_data.messages[-1].content}':\n\n{draft}")]
        }))
        final_req = request_data.model_copy(update={
            "messages": request_data.messages + [Message(role="user", content=f"Final answer based on critique:\n{critique}\n\nDraft:\n{draft}")]
        })
        async for chunk in self._stream_with_failover(final_req, self._model_router.resolve_candidates(target)):
            if "message_start" in chunk: continue
            yield chunk

    async def _run_research_routing(self, request_data: MessagesRequest, target: str) -> AsyncIterator[str]:
        fast = self._settings.model_haiku or self._settings.model
        queries_raw = await self._orchestrator._get_full_response(fast, request_data.model_copy(update={
            "messages": [Message(role="user", content=f"List 3 research queries for: {request_data.messages[-1].content}. JSON list.")]
        }))
        try: queries = json.loads(re.search(r'\[.*\]', queries_raw, re.DOTALL).group(0))
        except: queries = [request_data.messages[-1].content[:50]]
        results = [f"Found for {q}: [Simulated result]" for q in queries]
        final_req = request_data.model_copy(update={
            "messages": request_data.messages + [Message(role="user", content=f"Answer using research:\n" + "\n".join(results))]
        })
        async for chunk in self._stream_with_failover(final_req, self._model_router.resolve_candidates(target)):
            if "message_start" in chunk: continue
            yield chunk

    async def _run_rag_routing(self, request_data: MessagesRequest, target: str) -> AsyncIterator[str]:
        if len(request_data.messages) <= 6: return self._stream_with_failover(request_data, self._model_router.resolve_candidates(target))
        history = request_data.messages[:-2]
        retrieval_prompt = f"Pick relevant history indices for '{request_data.messages[-1].content}':\n" + "\n".join(f"[{i}]: {str(m.content)[:100]}" for i, m in enumerate(history))
        indices_raw = await self._orchestrator._get_full_response(self._settings.model_haiku or self._settings.model, request_data.model_copy(update={
            "messages": [Message(role="user", content=retrieval_prompt)]
        }))
        try:
            indices = [int(i.strip()) for i in indices_raw.split(",") if i.strip().isdigit()]
            retrieved = [history[i] for i in indices if 0 <= i < len(history)]
        except: retrieved = history[-3:]
        rag_req = request_data.model_copy(update={"messages": retrieved + request_data.messages[-2:]})
        return self._stream_with_failover(rag_req, self._model_router.resolve_candidates(target))

    async def _run_hybrid_routing(self, request_data: MessagesRequest, cloud: str) -> AsyncIterator[str]:
        intent_raw = await self._orchestrator._get_full_response(self._settings.model_haiku or self._settings.model, request_data.model_copy(update={
            "messages": [Message(role="user", content=f"Is this SIMPLE or COMPLEX? {request_data.messages[-1].content}")]
        }))
        if "SIMPLE" in intent_raw.upper():
            try: return self._stream_with_failover(request_data, self._model_router.resolve_candidates("ollama/llama3"))
            except: pass
        return self._stream_with_failover(request_data, self._model_router.resolve_candidates(cloud))

    async def _run_shadow_routing(self, request_data: MessagesRequest, primary_ref: str, shadow_ref: str) -> AsyncIterator[str]:
        primary_fut = asyncio.Future()
        async def _shadow(pfut):
            try:
                shadow_text = await self._orchestrator._get_full_response(shadow_ref, request_data)
                ptext = await pfut; await self._distill_feedback(ptext, shadow_text, shadow_ref, request_data)
            except: pass
        asyncio.create_task(_shadow(primary_fut))
        full_text = []
        async def _wrapper() -> AsyncIterator[str]:
            async for chunk in self._stream_with_failover(request_data, self._model_router.resolve_candidates(primary_ref)):
                if "text" in chunk:
                    try:
                        for line in chunk.splitlines():
                            if line.startswith("data: "):
                                d = json.loads(line[6:].strip())
                                if d.get("type") == "content_block_delta": full_text.append(d.get("delta", {}).get("text", ""))
                    except: pass
                yield chunk
            primary_fut.set_result("".join(full_text))
        return _wrapper()

    async def _distill_feedback(self, p_text: str, s_text: str, s_model: str, req: MessagesRequest):
        judge = self._settings.model_opus or self._settings.model
        score_raw = await self._orchestrator._get_full_response(judge, req.model_copy(update={
            "messages": [Message(role="user", content=f"Grade shadow vs primary (0.0 to 1.0). Primary: {p_text}\n\nShadow: {s_text}")]
        }))
        m = re.search(r'\d+\.\d+', score_raw)
        if m: performance_tracker.record_quality_score(s_model, float(m.group(0)))

    async def _run_ensemble_voting(self, request_data: MessagesRequest, models: list[str]) -> AsyncIterator[str]:
        tasks = [self._orchestrator._get_full_response(m, request_data) for m in models]
        resps = await asyncio.gather(*tasks)
        prompt = f"Pick winning response for: {request_data.messages[-1].content}\n\n" + "\n\n".join(f"C{i+1}: {r}" for i, r in enumerate(resps))
        winner = await self._orchestrator._get_full_response(self._settings.model_opus or self._settings.model, request_data.model_copy(update={
            "messages": [Message(role="user", content=prompt)]
        }))
        message_id = f"msg_vote_{uuid.uuid4().hex[:12]}"
        sse = SSEBuilder(message_id, request_data.model)
        yield sse.message_start(); yield sse.start_text_block(); yield sse.emit_text_delta(winner); yield sse.stop_text_block(); yield sse.message_delta("end_turn", 1); yield sse.message_stop()

    async def _run_auto_repair_routing(self, request_data: MessagesRequest, target: str) -> AsyncIterator[str]:
        candidates = self._model_router.resolve_candidates(target)
        async def _repair():
            async for chunk in self._stream_with_failover(request_data, candidates):
                if "error" in chunk and "tool_use" in chunk:
                    repair_req = request_data.model_copy(update={"messages": request_data.messages + [Message(role="user", content="Prior tool call failed. Fix it.")]})
                    async for rc in self._stream_with_failover(repair_req, candidates): yield rc
                    return
                yield chunk
        return _repair()

    async def _compress_context(self, request_data: MessagesRequest) -> MessagesRequest:
        if len(request_data.messages) <= 5: return request_data
        summary = await self._orchestrator._get_full_response(self._settings.model_haiku or self._settings.model, MessagesRequest(
            model=self._settings.model_haiku or self._settings.model,
            messages=[Message(role="user", content=f"Summarize history:\n{request_data.messages[:-3]}")], stream=False
        ))
        return request_data.model_copy(update={"messages": [Message(role="user", content=f"Summary: {summary}")] + request_data.messages[-3:]})

    def count_tokens(self, request_data: TokenCountRequest) -> TokenCountResponse:
        """Count tokens for a request after applying configured model routing."""
        request_id = f"req_{uuid.uuid4().hex[:12]}"
        with logger.contextualize(request_id=request_id):
            try:
                _require_non_empty_messages(request_data.messages)
                candidates = self._model_router.resolve_candidates(request_data.model)
                if not candidates:
                    raise InvalidRequestError(f"Model {request_data.model} not found")
                resolved = candidates[0]
                routed_request = request_data.model_copy(update={"model": resolved.provider_model}, deep=True)
                tokens = self._token_counter(routed_request.messages, routed_request.system, routed_request.tools)
                return TokenCountResponse(input_tokens=tokens)
            except (ProviderError, HTTPException):
                raise
            except Exception as e:
                _log_unexpected_service_exception(self._settings, e, context="COUNT_TOKENS_ERROR", request_id=request_id)
                detail = get_user_facing_error_message(e)
                if self._settings.log_api_error_tracebacks:
                    detail = str(e)
                raise HTTPException(status_code=500, detail=detail) from e
