"""ChatGPT Codex backend provider using OpenAI Responses."""

import asyncio
import json
import sys
import uuid
from collections.abc import AsyncIterator
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import httpx

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.core.diagnostics import (
    ERROR_DETAIL_DISPLAY_CAP_BYTES,
    attach_upstream_error_body,
    extract_upstream_error_detail,
)
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.inference import (
    InferenceEvent,
    InferenceRequest,
    ReplayCompatibilityScope,
)
from free_claude_code.core.openai_responses import (
    ResponsesConversionError,
)
from free_claude_code.core.reasoning import (
    DEFAULT_REASONING_POLICY,
    ReasoningPolicy,
)
from free_claude_code.core.trace import trace_event
from free_claude_code.providers.admission import (
    ProviderAdmissionController,
    ProviderCorrectionAction,
    ProviderExecution,
    ProviderOperationKind,
)
from free_claude_code.providers.base import BaseProvider, ProviderConfig
from free_claude_code.providers.failure_policy import (
    RetryableProviderProtocolError,
    classify_provider_failure,
    is_retryable_stream_error,
)
from free_claude_code.providers.http import ProviderAttemptScope, maybe_await_aclose
from free_claude_code.providers.openai_compat import (
    OpenAIToolNameCodec,
    openai_replay_scope,
)
from free_claude_code.providers.openai_responses import (
    ResponsesEventDecoder,
    ResponsesRequestEncodingError,
    ResponsesStreamFailure,
    build_responses_request_body,
)
from free_claude_code.providers.stream_recovery import (
    RecoveryController,
    RecoveryFailureAction,
)

from .auth import OpenAIAccess, OpenAIAuthManager, OpenAIReconnectRequired
from .login import OPENAI_CODEX_ORIGINATOR

try:
    FCC_VERSION = version("free-claude-code")
except PackageNotFoundError:
    FCC_VERSION = "dev"


class _TruncatedResponsesStream(RetryableProviderProtocolError):
    """A Responses stream ended without a terminal lifecycle event."""


class OpenAICodexProvider(BaseProvider):
    """Use a ChatGPT subscription through OpenAI's Codex backend."""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        auth: OpenAIAuthManager,
        admission: ProviderAdmissionController,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(config)
        self._auth = auth
        self._admission = admission
        self._client_headers = {
            "User-Agent": f"{OPENAI_CODEX_ORIGINATOR}/{FCC_VERSION}",
            "originator": OPENAI_CODEX_ORIGINATOR,
            "version": FCC_VERSION,
        }
        self._client = client or httpx.AsyncClient(
            base_url=f"{config.base_url.rstrip('/')}/",
            proxy=config.proxy,
            timeout=httpx.Timeout(
                config.http_read_timeout,
                connect=config.http_connect_timeout,
                write=config.http_write_timeout,
            ),
            headers=self._client_headers,
        )
        self._owns_client = client is None

    def preflight_stream(
        self,
        request: InferenceRequest,
        *,
        provider_model: str,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
    ) -> None:
        """Validate and adapt the private Codex request before upstream I/O."""

        self._build_body(
            request,
            provider_model=provider_model,
            reasoning=reasoning,
        )

    async def cleanup(self) -> None:
        """Close only provider-owned transport resources."""

        if self._owns_client:
            await self._client.aclose()

    async def list_model_infos(self) -> frozenset[ProviderModelInfo]:
        """Discover models visible to the currently connected ChatGPT account."""
        return _model_infos(await self._list_models_payload())

    async def _list_models_payload(self) -> Any:
        """Fetch one Codex model catalog with each provider GET admitted once."""
        execution = self._admission.start_execution()
        authentication_recovered = False
        while execution.can_attempt:
            scope: ProviderAttemptScope | None = None
            try:
                access = await self._auth.access()
                attempt = await execution.open_attempt(
                    ProviderOperationKind.MODEL_DISCOVERY
                )
                scope = ProviderAttemptScope(
                    attempt,
                    provider_name="OpenAI",
                    request_id=execution.request_id,
                )
                response = scope.retain(
                    await self._client.get(
                        "models",
                        params={"client_version": FCC_VERSION},
                        headers={**self._client_headers, **_auth_headers(access)},
                    )
                )
                if response.status_code == 401 and not authentication_recovered:
                    error = await _response_status_error(response)
                    correction = await scope.attempt.correct(error)
                    closing_scope = scope
                    scope = None
                    await closing_scope.aclose(active_error=error)
                    if correction is ProviderCorrectionAction.FINAL:
                        raise error
                    await self._auth.recover_unauthorized(access.access_token)
                    authentication_recovered = True
                    continue
                if not response.is_success:
                    raise await _response_status_error(response)
                payload = response.json()
                await scope.attempt.accept()
                execution.succeed()
                return payload
            except asyncio.CancelledError:
                execution.abandon()
                raise
            except Exception as error:
                if scope is not None and not scope.attempt.accepted:
                    decision = await scope.attempt.fail(error)
                    if decision.retry_allowed:
                        continue
                execution.fail(error)
                raise
            finally:
                if scope is not None:
                    await scope.aclose(active_error=sys.exception())

        if execution.last_failure is not None:
            raise execution.last_failure
        execution.abandon()
        raise RuntimeError("OpenAI model discovery ended without an attempt outcome")

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
        """Stream private Responses output as canonical inference events."""

        tool_names = OpenAIToolNameCodec.from_request(request)
        replay_scope = self._replay_scope(provider_model)
        body = self._build_body(
            request,
            provider_model=provider_model,
            reasoning=reasoning,
        )
        return self._run_stream(
            body,
            input_tokens=input_tokens,
            request_id=request_id,
            response_model=response_model or request.model,
            tool_names=tool_names,
            replay_scope=replay_scope,
        )

    def _build_body(
        self,
        request: InferenceRequest,
        *,
        provider_model: str,
        reasoning: ReasoningPolicy,
    ) -> dict[str, Any]:
        try:
            body = build_responses_request_body(
                request,
                provider_model=provider_model,
                reasoning=reasoning,
                tool_names=OpenAIToolNameCodec.from_request(request),
                replay_scope=self._replay_scope(provider_model),
            )
        except (ResponsesConversionError, ResponsesRequestEncodingError) as exc:
            raise InvalidRequestError(str(exc)) from exc
        # The private Codex backend rejects these public Responses fields.
        # Codex itself omits the output cap and uses separate internal metadata.
        body.pop("max_output_tokens", None)
        body.pop("metadata", None)
        return body

    @staticmethod
    def _replay_scope(provider_model: str) -> ReplayCompatibilityScope:
        return openai_replay_scope(
            "openai",
            provider_model,
            replay_format="codex-responses",
        )

    async def _run_stream(
        self,
        body: dict[str, Any],
        *,
        input_tokens: int,
        request_id: str | None,
        response_model: str,
        tool_names: OpenAIToolNameCodec,
        replay_scope: ReplayCompatibilityScope,
    ) -> AsyncIterator[InferenceEvent]:
        execution = self._admission.start_execution(request_id=request_id)
        provider_stream = self._run_stream_execution(
            body,
            input_tokens=input_tokens,
            request_id=request_id,
            response_model=response_model,
            tool_names=tool_names,
            replay_scope=replay_scope,
            execution=execution,
        )
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

    async def _run_stream_execution(
        self,
        body: dict[str, Any],
        *,
        input_tokens: int,
        request_id: str | None,
        response_model: str,
        tool_names: OpenAIToolNameCodec,
        replay_scope: ReplayCompatibilityScope,
        execution: ProviderExecution,
    ) -> AsyncIterator[InferenceEvent]:
        """Run one Codex execution while retaining Responses transport ownership."""
        recovery = RecoveryController()
        response_id = f"response_{uuid.uuid4().hex}"
        session_id = str(uuid.uuid4())
        authentication_recovered = False
        trace_event(
            stage="provider",
            event="provider.request.sent",
            source="provider",
            provider="openai",
            request_id=request_id,
            execution_id=execution.execution_id,
            gateway_model=response_model,
            downstream_model=body.get("model"),
            item_count=len(body.get("input", [])),
            tool_count=len(body.get("tools", [])),
        )

        while execution.can_attempt:
            stream = ResponsesEventDecoder(
                response_id=response_id,
                model=response_model,
                input_tokens=input_tokens,
                tool_names=tool_names,
                replay_scope=replay_scope,
            )
            for event in stream.start():
                for held in recovery.push(event):
                    yield held

            scope: ProviderAttemptScope | None = None
            stream_opened = False
            try:
                access = await self._auth.access()
                attempt = await execution.open_attempt(ProviderOperationKind.GENERATION)
                scope = ProviderAttemptScope(
                    attempt,
                    provider_name="OpenAI",
                    request_id=request_id,
                )
                response = scope.retain(
                    await self._client.send(
                        self._client.build_request(
                            "POST",
                            "responses",
                            json=body,
                            headers={
                                **self._client_headers,
                                **_auth_headers(access),
                                "Accept": "text/event-stream",
                                "session_id": session_id,
                            },
                        ),
                        stream=True,
                    )
                )
                if response.status_code == 401 and not authentication_recovered:
                    error = await _response_status_error(response)
                    correction = await scope.attempt.correct(error)
                    closing_scope = scope
                    scope = None
                    await closing_scope.aclose(active_error=error)
                    if correction is ProviderCorrectionAction.FINAL:
                        raise error
                    recovery.discard()
                    await self._auth.recover_unauthorized(access.access_token)
                    authentication_recovered = True
                    continue
                if not response.is_success:
                    raise await _response_status_error(response)
                content_type = response.headers.get("content-type")
                if content_type and "text/event-stream" not in content_type.lower():
                    body_bytes, body_truncated = await _read_bounded_body(response)
                    error = _TruncatedResponsesStream(
                        "OpenAI returned a non-streaming Responses payload."
                    )
                    attach_upstream_error_body(
                        error,
                        body_bytes,
                        truncated=body_truncated,
                    )
                    raise error
                stream_opened = True

                async for event_type, payload in _iter_sse(response):
                    if not scope.attempt.accepted:
                        await scope.attempt.accept()
                    for event in stream.feed(event_type, payload):
                        for held in recovery.push(event):
                            yield held
                if not stream.completed:
                    raise _TruncatedResponsesStream(
                        "OpenAI Responses stream ended without a terminal event."
                    )
                for event in recovery.flush():
                    yield event
                trace_event(
                    stage="provider",
                    event="provider.response.completed",
                    source="provider",
                    provider="openai",
                    request_id=request_id,
                )
                return
            except asyncio.CancelledError, GeneratorExit:
                raise
            except Exception as raw_error:
                error = _effective_error(raw_error)
                attempt_failure = None
                if scope is not None and not scope.attempt.accepted:
                    attempt_failure = await scope.attempt.fail(error)
                if attempt_failure is not None and attempt_failure.retry_allowed:
                    recovery.discard()
                    trace_event(
                        stage="provider",
                        event="provider.recovery.early_retry",
                        source="provider",
                        provider="openai",
                        request_id=request_id,
                        attempts_started=execution.attempts_started,
                        max_attempts=execution.max_attempts,
                    )
                    continue
                retryable = (
                    attempt_failure.retryable
                    if attempt_failure is not None
                    else is_retryable_stream_error(error)
                )
                decision = recovery.advance_failure(
                    retryable=retryable,
                    stream_opened=stream_opened,
                    generated_output=recovery.committed,
                    complete_tool_salvageable=False,
                    attempts_remaining=execution.attempts_remaining,
                )
                if decision.action is RecoveryFailureAction.EARLY_RETRY:
                    recovery.discard()
                    trace_event(
                        stage="provider",
                        event="provider.recovery.early_retry",
                        source="provider",
                        provider="openai",
                        request_id=request_id,
                        attempts_started=execution.attempts_started,
                        max_attempts=execution.max_attempts,
                    )
                    continue

                failure = classify_provider_failure(
                    error,
                    provider_name="OpenAI",
                    read_timeout_s=self._config.http_read_timeout,
                    request_id=request_id,
                )
                self._log_stream_transport_error(
                    "OPENAI",
                    f" request_id={request_id}" if request_id else "",
                    error,
                    request_id=request_id,
                )
                if not decision.committed:
                    recovery.discard()
                    raise failure from raw_error
                for event in stream.ledger.close_unclosed_blocks():
                    yield event
                raise failure from raw_error
            finally:
                if scope is not None:
                    await scope.aclose(active_error=sys.exception())

        if execution.last_failure is not None:
            raise execution.last_failure
        raise RuntimeError("OpenAI execution ended without a terminal result.")


async def _iter_sse(
    response: httpx.Response,
) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    event_type = ""
    data_lines: list[str] = []
    async for line in response.aiter_lines():
        if not line:
            if not data_lines:
                event_type = ""
                continue
            raw_data = "\n".join(data_lines)
            data_lines = []
            if raw_data == "[DONE]":
                return
            try:
                payload = json.loads(raw_data)
            except json.JSONDecodeError as exc:
                raise _TruncatedResponsesStream(
                    "OpenAI returned malformed Responses SSE."
                ) from exc
            if not isinstance(payload, dict):
                raise _TruncatedResponsesStream(
                    "OpenAI returned a non-object Responses event."
                )
            resolved_type = event_type or payload.get("type")
            event_type = ""
            if isinstance(resolved_type, str) and resolved_type:
                yield resolved_type, payload
            continue
        if line.startswith("event:"):
            event_type = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    if data_lines:
        raise _TruncatedResponsesStream(
            "OpenAI Responses stream ended during an SSE event."
        )


async def _read_bounded_body(
    response: httpx.Response,
) -> tuple[bytes, bool]:
    limit = ERROR_DETAIL_DISPLAY_CAP_BYTES
    body = bytearray()
    async for chunk in response.aiter_bytes():
        remaining = limit + 1 - len(body)
        if remaining <= 0:
            break
        body.extend(chunk[:remaining])
        if len(body) > limit:
            break
    truncated = len(body) > limit
    return bytes(body[:limit]), truncated


async def _response_status_error(response: httpx.Response) -> httpx.HTTPStatusError:
    """Return one bounded, diagnostic-preserving HTTP status failure."""
    body, truncated = await _read_bounded_body(response)
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as error:
        attach_upstream_error_body(error, body, truncated=truncated)
        return error
    raise RuntimeError("response status is successful")


def _auth_headers(access: OpenAIAccess) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {access.access_token}",
        "ChatGPT-Account-ID": access.account_id,
    }
    if access.fedramp:
        headers["X-OpenAI-Fedramp"] = "true"
    return headers


def _model_infos(payload: Any) -> frozenset[ProviderModelInfo]:
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
        raise ValueError("OpenAI model-list response is missing the models array.")
    infos: set[ProviderModelInfo] = set()
    for model in payload["models"]:
        if not isinstance(model, dict):
            continue
        model_id = model.get("slug")
        visibility = model.get("visibility")
        if (
            not isinstance(model_id, str)
            or not model_id.strip()
            or visibility != "list"
        ):
            continue
        efforts = model.get(
            "supported_reasoning_levels",
            model.get("supported_reasoning_efforts"),
        )
        infos.add(
            ProviderModelInfo(
                model_id=model_id,
                supports_thinking=bool(efforts) if isinstance(efforts, list) else None,
            )
        )
    if not infos:
        raise ValueError("OpenAI did not advertise any visible models.")
    return frozenset(infos)


def _effective_error(error: Exception) -> Exception:
    if isinstance(error, OpenAIReconnectRequired):
        return ExecutionFailure(
            kind=FailureKind.AUTHENTICATION,
            status_code=401,
            message=str(error),
            retryable=False,
        )
    if isinstance(error, ResponsesStreamFailure):
        message = (
            extract_upstream_error_detail(error).exception_text
            or "OpenAI response failed."
        )
        code = (error.code or "").lower()
        if "rate" in code or "429" in code:
            return ExecutionFailure(FailureKind.RATE_LIMIT, 429, message, True)
        if any(marker in code for marker in ("overload", "capacity", "529")):
            return ExecutionFailure(FailureKind.OVERLOADED, 529, message, True)
        retryable = any(
            marker in code
            for marker in ("server", "internal", "unavailable", "timeout")
        )
        return ExecutionFailure(FailureKind.UPSTREAM, 502, message, retryable)
    return error
