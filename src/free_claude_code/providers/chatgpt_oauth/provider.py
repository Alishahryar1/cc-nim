"""Direct ChatGPT/Codex OAuth provider using the Responses API."""

from __future__ import annotations

import platform
import re
import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx
from loguru import logger

from free_claude_code.application.errors import ApplicationUnavailableError
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.constants import HTTP_CONNECT_TIMEOUT_DEFAULT
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.anthropic.streaming import AnthropicStreamLedger
from free_claude_code.core.diagnostics import (
    exception_cause_types,
    redacted_exception_traceback,
)
from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.core.reasoning import DEFAULT_REASONING_POLICY, ReasoningPolicy
from free_claude_code.core.trace import trace_event
from free_claude_code.core.version import package_version
from free_claude_code.providers.base import BaseProvider, ProviderConfig
from free_claude_code.providers.failure_policy import classify_provider_failure
from free_claude_code.providers.model_listing import model_infos_from_ids
from free_claude_code.providers.rate_limit import ProviderRateLimiter

from .conversion import build_chatgpt_oauth_request_body
from .credentials import ChatGPTOAuthError, load_chatgpt_oauth_credentials
from .streaming import ChatGPTOAuthStreamConverter, iter_chatgpt_oauth_sse_events

CHATGPT_OAUTH_DEFAULT_BASE = "https://chatgpt.com/backend-api"

# Model allowlist aligned with OpenCode's ChatGPT/Codex OAuth filter.
# https://github.com/anomalyco/opencode/blob/main/packages/opencode/src/plugin/openai/codex.ts
_CHATGPT_OAUTH_ALLOWED_MODELS = frozenset({
    "gpt-5.5",
    "gpt-5.3-codex-spark",
    "gpt-5.4",
    "gpt-5.4-mini",
})
_CHATGPT_OAUTH_DISALLOWED_MODELS = frozenset({"gpt-5.5-pro"})
_CHATGPT_OAUTH_GPT_VERSION_RE = re.compile(r"^gpt-(\d+\.\d+)")


def _user_agent() -> str:
    """Return a User-Agent string matching the shape used by OpenCode."""
    version = package_version()
    return (
        f"free-claude-code/{version} "
        f"({platform.system()} {platform.release()}; {platform.machine()})"
    )


def _is_chatgpt_oauth_model(model_id: str) -> bool:
    """Return True when ``model_id`` is exposed by the ChatGPT/Codex backend.

    Mirrors OpenCode's model filter: a small allowlist, an explicit blocklist,
    and a version heuristic for future GPT-5.x models.
    """
    if model_id in _CHATGPT_OAUTH_DISALLOWED_MODELS:
        return False
    if model_id == "gpt-5.6":
        return False
    if model_id in _CHATGPT_OAUTH_ALLOWED_MODELS:
        return True
    match = _CHATGPT_OAUTH_GPT_VERSION_RE.match(model_id)
    if match:
        return float(match.group(1)) > 5.4
    return False


def _build_headers(credentials: Any, session_id: str) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {credentials.access_token}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "OpenAI-Beta": "responses=experimental",
        "originator": "free-claude-code",
        "User-Agent": _user_agent(),
        "session-id": session_id,
    }
    if credentials.account_id:
        headers["ChatGPT-Account-ID"] = credentials.account_id
    return headers


class ChatGPTOAuthProvider(BaseProvider):
    """ChatGPT/Codex OAuth provider using the Responses API."""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        rate_limiter: ProviderRateLimiter,
        account_id: str = "",
    ):
        super().__init__(config)
        self._rate_limiter = rate_limiter
        self._base_url = (config.base_url or CHATGPT_OAUTH_DEFAULT_BASE).rstrip("/")
        self._account_id = account_id
        self._api_key = config.api_key
        self._proxy = config.proxy
        self._session_id = str(uuid.uuid4())
        self._client = httpx.AsyncClient(
            proxy=config.proxy if config.proxy else None,
            timeout=httpx.Timeout(
                config.http_read_timeout,
                connect=config.http_connect_timeout or HTTP_CONNECT_TIMEOUT_DEFAULT,
                read=config.http_read_timeout,
                write=config.http_write_timeout,
            ),
        )

    async def cleanup(self) -> None:
        await self._client.aclose()

    async def list_model_ids(self) -> frozenset[str]:
        """Return a static set of known ChatGPT/Codex OAuth model ids.

        The Responses API models endpoint requires a valid session and is not
        reliably probe-able during discovery, so we expose the ids that the
        upstream documentation and community implementations reference.
        The list is filtered to match OpenCode's ChatGPT/Codex OAuth allowlist.
        """
        candidates = {
            "gpt-5",
            "gpt-5.2",
            "gpt-5.4",
            "gpt-5.5",
            "gpt-5.6",
            "gpt-5-codex",
            "gpt-5.1-codex",
            "gpt-5.2-codex",
            "gpt-5.3-codex",
            "gpt-5.3-codex-spark",
            "gpt-5.4-mini",
            "gpt-5.5-pro",
            "codex-mini-latest",
        }
        return frozenset(m for m in candidates if _is_chatgpt_oauth_model(m))

    async def list_model_infos(self) -> frozenset[ProviderModelInfo]:
        return model_infos_from_ids(await self.list_model_ids())

    def preflight_stream(
        self,
        request: MessagesRequest,
        *,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
    ) -> None:
        """Validate the upstream request before streaming."""
        build_chatgpt_oauth_request_body(request)

    async def _send_stream_request(
        self,
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
    ) -> httpx.Response:
        """Build and send a streaming POST, raising on HTTP errors.

        ``httpx.AsyncClient.stream`` returns an async context manager, which is
        not awaitable and therefore cannot be passed directly to the retry
        helper. We instead build the request and call ``send(..., stream=True)``,
        which returns an awaitable ``Response`` while still keeping the body
        stream open until we explicitly close it.
        """
        request = self._client.build_request("POST", url, headers=headers, json=body)
        response = await self._client.send(request, stream=True)
        if response.status_code >= 400:
            error_body = await response.aread()
            await response.aclose()
            error_text = error_body.decode("utf-8", errors="replace")
            raise httpx.HTTPStatusError(
                f"ChatGPT OAuth API error {response.status_code}: {error_text[:1000]}",
                request=request,
                response=response,
            )
        return response

    def stream_response(
        self,
        request: MessagesRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
    ) -> AsyncIterator[str]:
        tag = "CHATGPT_OAUTH"
        req_tag = f" request_id={request_id}" if request_id else ""
        logger.debug("{}_STREAM: starting{}", tag, req_tag)

        try:
            credentials = load_chatgpt_oauth_credentials(
                access_token=self._api_key or None,
                account_id=self._account_id or None,
            )
        except ChatGPTOAuthError as exc:
            logger.error("{}_ERROR:{} {}", tag, req_tag, exc)
            raise ApplicationUnavailableError(str(exc)) from exc

        body = build_chatgpt_oauth_request_body(request)
        url = f"{self._base_url}/codex/responses"
        headers = _build_headers(credentials, self._session_id)

        trace_event(
            stage="provider",
            event="provider.request.sent",
            source="provider",
            provider=tag,
            request_id=request_id,
            gateway_model=request.model,
            downstream_model=body.get("model"),
            message_count=len(body.get("input", [])),
            tool_count=len(body.get("tools", [])),
            body={
                "model": body.get("model"),
                "input_count": len(body.get("input", [])),
                "tool_count": len(body.get("tools", [])),
            },
        )

        async def _stream() -> AsyncIterator[str]:
            message_id = f"msg_{uuid.uuid4()}"
            ledger = AnthropicStreamLedger(
                message_id,
                request.model,
                input_tokens,
                log_raw_events=self._config.log_raw_sse_events,
            )
            converter = ChatGPTOAuthStreamConverter(
                ledger,
                log_raw_events=self._config.log_raw_sse_events,
            )

            async with self._rate_limiter.concurrency_slot():
                try:
                    response = await self._rate_limiter.execute_with_retry(
                        self._send_stream_request,
                        provider_failure_override=self._provider_failure_override,
                        url=url,
                        headers=headers,
                        body=body,
                    )
                    try:
                        if response.status_code >= 400:
                            self._log_error(tag, req_tag, None, request_id)
                            raise ApplicationUnavailableError(
                                f"ChatGPT OAuth API error {response.status_code}"
                            )

                        yield ledger.message_start()
                        async for event in iter_chatgpt_oauth_sse_events(response.aiter_raw()):
                            for sse_event in converter.feed(event):
                                yield sse_event

                        for sse_event in converter.finish():
                            yield sse_event
                    finally:
                        await response.aclose()

                except ApplicationUnavailableError:
                    raise
                except Exception as error:
                    self._log_error(tag, req_tag, error, request_id)
                    failure = classify_provider_failure(
                        error,
                        provider_name=tag,
                        read_timeout_s=self._config.http_read_timeout,
                        request_id=request_id,
                        mark_rate_limited=self._rate_limiter.extend_reactive_block,
                        provider_failure_override=self._provider_failure_override,
                    )
                    trace_event(
                        stage="provider",
                        event="provider.response.error",
                        source="provider",
                        provider=tag,
                        request_id=request_id,
                        exc_type=type(error).__name__,
                        failure_kind=failure.kind.value,
                        status_code=failure.status_code,
                        provider_retryable=failure.retryable,
                    )
                    raise failure from error

        return _stream()

    def _provider_failure_override(self, error: Exception) -> ExecutionFailure | None:
        return None

    def _log_error(
        self,
        tag: str,
        req_tag: str,
        error: Exception | None,
        request_id: str | None,
    ) -> None:
        if error is None:
            logger.error("{}_ERROR:{} transport error", tag, req_tag)
            return
        if self._config.log_api_error_tracebacks:
            logger.error(
                "{}_ERROR:{} exc_type={}\n{}",
                tag,
                req_tag,
                type(error).__name__,
                redacted_exception_traceback(error),
            )
        else:
            logger.error(
                "{}_ERROR:{} exc_type={} cause_types={}",
                tag,
                req_tag,
                type(error).__name__,
                ",".join(exception_cause_types(error)),
            )
