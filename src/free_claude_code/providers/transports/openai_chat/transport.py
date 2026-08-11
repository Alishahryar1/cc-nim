"""OpenAI-compatible chat transport base."""

import json
from abc import abstractmethod
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx
from loguru import logger
from openai import AsyncOpenAI

from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.anthropic.streaming import AnthropicStreamLedger
from free_claude_code.core.reasoning import DEFAULT_REASONING_POLICY, ReasoningPolicy
from free_claude_code.providers.admission import ProviderAdmissionController
from free_claude_code.providers.base import BaseProvider, ProviderConfig
from free_claude_code.providers.error_mapping import (
    extract_provider_error_detail,
    map_error,
    user_visible_message_for_mapped_provider_error,
)
from free_claude_code.providers.model_listing import (
    ProviderModelInfo,
    extract_openai_model_ids,
    extract_openai_model_infos,
)
from free_claude_code.providers.rate_limit import GlobalRateLimiter

from .stream import OpenAIChatStreamAdapter


def _clean_dict_thinking_fields(d: dict[str, Any]) -> bool:
    """Recursively strip fields related to thinking budget, reasoning effort, etc."""
    changed = False

    # Clean reasoning_effort
    if "reasoning_effort" in d:
        d.pop("reasoning_effort", None)
        changed = True

    # Clean reasoning_budget
    if "reasoning_budget" in d:
        d.pop("reasoning_budget", None)
        changed = True

    # Clean thinking_config
    if "thinking_config" in d:
        d.pop("thinking_config", None)
        changed = True

    # Clean include_thoughts if it exists
    if "include_thoughts" in d:
        d.pop("include_thoughts", None)
        changed = True

    # Recurse into nested dictionaries
    for key in list(d.keys()):
        val = d[key]
        if isinstance(val, dict):
            if _clean_dict_thinking_fields(val):
                changed = True
            # Clean up empty dictionaries
            if not val:
                d.pop(key, None)
                changed = True

    return changed


def _clone_body_without_thinking_budget(body: dict[str, Any]) -> dict[str, Any] | None:
    """Clone a request body and strip fields related to thinking budget/reasoning config."""
    from copy import deepcopy

    cloned_body = deepcopy(body)

    changed = _clean_dict_thinking_fields(cloned_body)

    # Clean reasoning_content in assistant messages
    messages = cloned_body.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict) and "reasoning_content" in message:
                message.pop("reasoning_content", None)
                changed = True

    return cloned_body if changed else None


class OpenAIChatTransport(BaseProvider):
    """Base for OpenAI-compatible ``/chat/completions`` adapters."""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        provider_name: str,
        base_url: str,
        api_key: str,
        admission: ProviderAdmissionController | None = None,
    ):
        super().__init__(config, admission=admission)
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

    def preflight_stream(
        self,
        request: MessagesRequest,
        *,
        reasoning: Any = None,
    ) -> None:
        """Validate request conversion before streaming."""
        pass

    async def list_model_infos(self) -> frozenset[ProviderModelInfo]:
        """Fetch model metadata from the OpenAI-compatible models endpoint."""
        try:
            response = await self._client.models.list()
            return extract_openai_model_infos(
                response, provider_name=self._provider_name
            )
        except Exception:
            return frozenset()

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
        self, request: MessagesRequest, thinking_enabled: bool | None = None
    ) -> dict:
        """Build request body. Must be implemented by subclasses."""

    def _handle_extra_reasoning(
        self, delta: Any, ledger: AnthropicStreamLedger, *, thinking_enabled: bool
    ) -> Iterator[str]:
        """Hook for provider-specific reasoning."""
        return iter(())

    def _get_retry_request_body(self, error: Exception, body: dict) -> dict | None:
        """Return a modified request body for one retry, or None."""
        import openai

        status_code = getattr(error, "status_code", None)
        if not isinstance(error, openai.BadRequestError) and status_code != 400:
            return None

        error_text = str(error)
        error_body = getattr(error, "body", None)
        if error_body is not None:
            error_text = f"{error_text} {json.dumps(error_body, default=str)}"
        error_text = error_text.lower()

        # Catch thinking-budget/reasoning related errors
        if (
            "thinking budget" in error_text
            or "thinking_budget" in error_text
            or "reasoning_budget" in error_text
            or "reasoning_effort" in error_text
            or "reasoning effort" in error_text
            or "thinking_config" in error_text
            or "thinking config" in error_text
        ):
            retry_body = _clone_body_without_thinking_budget(body)
            if retry_body is not None:
                logger.warning(
                    "{}_STREAM: retrying without thinking budget after HTTP 400 error",
                    self._provider_name,
                )
                return retry_body

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

    def _openai_error_message(self, error: Exception, request_id: str | None) -> str:
        mapped_error = map_error(error, rate_limiter=self._global_rate_limiter)
        return user_visible_message_for_mapped_provider_error(
            mapped_error,
            provider_name=self._provider_name,
            read_timeout_s=self._config.http_read_timeout,
            detail=extract_provider_error_detail(error),
            request_id=request_id,
        )

    async def stream_response(
        self,
        request: MessagesRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
        thinking_enabled: bool | None = None,
    ) -> AsyncIterator[str]:
        """Stream response in Anthropic SSE format."""
        adapter = OpenAIChatStreamAdapter(
            self,
            request=request,
            input_tokens=input_tokens,
            request_id=request_id,
            thinking_enabled=thinking_enabled,
        )
        async for event in adapter.run():
            yield event
