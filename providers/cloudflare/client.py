"""Cloudflare AI REST provider using Anthropic-compatible Messages."""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

from providers.base import ProviderConfig
from providers.defaults import CLOUDFLARE_AI_REST_ROOT
from providers.exceptions import AuthenticationError
from providers.transports.anthropic_messages import (
    AnthropicMessagesTransport,
    NativeMessagesRequestPolicy,
    build_native_messages_request_body,
)

_ANTHROPIC_VERSION = "2023-06-01"
_REQUEST_POLICY = NativeMessagesRequestPolicy(
    provider_name="CLOUDFLARE",
    extra_body="reject",
    reject_extra_body_message=(
        "Cloudflare native Messages API does not support extra_body on requests."
    ),
)


def cloudflare_ai_base_url(api_root: str | None, account_id: str) -> str:
    """Return the account-scoped Cloudflare AI REST base URL."""

    stripped_account = account_id.strip()
    if not stripped_account:
        raise AuthenticationError(
            "CLOUDFLARE_ACCOUNT_ID is not set. Add it to your .env file."
        )
    root = (api_root or CLOUDFLARE_AI_REST_ROOT).rstrip("/")
    encoded_account = quote(stripped_account, safe="")
    return f"{root}/accounts/{encoded_account}/ai/v1"


class CloudflareProvider(AnthropicMessagesTransport):
    """Cloudflare account-scoped AI REST provider."""

    def __init__(self, config: ProviderConfig, *, account_id: str):
        base_url = cloudflare_ai_base_url(config.base_url, account_id)
        super().__init__(
            config.model_copy(update={"base_url": base_url}),
            provider_name="CLOUDFLARE",
            default_base_url=base_url,
        )

    def _build_request_body(
        self, request: Any, thinking_enabled: bool | None = None
    ) -> dict:
        return build_native_messages_request_body(
            request,
            thinking_enabled=False,
            policy=_REQUEST_POLICY,
        )

    def _request_headers(self) -> dict[str, str]:
        return {
            "Accept": "text/event-stream",
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "anthropic-version": _ANTHROPIC_VERSION,
        }

    def _model_list_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}
