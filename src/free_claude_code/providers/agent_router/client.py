"""AgentRouter provider implementation (native Anthropic-compatible Messages)."""

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.provider_catalog import AGENTROUTER_DEFAULT_BASE
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.providers.admission import ProviderAdmissionController
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.model_listing import model_infos_from_ids
from free_claude_code.providers.transports.anthropic_messages import (
    AnthropicMessagesTransport,
    NativeMessagesRequestPolicy,
    build_native_messages_request_body,
)

_ANTHROPIC_VERSION = "2023-06-01"
_REQUEST_POLICY = NativeMessagesRequestPolicy(
    provider_name="AGENTROUTER",
)

DEFAULT_AGENTROUTER_MODELS: tuple[str, ...] = (
    "claude-fable-5",
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-6",
    "glm-5.2",
    "gpt-5.5",
    "gpt-5.6-sol",
    "kimi-k3",
)

_CLAUDE_CODE_CLIENT_HEADERS: dict[str, str] = {
    "Accept": "application/json",
    "User-Agent": "claude-cli/2.1.221 (external, sdk-cli)",
    "X-Stainless-Arch": "x64",
    "X-Stainless-Lang": "js",
    "X-Stainless-OS": "Linux",
    "X-Stainless-Package-Version": "0.94.0",
    "X-Stainless-Retry-Count": "0",
    "X-Stainless-Runtime": "node",
    "X-Stainless-Runtime-Version": "v26.3.0",
    "X-Stainless-Timeout": "600",
    "anthropic-beta": (
        "claude-code-20250219,interleaved-thinking-2025-05-14,"
        "thinking-token-count-2026-05-13,context-management-2025-06-27,"
        "prompt-caching-scope-2026-01-05,mid-conversation-system-2026-04-07,"
        "effort-2025-11-24,structured-outputs-2025-12-15"
    ),
    "anthropic-dangerous-direct-browser-access": "true",
    "x-app": "cli",
}

_CLAUDE_CODE_REQUEST_HEADERS = frozenset(
    {
        "accept",
        "user-agent",
        "x-claude-code-session-id",
        "x-stainless-arch",
        "x-stainless-lang",
        "x-stainless-os",
        "x-stainless-package-version",
        "x-stainless-retry-count",
        "x-stainless-runtime",
        "x-stainless-runtime-version",
        "x-stainless-timeout",
        "anthropic-beta",
        "anthropic-dangerous-direct-browser-access",
        "anthropic-version",
        "x-app",
    }
)


class AgentRouterProvider(AnthropicMessagesTransport):
    """AgentRouter provider using Anthropic-compatible Messages at https://ps.air-outer.com."""

    def __init__(
        self,
        config: ProviderConfig,
        admission: ProviderAdmissionController | None = None,
    ):
        super().__init__(
            config,
            admission=admission,
            provider_name="AGENTROUTER",
            default_base_url=AGENTROUTER_DEFAULT_BASE,
        )

    async def list_model_infos(self) -> frozenset[ProviderModelInfo]:
        """Return model metadata, falling back to static model list if endpoint fails."""
        try:
            return await super().list_model_infos()
        except Exception:
            return model_infos_from_ids(DEFAULT_AGENTROUTER_MODELS)

    def _build_request_body(
        self, request: MessagesRequest, thinking_enabled: bool | None = None
    ) -> dict:
        return build_native_messages_request_body(
            request,
            thinking_enabled=self._is_thinking_enabled(request, thinking_enabled),
            policy=_REQUEST_POLICY,
        )

    def _messages_path(self, request: MessagesRequest | None = None) -> str:
        return "/messages?beta=true"

    def _request_headers(self) -> dict[str, str]:
        return self._request_headers_for_request(None)

    def _request_headers_for_request(
        self, request: MessagesRequest | None
    ) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "anthropic-version": _ANTHROPIC_VERSION,
            **_CLAUDE_CODE_CLIENT_HEADERS,
        }
        client_headers = getattr(request, "client_headers", None)
        if isinstance(client_headers, dict):
            for key, value in client_headers.items():
                normalized_key = key.lower()
                if normalized_key not in _CLAUDE_CODE_REQUEST_HEADERS:
                    continue
                if not isinstance(value, str) or not value:
                    continue
                existing_key = next(
                    (
                        header_key
                        for header_key in headers
                        if header_key.lower() == normalized_key
                    ),
                    key,
                )
                headers[existing_key] = value
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
            headers["x-api-key"] = self._api_key
        return headers

    def _model_list_headers(self) -> dict[str, str]:
        headers = dict(_CLAUDE_CODE_CLIENT_HEADERS)
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
            headers["x-api-key"] = self._api_key
        return headers
