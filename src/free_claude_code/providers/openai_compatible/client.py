"""Generic OpenAI-compatible provider implementation."""

from free_claude_code.config.provider_catalog import OPENAI_COMPATIBLE_DEFAULT_BASE
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.providers.admission import ProviderAdmissionController
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.transports.openai_chat import (
    OpenAIChatRequestPolicy,
    OpenAIChatTransport,
    build_openai_chat_request_body,
)

_REQUEST_POLICY = OpenAIChatRequestPolicy(
    provider_name="OPENAI_COMPATIBLE",
    include_extra_body=True,
    max_tokens_field="max_completion_tokens",
    strip_message_names=True,
    normalize_n_to_one=True,
)


class OpenAICompatibleProvider(OpenAIChatTransport):
    """Generic OpenAI-compatible provider using configurable base_url."""

    def __init__(
        self,
        config: ProviderConfig,
        admission: ProviderAdmissionController | None = None,
    ):
        super().__init__(
            config,
            admission=admission,
            provider_name="OPENAI_COMPATIBLE",
            base_url=config.base_url or OPENAI_COMPATIBLE_DEFAULT_BASE,
            api_key=config.api_key or "dummy",
        )

    def _build_request_body(
        self, request: MessagesRequest, thinking_enabled: bool | None = None
    ) -> dict:
        return build_openai_chat_request_body(
            request,
            thinking_enabled=self._is_thinking_enabled(request, thinking_enabled),
            policy=_REQUEST_POLICY,
        )
