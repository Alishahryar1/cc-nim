"""TokenRouter provider implementation (OpenAI-compatible chat completions)."""

from free_claude_code.config.provider_catalog import TOKENROUTER_DEFAULT_BASE
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.providers.admission import ProviderAdmissionController
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.transports.openai_chat import (
    OpenAIChatRequestPolicy,
    OpenAIChatTransport,
    build_openai_chat_request_body,
)

_REQUEST_POLICY = OpenAIChatRequestPolicy(
    provider_name="TOKENROUTER",
    include_extra_body=True,
    max_tokens_field="max_completion_tokens",
    strip_message_names=True,
    normalize_n_to_one=True,
)


class TokenRouterProvider(OpenAIChatTransport):
    """TokenRouter API using ``https://api.tokenrouter.com/v1/chat/completions``."""

    def __init__(
        self,
        config: ProviderConfig,
        admission: ProviderAdmissionController | None = None,
    ):
        super().__init__(
            config,
            admission=admission,
            provider_name="TOKENROUTER",
            base_url=config.base_url or TOKENROUTER_DEFAULT_BASE,
            api_key=config.api_key,
        )

    def _build_request_body(
        self, request: MessagesRequest, thinking_enabled: bool | None = None
    ) -> dict:
        return build_openai_chat_request_body(
            request,
            thinking_enabled=self._is_thinking_enabled(request, thinking_enabled),
            policy=_REQUEST_POLICY,
        )
