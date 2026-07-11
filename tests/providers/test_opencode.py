"""Tests for the OpenCode OpenAI-compatible provider."""

from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.opencode import OpenCodeProvider
from tests.providers.support import passthrough_rate_limiter


def test_build_request_body_preserves_empty_reasoning_content() -> None:
    provider = OpenCodeProvider(
        ProviderConfig(
            api_key="test_opencode_key",
            base_url="https://example.invalid/v1",
            rate_limit=1,
            rate_window=1,
            enable_thinking=True,
        ),
        rate_limiter=passthrough_rate_limiter(),
    )
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": "visible",
                    "reasoning_content": "",
                }
            ],
            "thinking": {"type": "enabled"},
        }
    )

    body = provider._build_request_body(request)

    assert body["messages"][0] == {
        "role": "assistant",
        "content": "visible",
        "reasoning_content": "",
    }
