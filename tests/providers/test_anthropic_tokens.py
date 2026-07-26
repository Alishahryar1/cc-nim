from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import anthropic
import httpx
import pytest

from free_claude_code.core.anthropic.models import Message, SystemContent, Tool
from free_claude_code.providers.anthropic_tokens import (
    AnthropicTokenCountUnavailable,
    count_tokens_via_anthropic_api,
)


def _mock_client(input_tokens: int = 42) -> MagicMock:
    client = MagicMock()
    client.messages.count_tokens.return_value = SimpleNamespace(
        input_tokens=input_tokens
    )
    return client


def test_count_tokens_via_anthropic_api_returns_exact_count():
    with patch(
        "free_claude_code.providers.anthropic_tokens.anthropic.Anthropic"
    ) as mock_anthropic:
        mock_anthropic.return_value = _mock_client(input_tokens=17)
        tokens = count_tokens_via_anthropic_api(
            api_key="sk-test",
            model="claude-sonnet-4-5-20250929",
            messages=[Message(role="user", content="hi")],
            system=None,
            tools=None,
            timeout=30.0,
        )

    assert tokens == 17
    mock_anthropic.assert_called_once_with(api_key="sk-test", timeout=30.0)


def test_count_tokens_via_anthropic_api_omits_absent_system_and_tools():
    client = _mock_client()
    with patch(
        "free_claude_code.providers.anthropic_tokens.anthropic.Anthropic",
        return_value=client,
    ):
        count_tokens_via_anthropic_api(
            api_key="sk-test",
            model="claude-sonnet-4-5-20250929",
            messages=[Message(role="user", content="hi")],
            system=None,
            tools=None,
            timeout=30.0,
        )

    call_kwargs = client.messages.count_tokens.call_args.kwargs
    assert call_kwargs["system"] is anthropic.omit
    assert call_kwargs["tools"] is anthropic.omit
    assert call_kwargs["messages"] == [{"role": "user", "content": "hi"}]


def test_count_tokens_via_anthropic_api_dumps_string_system_and_tools():
    client = _mock_client()
    with patch(
        "free_claude_code.providers.anthropic_tokens.anthropic.Anthropic",
        return_value=client,
    ):
        count_tokens_via_anthropic_api(
            api_key="sk-test",
            model="claude-sonnet-4-5-20250929",
            messages=[Message(role="user", content="hi")],
            system="Be concise.",
            tools=[Tool(name="lookup", input_schema={"type": "object"})],
            timeout=30.0,
        )

    call_kwargs = client.messages.count_tokens.call_args.kwargs
    assert call_kwargs["system"] == "Be concise."
    assert call_kwargs["tools"] == [
        {"name": "lookup", "input_schema": {"type": "object"}}
    ]


def test_count_tokens_via_anthropic_api_dumps_block_system():
    client = _mock_client()
    with patch(
        "free_claude_code.providers.anthropic_tokens.anthropic.Anthropic",
        return_value=client,
    ):
        count_tokens_via_anthropic_api(
            api_key="sk-test",
            model="claude-sonnet-4-5-20250929",
            messages=[Message(role="user", content="hi")],
            system=[SystemContent(type="text", text="Be concise.")],
            tools=None,
            timeout=30.0,
        )

    call_kwargs = client.messages.count_tokens.call_args.kwargs
    assert call_kwargs["system"] == [{"type": "text", "text": "Be concise."}]


def test_count_tokens_via_anthropic_api_raises_on_sdk_failure():
    request = httpx.Request(
        "POST", "https://api.anthropic.com/v1/messages/count_tokens"
    )
    client = MagicMock()
    client.messages.count_tokens.side_effect = anthropic.APIConnectionError(
        request=request
    )

    with (
        patch(
            "free_claude_code.providers.anthropic_tokens.anthropic.Anthropic",
            return_value=client,
        ),
        pytest.raises(AnthropicTokenCountUnavailable),
    ):
        count_tokens_via_anthropic_api(
            api_key="sk-test",
            model="claude-sonnet-4-5-20250929",
            messages=[Message(role="user", content="hi")],
            system=None,
            tools=None,
            timeout=30.0,
        )
