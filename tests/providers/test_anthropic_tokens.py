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


def test_count_tokens_via_anthropic_api_error_message_is_preserved():
    """The raised exception carries a readable message and the original cause."""
    request = httpx.Request(
        "POST", "https://api.anthropic.com/v1/messages/count_tokens"
    )
    client = MagicMock()
    sdk_error = anthropic.APIConnectionError(
        message="connection refused", request=request
    )
    client.messages.count_tokens.side_effect = sdk_error

    with (
        patch(
            "free_claude_code.providers.anthropic_tokens.anthropic.Anthropic",
            return_value=client,
        ),
        pytest.raises(AnthropicTokenCountUnavailable) as exc_info,
    ):
        count_tokens_via_anthropic_api(
            api_key="sk-test",
            model="claude-sonnet-4-5-20250929",
            messages=[Message(role="user", content="hi")],
            system=None,
            tools=None,
            timeout=30.0,
        )

    assert str(exc_info.value)
    assert exc_info.value.__cause__ is sdk_error


def test_count_tokens_via_anthropic_api_non_api_error_propagates_unwrapped():
    """Only ``anthropic.APIError`` is translated; other errors bubble up raw."""
    client = MagicMock()
    client.messages.count_tokens.side_effect = ValueError("boom")

    with (
        patch(
            "free_claude_code.providers.anthropic_tokens.anthropic.Anthropic",
            return_value=client,
        ),
        pytest.raises(ValueError, match="boom"),
    ):
        count_tokens_via_anthropic_api(
            api_key="sk-test",
            model="claude-sonnet-4-5-20250929",
            messages=[Message(role="user", content="hi")],
            system=None,
            tools=None,
            timeout=30.0,
        )


def test_count_tokens_via_anthropic_api_empty_tools_list_is_omitted():
    """An empty tools list is treated the same as absent (omitted), not sent as []."""
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
            tools=[],
            timeout=30.0,
        )

    call_kwargs = client.messages.count_tokens.call_args.kwargs
    assert call_kwargs["tools"] is anthropic.omit


def test_count_tokens_via_anthropic_api_preserves_multiple_message_order():
    """Multiple messages are dumped in the same order they were provided."""
    client = _mock_client()
    with patch(
        "free_claude_code.providers.anthropic_tokens.anthropic.Anthropic",
        return_value=client,
    ):
        count_tokens_via_anthropic_api(
            api_key="sk-test",
            model="claude-sonnet-4-5-20250929",
            messages=[
                Message(role="user", content="first"),
                Message(role="assistant", content="second"),
                Message(role="user", content="third"),
            ],
            system=None,
            tools=None,
            timeout=30.0,
        )

    call_kwargs = client.messages.count_tokens.call_args.kwargs
    assert call_kwargs["messages"] == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "second"},
        {"role": "user", "content": "third"},
    ]


def test_count_tokens_via_anthropic_api_excludes_none_message_fields():
    """``exclude_none`` keeps the payload minimal (no ``reasoning_content: null``)."""
    client = _mock_client()
    with patch(
        "free_claude_code.providers.anthropic_tokens.anthropic.Anthropic",
        return_value=client,
    ):
        count_tokens_via_anthropic_api(
            api_key="sk-test",
            model="claude-sonnet-4-5-20250929",
            messages=[Message(role="user", content="hi", reasoning_content=None)],
            system=None,
            tools=None,
            timeout=30.0,
        )

    call_kwargs = client.messages.count_tokens.call_args.kwargs
    assert call_kwargs["messages"] == [{"role": "user", "content": "hi"}]
    assert "reasoning_content" not in call_kwargs["messages"][0]


def test_count_tokens_via_anthropic_api_passes_model_through_unchanged():
    """The model id is forwarded to the SDK exactly as provided."""
    client = _mock_client()
    with patch(
        "free_claude_code.providers.anthropic_tokens.anthropic.Anthropic",
        return_value=client,
    ):
        count_tokens_via_anthropic_api(
            api_key="sk-test",
            model="claude-opus-4-1-20250805",
            messages=[Message(role="user", content="hi")],
            system=None,
            tools=None,
            timeout=30.0,
        )

    assert (
        client.messages.count_tokens.call_args.kwargs["model"]
        == "claude-opus-4-1-20250805"
    )
