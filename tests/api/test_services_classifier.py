"""Tests for disabling thinking on auto-mode safety-classifier requests."""

from api.model_router import ResolvedModel, RoutedMessagesRequest
from api.models.anthropic import Message, MessagesRequest
from api.services import ClaudeProxyService

_CLASSIFIER_SYSTEM = (
    "You are a security monitor. Respond with <block>yes</block> or <block>no</block>."
)
_CLASSIFIER_USER = (
    "<transcript>\nUser: review the repo\nWebFetch https://example.com: fetch\n"
    "</transcript>\n<block> immediately."
)


def _routed(
    request: MessagesRequest, *, thinking_enabled: bool
) -> RoutedMessagesRequest:
    resolved = ResolvedModel(
        original_model=request.model,
        provider_id="lmstudio",
        provider_model="openai/gpt-oss-20b",
        provider_model_ref="lmstudio/openai/gpt-oss-20b",
        thinking_enabled=thinking_enabled,
    )
    return RoutedMessagesRequest(request=request, resolved=resolved)


def _make_request(content: str, **kwargs) -> MessagesRequest:
    return MessagesRequest(
        model="lmstudio/openai/gpt-oss-20b",
        max_tokens=2112,
        messages=[Message(role="user", content=content)],
        **kwargs,
    )


def test_thinking_disabled_for_classifier_request():
    """A classifier request with thinking on has thinking forced off."""
    routed = _routed(
        _make_request(_CLASSIFIER_USER, system=_CLASSIFIER_SYSTEM),
        thinking_enabled=True,
    )
    result = ClaudeProxyService._disable_thinking_for_safety_classifier(routed)
    assert result.resolved.thinking_enabled is False
    # The request itself is untouched; only the routing decision changes.
    assert result.request is routed.request


def test_thinking_preserved_for_non_classifier_request():
    """An ordinary request keeps thinking enabled."""
    routed = _routed(_make_request("hola"), thinking_enabled=True)
    result = ClaudeProxyService._disable_thinking_for_safety_classifier(routed)
    assert result.resolved.thinking_enabled is True
    assert result is routed


def test_no_op_when_thinking_already_disabled():
    """When thinking is already off, the routed request is returned unchanged."""
    routed = _routed(
        _make_request(_CLASSIFIER_USER, system=_CLASSIFIER_SYSTEM),
        thinking_enabled=False,
    )
    result = ClaudeProxyService._disable_thinking_for_safety_classifier(routed)
    assert result is routed
