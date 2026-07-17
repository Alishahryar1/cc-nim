import pytest

from free_claude_code.application.reasoning import (
    client_reasoning_policy,
    resolve_reasoning_policy,
)
from free_claude_code.config.reasoning import ReasoningPreference
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.reasoning import (
    ReasoningControl,
    ReasoningEffort,
    ReasoningPolicy,
)


def _request(**overrides) -> MessagesRequest:
    payload = {
        "model": "provider/model",
        "messages": [{"role": "user", "content": "hello"}],
    }
    payload.update(overrides)
    return MessagesRequest.model_validate(payload)


def test_client_without_reasoning_control_uses_provider_default() -> None:
    assert client_reasoning_policy(_request()) == ReasoningPolicy.provider_default()


def test_client_reasoning_preserves_effort_and_exact_budget() -> None:
    policy = client_reasoning_policy(
        _request(
            thinking={"type": "enabled", "budget_tokens": 4096},
            output_config={"effort": "xhigh"},
        )
    )

    assert policy == ReasoningPolicy.on(
        effort=ReasoningEffort.XHIGH,
        budget_tokens=4096,
    )


def test_named_effort_does_not_invent_a_token_budget() -> None:
    policy = client_reasoning_policy(_request(output_config={"effort": "high"}))

    assert policy == ReasoningPolicy(
        control=ReasoningControl.DEFAULT,
        effort=ReasoningEffort.HIGH,
    )
    assert policy.budget_tokens is None
    assert policy.requests_reasoning is True


def test_invalid_budget_does_not_implicitly_enable_reasoning() -> None:
    policy = client_reasoning_policy(_request(thinking={"budget_tokens": 0}))

    assert policy == ReasoningPolicy.provider_default()


@pytest.mark.parametrize(
    "messages_request",
    [
        _request(thinking={"type": "disabled"}),
        _request(output_config={"effort": "none"}),
    ],
)
def test_client_disable_is_explicit(messages_request: MessagesRequest) -> None:
    policy = client_reasoning_policy(messages_request)

    assert policy.control is ReasoningControl.OFF
    assert policy.output_enabled is False
    assert policy.requests_reasoning is False


def test_disabled_thinking_preserves_independent_effort_intent() -> None:
    policy = client_reasoning_policy(
        _request(
            thinking={"type": "disabled"},
            output_config={"effort": "medium"},
        )
    )

    assert policy == ReasoningPolicy(
        control=ReasoningControl.OFF,
        effort=ReasoningEffort.MEDIUM,
    )
    assert policy.requests_reasoning is False


def test_fixed_route_effort_overrides_client_disable() -> None:
    policy = resolve_reasoning_policy(
        _request(thinking={"type": "disabled"}),
        ReasoningPreference.MAX,
    )

    assert policy == ReasoningPolicy.on(effort=ReasoningEffort.MAX)


def test_fixed_off_overrides_client_enable() -> None:
    policy = resolve_reasoning_policy(
        _request(thinking={"type": "enabled", "budget_tokens": 1024}),
        ReasoningPreference.OFF,
    )

    assert policy == ReasoningPolicy.off()


def test_client_preference_preserves_client_policy() -> None:
    request = _request(output_config={"effort": "low"})

    assert resolve_reasoning_policy(
        request, ReasoningPreference.CLIENT
    ) == client_reasoning_policy(request)


def test_unresolved_inherit_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be resolved"):
        resolve_reasoning_policy(_request(), ReasoningPreference.INHERIT)


@pytest.mark.parametrize("budget", [0, -1, True])
def test_reasoning_budget_requires_a_positive_integer(budget: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        ReasoningPolicy.on(budget_tokens=budget)


def test_reasoning_budget_requires_explicit_on_control() -> None:
    with pytest.raises(ValueError, match="control to be on"):
        ReasoningPolicy(budget_tokens=100)
