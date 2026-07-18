""" Extract the client's requested reasoning configuration without applying user preference overrides. """

from collections.abc import Mapping
from typing import Any

from free_claude_code.config.reasoning import ReasoningPreference
from free_claude_code.core.anthropic.models import MessagesRequest, ThinkingConfig
from free_claude_code.core.reasoning import (
    ReasoningControl,
    ReasoningEffort,
    ReasoningPolicy,
)


def resolve_reasoning_policy(
    request: MessagesRequest,
    preference: ReasoningPreference,
) -> ReasoningPolicy:
    """Apply one resolved configuration preference to the client request."""

    if preference is ReasoningPreference.INHERIT:
        raise ValueError("Reasoning preference must be resolved before application.")
    if preference is ReasoningPreference.OFF:
        return ReasoningPolicy.off()
    if preference is not ReasoningPreference.CLIENT:
        return ReasoningPolicy.on(effort=ReasoningEffort(preference.value))
    return client_reasoning_policy(request)



def _build_reasoning_policy(
    thinking_control: ReasoningControl,
    effort: ReasoningEffort | None,
    budget_tokens: int | None,
) -> ReasoningPolicy:
    """Create the final reasoning policy from resolved client settings."""

    if thinking_control is ReasoningControl.OFF:
        return ReasoningPolicy(
            control=ReasoningControl.OFF,
            effort=effort,
        )

    if thinking_control is ReasoningControl.ON:
        return ReasoningPolicy.on(
            effort=effort,
            budget_tokens=budget_tokens,
        )

    return ReasoningPolicy(
        control=ReasoningControl.DEFAULT,
        effort=effort,
    )


def client_reasoning_policy(request: MessagesRequest) -> ReasoningPolicy:
    """Return the lossless reasoning intent expressed by one client request."""

    budget_tokens = _positive_budget(request.thinking)
    thinking_control = _thinking_control(
        request.thinking,
        budget_tokens=budget_tokens,
    )
    effort, effort_disables = _output_effort(request.output_config)

    if effort_disables:
        return ReasoningPolicy.off()

    return _build_reasoning_policy(
        thinking_control,
        effort,
        budget_tokens,
    )

def _thinking_control(
    thinking: ThinkingConfig | None,
    *,
    budget_tokens: int | None,
    ) -> ReasoningControl:
    if thinking is None:
        return ReasoningControl.DEFAULT
    if thinking.type == "disabled" or (
        "enabled" in thinking.model_fields_set and thinking.enabled is False
    ):
        return ReasoningControl.OFF
    if (
        thinking.type in {"adaptive", "enabled"}
        or ("enabled" in thinking.model_fields_set and thinking.enabled is True)
        or budget_tokens is not None
    ):
        return ReasoningControl.ON
    return ReasoningControl.DEFAULT


def _output_effort(...) -> OutputEffort:
    OutputEffort = tuple[ReasoningEffort | None, bool] # to increase the readability
    if not isinstance(value, Mapping):
        return None, False
    raw_effort = value.get("effort")
    if not isinstance(raw_effort, str):
        return None, False
    normalized_effort = raw_effort.strip().lower()
    if normalized_effort == "none":
        return None, True
    try:
        return ReasoningEffort(normalized), False
    except ValueError:
        return None, False


def _positive_budget(thinking: ThinkingConfig | None) -> int | None:
    if thinking is None:
        return None
    value = thinking.budget_tokens
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None
