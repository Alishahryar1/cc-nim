"""Resolve client reasoning input and FCC configuration exactly once."""

from free_claude_code.config.reasoning import ReasoningPreference
from free_claude_code.core.inference import ClientReasoningIntent
from free_claude_code.core.reasoning import (
    ReasoningControl,
    ReasoningEffort,
    ReasoningPolicy,
)


def resolve_reasoning_policy(
    intent: ClientReasoningIntent,
    preference: ReasoningPreference,
) -> ReasoningPolicy:
    """Apply one resolved configuration preference to the client request."""

    if preference is ReasoningPreference.INHERIT:
        raise ValueError("Reasoning preference must be resolved before application.")
    if preference is ReasoningPreference.OFF:
        return ReasoningPolicy.off()
    if preference is not ReasoningPreference.CLIENT:
        return ReasoningPolicy.on(effort=ReasoningEffort(preference.value))
    return client_reasoning_policy(intent)


def client_reasoning_policy(intent: ClientReasoningIntent) -> ReasoningPolicy:
    """Return the lossless reasoning intent expressed by one client request."""

    if intent.control is ReasoningControl.OFF:
        return ReasoningPolicy(
            control=ReasoningControl.OFF,
            effort=intent.effort,
        )
    if intent.control is ReasoningControl.ON or intent.budget_tokens is not None:
        return ReasoningPolicy.on(
            effort=intent.effort,
            budget_tokens=intent.budget_tokens,
        )
    return ReasoningPolicy(
        control=ReasoningControl.DEFAULT,
        effort=intent.effort,
    )
