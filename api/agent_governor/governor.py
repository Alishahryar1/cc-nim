"""Main governor orchestrator.

The governor is invoked from ``api/services.py`` once per ``/v1/messages``
request, after context-trim and before provider streaming. It returns a
:class:`Verdict` describing what the proxy should do next:

* ``pass``    — forward the (possibly augmented) request to the provider.
* ``augment`` — same, but the request was modified (system, tools, messages).
* ``abort``   — short-circuit with a forced text response.

The governor itself is stateless. All decisions are derived from the request's
own message history; Claude Code re-sends the full conversation each turn.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from api.models.anthropic import MessagesRequest
from api.models.responses import MessagesResponse, Usage

from .config import GovernorConfig, load_governor_config
from .conversation_stats import ConversationStats, compute_stats
from .loop_detect import detect_repeated_signature
from .system_augmentation import with_plan_mode, with_termination_hint
from .telemetry import log_decision, log_skipped
from .tiers import ModelTier
from .tool_cull import CullResult, apply_cull, cull_tools


class GovernorAction(StrEnum):
    PASS = "pass"
    AUGMENT = "augment"
    ABORT = "abort"


@dataclass(slots=True)
class Verdict:
    """Outcome of one governor review."""

    action: GovernorAction
    augmented_request: MessagesRequest | None = None
    short_circuit_response: MessagesResponse | None = None
    interventions: tuple[str, ...] = field(default_factory=tuple)


class AgentGovernor:
    """Run the configured interventions against one request."""

    def __init__(
        self,
        config: GovernorConfig | None = None,
    ):
        self._config = config or load_governor_config()

    @property
    def config(self) -> GovernorConfig:
        return self._config

    def review(
        self,
        request: MessagesRequest,
        *,
        request_id: str,
        provider_id: str,
    ) -> Verdict:
        """Inspect a request, decide on an intervention, return a Verdict."""
        tier = self._config.tier_for(request.model)
        cfg = self._config.for_model(request.model)

        if not cfg.enabled:
            log_skipped(request_id, request.model, "disabled")
            return Verdict(action=GovernorAction.PASS)

        if tier == ModelTier.FRONTIER:
            log_skipped(request_id, request.model, "strong_model_bypass")
            return Verdict(action=GovernorAction.PASS)

        # Any non-frontier model gets the augmentation pipeline. Tier defaults
        # already decided whether plan_mode / termination_hint are appropriate
        # for the size class, so we just respect cfg.{plan_mode,termination_hint}_for_weak.
        is_weak = True
        stats = compute_stats(request.messages)
        tools_in_request = len(request.tools or [])

        # ---- 1. Hard abort: total tool budget exceeded
        if (
            cfg.max_total_tool_calls > 0
            and stats.total_tool_calls >= cfg.max_total_tool_calls
        ):
            response = self._build_abort_response(
                request,
                reason="total_tool_budget_exceeded",
                detail=(
                    f"This conversation has used {stats.total_tool_calls} tool "
                    f"calls, exceeding the configured budget of "
                    f"{cfg.max_total_tool_calls}. Stopping to prevent runaway "
                    f"loops. Start a fresh session to continue, ideally with a "
                    f"narrower task and fewer tools loaded."
                ),
            )
            log_decision(
                request_id,
                request.model,
                action=GovernorAction.ABORT,
                consecutive_tool_calls=stats.consecutive_tool_calls,
                total_tool_calls=stats.total_tool_calls,
                tools_in_request=tools_in_request,
                reason="total_tool_budget_exceeded",
                extra={"tier": tier.value},
            )
            return Verdict(
                action=GovernorAction.ABORT,
                short_circuit_response=response,
                interventions=("total_budget_abort",),
            )

        # ---- 2. Hard abort: detected repeating tool+args signature
        repeat = detect_repeated_signature(
            stats, threshold=cfg.loop_repeat_threshold
        )
        if repeat is not None:
            response = self._build_abort_response(
                request,
                reason="loop_signature_detected",
                detail=(
                    f"Detected {cfg.loop_repeat_threshold}+ identical calls to "
                    f"'{repeat.name}' with the same arguments in the recent "
                    f"history. The result is not changing — stopping the loop. "
                    f"Try a different approach or report what you found so far."
                ),
            )
            log_decision(
                request_id,
                request.model,
                action=GovernorAction.ABORT,
                consecutive_tool_calls=stats.consecutive_tool_calls,
                total_tool_calls=stats.total_tool_calls,
                tools_in_request=tools_in_request,
                intervention="loop_signature",
                reason=f"repeat:{repeat.name}",
                extra={"tier": tier.value},
            )
            return Verdict(
                action=GovernorAction.ABORT,
                short_circuit_response=response,
                interventions=("loop_signature_abort",),
            )

        # ---- 3. Soft abort: too many consecutive tool calls without text
        if (
            cfg.max_consecutive_tool_calls > 0
            and stats.consecutive_tool_calls >= cfg.max_consecutive_tool_calls
        ):
            response = self._build_abort_response(
                request,
                reason="consecutive_tool_calls_exceeded",
                detail=(
                    f"You have made {stats.consecutive_tool_calls} consecutive "
                    f"tool calls without producing a text answer. Stop now and "
                    f"summarise what you've found, or state clearly that you "
                    f"cannot complete the task. Pause, review, and answer."
                ),
            )
            log_decision(
                request_id,
                request.model,
                action=GovernorAction.ABORT,
                consecutive_tool_calls=stats.consecutive_tool_calls,
                total_tool_calls=stats.total_tool_calls,
                tools_in_request=tools_in_request,
                intervention="consecutive_tools",
                reason="cap_reached",
                extra={"tier": tier.value},
            )
            return Verdict(
                action=GovernorAction.ABORT,
                short_circuit_response=response,
                interventions=("consecutive_tools_abort",),
            )

        # ---- 4. Augment: tool culling, plan mode, termination hint
        augmented = request
        interventions: list[str] = []
        cull_outcome: CullResult | None = None

        if (
            cfg.tool_cull_enabled
            and request.tools
            and len(request.tools) > cfg.tool_cull_max
        ):
            cull_outcome = cull_tools(
                request.tools,
                request.messages,
                request.system,
                max_keep=cfg.tool_cull_max,
            )
            if cull_outcome.dropped_count > 0:
                new_tools = apply_cull(request.tools, cull_outcome)
                augmented = augmented.model_copy(update={"tools": new_tools})
                interventions.append(f"tool_cull(-{cull_outcome.dropped_count})")

        if is_weak and cfg.plan_mode_for_weak:
            new_system = with_plan_mode(augmented.system)
            if new_system is not augmented.system:
                augmented = augmented.model_copy(update={"system": new_system})
                interventions.append("plan_mode")

        if is_weak and cfg.termination_hint_for_weak:
            new_system = with_termination_hint(augmented.system)
            if new_system is not augmented.system:
                augmented = augmented.model_copy(update={"system": new_system})
                interventions.append("termination_hint")

        if not interventions:
            log_decision(
                request_id,
                request.model,
                action=GovernorAction.PASS,
                consecutive_tool_calls=stats.consecutive_tool_calls,
                total_tool_calls=stats.total_tool_calls,
                tools_in_request=tools_in_request,
                extra={"tier": tier.value},
            )
            return Verdict(action=GovernorAction.PASS)

        log_decision(
            request_id,
            request.model,
            action=GovernorAction.AUGMENT,
            consecutive_tool_calls=stats.consecutive_tool_calls,
            total_tool_calls=stats.total_tool_calls,
            tools_in_request=tools_in_request,
            tools_after_cull=len(augmented.tools or []) if cull_outcome else None,
            intervention=",".join(interventions),
            extra={"tier": tier.value},
        )
        return Verdict(
            action=GovernorAction.AUGMENT,
            augmented_request=augmented,
            interventions=tuple(interventions),
        )

    @staticmethod
    def _build_abort_response(
        request: MessagesRequest, *, reason: str, detail: str
    ) -> MessagesResponse:
        """Return an Anthropic-shape response that ends the conversation cleanly."""
        text = (
            f"[agent governor: {reason}]\n\n{detail}"
        )
        return MessagesResponse(
            id=f"msg_{uuid.uuid4().hex}",
            model=request.model,
            content=[{"type": "text", "text": text}],
            stop_reason="end_turn",
            usage=Usage(input_tokens=0, output_tokens=len(text) // 4),
        )


__all__ = ["AgentGovernor", "GovernorAction", "Verdict"]
