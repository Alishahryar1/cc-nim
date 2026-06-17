"""Append governor preambles to the system prompt for weak models.

Two preambles, both opt-in via env:

* **plan_mode** — instructs the model to write a 3-step plan before any tool
  call. Acts as a soft termination signal: once the plan completes, the model
  has license to stop.
* **termination_hint** — explicitly tells the model that text-only output is a
  valid answer; it does NOT need to call a tool to terminate.
"""

from __future__ import annotations

from typing import Any


PLAN_MODE_PREAMBLE = """\
[Operating mode: plan-then-execute]

Before calling any tool, write a numbered plan with at most 5 short steps. \
Each step should be one sentence describing a single concrete action. After \
the plan, execute the steps in order. After completing the plan, write a \
final answer summarising what you found or did. Do not call additional tools \
beyond the plan unless absolutely necessary."""


TERMINATION_HINT_PREAMBLE = """\
[Termination policy]

When you have answered the user's question, completed the requested task, OR \
when you cannot make further progress, output your final answer as plain \
text and stop. Do not keep calling tools to "double-check" — one verification \
is enough. If a tool returned an error or unhelpful result twice in a row, \
state what you tried and stop. Tool calls are expensive; text answers are free."""


def _coerce_system_to_string(system: Any) -> tuple[str, str]:
    """Return (prefix_for_concatenation, marker_to_check_for) used by callers.

    fcc preserves Anthropic's structured ``system`` field — it can be ``None``,
    a plain string, or a list of SystemContent blocks. The governor stays
    structure-preserving by editing or appending blocks, not by flattening.
    Callers receive the original system unchanged from this helper; this exists
    only for the existence-check below.
    """
    if isinstance(system, str):
        return system, system
    if isinstance(system, list):
        joined = " ".join(
            getattr(block, "text", "") or ""
            for block in system
            if hasattr(block, "text")
        )
        joined += " " + " ".join(
            block.get("text", "") if isinstance(block, dict) else ""
            for block in system
        )
        return joined, joined
    return "", ""


def _has_marker(system: Any, marker: str) -> bool:
    _, joined = _coerce_system_to_string(system)
    return marker in joined


def with_plan_mode(system: Any) -> Any:
    """Append the plan-mode preamble unless already present."""
    return _append_preamble(system, PLAN_MODE_PREAMBLE, marker="[Operating mode: plan-then-execute]")


def with_termination_hint(system: Any) -> Any:
    """Append the termination-hint preamble unless already present."""
    return _append_preamble(system, TERMINATION_HINT_PREAMBLE, marker="[Termination policy]")


def _append_preamble(system: Any, preamble: str, *, marker: str) -> Any:
    if _has_marker(system, marker):
        return system
    if system is None:
        return preamble
    if isinstance(system, str):
        return system.rstrip() + "\n\n" + preamble
    if isinstance(system, list):
        # Append a new structured text block so we don't disturb cache_control
        # or other attributes on existing blocks.
        new_block = _build_text_block(system, preamble)
        return list(system) + [new_block]
    # Unknown shape — fall back to leaving system alone.
    return system


def _build_text_block(existing: list[Any], text: str) -> Any:
    """Build a SystemContent-shaped block matching whatever the existing list uses.

    Callers may pass pydantic models or dicts depending on origin. We mimic
    whichever form is already present so the request remains serialisable.
    """
    for item in existing:
        if hasattr(item, "model_copy"):
            try:
                clone = item.model_copy(update={"text": text})
                return clone
            except Exception:
                continue
        if isinstance(item, dict):
            return {"type": item.get("type", "text"), "text": text}
    return {"type": "text", "text": text}
