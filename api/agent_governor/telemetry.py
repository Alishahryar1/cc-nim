"""Structured logging for governor decisions."""

from __future__ import annotations

from typing import Any

from loguru import logger


def log_decision(
    request_id: str,
    model: str,
    action: str,
    *,
    consecutive_tool_calls: int,
    total_tool_calls: int,
    tools_in_request: int,
    tools_after_cull: int | None = None,
    intervention: str | None = None,
    reason: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Emit a single GOVERNOR: log line with all decision metadata."""
    parts = [
        f"request_id={request_id}",
        f"model={model}",
        f"action={action}",
        f"consecutive_tools={consecutive_tool_calls}",
        f"total_tools={total_tool_calls}",
        f"tools_in={tools_in_request}",
    ]
    if tools_after_cull is not None and tools_after_cull != tools_in_request:
        parts.append(f"tools_after_cull={tools_after_cull}")
    if intervention:
        parts.append(f"intervention={intervention}")
    if reason:
        parts.append(f"reason={reason}")
    if extra:
        for key, value in extra.items():
            parts.append(f"{key}={value}")
    logger.info("GOVERNOR: " + " ".join(parts))


def log_skipped(request_id: str, model: str, reason: str) -> None:
    """Emit a single line when the governor skips intervention entirely."""
    logger.debug(
        "GOVERNOR: request_id={} model={} action=skip reason={}",
        request_id,
        model,
        reason,
    )
