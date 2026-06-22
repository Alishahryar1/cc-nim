"""Native Anthropic Messages request builder for OpenRouter."""

from __future__ import annotations

import os
from typing import Any

from loguru import logger

from config.constants import (
    ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS as OPENROUTER_DEFAULT_MAX_TOKENS,
)
from core.anthropic.native_messages_request import (
    OpenRouterExtraBodyError,
    build_openrouter_native_request_body,
)
from providers.exceptions import InvalidRequestError


def _inject_extra_system_prompt(
    body: dict[str, Any], *, thinking_enabled: bool = False
) -> None:
    """Append language enforcement + routing banner to OpenRouter's ``system``.

    Parity with ``providers/deepseek`` and ``providers/zai_anthropic``: those
    inject the same two segments so every turn (a) obeys ``EXTRA_SYSTEM_PROMPT``
    (e.g. force Spanish) and (b) starts with a ``🤖 Modelo: ... | Thinking: ...``
    banner. OpenRouter's native endpoint flattens ``system`` to a *string*
    (see ``_normalize_system_prompt_for_openrouter``), so the segments are
    appended as text joined by blank lines, not as Anthropic content blocks.
    """
    segments: list[str] = []

    extra = os.environ.get("EXTRA_SYSTEM_PROMPT", "").strip()
    if extra:
        segments.append(extra)

    if os.environ.get("ENABLE_ROUTING_BANNER", "true").lower() in ("1", "true", "yes"):
        model_id = str(body.get("model") or "?")
        model_name = model_id.rsplit("/", 1)[-1] if "/" in model_id else model_id
        thinking_status = "ON" if thinking_enabled else "OFF"
        banner_line = f"🤖 Modelo: {model_name} | Thinking: {thinking_status}"
        segments.append(
            "INSTRUCCIÓN OBLIGATORIA DE BANNER DE ROUTING: Antes de cualquier "
            "otro contenido, tu respuesta DEBE comenzar EXACTAMENTE con la "
            "siguiente línea literal (incluyendo el emoji, sin bloque de "
            "código, sin variaciones, sin traducir):\n\n"
            f"{banner_line}\n\n"
            "Después deja una línea en blanco y continúa con tu respuesta "
            "normal. Esta regla es OBLIGATORIA en TODAS las respuestas, "
            "incluidas las muy cortas (sí, no, OK). Si la respuesta es de "
            "una sola línea, el banner sigue siendo la primera línea. NO "
            "modifiques el texto del banner."
        )

    if not segments:
        return

    appended = "\n\n".join(segments)
    existing = body.get("system")
    if existing is None or existing == "":
        body["system"] = appended
    elif isinstance(existing, str):
        body["system"] = f"{existing}\n\n{appended}"
    elif isinstance(existing, list):
        text_parts = [
            block["text"]
            for block in existing
            if isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        ]
        joined = "\n\n".join(text_parts)
        body["system"] = f"{joined}\n\n{appended}" if joined else appended
    else:
        logger.warning(
            "OPENROUTER_REQUEST: cannot inject extras, unexpected system type: {}",
            type(existing).__name__,
        )


def build_request_body(request_data: Any, *, thinking_enabled: bool) -> dict:
    """Build an Anthropic-format request body for OpenRouter's messages API."""
    logger.debug(
        "OPENROUTER_REQUEST: conversion start model={} msgs={}",
        getattr(request_data, "model", "?"),
        len(getattr(request_data, "messages", [])),
    )

    try:
        body = build_openrouter_native_request_body(
            request_data,
            thinking_enabled=thinking_enabled,
            default_max_tokens=OPENROUTER_DEFAULT_MAX_TOKENS,
        )
    except OpenRouterExtraBodyError as exc:
        raise InvalidRequestError(str(exc)) from exc

    # Restore language enforcement + routing banner (parity with DeepSeek/Z.ai).
    # OpenRouter flattens ``system`` to a string, so this appends string segments.
    _inject_extra_system_prompt(body, thinking_enabled=thinking_enabled)

    logger.info(
        "MODEL_ROUTING: model={} thinking={} provider=open_router",
        body.get("model"),
        thinking_enabled,
    )

    logger.debug(
        "OPENROUTER_REQUEST: conversion done model={} msgs={} tools={}",
        body.get("model"),
        len(body.get("messages", [])),
        len(body.get("tools", [])),
    )
    return body
