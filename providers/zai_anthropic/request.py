"""Request builder for Z.ai Anthropic-compatible Messages API (GLM 5.2).

GLM 5.2 via ``https://api.z.ai/api/anthropic`` accepts standard Anthropic
Messages format with a few caveats: no vision (image/document blocks must be
stripped), and the bridge translates "95%+ of common patterns" but may lose
nested blocks in long agentic loops.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any

from loguru import logger

from config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from core.anthropic.native_messages_request import dump_raw_messages_request

# ---------------------------------------------------------------------------
# Block-type policies
# ---------------------------------------------------------------------------

# Block types not supported on GLM 5.2 Anthropic API (no vision).
_UNSUPPORTED_MESSAGE_BLOCK_TYPES = frozenset(
    {"image", "document", "server_tool_use", "web_search_tool_result",
     "web_fetch_tool_result"}
)

# Block types silently stripped (client attaches PDFs as document blocks
# alongside a Read tool_result that already contains the extracted text).
_STRIPPABLE_MESSAGE_BLOCK_TYPES = frozenset({"image", "document"})
_OMITTED_ATTACHMENT_TEXT = (
    "[attachment omitted: GLM 5.2 does not support image or document inputs]"
)
_OMITTED_ATTACHMENT_BLOCK = {"type": "text", "text": _OMITTED_ATTACHMENT_TEXT}

# Placeholder used when a tool call's real result was trimmed from context.
_TRIMMED_TOOL_RESULT_TEXT = (
    "[tool result omitted: trimmed from conversation context before reaching the provider]"
)


# ---------------------------------------------------------------------------
# Attachment stripping (same pattern as DeepSeek)
# ---------------------------------------------------------------------------

def _strip_unsupported_attachment_blocks(messages: Any) -> Any:
    """Remove image/document blocks that GLM 5.2 cannot process."""
    if not isinstance(messages, list):
        return messages

    stripped: list[Any] = []
    top_level_dropped: dict[str, int] = {}
    nested_dropped: dict[str, int] = {}
    placeholder_replacements = 0

    for message in messages:
        if not isinstance(message, dict):
            stripped.append(message)
            continue
        content = message.get("content")
        if not isinstance(content, list):
            stripped.append(message)
            continue

        new_content: list[Any] = []
        message_dropped_attachment = False
        for block in content:
            if isinstance(block, dict):
                btype = block.get("type")
                if btype in _STRIPPABLE_MESSAGE_BLOCK_TYPES:
                    top_level_dropped[btype] = top_level_dropped.get(btype, 0) + 1
                    message_dropped_attachment = True
                    continue
                if btype == "tool_result":
                    inner = block.get("content")
                    if isinstance(inner, list):
                        filtered_inner: list[Any] = []
                        for sub in inner:
                            if (
                                isinstance(sub, dict)
                                and sub.get("type") in _STRIPPABLE_MESSAGE_BLOCK_TYPES
                            ):
                                sub_type = sub["type"]
                                nested_dropped[sub_type] = (
                                    nested_dropped.get(sub_type, 0) + 1
                                )
                                continue
                            filtered_inner.append(sub)
                        if not filtered_inner:
                            filtered_inner = [_OMITTED_ATTACHMENT_BLOCK]
                            placeholder_replacements += 1
                        new_block = dict(block)
                        new_block["content"] = filtered_inner
                        new_content.append(new_block)
                        continue
            new_content.append(block)
        if not new_content and message_dropped_attachment:
            new_content = [_OMITTED_ATTACHMENT_BLOCK]
            placeholder_replacements += 1
        new_msg = dict(message)
        new_msg["content"] = new_content
        stripped.append(new_msg)

    if top_level_dropped or nested_dropped:
        logger.warning(
            "ZAI_ANTHROPIC_REQUEST: stripped unsupported attachment blocks "
            "(top_level={} nested_in_tool_result={} placeholder_tool_results={}). "
            "GLM 5.2 has no vision support; the model will not see this content.",
            dict(top_level_dropped),
            dict(nested_dropped),
            placeholder_replacements,
        )
    return stripped


# ---------------------------------------------------------------------------
# Thinking sanitization
# ---------------------------------------------------------------------------

def _sanitize_thinking_blocks(
    messages: Any, *, thinking_enabled: bool
) -> Any:
    """Filter assistant content for GLM 5.2: keep unsigned thinking when enabled,
    drop redacted_thinking always."""
    if not isinstance(messages, list):
        return messages

    sanitized: list[Any] = []
    for message in messages:
        if not isinstance(message, dict):
            sanitized.append(message)
            continue
        if message.get("role") != "assistant":
            sanitized.append(message)
            continue
        content = message.get("content")
        if not isinstance(content, list):
            sanitized.append(message)
            continue

        if not thinking_enabled:
            filtered = [
                block
                for block in content
                if not (
                    isinstance(block, dict)
                    and block.get("type") in ("thinking", "redacted_thinking")
                )
            ]
        else:
            filtered = [
                block
                for block in content
                if not (
                    isinstance(block, dict)
                    and block.get("type") == "redacted_thinking"
                )
            ]
        new_msg = dict(message)
        new_msg["content"] = filtered or ""
        sanitized.append(new_msg)
    return sanitized


def _strip_reasoning_content(messages: Any) -> Any:
    """``reasoning_content`` is OpenAI-helper metadata; remove from native body."""
    if not isinstance(messages, list):
        return messages
    out: list[Any] = []
    for m in messages:
        if not isinstance(m, dict):
            out.append(m)
            continue
        msg = {k: v for k, v in m.items() if k != "reasoning_content"}
        out.append(msg)
    return out


# ---------------------------------------------------------------------------
# Tool-pair reconciliation (same pattern as DeepSeek)
# ---------------------------------------------------------------------------

def _serialize_tool_result_content(content: Any) -> str:
    """Serialize tool_result content to string for API compatibility."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, dict):
                parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _normalize_tool_result_content(messages: Any) -> Any:
    """Normalize tool_result content to strings for API compatibility."""
    if not isinstance(messages, list):
        return messages

    normalized: list[Any] = []
    for message in messages:
        if not isinstance(message, dict):
            normalized.append(message)
            continue
        content = message.get("content")
        if not isinstance(content, list):
            normalized.append(message)
            continue

        new_content: list[Any] = []
        for block in content:
            if not isinstance(block, dict):
                new_content.append(block)
                continue
            if block.get("type") == "tool_result":
                normalized_block = dict(block)
                normalized_block["content"] = _serialize_tool_result_content(
                    block.get("content")
                )
                new_content.append(normalized_block)
            else:
                new_content.append(block)

        new_msg = dict(message)
        new_msg["content"] = new_content
        normalized.append(new_msg)

    return normalized


# ---------------------------------------------------------------------------
# Tool-pair reconciliation
# ---------------------------------------------------------------------------

def _assistant_tool_use_ids(message: Any) -> list[str]:
    ids: list[str] = []
    if not isinstance(message, dict) or message.get("role") != "assistant":
        return ids
    content = message.get("content")
    if not isinstance(content, list):
        return ids
    for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            tid = block.get("id")
            if isinstance(tid, str):
                ids.append(tid)
    return ids


def _user_tool_result_ids(message: Any) -> set[str]:
    ids: set[str] = set()
    if not isinstance(message, dict) or message.get("role") != "user":
        return ids
    content = message.get("content")
    if not isinstance(content, list):
        return ids
    for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_result":
            tid = block.get("tool_use_id")
            if isinstance(tid, str):
                ids.add(tid)
    return ids


def _placeholder_tool_result(tool_use_id: str) -> dict[str, str]:
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": _TRIMMED_TOOL_RESULT_TEXT,
    }


def _orphan_tool_result_as_text(block: dict[str, Any]) -> dict[str, str]:
    raw = block.get("content")
    text = raw if isinstance(raw, str) and raw.strip() else _TRIMMED_TOOL_RESULT_TEXT
    return {"type": "text", "text": text}


def _reconcile_tool_pairs(messages: Any) -> tuple[Any, dict[str, int]]:
    """Repair orphaned tool_use/tool_result pairs before forwarding.

    GLM 5.2 Anthropic endpoint enforces the Messages tool-pairing contract.
    """
    stats = {"added_tool_results": 0, "orphan_tool_results_as_text": 0}
    if not isinstance(messages, list) or not messages:
        return messages, stats

    # Pass A: reframe orphan tool_result blocks as text
    pass_a: list[Any] = []
    for i, message in enumerate(messages):
        if (
            isinstance(message, dict)
            and message.get("role") == "user"
            and isinstance(message.get("content"), list)
        ):
            prev = messages[i - 1] if i > 0 else None
            valid_ids = set(_assistant_tool_use_ids(prev))
            new_content: list[Any] = []
            for block in message["content"]:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "tool_result"
                    and block.get("tool_use_id") not in valid_ids
                ):
                    new_content.append(_orphan_tool_result_as_text(block))
                    stats["orphan_tool_results_as_text"] += 1
                    continue
                new_content.append(block)
            new_msg = dict(message)
            new_msg["content"] = new_content
            pass_a.append(new_msg)
        else:
            pass_a.append(message)

    # Pass B: every assistant tool_use must be answered in next user message
    result: list[Any] = []
    n = len(pass_a)
    i = 0
    while i < n:
        message = pass_a[i]
        result.append(message)
        ids = _assistant_tool_use_ids(message)
        if ids:
            nxt = pass_a[i + 1] if i + 1 < n else None
            if isinstance(nxt, dict) and nxt.get("role") == "user":
                missing = [tid for tid in ids if tid not in _user_tool_result_ids(nxt)]
                if missing:
                    placeholders = [_placeholder_tool_result(tid) for tid in missing]
                    new_next = dict(nxt)
                    content = new_next.get("content")
                    if isinstance(content, list):
                        new_next["content"] = placeholders + list(content)
                    elif isinstance(content, str) and content:
                        new_next["content"] = placeholders + [
                            {"type": "text", "text": content}
                        ]
                    else:
                        new_next["content"] = placeholders
                    stats["added_tool_results"] += len(missing)
                    result.append(new_next)
                    i += 2
                    continue
            else:
                placeholders = [_placeholder_tool_result(tid) for tid in ids]
                result.append({"role": "user", "content": placeholders})
                stats["added_tool_results"] += len(ids)
        i += 1

    return result, stats


# ---------------------------------------------------------------------------
# Extra system prompt injection (language + routing banner)
# ---------------------------------------------------------------------------

def _inject_extra_system_prompt(
    data: dict[str, Any], *, thinking_enabled: bool = False
) -> None:
    """Append extras to the system prompt sent to GLM 5.2:
    1. ``EXTRA_SYSTEM_PROMPT`` env var content (force Spanish responses).
    2. Dynamic routing banner directive.
    """
    blocks_to_append: list[dict[str, str]] = []

    extra = os.environ.get("EXTRA_SYSTEM_PROMPT", "").strip()
    if extra:
        blocks_to_append.append({"type": "text", "text": extra})

    if os.environ.get("ENABLE_ROUTING_BANNER", "true").lower() in ("1", "true", "yes"):
        model_id = str(data.get("model") or "?")
        model_name = model_id.rsplit("/", 1)[-1] if "/" in model_id else model_id
        thinking_status = "ON" if thinking_enabled else "OFF"
        banner_line = f"🤖 Modelo: {model_name} | Thinking: {thinking_status}"
        banner_directive = (
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
        blocks_to_append.append({"type": "text", "text": banner_directive})

    if not blocks_to_append:
        return

    existing = data.get("system")
    if existing is None or existing == "":
        data["system"] = blocks_to_append
    elif isinstance(existing, str):
        data["system"] = [{"type": "text", "text": existing}] + blocks_to_append
    elif isinstance(existing, list):
        data["system"] = list(existing) + blocks_to_append
    else:
        logger.warning(
            "ZAI_ANTHROPIC_REQUEST: cannot inject extras, unexpected system type: {}",
            type(existing).__name__,
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_request_body(request_data: Any, *, thinking_enabled: bool) -> dict:
    """Build an Anthropic-format request body for Z.ai GLM 5.2 /api/anthropic."""
    logger.debug(
        "ZAI_ANTHROPIC_REQUEST: build start model={} msgs={}",
        getattr(request_data, "model", "?"),
        len(getattr(request_data, "messages", [])),
    )

    data = dump_raw_messages_request(request_data)
    data.pop("extra_body", None)

    # Strip image/document blocks (no vision support in GLM 5.2)
    if "messages" in data:
        data["messages"] = _strip_unsupported_attachment_blocks(data["messages"])

    # Inject extra system prompt (language + routing banner)
    _inject_extra_system_prompt(data, thinking_enabled=thinking_enabled)

    logger.info(
        "MODEL_ROUTING: model={} thinking={} provider=zai_anthropic",
        data.get("model"),
        thinking_enabled,
    )

    # Thinking configuration
    thinking_cfg = data.pop("thinking", None)
    if thinking_enabled and isinstance(thinking_cfg, dict):
        thinking_payload: dict[str, Any] = {"type": "enabled"}
        budget_tokens = thinking_cfg.get("budget_tokens")
        if isinstance(budget_tokens, int):
            thinking_payload["budget_tokens"] = budget_tokens
        data["thinking"] = thinking_payload

    # Sanitize + normalize messages
    if "messages" in data:
        data["messages"] = _strip_reasoning_content(
            _normalize_tool_result_content(
                _sanitize_thinking_blocks(
                    data["messages"],
                    thinking_enabled=thinking_enabled,
                )
            )
        )
        # Reconcile orphaned tool pairs
        data["messages"], tool_pair_stats = _reconcile_tool_pairs(data["messages"])
        if any(tool_pair_stats.values()):
            logger.warning(
                "ZAI_ANTHROPIC_REQUEST: reconciled orphaned tool pairs "
                "(added_tool_results={added_tool_results} "
                "orphan_tool_results_as_text={orphan_tool_results_as_text})",
                **tool_pair_stats,
            )

    if "max_tokens" not in data or data.get("max_tokens") is None:
        data["max_tokens"] = ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS

    data["stream"] = True

    logger.debug(
        "ZAI_ANTHROPIC_REQUEST: build done model={} msgs={} tools={}",
        data.get("model"),
        len(data.get("messages", [])),
        len(data.get("tools", [])),
    )
    return data
