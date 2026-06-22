"""Native Anthropic Messages request builder for Kimi (Moonshot)."""

from __future__ import annotations

import copy
from typing import Any

from loguru import logger

from config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from core.anthropic.native_messages_request import (
    build_base_native_anthropic_request_body,
)
from providers.exceptions import InvalidRequestError


def _resolve_ref(ref: str, root: dict) -> dict | None:
    """Resolve a local JSON pointer like '#/$defs/Foo' against root."""
    if not isinstance(ref, str) or not ref.startswith("#/"):
        return None
    node: Any = root
    for part in ref[2:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return None
    return node if isinstance(node, dict) else None


def _inline_refs(node: Any, root: dict, seen: frozenset = frozenset()) -> Any:
    """Recursively inline $ref pointers and drop sibling keywords next to $ref.

    Moonshot's flavored JSON Schema rejects siblings (description, etc.) next
    to $ref because expansion produces conflicting keywords. Resolve refs
    against root and replace the wrapping object outright.
    """
    if isinstance(node, dict):
        if "$ref" in node:
            ref = node["$ref"]
            if ref in seen:
                clone = {k: v for k, v in node.items() if k != "$ref"}
                clone["type"] = clone.get("type", "object")
                return clone
            target = _resolve_ref(ref, root)
            if target is not None:
                merged = _inline_refs(target, root, seen | {ref})
                if isinstance(merged, dict):
                    out = dict(merged)
                    for k, v in node.items():
                        if k == "$ref":
                            continue
                        out.setdefault(k, v)
                    return out
                return merged
            return {k: v for k, v in node.items() if k != "$ref"}
        return {k: _inline_refs(v, root, seen) for k, v in node.items()}
    if isinstance(node, list):
        return [_inline_refs(item, root, seen) for item in node]
    return node


def _sanitize_tool_parameters(params: dict) -> dict:
    """Inline $ref/$defs and strip sibling-conflict keys for Moonshot."""
    if not isinstance(params, dict):
        return params
    resolved = _inline_refs(copy.deepcopy(params), params)
    if isinstance(resolved, dict):
        resolved.pop("$defs", None)
        resolved.pop("definitions", None)
    return resolved


def _sanitize_tools(tools: list) -> list:
    out = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            out.append(tool)
            continue
        new_tool = dict(tool)
        fn = new_tool.get("function")
        if isinstance(fn, dict):
            new_fn = dict(fn)
            params = new_fn.get("parameters")
            if isinstance(params, dict):
                new_fn["parameters"] = _sanitize_tool_parameters(params)
            new_tool["function"] = new_fn
        out.append(new_tool)
    return out


def build_request_body(request_data: Any, *, thinking_enabled: bool) -> dict:
    """Build JSON for Kimi Anthropic-compat ``POST …/messages``."""
    logger.debug(
        "KIMI_REQUEST: native build model={} msgs={}",
        getattr(request_data, "model", "?"),
        len(getattr(request_data, "messages", [])),
    )

    body = build_base_native_anthropic_request_body(
        request_data,
        default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
        thinking_enabled=thinking_enabled,
    )
    extra = getattr(request_data, "extra_body", None)
    if extra:
        raise InvalidRequestError(
            "Kimi native Messages API does not support extra_body on requests."
        )
    body["stream"] = True

    if "tools" in body:
        body["tools"] = _sanitize_tools(body["tools"])

    logger.debug(
        "KIMI_REQUEST: build done model={} msgs={} tools={}",
        body.get("model"),
        len(body.get("messages", [])),
        len(body.get("tools", [])),
    )
    return body
