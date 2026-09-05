"""Versioned native-thinking replay carried by Responses reasoning items."""

import base64
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from free_claude_code.core.json_types import JsonObject, JsonValue

from .errors import ResponsesConversionError

_REASONING_FAMILY = "fcc:anthropic-reasoning:"
_REASONING_PREFIX = _REASONING_FAMILY + "v1:"
_MAX_REPLAY_DEPTH = 128


@dataclass(frozen=True, slots=True)
class MessagesReplayOrigin:
    replay_scope: str
    source_model: str


def is_messages_reasoning_carrier(value: object) -> bool:
    return isinstance(value, str) and value.startswith(_REASONING_FAMILY)


def _validate_block(block: JsonValue) -> JsonObject:
    if not isinstance(block, Mapping):
        raise ResponsesConversionError(
            "Native reasoning replay block must be an object."
        )
    kind = block.get("type")
    if kind == "thinking":
        if not isinstance(block.get("thinking"), str):
            raise ResponsesConversionError(
                "Native thinking replay requires thinking text."
            )
        if not isinstance(block.get("signature"), str) or not block["signature"]:
            raise ResponsesConversionError(
                "Native thinking replay requires its signature."
            )
    elif kind == "redacted_thinking":
        if not isinstance(block.get("data"), str) or not block["data"]:
            raise ResponsesConversionError(
                "Redacted native thinking requires its data."
            )
    else:
        raise ResponsesConversionError("Unsupported native reasoning replay block.")
    pending: list[tuple[JsonValue, int]] = [(block, 0)]
    while pending:
        value, depth = pending.pop()
        if depth > _MAX_REPLAY_DEPTH:
            raise ResponsesConversionError(
                "Native reasoning replay is too deeply nested."
            )
        if isinstance(value, Mapping):
            pending.extend((child, depth + 1) for child in value.values())
        elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
            pending.extend((child, depth + 1) for child in value)
    try:
        # A carrier can hide malformed JSON inside an otherwise ordinary string.
        # Validate finite numbers and nesting before HTTP serialization.
        return cast(
            JsonObject,
            json.loads(
                json.dumps(dict(block), ensure_ascii=False, allow_nan=False).encode(
                    "utf-8"
                )
            ),
        )
    except (TypeError, ValueError, RecursionError) as exc:
        raise ResponsesConversionError(
            "Native reasoning replay must contain finite, representable JSON."
        ) from exc


def encode_messages_reasoning(
    block: JsonObject, *, origin: MessagesReplayOrigin
) -> str:
    """Serialize native signed data; this does not add encryption or authentication."""

    if not origin.replay_scope or not origin.source_model:
        raise ResponsesConversionError(
            "Native reasoning replay requires provider provenance."
        )
    payload: JsonObject = {
        "protocol": "anthropic_messages",
        "replay_scope": origin.replay_scope,
        "source_model": origin.source_model,
        "block": _validate_block(block),
    }
    try:
        encoded = json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ResponsesConversionError(
            "Malformed native Messages reasoning replay."
        ) from exc
    return _REASONING_PREFIX + base64.urlsafe_b64encode(encoded).decode("ascii").rstrip(
        "="
    )


def decode_messages_reasoning(value: str, *, replay_scope: str) -> JsonObject:
    """Validate provenance and return the exact authoritative replay block."""

    if not value.startswith(_REASONING_PREFIX):
        raise ResponsesConversionError(
            "This reasoning item is not a supported native Messages replay carrier."
        )
    encoded = value[len(_REASONING_PREFIX) :]
    try:
        raw = base64.b64decode(
            encoded + "=" * (-len(encoded) % 4), altchars=b"-_", validate=True
        )
        payload = cast(JsonValue, json.loads(raw.decode("utf-8")))
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ResponsesConversionError(
            "Malformed native Messages reasoning replay."
        ) from exc
    if (
        not isinstance(payload, Mapping)
        or set(payload) != {"protocol", "replay_scope", "source_model", "block"}
        or payload.get("protocol") != "anthropic_messages"
        or payload.get("replay_scope") != replay_scope
        or not isinstance(payload.get("source_model"), str)
        or not payload["source_model"]
    ):
        raise ResponsesConversionError(
            "Native reasoning replay has incompatible provenance."
        )
    return _validate_block(payload.get("block"))


def reject_messages_reasoning_for_other_egress(value: JsonValue) -> None:
    """Prevent routing native signatures into a different reasoning protocol."""

    items: Sequence[JsonValue]
    if isinstance(value, Mapping):
        items = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
        items = value
    else:
        return
    for item in items:
        if (
            isinstance(item, Mapping)
            and item.get("type") == "reasoning"
            and is_messages_reasoning_carrier(item.get("encrypted_content"))
        ):
            raise ResponsesConversionError(
                "Native Messages reasoning history requires a Messages upstream model."
            )
