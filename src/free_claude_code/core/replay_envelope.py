"""Versioned public envelope for opaque inference replay artifacts."""

import base64
import binascii
import json
import math
from collections.abc import Mapping, Sequence

from free_claude_code.core.inference import (
    ReplayArtifact,
    ReplayArtifactKind,
    ReplayArtifactOrigin,
    ReplayAttachment,
    ReplayCompatibilityScope,
)
from free_claude_code.core.json_types import JsonValue

REPLAY_ENVELOPE_PREFIX = "fcc-replay-v1:"
_REPLAY_ENVELOPE_VERSION = 1
_MAX_REPLAY_ENVELOPE_BYTES = 1_048_576
_ARTIFACT_KEYS = frozenset({"origin", "kind", "attachment", "scope", "payload"})


class ReplayEnvelopeError(ValueError):
    """A recognized FCC replay envelope is malformed or unsupported."""


def encode_replay_envelope(artifacts: tuple[ReplayArtifact, ...]) -> str:
    """Encode ordered replay artifacts into the deterministic v1 carrier."""

    if not artifacts:
        raise ValueError("a replay envelope requires at least one artifact")
    payload = {
        "version": _REPLAY_ENVELOPE_VERSION,
        "artifacts": [
            {
                "origin": artifact.origin.value,
                "kind": artifact.kind.value,
                "attachment": artifact.attachment.value,
                "scope": (artifact.scope.value if artifact.scope is not None else None),
                "payload": _json_copy(artifact.payload),
            }
            for artifact in artifacts
        ],
    }
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if len(encoded) > _MAX_REPLAY_ENVELOPE_BYTES:
        raise ValueError("replay envelope exceeds the supported size limit")
    carrier = base64.urlsafe_b64encode(encoded).decode("ascii").rstrip("=")
    return f"{REPLAY_ENVELOPE_PREFIX}{carrier}"


def decode_replay_envelope(
    value: str,
    *,
    attachment: ReplayAttachment,
) -> tuple[ReplayArtifact, ...] | None:
    """Decode a recognized carrier, or return ``None`` for a legacy raw value."""

    if not value.startswith(REPLAY_ENVELOPE_PREFIX):
        return None
    token = value.removeprefix(REPLAY_ENVELOPE_PREFIX)
    if not token:
        raise ReplayEnvelopeError("FCC replay envelope payload is empty")
    padding = "=" * (-len(token) % 4)
    try:
        raw = base64.b64decode(
            token + padding,
            altchars=b"-_",
            validate=True,
        )
    except (binascii.Error, ValueError) as exc:
        raise ReplayEnvelopeError("FCC replay envelope is not valid base64url") from exc
    if len(raw) > _MAX_REPLAY_ENVELOPE_BYTES:
        raise ReplayEnvelopeError(
            "FCC replay envelope exceeds the supported size limit"
        )
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReplayEnvelopeError(
            "FCC replay envelope is not valid UTF-8 JSON"
        ) from exc
    if not isinstance(payload, dict) or set(payload) != {"version", "artifacts"}:
        raise ReplayEnvelopeError("FCC replay envelope has an invalid top-level schema")
    if payload.get("version") != _REPLAY_ENVELOPE_VERSION:
        raise ReplayEnvelopeError("FCC replay envelope version is unsupported")
    raw_artifacts = payload.get("artifacts")
    if not isinstance(raw_artifacts, list) or not raw_artifacts:
        raise ReplayEnvelopeError(
            "FCC replay envelope artifacts must be a non-empty list"
        )
    artifacts = tuple(
        _decode_artifact(raw_artifact, index=index, attachment=attachment)
        for index, raw_artifact in enumerate(raw_artifacts)
    )
    return artifacts


def _decode_artifact(
    value: object,
    *,
    index: int,
    attachment: ReplayAttachment,
) -> ReplayArtifact:
    path = f"artifacts[{index}]"
    if not isinstance(value, dict) or set(value) != _ARTIFACT_KEYS:
        raise ReplayEnvelopeError(f"FCC replay envelope {path} has an invalid schema")
    try:
        origin = ReplayArtifactOrigin(value["origin"])
        kind = ReplayArtifactKind(value["kind"])
        decoded_attachment = ReplayAttachment(value["attachment"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ReplayEnvelopeError(
            f"FCC replay envelope {path} has an unknown artifact value"
        ) from exc
    if decoded_attachment is not attachment:
        raise ReplayEnvelopeError(
            f"FCC replay envelope {path}.attachment does not match its carrier"
        )
    raw_scope = value.get("scope")
    if raw_scope is not None and (
        not isinstance(raw_scope, str) or not raw_scope.strip()
    ):
        raise ReplayEnvelopeError(
            f"FCC replay envelope {path}.scope must be null or a non-empty string"
        )
    raw_payload = value.get("payload")
    if not _is_json_value(raw_payload):
        raise ReplayEnvelopeError(
            f"FCC replay envelope {path}.payload is not valid JSON"
        )
    return ReplayArtifact(
        origin=origin,
        kind=kind,
        attachment=decoded_attachment,
        payload=raw_payload,
        scope=(
            ReplayCompatibilityScope(raw_scope) if isinstance(raw_scope, str) else None
        ),
    )


def _is_json_value(value: object) -> bool:
    if value is None or isinstance(value, bool | int | str):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_is_json_value(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _is_json_value(item) for key, item in value.items()
        )
    return False


def _json_copy(value: JsonValue) -> JsonValue:
    if isinstance(value, Mapping):
        return {str(key): _json_copy(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_json_copy(item) for item in value]
    return value
