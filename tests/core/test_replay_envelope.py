import base64
import json
from collections.abc import Mapping

import pytest

from free_claude_code.core.inference import (
    ReplayArtifact,
    ReplayArtifactKind,
    ReplayArtifactOrigin,
    ReplayAttachment,
    ReplayCompatibilityScope,
)
from free_claude_code.core.replay_envelope import (
    REPLAY_ENVELOPE_PREFIX,
    ReplayEnvelopeError,
    decode_replay_envelope,
    encode_replay_envelope,
)


def _carrier(payload: object) -> str:
    raw = json.dumps(payload, separators=(",", ":")).encode()
    token = base64.urlsafe_b64encode(raw).decode().rstrip("=")
    return f"{REPLAY_ENVELOPE_PREFIX}{token}"


def _artifact_payload(
    *,
    origin: str = "openai",
    kind: str = "encrypted_reasoning",
    attachment: str = "reasoning",
    scope: object = "openai_responses:model",
    payload: object = "opaque",
) -> dict[str, object]:
    return {
        "origin": origin,
        "kind": kind,
        "attachment": attachment,
        "scope": scope,
        "payload": payload,
    }


@pytest.mark.parametrize(
    ("kind", "attachment"),
    [
        (ReplayArtifactKind.THINKING_SIGNATURE, ReplayAttachment.REASONING),
        (ReplayArtifactKind.REDACTED_THINKING, ReplayAttachment.REASONING),
        (ReplayArtifactKind.ENCRYPTED_REASONING, ReplayAttachment.REASONING),
        (ReplayArtifactKind.REASONING_DETAILS, ReplayAttachment.REASONING),
        (ReplayArtifactKind.THOUGHT_SIGNATURE, ReplayAttachment.TOOL_CALL),
        (ReplayArtifactKind.TOOL_EXTRA_CONTENT, ReplayAttachment.TOOL_CALL),
    ],
)
def test_each_replay_artifact_kind_round_trips(
    kind: ReplayArtifactKind,
    attachment: ReplayAttachment,
) -> None:
    artifact = ReplayArtifact(
        origin=ReplayArtifactOrigin.OPENAI_COMPATIBLE,
        kind=kind,
        attachment=attachment,
        scope=ReplayCompatibilityScope("provider:model"),
        payload={"opaque": [kind.value, 1, True, None]},
    )

    assert decode_replay_envelope(
        encode_replay_envelope((artifact,)), attachment=attachment
    ) == (artifact,)


@pytest.mark.parametrize(
    "attachment",
    [ReplayAttachment.REASONING, ReplayAttachment.TOOL_CALL],
)
def test_multiple_artifacts_keep_order_and_encode_deterministically(
    attachment: ReplayAttachment,
) -> None:
    origins = tuple(ReplayArtifactOrigin)
    kinds = (
        (
            ReplayArtifactKind.THINKING_SIGNATURE,
            ReplayArtifactKind.REDACTED_THINKING,
            ReplayArtifactKind.ENCRYPTED_REASONING,
            ReplayArtifactKind.REASONING_DETAILS,
        )
        if attachment is ReplayAttachment.REASONING
        else (
            ReplayArtifactKind.THOUGHT_SIGNATURE,
            ReplayArtifactKind.TOOL_EXTRA_CONTENT,
        )
    )
    artifacts = tuple(
        ReplayArtifact(
            origin=origins[index % len(origins)],
            kind=kind,
            attachment=attachment,
            scope=(ReplayCompatibilityScope("provider:model") if index else None),
            payload={"index": index},
        )
        for index, kind in enumerate(kinds)
    )

    first = encode_replay_envelope(artifacts)
    second = encode_replay_envelope(artifacts)

    assert first == second
    assert decode_replay_envelope(first, attachment=attachment) == artifacts


def test_legacy_raw_replay_value_is_not_claimed_by_the_envelope_codec() -> None:
    assert (
        decode_replay_envelope(
            "provider-owned-opaque-value",
            attachment=ReplayAttachment.REASONING,
        )
        is None
    )


def test_empty_artifact_sequence_cannot_be_encoded() -> None:
    with pytest.raises(ValueError, match="at least one artifact"):
        encode_replay_envelope(())


@pytest.mark.parametrize(
    ("value", "match"),
    [
        (REPLAY_ENVELOPE_PREFIX, "payload is empty"),
        (f"{REPLAY_ENVELOPE_PREFIX}*", "valid base64url"),
        (
            f"{REPLAY_ENVELOPE_PREFIX}"
            + base64.urlsafe_b64encode(b"\xff").decode().rstrip("="),
            "valid UTF-8 JSON",
        ),
        (
            f"{REPLAY_ENVELOPE_PREFIX}"
            + base64.urlsafe_b64encode(b"{").decode().rstrip("="),
            "valid UTF-8 JSON",
        ),
        (_carrier([]), "top-level schema"),
        (_carrier({"version": 1}), "top-level schema"),
        (_carrier({"version": 2, "artifacts": []}), "version is unsupported"),
        (
            _carrier({"version": 1, "artifacts": []}),
            "artifacts must be a non-empty list",
        ),
        (
            _carrier({"version": 1, "artifacts": ["opaque"]}),
            "artifacts\\[0\\] has an invalid schema",
        ),
        (
            _carrier(
                {
                    "version": 1,
                    "artifacts": [{**_artifact_payload(), "extra": True}],
                }
            ),
            "artifacts\\[0\\] has an invalid schema",
        ),
        (
            _carrier(
                {
                    "version": 1,
                    "artifacts": [_artifact_payload(origin="unknown")],
                }
            ),
            "unknown artifact value",
        ),
        (
            _carrier(
                {
                    "version": 1,
                    "artifacts": [_artifact_payload(kind="unknown")],
                }
            ),
            "unknown artifact value",
        ),
        (
            _carrier(
                {
                    "version": 1,
                    "artifacts": [_artifact_payload(attachment="unknown")],
                }
            ),
            "unknown artifact value",
        ),
        (
            _carrier(
                {
                    "version": 1,
                    "artifacts": [_artifact_payload(scope="")],
                }
            ),
            "scope must be null or a non-empty string",
        ),
        (
            _carrier(
                {
                    "version": 1,
                    "artifacts": [_artifact_payload(scope=42)],
                }
            ),
            "scope must be null or a non-empty string",
        ),
        (
            _carrier(
                {
                    "version": 1,
                    "artifacts": [_artifact_payload(payload=float("nan"))],
                }
            ),
            "payload is not valid JSON",
        ),
    ],
)
def test_malformed_envelopes_fail_with_stable_safe_errors(
    value: str,
    match: str,
) -> None:
    with pytest.raises(ReplayEnvelopeError, match=match):
        decode_replay_envelope(value, attachment=ReplayAttachment.REASONING)


def test_carrier_attachment_must_match_every_artifact() -> None:
    value = _carrier(
        {
            "version": 1,
            "artifacts": [_artifact_payload(attachment="tool_call")],
        }
    )

    with pytest.raises(ReplayEnvelopeError, match="does not match its carrier"):
        decode_replay_envelope(value, attachment=ReplayAttachment.REASONING)


def test_encode_rejects_non_finite_json_numbers() -> None:
    artifact = ReplayArtifact(
        origin=ReplayArtifactOrigin.OPENAI,
        kind=ReplayArtifactKind.ENCRYPTED_REASONING,
        attachment=ReplayAttachment.REASONING,
        payload=float("inf"),
    )

    with pytest.raises(ValueError, match="Out of range float values"):
        encode_replay_envelope((artifact,))


def test_encoded_and_decoded_size_limits_are_enforced() -> None:
    artifact = ReplayArtifact(
        origin=ReplayArtifactOrigin.OPENAI,
        kind=ReplayArtifactKind.ENCRYPTED_REASONING,
        attachment=ReplayAttachment.REASONING,
        payload="x" * 1_048_576,
    )
    with pytest.raises(ValueError, match="size limit"):
        encode_replay_envelope((artifact,))

    oversized = base64.urlsafe_b64encode(b"x" * 1_048_577).decode().rstrip("=")
    with pytest.raises(ReplayEnvelopeError, match="size limit"):
        decode_replay_envelope(
            f"{REPLAY_ENVELOPE_PREFIX}{oversized}",
            attachment=ReplayAttachment.REASONING,
        )


def test_errors_and_logs_do_not_expose_replay_payload(
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "private-replay-material"
    raw_artifact = _artifact_payload(payload={"secret": secret})
    assert isinstance(raw_artifact, Mapping)
    value = _carrier(
        {
            "version": 1,
            "artifacts": [{**raw_artifact, "unexpected": True}],
        }
    )

    with pytest.raises(ReplayEnvelopeError) as captured:
        decode_replay_envelope(value, attachment=ReplayAttachment.REASONING)

    assert secret not in str(captured.value)
    assert secret not in caplog.text
