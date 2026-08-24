"""Route-scoped replay identity for OpenAI-compatible transports."""

import hashlib

from free_claude_code.core.inference import ReplayCompatibilityScope


def openai_replay_scope(
    provider_identity: str,
    provider_model: str,
    *,
    replay_format: str,
) -> ReplayCompatibilityScope:
    """Return an opaque deterministic scope for one replay-compatible route."""

    material = "\0".join((provider_identity, provider_model, replay_format))
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
    return ReplayCompatibilityScope(f"openai-{digest}")
