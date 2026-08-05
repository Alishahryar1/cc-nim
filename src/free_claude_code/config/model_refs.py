"""Provider-prefixed model reference helpers."""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol

# Internal marker used to persist a provider that is restricted to zero models
# (i.e. disabled) in the FCC_PROVIDER_MODEL_ALLOWLIST env value without colliding
# with the "all models allowed" absent state.
RESTRICTED_EMPTY_MARKER = "__fcc_no_models__"


@dataclass(frozen=True, slots=True)
class ConfiguredChatModelRef:
    """A unique configured chat model reference."""

    model_ref: str
    provider_id: str
    model_id: str


class ChatModelConfig(Protocol):
    model: str
    model_fable: str | None
    model_opus: str | None
    model_sonnet: str | None
    model_haiku: str | None


def parse_provider_type(model_ref: str) -> str:
    """Extract provider type from any 'provider/model' string."""

    return model_ref.split("/", 1)[0]


def parse_model_name(model_ref: str) -> str:
    """Extract model name from any 'provider/model' string."""

    return model_ref.split("/", 1)[1]


def configured_chat_model_refs(
    settings: ChatModelConfig,
) -> tuple[ConfiguredChatModelRef, ...]:
    """Return unique configured chat provider/model refs."""

    model_refs = dict.fromkeys(
        model_ref
        for model_ref in (
            settings.model,
            settings.model_fable,
            settings.model_opus,
            settings.model_sonnet,
            settings.model_haiku,
        )
        if model_ref is not None
    )

    return tuple(
        ConfiguredChatModelRef(
            model_ref=model_ref,
            provider_id=parse_provider_type(model_ref),
            model_id=parse_model_name(model_ref),
        )
        for model_ref in model_refs
    )


def model_ref_allowed(
    settings: Any,
    provider_model_ref: str,
    *,
    alt_ids: Iterable[str] = (),
) -> bool:
    """Return whether a provider model reference passes allowlist filtering.

    Semantics:
    - If the provider_id has a per-provider allowlist, the model_name must be in it.
    - Otherwise, if the global model_allowlist is empty, all models are allowed.
    - Otherwise, the provider_model_ref or any alt_id must be in the global allowlist.
    """
    provider_id, separator, model_name = provider_model_ref.partition("/")
    provider_allowlists = settings.provider_model_allowlists
    if provider_id in provider_allowlists:
        if not separator:
            return False
        return model_name in provider_allowlists[provider_id]

    allowlist = settings.model_allowlist
    if not allowlist:
        return True
    return provider_model_ref in allowlist or any(alt in allowlist for alt in alt_ids)
