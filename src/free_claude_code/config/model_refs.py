"""Provider-prefixed model reference helpers."""

from dataclasses import dataclass
from typing import Protocol


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
    model_fallbacks: tuple[str, ...] | None


def split_provider_model_ref(model_ref: str) -> tuple[str, str]:
    """Split one complete ``provider/model`` reference."""

    provider_id, separator, model_id = model_ref.partition("/")
    if not separator or not provider_id or not model_id:
        raise ValueError("Model reference must contain provider and model names.")
    return provider_id, model_id


def parse_provider_type(model_ref: str) -> str:
    """Extract provider type from any 'provider/model' string."""

    return split_provider_model_ref(model_ref)[0]


def parse_model_name(model_ref: str) -> str:
    """Extract model name from any 'provider/model' string."""

    return split_provider_model_ref(model_ref)[1]


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
            *(settings.model_fallbacks or ()),
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
