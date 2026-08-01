"""Provider-prefixed model reference helpers."""

from dataclasses import dataclass
from typing import Protocol

# Each backup slot, the primary route its refs must differ from, and the env
# names both are configured under.
BACKUP_MODEL_SLOTS: tuple[tuple[str, str, str, str], ...] = (
    ("model_fable_backup", "model_fable", "MODEL_FABLE_BACKUP", "MODEL_FABLE"),
    ("model_opus_backup", "model_opus", "MODEL_OPUS_BACKUP", "MODEL_OPUS"),
    ("model_sonnet_backup", "model_sonnet", "MODEL_SONNET_BACKUP", "MODEL_SONNET"),
    ("model_haiku_backup", "model_haiku", "MODEL_HAIKU_BACKUP", "MODEL_HAIKU"),
    ("model_backup", "model", "MODEL_BACKUP", "MODEL"),
)


@dataclass(frozen=True, slots=True)
class ConfiguredChatModelRef:
    """A unique configured chat model reference."""

    model_ref: str
    provider_id: str
    model_id: str


@dataclass(frozen=True, slots=True)
class ConfiguredFailoverRoute:
    """One tier's primary route and the ordered chain that will back it up.

    ``backup_refs`` is the *effective* chain: a tier without its own chain
    inherits the global one, and any entry equal to the tier's own primary is
    dropped, so this mirrors what routing actually resolves.
    """

    primary_env: str
    backup_env: str
    primary_ref: str
    backup_refs: tuple[str, ...]
    inherited: bool = False


class ChatModelConfig(Protocol):
    model: str
    model_fable: str | None
    model_opus: str | None
    model_sonnet: str | None
    model_haiku: str | None
    model_fable_backup: str | None
    model_opus_backup: str | None
    model_sonnet_backup: str | None
    model_haiku_backup: str | None
    model_backup: str | None


def parse_provider_type(model_ref: str) -> str:
    """Extract provider type from any 'provider/model' string."""

    return model_ref.split("/", 1)[0]


def parse_model_name(model_ref: str) -> str:
    """Extract model name from any 'provider/model' string."""

    return model_ref.split("/", 1)[1]


def parse_model_ref_chain(value: str | None) -> tuple[str, ...]:
    """Split a comma-separated backup chain into ordered unique refs."""

    if not value:
        return ()
    return tuple(
        dict.fromkeys(part.strip() for part in value.split(",") if part.strip())
    )


def configured_failover_routes(
    settings: ChatModelConfig,
) -> tuple[ConfiguredFailoverRoute, ...]:
    """Return each tier's effective primary ref and its configured backup chain."""

    global_chain = parse_model_ref_chain(settings.model_backup)
    routes: list[ConfiguredFailoverRoute] = []
    for backup_attr, primary_attr, backup_env, primary_env in BACKUP_MODEL_SLOTS:
        own_chain = parse_model_ref_chain(getattr(settings, backup_attr))
        inherited = not own_chain and backup_attr != "model_backup"
        primary_ref = getattr(settings, primary_attr) or settings.model
        chain = global_chain if inherited else own_chain
        routes.append(
            ConfiguredFailoverRoute(
                primary_env=primary_env,
                backup_env="MODEL_BACKUP" if inherited else backup_env,
                primary_ref=primary_ref,
                backup_refs=tuple(ref for ref in chain if ref != primary_ref),
                inherited=inherited,
            )
        )
    return tuple(routes)


def configured_chat_model_refs(
    settings: ChatModelConfig,
) -> tuple[ConfiguredChatModelRef, ...]:
    """Return unique configured chat provider/model refs."""

    primaries = (
        settings.model,
        settings.model_fable,
        settings.model_opus,
        settings.model_sonnet,
        settings.model_haiku,
    )
    backups = (
        parse_model_ref_chain(getattr(settings, backup_attr))
        for backup_attr, *_ in BACKUP_MODEL_SLOTS
    )
    model_refs = dict.fromkeys(
        (
            *(model_ref for model_ref in primaries if model_ref is not None),
            *(model_ref for chain in backups for model_ref in chain),
        )
    )

    return tuple(
        ConfiguredChatModelRef(
            model_ref=model_ref,
            provider_id=parse_provider_type(model_ref),
            model_id=parse_model_name(model_ref),
        )
        for model_ref in model_refs
    )
