"""Gateway-safe model ID encoding shared by API and CLI adapters."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

GATEWAY_MODEL_ID_PREFIX = "anthropic"

# Claude Code currently treats any model id containing ``claude-3-`` as not
# supporting thinking. This intentionally uses that client-side capability
# heuristic while keeping the real provider/model ref reversible for routing.
NO_THINKING_GATEWAY_MODEL_ID_PREFIX = "claude-3-freecc-no-thinking"

PICKER_ALIAS_PREFIX = "claude-sonnet-nim"
_PICKER_ALIAS_NO_THINKING_SUFFIX = "-no-thinking"
_PICKER_ALIAS_WIDTH = 4


@dataclass(frozen=True, slots=True)
class _PickerAliasMaps:
    """Immutable alias snapshot shared by catalog output and request routing.

    Published as a single module-global assignment so concurrent readers can
    never observe a torn state mixing refs from two different inventories.
    """

    alias_to_ref: Mapping[str, str]
    alias_to_ref_no_thinking: Mapping[str, str]
    ref_to_alias: Mapping[str, str]
    ref_to_alias_no_thinking: Mapping[str, str]


_EMPTY_PICKER_ALIAS_MAPS = _PickerAliasMaps({}, {}, {}, {})

# Seeded at startup with the sorted refs from the runtime cached model catalog,
# then republished whenever the model cache is mutated. Tests call
# ``clear_picker_aliases()`` between runs to restore the inert empty snapshot.
_picker_aliases = _EMPTY_PICKER_ALIAS_MAPS


@dataclass(frozen=True, slots=True)
class DecodedGatewayModelId:
    provider_id: str
    provider_model: str
    force_reasoning_off: bool = False


def gateway_model_id(provider_model_ref: str) -> str:
    """Return the normal Claude Code-discoverable id for a provider/model ref."""
    return f"{GATEWAY_MODEL_ID_PREFIX}/{provider_model_ref}"


def no_thinking_gateway_model_id(provider_model_ref: str) -> str:
    """Return a Claude Code-discoverable id that disables client thinking."""
    return f"{NO_THINKING_GATEWAY_MODEL_ID_PREFIX}/{provider_model_ref}"


def decode_gateway_model_id(model_name: str) -> DecodedGatewayModelId | None:
    """Decode a model id advertised by this gateway, if it is one."""
    prefix, separator, remainder = model_name.partition("/")
    if not separator:
        return None

    if prefix == GATEWAY_MODEL_ID_PREFIX:
        force_reasoning_off = False
    elif prefix == NO_THINKING_GATEWAY_MODEL_ID_PREFIX:
        force_reasoning_off = True
    else:
        return None

    provider_id, provider_separator, provider_model = remainder.partition("/")
    if not provider_separator or not provider_model:
        return None

    return DecodedGatewayModelId(
        provider_id=provider_id,
        provider_model=provider_model,
        force_reasoning_off=force_reasoning_off,
    )


def _format_picker_alias(index: int, *, no_thinking: bool) -> str:
    suffix = _PICKER_ALIAS_NO_THINKING_SUFFIX if no_thinking else ""
    return f"{PICKER_ALIAS_PREFIX}-{index:0{_PICKER_ALIAS_WIDTH}d}{suffix}"


def seed_picker_aliases(provider_model_refs: Iterable[str]) -> None:
    """Publish a fresh immutable alias snapshot built from the supplied refs.

    Stable across restarts: counter is assigned by ``sorted(refs)``, so the
    same input order produces the same numbering.

    The snapshot stays empty when the iterable is empty so a cold-start
    ``/v1/models`` request still falls back to the canonical
    ``gateway_model_id`` wrappers.
    """
    thinking_aliases: dict[str, str] = {}
    no_thinking_aliases: dict[str, str] = {}
    ref_to_thinking: dict[str, str] = {}
    ref_to_no_thinking: dict[str, str] = {}

    for index, ref in enumerate(sorted(provider_model_refs), start=1):
        if not ref:
            continue
        alias = _format_picker_alias(index, no_thinking=False)
        no_thinking_alias = _format_picker_alias(index, no_thinking=True)
        thinking_aliases[alias] = ref
        no_thinking_aliases[no_thinking_alias] = ref
        ref_to_thinking[ref] = alias
        ref_to_no_thinking[ref] = no_thinking_alias

    global _picker_aliases
    _picker_aliases = _PickerAliasMaps(
        alias_to_ref=thinking_aliases,
        alias_to_ref_no_thinking=no_thinking_aliases,
        ref_to_alias=ref_to_thinking,
        ref_to_alias_no_thinking=ref_to_no_thinking,
    )


def picker_alias_for(
    provider_model_ref: str, *, force_reasoning_off: bool = False
) -> str | None:
    """Return the picker alias for ``provider_model_ref``, if seeded."""
    maps = _picker_aliases
    if force_reasoning_off:
        return maps.ref_to_alias_no_thinking.get(provider_model_ref)
    return maps.ref_to_alias.get(provider_model_ref)


def resolve_picker_alias(model_name: str) -> tuple[str, bool] | None:
    """Reverse-lookup alias. Returns ``(provider_ref, force_reasoning_off)``."""
    if not model_name:
        return None
    maps = _picker_aliases
    ref = maps.alias_to_ref_no_thinking.get(model_name)
    if ref is not None:
        return ref, True
    ref = maps.alias_to_ref.get(model_name)
    if ref is not None:
        return ref, False
    return None


def has_picker_aliases() -> bool:
    """Whether ``seed_picker_aliases`` has populated the snapshot."""
    maps = _picker_aliases
    return bool(maps.alias_to_ref) or bool(maps.alias_to_ref_no_thinking)


def clear_picker_aliases() -> None:
    """Drop every alias entry. Used by tests and hardening paths."""
    global _picker_aliases
    _picker_aliases = _EMPTY_PICKER_ALIAS_MAPS
