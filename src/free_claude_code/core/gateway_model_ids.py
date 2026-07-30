"""Gateway-safe model ID encoding shared by API and CLI adapters."""

from collections.abc import Iterable
from dataclasses import dataclass

GATEWAY_MODEL_ID_PREFIX = "anthropic"

# Claude Code currently treats any model id containing ``claude-3-`` as not
# supporting thinking. This intentionally uses that client-side capability
# heuristic while keeping the real provider/model ref reversible for routing.
NO_THINKING_GATEWAY_MODEL_ID_PREFIX = "claude-3-freecc-no-thinking"

PICKER_ALIAS_PREFIX = "claude-sonnet-nim"
_PICKER_ALIAS_NO_THINKING_SUFFIX = "-no-thinking"
_PICKER_ALIAS_WIDTH = 4

# Module-scope alias maps. Seeded once at startup with the sorted refs from the
# runtime cached model catalog. Read-only after seed in production; tests
# ``clear_picker_aliases()`` between runs to keep the maps empty.
_picker_alias_to_ref: dict[str, str] = {}
_picker_alias_to_ref_no_thinking: dict[str, str] = {}
_picker_ref_to_alias: dict[str, str] = {}
_picker_ref_to_alias_no_thinking: dict[str, str] = {}


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
    """Reset and rebuild the alias maps from the supplied refs.

    Stable across restarts: counter is assigned by ``sorted(refs)``, so the
    same input order produces the same numbering.

    Maps stay empty when the iterable is empty so a cold-start ``/v1/models``
    request still falls back to the canonical ``gateway_model_id`` wrappers.
    """
    _seed_picker_aliases_internal(provider_model_refs, append=False)


def extend_picker_aliases(provider_model_refs: Iterable[str]) -> None:
    """Add aliases for new refs only; preserve existing alias assignments.

    Used by ``refresh_models`` to append newly discovered models without
    remapping aliases that Claude Desktop may have already cached.
    """
    _seed_picker_aliases_internal(provider_model_refs, append=True)


def _seed_picker_aliases_internal(
    provider_model_refs: Iterable[str], *, append: bool
) -> None:
    """Internal helper for both seed and extend modes."""
    global _picker_alias_to_ref, _picker_alias_to_ref_no_thinking
    global _picker_ref_to_alias, _picker_ref_to_alias_no_thinking

    if not append:
        _picker_alias_to_ref = {}
        _picker_alias_to_ref_no_thinking = {}
        _picker_ref_to_alias = {}
        _picker_ref_to_alias_no_thinking = {}

    # Determine next counter value from existing maps
    next_index = len(_picker_ref_to_alias) + 1

    for ref in sorted(provider_model_refs):
        if not ref:
            continue
        # Skip if already has an alias assigned
        if ref in _picker_ref_to_alias:
            continue
        alias = _format_picker_alias(next_index, no_thinking=False)
        no_thinking_alias = _format_picker_alias(next_index, no_thinking=True)
        _picker_alias_to_ref[alias] = ref
        _picker_alias_to_ref_no_thinking[no_thinking_alias] = ref
        _picker_ref_to_alias[ref] = alias
        _picker_ref_to_alias_no_thinking[ref] = no_thinking_alias
        next_index += 1


def picker_alias_for(
    provider_model_ref: str, *, force_reasoning_off: bool = False
) -> str | None:
    """Return the picker alias for ``provider_model_ref``, if seeded."""
    if force_reasoning_off:
        return _picker_ref_to_alias_no_thinking.get(provider_model_ref)
    return _picker_ref_to_alias.get(provider_model_ref)


def resolve_picker_alias(model_name: str) -> tuple[str, bool] | None:
    """Reverse-lookup alias. Returns ``(provider_ref, force_reasoning_off)``."""
    if not model_name:
        return None
    ref = _picker_alias_to_ref_no_thinking.get(model_name)
    if ref is not None:
        return ref, True
    ref = _picker_alias_to_ref.get(model_name)
    if ref is not None:
        return ref, False
    return None


def has_picker_aliases() -> bool:
    """Whether ``seed_picker_aliases`` has populated the maps."""
    return bool(_picker_alias_to_ref) or bool(_picker_alias_to_ref_no_thinking)


def clear_picker_aliases() -> None:
    """Drop every alias entry. Used by tests and hardening paths."""
    global _picker_alias_to_ref, _picker_alias_to_ref_no_thinking
    global _picker_ref_to_alias, _picker_ref_to_alias_no_thinking
    _picker_alias_to_ref = {}
    _picker_alias_to_ref_no_thinking = {}
    _picker_ref_to_alias = {}
    _picker_ref_to_alias_no_thinking = {}
