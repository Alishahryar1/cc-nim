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

# Sticky ref-to-alias assignments that survive reseeds: once an alias has been
# advertised for a ref, that alias is never re-pointed to a different ref, so
# a desktop picker holding a stale catalog cannot silently route to another
# model. Numbers of retired refs stay consumed; a returning ref regains its
# previous alias. Reset only by ``clear_picker_aliases()``.
_sticky_ref_to_alias: dict[str, str] = {}
_used_alias_numbers: set[int] = set()


def _next_unused_alias_number(used_numbers: set[int]) -> int:
    candidate = 1
    while candidate in used_numbers:
        candidate += 1
    return candidate


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

    Aliases are stable across reseeds: a ref that already holds an alias keeps
    it, and new refs take previously unused numbers, so an alias already shown
    in a client picker can never silently resolve to a different model.
    Numbering is deterministic on restart when starting from the same initial
    inventory (counters assigned by ``sorted(refs)`` order).

    An empty inventory publishes the inert empty snapshot so a cold-start
    ``/v1/models`` request still falls back to the canonical
    ``gateway_model_id`` wrappers.
    """
    global _picker_aliases, _sticky_ref_to_alias, _used_alias_numbers

    wanted = [ref for ref in sorted(set(provider_model_refs)) if ref]
    if not wanted:
        _sticky_ref_to_alias = {}
        _used_alias_numbers = set()
        _picker_aliases = _EMPTY_PICKER_ALIAS_MAPS
        return

    thinking_aliases: dict[str, str] = {}
    no_thinking_aliases: dict[str, str] = {}
    ref_to_thinking: dict[str, str] = {}
    ref_to_no_thinking: dict[str, str] = {}
    used_numbers = set(_used_alias_numbers)

    for ref in wanted:
        alias = _sticky_ref_to_alias.get(ref)
        if alias is None:
            number = _next_unused_alias_number(used_numbers)
            used_numbers.add(number)
            alias = _format_picker_alias(number, no_thinking=False)
        thinking_aliases[alias] = ref
        alias_no_thinking = f"{alias}{_PICKER_ALIAS_NO_THINKING_SUFFIX}"
        no_thinking_aliases[alias_no_thinking] = ref
        ref_to_thinking[ref] = alias
        ref_to_no_thinking[ref] = alias_no_thinking

    _sticky_ref_to_alias = {**_sticky_ref_to_alias, **ref_to_thinking}
    _used_alias_numbers = used_numbers
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
    """Drop every alias entry and sticky assignment. Tests and hardening."""
    global _picker_aliases, _sticky_ref_to_alias, _used_alias_numbers
    _picker_aliases = _EMPTY_PICKER_ALIAS_MAPS
    _sticky_ref_to_alias = {}
    _used_alias_numbers = set()
