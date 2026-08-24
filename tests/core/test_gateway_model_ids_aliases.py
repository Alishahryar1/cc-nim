"""Tests for picker aliasing in ``free_claude_code.core.gateway_model_ids``."""

import pytest

from free_claude_code.core import gateway_model_ids as gmi


@pytest.fixture(autouse=True)
def _isolate_alias_state():
    """Reset module-scope alias maps before/after each test."""
    gmi.clear_picker_aliases()
    yield
    gmi.clear_picker_aliases()


def test_picker_alias_prefix_is_anthropic_safe():
    assert gmi.PICKER_ALIAS_PREFIX == "claude-sonnet-nim"


def test_has_picker_aliases_false_before_seed():
    assert gmi.has_picker_aliases() is False


def test_seed_populates_maps_and_marks_seeded():
    gmi.seed_picker_aliases(
        [
            "nvidia_nim/01-ai/yi-large",
            "nvidia_nim/meta/llama-3.3-70b-instruct",
        ]
    )

    assert gmi.has_picker_aliases() is True


def test_alias_counter_is_deterministic_from_sorted_input():
    first = ["nvidia_nim/z-provider/z-model", "nvidia_nim/a-provider/a-model"]
    second = list(reversed(first))

    gmi.seed_picker_aliases(first)
    a_first = gmi.picker_alias_for("nvidia_nim/a-provider/a-model")
    gmi.clear_picker_aliases()

    gmi.seed_picker_aliases(second)
    a_second = gmi.picker_alias_for("nvidia_nim/a-provider/a-model")

    assert a_first == a_second == "claude-sonnet-nim-0001"
    assert (
        gmi.picker_alias_for("nvidia_nim/z-provider/z-model")
        == "claude-sonnet-nim-0002"
    )


def test_alias_lookup_thinking_and_no_thinking_variants():
    gmi.seed_picker_aliases(["nvidia_nim/01-ai/yi-large"])

    assert gmi.picker_alias_for("nvidia_nim/01-ai/yi-large") == "claude-sonnet-nim-0001"
    assert (
        gmi.picker_alias_for("nvidia_nim/01-ai/yi-large", force_reasoning_off=True)
        == "claude-sonnet-nim-0001-no-thinking"
    )


def test_alias_lookup_unknown_ref_returns_none():
    gmi.seed_picker_aliases(["nvidia_nim/01-ai/yi-large"])

    assert gmi.picker_alias_for("nvidia_nim/missing/model") is None
    assert (
        gmi.picker_alias_for("nvidia_nim/missing/model", force_reasoning_off=True)
        is None
    )


def test_resolve_picker_alias_round_trip():
    gmi.seed_picker_aliases(["nvidia_nim/meta/llama-3.3-70b-instruct"])

    assert gmi.resolve_picker_alias("claude-sonnet-nim-0001") == (
        "nvidia_nim/meta/llama-3.3-70b-instruct",
        False,
    )
    assert gmi.resolve_picker_alias("claude-sonnet-nim-0001-no-thinking") == (
        "nvidia_nim/meta/llama-3.3-70b-instruct",
        True,
    )


def test_resolve_picker_alias_unknown_returns_none():
    gmi.seed_picker_aliases(["nvidia_nim/01-ai/yi-large"])

    assert gmi.resolve_picker_alias("claude-sonnet-nim-9999") is None
    assert gmi.resolve_picker_alias("claude-sonnet-nim-9999-no-thinking") is None
    assert gmi.resolve_picker_alias("claude-haiku-nim-0001") is None


def test_resolve_picker_alias_for_prefix_only_when_seeded_empty():
    # Maps are empty by default (cold-start window).
    assert gmi.resolve_picker_alias("claude-sonnet-nim-0001") is None


def test_clear_resets_state():
    gmi.seed_picker_aliases(["nvidia_nim/01-ai/yi-large"])
    assert gmi.has_picker_aliases() is True

    gmi.clear_picker_aliases()

    assert gmi.has_picker_aliases() is False
    assert gmi.picker_alias_for("nvidia_nim/01-ai/yi-large") is None
    assert gmi.resolve_picker_alias("claude-sonnet-nim-0001") is None


def test_seed_empty_iterable_keeps_state_empty():
    gmi.seed_picker_aliases([])

    assert gmi.has_picker_aliases() is False


def test_seed_replaces_previous_aliases():
    gmi.seed_picker_aliases(["nvidia_nim/01-ai/yi-large"])
    assert gmi.picker_alias_for("nvidia_nim/01-ai/yi-large") == "claude-sonnet-nim-0001"

    gmi.seed_picker_aliases(["nvidia_nim/meta/llama-3.3-70b-instruct"])

    assert gmi.picker_alias_for("nvidia_nim/01-ai/yi-large") is None
    assert (
        gmi.picker_alias_for("nvidia_nim/meta/llama-3.3-70b-instruct")
        == "claude-sonnet-nim-0001"
    )


def test_aliases_are_four_digit_zero_padded():
    refs = [f"openai_chat/provider-{i}/model" for i in range(15)]
    gmi.seed_picker_aliases(refs)

    aliases = [
        alias for ref in refs if (alias := gmi.picker_alias_for(ref)) is not None
    ]

    assert len(aliases) == 15
    for alias in aliases:
        # All aliases share the same prefix and 4-digit counter width.
        assert alias.startswith("claude-sonnet-nim-")
        counter = alias.removeprefix("claude-sonnet-nim-").split("-")[0]
        assert len(counter) == 4
        assert counter.isdigit()


def test_seed_assigns_unique_aliases_per_ref():
    refs = [f"openai_chat/provider-{i}/model" for i in range(10)]
    gmi.seed_picker_aliases(refs)

    aliases = [gmi.picker_alias_for(ref) for ref in refs]

    assert len(set(aliases)) == len(refs)
    assert all(alias is not None for alias in aliases)
