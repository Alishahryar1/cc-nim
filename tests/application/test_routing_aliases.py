"""Picker alias decoding routed through ``application.routing.ModelRouter``."""

import pytest

from free_claude_code.application.routing import ModelRouter
from free_claude_code.config.reasoning import ReasoningPreference
from free_claude_code.config.settings import Settings
from free_claude_code.core import gateway_model_ids as gmi


@pytest.fixture
def settings():
    settings = Settings()
    settings.model = "nvidia_nim/fallback-model"
    settings.model_fable = None
    settings.model_opus = None
    settings.model_sonnet = None
    settings.model_haiku = None
    settings.reasoning_policy = ReasoningPreference.CLIENT
    return settings


@pytest.fixture(autouse=True)
def _isolate_alias_state():
    gmi.clear_picker_aliases()
    yield
    gmi.clear_picker_aliases()


def test_resolves_alias_before_gateway_prefix(settings):
    gmi.seed_picker_aliases(["nvidia_nim/meta/llama-3.3-70b-instruct"])

    resolved = ModelRouter(settings).resolve("claude-sonnet-nim-0001")

    assert resolved.primary.provider_id == "nvidia_nim"
    assert resolved.primary.provider_model == "meta/llama-3.3-70b-instruct"
    assert resolved.original_model == "claude-sonnet-nim-0001"
    assert resolved.reasoning_preference is ReasoningPreference.CLIENT


def test_resolves_alias_no_thinking_variant_forces_reasoning_off(settings):
    gmi.seed_picker_aliases(["nvidia_nim/meta/llama-3.3-70b-instruct"])

    resolved = ModelRouter(settings).resolve("claude-sonnet-nim-0001-no-thinking")

    assert resolved.primary.provider_id == "nvidia_nim"
    assert resolved.primary.provider_model == "meta/llama-3.3-70b-instruct"
    assert resolved.reasoning_preference is ReasoningPreference.OFF


def test_alias_takes_priority_over_anthropic_prefix(settings):
    # The same underlying ref exists both as an alias and as an
    # ``anthropic/<ref>`` gateway id. The alias must win on the picker shape,
    # and both code paths resolve to the same upstream model.
    gmi.seed_picker_aliases(["nvidia_nim/meta/llama-3.3-70b-instruct"])

    alias_resolved = ModelRouter(settings).resolve("claude-sonnet-nim-0001")
    gateway_resolved = ModelRouter(settings).resolve(
        "anthropic/nvidia_nim/meta/llama-3.3-70b-instruct"
    )

    assert (
        alias_resolved.primary.provider_id
        == gateway_resolved.primary.provider_id
        == "nvidia_nim"
    )
    assert (
        alias_resolved.primary.provider_model
        == gateway_resolved.primary.provider_model
        == "meta/llama-3.3-70b-instruct"
    )


def test_alias_for_unknown_provider_falls_through_to_existing_chain(settings):
    # An alias pointing at an unsupported provider must not silently match;
    # the alias resolver returns the ref and routing layer validates it.
    # After validation, ``resolve`` falls through to the existing route
    # table, so ``claude-sonnet-nim-NNNN`` lands on the configured fallback.
    gmi.seed_picker_aliases(["bogus_provider/some-model"])

    resolved = ModelRouter(settings).resolve("claude-sonnet-nim-0001")

    # Not the bogus provider we accidentally aliased to.
    assert resolved.primary.provider_id != "bogus_provider"
    # Falls through to the configured default.
    assert resolved.primary.provider_id == "nvidia_nim"
    assert resolved.primary.provider_model == "fallback-model"


def test_unknown_alias_falls_through_to_existing_routing(settings):
    gmi.seed_picker_aliases(["nvidia_nim/01-ai/yi-large"])

    resolved = ModelRouter(settings).resolve("claude-sonnet-nim-9999")

    # 9999 is not a seeded alias; the routing layer should fall through.
    # With no alias match, unknown alias falls back to model name parsing,
    # which raises because ``claude-sonnet-nim-9999`` doesn't match a route.
    assert resolved.original_model == "claude-sonnet-nim-9999"


def test_unseeded_alias_returns_none_from_resolver(settings):
    # Cold-start path: no aliases seeded, alias-shaped id should not produce
    # a routing hit. Falls through to the existing fallback chain.
    resolved = ModelRouter(settings).resolve("claude-sonnet-nim-0500")

    # Falls through and lands on the configured default.
    assert resolved.primary.provider_id == "nvidia_nim"
    assert resolved.primary.provider_model == "fallback-model"
