"""Picker alias selection in ``free_claude_code.api.model_catalog``."""

import pytest

from free_claude_code.api import model_catalog
from free_claude_code.core import gateway_model_ids as gmi


class _StubRuntime:
    def cached_model_supports_thinking(self, provider_id, model_id):
        return None

    def cached_prefixed_model_infos(self):
        return []

    def cached_model_ids(self):
        return {}


@pytest.fixture(autouse=True)
def _isolate_alias_state():
    gmi.clear_picker_aliases()
    yield
    gmi.clear_picker_aliases()


def _settings_with_refs(*refs):
    from free_claude_code.config.reasoning import ReasoningPreference
    from free_claude_code.config.settings import Settings

    settings = Settings()
    settings.model = None
    settings.model_fable = None
    settings.model_opus = None
    settings.model_sonnet = None
    settings.model_haiku = None
    settings.reasoning_policy = ReasoningPreference.CLIENT
    slots = ("model", "model_fable", "model_opus", "model_sonnet", "model_haiku")
    for slot, ref in zip(slots, refs, strict=False):
        setattr(settings, slot, ref)
    return settings


def _ids(response: model_catalog.ModelsListResponse) -> list[str]:
    return [item.id for item in response.data]


def test_unseeded_state_uses_gateway_wrapper_ids():
    settings = _settings_with_refs("nvidia_nim/01-ai/yi-large")

    response = model_catalog.build_models_list_response(settings, _StubRuntime())

    picker_ids = [
        item.id for item in response.data if item.id.startswith("claude-sonnet-nim")
    ]
    assert picker_ids == []
    assert "anthropic/nvidia_nim/01-ai/yi-large" in _ids(response)
    assert "claude-3-freecc-no-thinking/nvidia_nim/01-ai/yi-large" in _ids(response)


def test_seeded_state_prefers_alias_for_thinking_variant():
    gmi.seed_picker_aliases(["nvidia_nim/01-ai/yi-large"])
    settings = _settings_with_refs("nvidia_nim/01-ai/yi-large")

    response = model_catalog.build_models_list_response(settings, _StubRuntime())

    ids = _ids(response)
    assert "claude-sonnet-nim-0001" in ids
    assert "claude-sonnet-nim-0001-no-thinking" in ids
    # Wrapper id is suppressed because the alias covers the same slot.
    assert "anthropic/nvidia_nim/01-ai/yi-large" not in ids


def test_seed_only_nim_uses_wrapper_for_other_providers():
    gmi.seed_picker_aliases(["nvidia_nim/01-ai/yi-large"])
    settings = _settings_with_refs("openai_chat/anthropic/claude-3-5-sonnet-20241022")

    response = model_catalog.build_models_list_response(settings, _StubRuntime())

    ids = _ids(response)
    # Non-NIM ref falls through to the original wrapper ids.
    assert "anthropic/openai_chat/anthropic/claude-3-5-sonnet-20241022" in ids
    assert (
        "claude-3-freecc-no-thinking/openai_chat/anthropic/claude-3-5-sonnet-20241022"
        in ids
    )
    # And no alias leaked from the seeded NIM ref into the unrelated entry.
    assert all(not item.startswith("claude-sonnet-nim") for item in ids)


def test_display_name_is_provider_model_ref_when_using_alias():
    gmi.seed_picker_aliases(["nvidia_nim/01-ai/yi-large"])
    settings = _settings_with_refs("nvidia_nim/01-ai/yi-large")

    response = model_catalog.build_models_list_response(settings, _StubRuntime())

    alias_entries = [
        item for item in response.data if item.id.startswith("claude-sonnet-nim-0001")
    ]
    # Both thinking and no-thinking entries expose the canonical provider ref
    # as the human-readable label.
    assert {item.display_name for item in alias_entries} == {
        "nvidia_nim/01-ai/yi-large",
    }
