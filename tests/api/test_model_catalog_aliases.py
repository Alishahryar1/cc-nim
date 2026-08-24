"""Picker alias selection in ``free_claude_code.api.model_catalog``."""

import pytest

from free_claude_code.api import model_catalog
from free_claude_code.application.ports import RequestRuntimeLease
from free_claude_code.config.reasoning import ReasoningPreference
from free_claude_code.config.settings import Settings
from free_claude_code.core import gateway_model_ids as gmi


class _StubRuntime:
    async def acquire(self) -> RequestRuntimeLease:
        raise NotImplementedError("catalog construction never acquires a lease")

    def current_settings(self) -> Settings:
        raise NotImplementedError

    def cached_model_supports_thinking(self, provider_id, model_id):
        return None

    def cached_prefixed_model_infos(self):
        return ()

    def cached_model_ids(self):
        return {}


@pytest.fixture(autouse=True)
def _isolate_alias_state():
    gmi.clear_picker_aliases()
    yield
    gmi.clear_picker_aliases()


_MODEL_SLOTS = ("model", "model_fable", "model_opus", "model_sonnet", "model_haiku")


def _settings_with_refs(*refs):
    slots = {
        slot: (refs[index] if index < len(refs) else None)
        for index, slot in enumerate(_MODEL_SLOTS)
    }
    return Settings.model_construct(
        **slots,
        reasoning_policy=ReasoningPreference.CLIENT,
    )


def _ids(response: model_catalog.ModelsListResponse) -> list[str]:
    return [item.id for item in response.data]


def test_unseeded_state_uses_gateway_wrapper_ids():
    settings = _settings_with_refs("nvidia_nim/01-ai/yi-large")

    response = model_catalog.build_models_list_response(
        settings, _StubRuntime(), picker_aliases=True
    )

    picker_ids = [
        item.id for item in response.data if item.id.startswith("claude-sonnet-nim")
    ]
    assert picker_ids == []
    assert "anthropic/nvidia_nim/01-ai/yi-large" in _ids(response)
    assert "claude-3-freecc-no-thinking/nvidia_nim/01-ai/yi-large" in _ids(response)


def test_seeded_state_prefers_alias_for_thinking_variant():
    gmi.seed_picker_aliases(["nvidia_nim/01-ai/yi-large"])
    settings = _settings_with_refs("nvidia_nim/01-ai/yi-large")

    response = model_catalog.build_models_list_response(
        settings, _StubRuntime(), picker_aliases=True
    )

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

    response = model_catalog.build_models_list_response(
        settings, _StubRuntime(), picker_aliases=True
    )

    alias_entries = [
        item for item in response.data if item.id.startswith("claude-sonnet-nim-0001")
    ]
    # Thinking entry exposes the canonical provider ref; the no-thinking
    # variant keeps the established "(no thinking)" label suffix.
    assert {item.display_name for item in alias_entries} == {
        "nvidia_nim/01-ai/yi-large",
        "nvidia_nim/01-ai/yi-large (no thinking)",
    }


def test_seeded_aliases_are_hidden_without_opt_in():
    """Claude Code / Codex / Pi path: aliases must never leak, even when seeded."""

    gmi.seed_picker_aliases(["nvidia_nim/01-ai/yi-large"])
    settings = _settings_with_refs("nvidia_nim/01-ai/yi-large")

    response = model_catalog.build_models_list_response(settings, _StubRuntime())

    ids = _ids(response)
    assert all(not item.startswith("claude-sonnet-nim") for item in ids)
    assert "anthropic/nvidia_nim/01-ai/yi-large" in ids
