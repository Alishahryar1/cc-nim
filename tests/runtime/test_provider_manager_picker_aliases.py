"""Picker-alias reseeding on model-catalog republication."""

import pytest

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.settings import Settings
from free_claude_code.core import gateway_model_ids as gmi
from free_claude_code.runtime.provider_manager import ProviderRuntimeManager


@pytest.fixture(autouse=True)
def _isolate_alias_state():
    gmi.clear_picker_aliases()
    yield
    gmi.clear_picker_aliases()


def _manager(model: str) -> ProviderRuntimeManager:
    settings = Settings().model_copy(
        update={"model": model, "nvidia_nim_api_key": "test-key"}
    )
    return ProviderRuntimeManager(settings)


def _info(model_id: str) -> ProviderModelInfo:
    return ProviderModelInfo(model_id=model_id, supports_thinking=None)


def test_refresh_picker_aliases_seeds_current_cache_snapshot() -> None:
    manager = _manager("nvidia_nim/one")
    manager.cache_model_infos("nvidia_nim", [_info("meta/llama-3.3-70b-instruct")])

    manager.refresh_picker_aliases()

    assert gmi.has_picker_aliases() is True
    assert (
        gmi.picker_alias_for("nvidia_nim/meta/llama-3.3-70b-instruct")
        == "claude-sonnet-nim-0001"
    )
    gmi.clear_picker_aliases()


def test_cache_publication_reseeds_aliases_atomically() -> None:
    manager = _manager("nvidia_nim/one")
    manager.cache_model_infos("nvidia_nim", [_info("meta/llama-3.3-70b-instruct")])
    assert gmi.has_picker_aliases() is True
    first_alias = gmi.picker_alias_for("nvidia_nim/meta/llama-3.3-70b-instruct")

    # A refreshed inventory keeps the sticky assignment and adds new refs.
    manager.cache_model_infos(
        "nvidia_nim",
        [
            _info("meta/llama-3.3-70b-instruct"),
            _info("01-ai/yi-large"),
        ],
    )

    assert gmi.picker_alias_for("nvidia_nim/meta/llama-3.3-70b-instruct") == first_alias
    assert gmi.picker_alias_for("nvidia_nim/01-ai/yi-large") is not None


def test_configured_ref_gets_alias_without_discovery_cache() -> None:
    manager = _manager("nvidia_nim/one")

    # No discovery cache yet: the configured model still gains a picker alias
    # so a cold /v1/models request never advertises a raw gateway identifier.
    manager.refresh_picker_aliases()

    assert gmi.has_picker_aliases() is True
    assert gmi.picker_alias_for("nvidia_nim/one") is not None


def test_empty_cache_retires_discovery_only_aliases_but_keeps_configured() -> None:
    manager = _manager("nvidia_nim/one")
    manager.cache_model_infos("nvidia_nim", [_info("meta/llama-3.3-70b-instruct")])
    assert gmi.has_picker_aliases() is True
    configured_alias = gmi.picker_alias_for("nvidia_nim/one")
    discovery_alias = gmi.picker_alias_for("nvidia_nim/meta/llama-3.3-70b-instruct")
    assert configured_alias is not None
    assert discovery_alias is not None

    manager.cache_model_infos("nvidia_nim", [])

    # The discovery-only ref retires; the configured ref keeps its alias.
    assert gmi.has_picker_aliases() is True
    assert gmi.resolve_picker_alias(discovery_alias) is None
    assert gmi.picker_alias_for("nvidia_nim/one") == configured_alias
