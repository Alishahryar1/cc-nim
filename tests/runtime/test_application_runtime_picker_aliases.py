"""Runtime seeding for picker aliases via ``ApplicationRuntime``."""

from typing import cast

import pytest

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.core import gateway_model_ids as gmi
from free_claude_code.runtime.application import ApplicationRuntime
from free_claude_code.runtime.provider_manager import ProviderRuntimeManager


@pytest.fixture(autouse=True)
def _isolate_alias_state():
    gmi.clear_picker_aliases()
    yield
    gmi.clear_picker_aliases()


class _SeedingProviderManager:
    """Stand-in honoring the manager's alias-refresh contract."""

    def __init__(self, model_ids: list[str]) -> None:
        self._model_ids = model_ids
        self.refresh_calls = 0

    def refresh_picker_aliases(self) -> None:
        self.refresh_calls += 1
        infos = tuple(
            ProviderModelInfo(model_id=model_id, supports_thinking=None)
            for model_id in self._model_ids
        )
        gmi.seed_picker_aliases(info.model_id for info in infos)


def _make_runtime_with_models(
    model_ids: list[str],
) -> tuple[ApplicationRuntime, _SeedingProviderManager]:
    provider_manager = _SeedingProviderManager(model_ids)
    return (
        ApplicationRuntime(
            cast(ProviderRuntimeManager, provider_manager), transcriber=None
        ),
        provider_manager,
    )


def test_seed_picker_aliases_from_cache_populates_maps():
    runtime, _manager = _make_runtime_with_models(
        ["nvidia_nim/01-ai/yi-large", "nvidia_nim/meta/llama-3.3-70b-instruct"]
    )

    runtime._seed_picker_aliases_from_cache()

    assert gmi.has_picker_aliases() is True
    assert gmi.picker_alias_for("nvidia_nim/01-ai/yi-large") == "claude-sonnet-nim-0001"
    assert (
        gmi.picker_alias_for("nvidia_nim/01-ai/yi-large", force_reasoning_off=True)
        == "claude-sonnet-nim-0001-no-thinking"
    )
    assert (
        gmi.picker_alias_for("nvidia_nim/meta/llama-3.3-70b-instruct")
        == "claude-sonnet-nim-0002"
    )


def test_seed_picker_aliases_with_empty_cache_leaves_state_inert():
    runtime, _manager = _make_runtime_with_models([])

    runtime._seed_picker_aliases_from_cache()

    assert gmi.has_picker_aliases() is False
    assert gmi.resolve_picker_alias("claude-sonnet-nim-0001") is None


def test_seed_picker_aliases_uses_sorted_input_for_counter_stability():
    first_runtime, first_manager = _make_runtime_with_models(
        [
            "nvidia_nim/z-provider/z-model",
            "nvidia_nim/a-provider/a-model",
        ]
    )
    first_runtime._seed_picker_aliases_from_cache()
    assert first_manager.refresh_calls == 1
    first_alias_for_a = gmi.picker_alias_for("nvidia_nim/a-provider/a-model")
    gmi.clear_picker_aliases()

    second_runtime, _manager = _make_runtime_with_models(
        [
            "nvidia_nim/a-provider/a-model",
            "nvidia_nim/z-provider/z-model",
        ]
    )
    second_runtime._seed_picker_aliases_from_cache()
    second_alias_for_a = gmi.picker_alias_for("nvidia_nim/a-provider/a-model")

    assert first_alias_for_a == second_alias_for_a == "claude-sonnet-nim-0001"
