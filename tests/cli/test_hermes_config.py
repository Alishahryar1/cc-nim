"""Contracts for the process-local Hermes configuration."""

import json

import pytest

from free_claude_code.cli.launchers.hermes_config import (
    build_hermes_managed_config,
)
from free_claude_code.cli.launchers.model_catalog import ClientModel


def _models() -> tuple[ClientModel, ...]:
    return (
        ClientModel(
            wire_slug="nvidia_nim/vendor/model",
            provider_model_ref="nvidia_nim/vendor/model",
            display_name="Nested model",
            allows_reasoning=True,
        ),
        ClientModel(
            wire_slug="claude-3-freecc-no-thinking/open_router/plain-model",
            provider_model_ref="open_router/plain-model",
            display_name="No-thinking model",
            allows_reasoning=False,
        ),
    )


def test_hermes_config_pins_responses_catalog_and_fallbacks() -> None:
    managed = build_hermes_managed_config(
        _models(),
        proxy_root_url="http://127.0.0.1:9191/",
        nonce="a1b2c3",
        selected_model="claude-3-freecc-no-thinking/open_router/plain-model",
    )

    assert managed.provider_key == "fcc-a1b2c3"
    assert managed.provider_ref == "custom:fcc-a1b2c3"
    assert managed.key_env == "FCC_HERMES_A1B2C3"
    assert managed.selected_model == (
        "claude-3-freecc-no-thinking/open_router/plain-model"
    )
    assert managed.config["providers"] == {
        "fcc-a1b2c3": {
            "name": "Free Claude Code",
            "api": "http://127.0.0.1:9191/v1",
            "key_env": "FCC_HERMES_A1B2C3",
            "transport": "codex_responses",
            "default_model": "claude-3-freecc-no-thinking/open_router/plain-model",
            "models": {
                "nvidia_nim/vendor/model": {},
                "claude-3-freecc-no-thinking/open_router/plain-model": {},
            },
            "discover_models": False,
        }
    }
    assert managed.config["model"] == {
        "provider": "custom:fcc-a1b2c3",
        "default": "claude-3-freecc-no-thinking/open_router/plain-model",
        "base_url": "",
        "api_key": "",
        "api_mode": "codex_responses",
    }
    assert managed.config["fallback_providers"] == []
    assert managed.config["fallback_model"] == []

    auxiliary = managed.config["auxiliary"]
    assert isinstance(auxiliary, dict)
    assert set(auxiliary) == {
        "vision",
        "web_extract",
        "compression",
        "skills_hub",
        "approval",
        "mcp",
        "title_generation",
        "memory_query_rewrite",
        "tts_audio_tags",
        "triage_specifier",
        "kanban_decomposer",
        "profile_describer",
        "goal_judge",
        "curator",
        "monitor",
        "background_review",
        "moa_reference",
        "moa_aggregator",
    }
    for task, task_config in auxiliary.items():
        assert isinstance(task_config, dict)
        expected = {
            "provider": "main",
            "model": "",
            "base_url": "",
            "api_key": "",
            "extra_body": {},
            "fallback_chain": [],
        }
        if task == "title_generation":
            expected["prefer_fast_model"] = False
        assert task_config == expected

    serialized = json.dumps(managed.config)
    assert "proxy-token" not in serialized
    assert "transient_retries" not in serialized
    assert "timeout" not in serialized


def test_hermes_config_uses_first_model_by_default() -> None:
    managed = build_hermes_managed_config(
        _models(),
        proxy_root_url="http://127.0.0.1:9191",
        nonce="ABC123",
    )

    assert managed.selected_model == "nvidia_nim/vendor/model"
    assert managed.provider_key == "fcc-abc123"
    assert managed.key_env == "FCC_HERMES_ABC123"


def test_hermes_config_rejects_empty_catalog() -> None:
    with pytest.raises(ValueError, match="at least one"):
        build_hermes_managed_config(
            (),
            proxy_root_url="http://127.0.0.1:9191",
            nonce="abc123",
        )


def test_hermes_config_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match="not in the FCC catalog"):
        build_hermes_managed_config(
            _models(),
            proxy_root_url="http://127.0.0.1:9191",
            nonce="abc123",
            selected_model="missing/model",
        )


@pytest.mark.parametrize("nonce", ["", "not-valid", "spaces here"])
def test_hermes_config_rejects_unsafe_nonce(nonce: str) -> None:
    with pytest.raises(ValueError, match="alphanumeric"):
        build_hermes_managed_config(
            _models(),
            proxy_root_url="http://127.0.0.1:9191",
            nonce=nonce,
        )
