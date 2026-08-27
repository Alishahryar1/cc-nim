"""Contracts for Aider's process-local FCC model files."""

import json

import pytest

from free_claude_code.cli.launchers.aider_config import build_aider_config
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
            wire_slug="ollama_cloud/qwen3-coder:480b",
            provider_model_ref="ollama_cloud/qwen3-coder:480b",
            display_name="Colon model",
            allows_reasoning=False,
        ),
    )


def test_aider_config_projects_messages_route_and_canonical_catalog() -> None:
    config = build_aider_config(
        _models(),
        messages_url="http://127.0.0.1:9191/v1/messages",
        api_key_env="FCC_AIDER_PROXY_AUTH_A1B2C3",
    )

    assert config.settings == [
        {
            "name": "aider/extra_params",
            "extra_params": {
                "api_base": "http://127.0.0.1:9191/v1/messages",
                "api_key": "os.environ/FCC_AIDER_PROXY_AUTH_A1B2C3",
            },
        }
    ]
    assert list(config.metadata) == [
        "anthropic/nvidia_nim/vendor/model",
        "anthropic/ollama_cloud/qwen3-coder:480b",
    ]
    assert config.metadata == {
        "anthropic/nvidia_nim/vendor/model": {
            "litellm_provider": "anthropic",
            "mode": "chat",
        },
        "anthropic/ollama_cloud/qwen3-coder:480b": {
            "litellm_provider": "anthropic",
            "mode": "chat",
        },
    }

    serialized = json.dumps({"settings": config.settings, "metadata": config.metadata})
    assert json.loads(serialized) == {
        "settings": config.settings,
        "metadata": config.metadata,
    }
    assert "proxy-token" not in serialized
    for fabricated_key in (
        "context_window",
        "max_tokens",
        "input_cost_per_token",
        "output_cost_per_token",
        "reasoning",
        "edit_format",
    ):
        assert fabricated_key not in serialized


def test_aider_config_rejects_empty_catalog() -> None:
    with pytest.raises(ValueError, match="at least one"):
        build_aider_config(
            (),
            messages_url="http://127.0.0.1:9191/v1/messages",
            api_key_env="FCC_AIDER_PROXY_AUTH_A1B2C3",
        )


@pytest.mark.parametrize(
    "api_key_env",
    [
        "",
        "FCC_AIDER_PROXY_AUTH_",
        "ANTHROPIC_API_KEY",
        "FCC_AIDER_PROXY_AUTH_lowercase",
        "FCC_AIDER_PROXY_AUTH_BAD-NAME",
        "FCC_AIDER_PROXY_AUTH_BAD/NAME",
    ],
)
def test_aider_config_rejects_invalid_api_key_environment_name(
    api_key_env: str,
) -> None:
    with pytest.raises(ValueError, match="environment"):
        build_aider_config(
            _models(),
            messages_url="http://127.0.0.1:9191/v1/messages",
            api_key_env=api_key_env,
        )
