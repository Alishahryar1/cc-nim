"""Tests for the Modal OpenAI-compatible provider (user-deployed Web Functions)."""

from dataclasses import replace

import pytest
from unittest.mock import patch

from free_claude_code.config.provider_catalog import PROVIDER_CATALOG
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.openai_chat import OpenAIChatProvider
from tests.providers.support import profiled_provider

MODAL_BASE_URL = "https://acme--inkling-nvfp4-server.us-west.modal.direct/v1"


@pytest.fixture
def modal_config() -> ProviderConfig:
    return ProviderConfig(
        api_key="wk-test-token-id",
        base_url=MODAL_BASE_URL,
        rate_limit=10,
        rate_window=60,
        modal_proxy_token_id="wk-test-token-id",
        modal_proxy_token_secret="ws-test-token-secret",
    )


def test_modal_descriptor_has_no_shared_default_and_requires_all_three_fields() -> None:
    descriptor = PROVIDER_CATALOG["modal"]

    assert descriptor.default_base_url is None
    assert descriptor.local is False
    assert descriptor.credential_env == "MODAL_PROXY_TOKEN_ID"
    assert descriptor.base_url_attr == "modal_base_url"
    assert descriptor.configuration_attrs() == (
        "modal_base_url",
        "modal_proxy_token_id",
        "modal_proxy_token_secret",
    )


def test_modal_provider_sends_proxy_auth_headers(modal_config: ProviderConfig) -> None:
    with patch(
        "free_claude_code.providers.openai_chat.provider.AsyncOpenAI"
    ) as mock_openai:
        provider = profiled_provider("modal", modal_config)

    assert isinstance(provider, OpenAIChatProvider)
    assert mock_openai.call_args.kwargs["base_url"] == MODAL_BASE_URL
    assert mock_openai.call_args.kwargs["default_headers"] == {
        "Modal-Key": "wk-test-token-id",
        "Modal-Secret": "ws-test-token-secret",
    }


def test_modal_provider_normalizes_base_url_missing_v1_suffix(
    modal_config: ProviderConfig,
) -> None:
    config = replace(
        modal_config,
        base_url="https://acme--inkling-nvfp4-server.us-west.modal.direct",
    )
    with patch("free_claude_code.providers.openai_chat.provider.AsyncOpenAI"):
        provider = profiled_provider("modal", config)

    assert provider._base_url == MODAL_BASE_URL


def test_other_openai_chat_providers_never_receive_modal_proxy_headers(
    modal_config: ProviderConfig,
) -> None:
    """Regression test: Modal-Key/Modal-Secret must stay scoped to the modal
    provider and must not leak onto other OpenAI-compatible providers just
    because Modal proxy tokens happen to be set in the shared settings."""
    other_config = replace(
        modal_config,
        api_key="other-provider-key",
        base_url="https://api.groq.com/openai/v1",
    )
    with patch(
        "free_claude_code.providers.openai_chat.provider.AsyncOpenAI"
    ) as mock_openai:
        profiled_provider("groq", other_config)

    headers = mock_openai.call_args.kwargs["default_headers"]
    assert headers is None or "Modal-Key" not in headers
    assert headers is None or "Modal-Secret" not in headers
