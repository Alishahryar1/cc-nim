"""Tests for OmniRoute's OpenAI-compatible model catalog."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.provider_catalog import OMNIROUTE_DEFAULT_BASE
from tests.providers.support import (
    immediate_admission,
    make_provider_config,
    profiled_provider,
)


@pytest.mark.asyncio
async def test_model_catalog_extracts_reasoning_and_context_limits() -> None:
    provider = profiled_provider(
        "omniroute",
        make_provider_config(
            api_key="test-omniroute-key",
            base_url=OMNIROUTE_DEFAULT_BASE,
            rate_limit=10,
            rate_window=60,
        ),
        admission=immediate_admission(provider_name="omniroute"),
    )
    provider._client.models.list = AsyncMock(
        return_value=SimpleNamespace(
            data=[
                {
                    "id": "auto/best-coding",
                    "context_length": 1000000,
                    "max_output_tokens": 384000,
                    "capabilities": {"reasoning": True},
                },
                {
                    "id": "auto/best-fast",
                    "context_length": 1000000,
                    "max_output_tokens": 384000,
                    "capabilities": {"reasoning": False},
                },
                {
                    "id": "plain",
                    "capabilities": {"reasoning": "true"},
                },
            ]
        )
    )
    try:
        infos = await provider.list_model_infos()
    finally:
        await provider.cleanup()

    assert infos == frozenset(
        {
            ProviderModelInfo(
                "auto/best-coding",
                supports_thinking=True,
                context_window_tokens=1000000,
                max_output_tokens=384000,
            ),
            ProviderModelInfo(
                "auto/best-fast",
                supports_thinking=False,
                context_window_tokens=1000000,
                max_output_tokens=384000,
            ),
            ProviderModelInfo("plain"),
        }
    )


@pytest.mark.asyncio
async def test_model_catalog_missing_id_rejected() -> None:
    provider = profiled_provider(
        "omniroute",
        make_provider_config(
            api_key="test-omniroute-key",
            base_url=OMNIROUTE_DEFAULT_BASE,
            rate_limit=10,
            rate_window=60,
        ),
        admission=immediate_admission(provider_name="omniroute"),
    )
    provider._client.models.list = AsyncMock(
        return_value=SimpleNamespace(
            data=[
                {
                    "context_length": 1000000,
                    "capabilities": {"reasoning": True},
                }
            ]
        )
    )
    try:
        with pytest.raises(ValueError):
            await provider.list_model_infos()
    finally:
        await provider.cleanup()
