"""Tests for free model labeling and filtering functionality."""

from unittest.mock import Mock

import pytest

from free_claude_code.api.model_catalog import (
    ModelsListResponse,
    _is_free_model,
    build_models_list_response,
)
from free_claude_code.config.settings import Settings


def test_free_models_list_validator_empty_string():
    """Test that FREE_MODELS_LIST=\"\" returns empty list."""
    # Test the validator function directly
    result = Settings._split_free_models("")
    assert result == []


def test_free_models_list_validator_whitespace_only():
    """Test that FREE_MODELS_LIST=\"  ,  \" returns empty list."""
    # Test the validator function directly
    result = Settings._split_free_models("  ,  ")
    assert result == []


def test_free_models_list_validator_with_whitespace():
    """Test that whitespace around commas is stripped."""
    # Test the validator function directly
    result = Settings._split_free_models(" model1 , model2 ")
    assert result == ["model1", "model2"]


def test_free_models_list_validator_already_list():
    """Test that when free_models_list is already a list, it's returned as-is."""
    # Test the validator function directly
    result = Settings._split_free_models(["pre-existing"])
    assert result == ["pre-existing"]


def test_is_free_model_suffix_free_colon():
    """Test model ref ending with :free is detected as free."""
    settings = Settings()
    settings.free_models_list = []  # Start with empty list
    assert _is_free_model("provider/model:free", settings) is True


def test_is_free_model_suffix_free_slash():
    """Test model ref ending with /free is detected as free."""
    settings = Settings()
    settings.free_models_list = []  # Start with empty list
    assert _is_free_model("provider/model/free", settings) is True


def test_is_free_model_suffix_free_dash():
    """Test model ref ending with -free is detected as free."""
    settings = Settings()
    settings.free_models_list = []  # Start with empty list
    assert _is_free_model("provider/model-free", settings) is True


def test_is_free_model_in_free_list():
    """Test model ref in FREE_MODELS_LIST is detected as free."""
    settings = Settings()
    settings.free_models_list = ["provider/model", "other/model"]
    assert _is_free_model("provider/model", settings) is True
    assert _is_free_model("other/model", settings) is True


def test_is_free_model_not_free():
    """Test model ref with no free indicators returns False."""
    settings = Settings()
    settings.free_models_list = ["other/model"]
    assert _is_free_model("provider/model", settings) is False


def test_is_free_model_case_sensitive_list():
    """Test FREE_MODELS_LIST matching is case-sensitive (current behavior)."""
    settings = Settings()
    settings.free_models_list = ["Provider/Model"]
    assert _is_free_model("provider/model", settings) is False  # Case sensitive
    assert _is_free_model("Provider/Model", settings) is True  # Exact match


def test_is_free_model_free_list_priority_over_suffix():
    """Test that explicit FREE_MODELS_LIST works alongside suffix detection."""
    settings = Settings()
    settings.free_models_list = ["provider/model"]
    # This would be True anyway due to suffix, but list matching should work
    assert _is_free_model("provider/model-free", settings) is True


def test_build_models_list_with_free_models_first():
    """Test that SHOW_FREE_MODELS_FIRST=true puts free models first."""
    settings = Settings.model_construct(
        anthropic_auth_token="",
    )
    # Make the default model free
    settings.free_models_list = ["nvidia_nim/nvidia/nemotron-3-super-120b-a12b"]
    settings.show_free_models_first = True
    settings.only_show_free_models = False

    # Mock provider registry with no models to simplify
    registry = Mock()
    registry.cached_model_supports_thinking.return_value = None
    registry.cached_prefixed_model_infos.return_value = ()

    response = build_models_list_response(settings, registry)

    # First model should be the free one
    assert (
        response.data[0].id == "anthropic/nvidia_nim/nvidia/nemotron-3-super-120b-a12b"
    )
    assert "(free)" in response.data[0].display_name


def test_build_models_list_only_free_models():
    """Test that ONLY_SHOW_FREE_MODELS=true filters to only free models."""
    settings = Settings.model_construct(
        MODEL_OPUS="open_router/anthropic/claude-opus",  # Not free
        MODEL_SONNET=None,
        MODEL_HAIKU=None,
        anthropic_auth_token="",
    )
    # Only make the default model free
    settings.free_models_list = ["nvidia_nim/nvidia/nemotron-3-super-120b-a12b"]
    settings.show_free_models_first = False
    settings.only_show_free_models = True

    registry = Mock()
    registry.cached_model_supports_thinking.return_value = None
    registry.cached_prefixed_model_infos.return_value = ()

    response = build_models_list_response(settings, registry)

    # Should only contain the free model and its thinking variants
    assert len(response.data) == 2
    # Both should be marked as free
    assert "(free)" in response.data[0].display_name
    assert "(free)" in response.data[1].display_name
    # Both should be based on the same model
    assert "nvidia_nim/nvidia/nemotron-3-super-120b-a12b" in response.data[0].id
    assert "nvidia_nim/nvidia/nemotron-3-super-120b-a12b" in response.data[1].id


def test_build_models_list_combined_flags():
    """Test SHOW_FREE_MODELS_FIRST and ONLY_SHOW_FREE_MODELS together."""
    settings = Settings.model_construct(
        MODEL_OPUS="open_router/anthropic/claude-opus",  # Not free
        MODEL_SONNET=None,
        MODEL_HAIKU=None,
        anthropic_auth_token="",
    )
    # Only make the default model free
    settings.free_models_list = ["nvidia_nim/nvidia/nemotron-3-super-120b-a12b"]
    settings.show_free_models_first = True
    settings.only_show_free_models = True

    registry = Mock()
    registry.cached_model_supports_thinking.return_value = None
    registry.cached_prefixed_model_infos.return_value = ()

    response = build_models_list_response(settings, registry)

    # Should only contain the free model and its thinking variants
    assert len(response.data) == 2
    # Both should be marked as free
    assert "(free)" in response.data[0].display_name
    assert "(free)" in response.data[1].display_name
    # Both should be based on the same model
    assert "nvidia_nim/nvidia/nemotron-3-super-120b-a12b" in response.data[0].id
    assert "nvidia_nim/nvidia/nemotron-3-super-120b-a12b" in response.data[1].id
    # When SHOW_FREE_MODELS_FIRST=True, free models should come first
    # Since we only have free models, they should be first anyway


def test_build_models_list_no_free_models():
    """Test behavior when no models are free."""
    settings = Settings.model_construct(
        MODEL="deepseek/deepseek-chat",
        MODEL_OPUS=None,
        MODEL_SONNET=None,
        MODEL_HAIKU=None,
        anthropic_auth_token="",
    )
    settings.free_models_list = []  # No free models
    settings.show_free_models_first = True
    settings.only_show_free_models = False

    registry = Mock()
    registry.cached_model_supports_thinking.return_value = None
    registry.cached_prefixed_model_infos.return_value = ()

    response = build_models_list_response(settings, registry)

    # Should still return models but none marked as free
    # Note: The actual behavior depends on how models are classified
    # At minimum, we should get a valid response
    assert isinstance(response, ModelsListResponse)
    assert response.has_more is False


def test_is_free_model_with_gateway_ids():
    """Test _is_free_model handles gateway ID normalization."""
    settings = Settings()
    settings.free_models_list = ["open_router/model"]

    # Direct match (existing behavior)
    assert _is_free_model("open_router/model", settings) is True

    # Gateway ID in list matching underlying model
    assert _is_free_model("anthropic/open_router/model", settings) is True

    # Underlying model in list matching gateway ID
    assert _is_free_model("open_router/model", settings) is True

    # No-thinking gateway ID
    assert (
        _is_free_model("claude-3-freecc-no-thinking/open_router/model", settings)
        is True
    )

    # Non-match
    assert _is_free_model("other/model", settings) is False


def test_is_free_model_suffixes():
    """Test _is_free_model still works with suffix-based free models."""
    settings = Settings()
    settings.free_models_list = []

    # Suffixes still work
    assert _is_free_model("model:free", settings) is True
    assert _is_free_model("model-free", settings) is True
    assert _is_free_model("model/free", settings) is True
    assert _is_free_model("model", settings) is False


def test_is_free_model_mixed_list():
    """Test _is_free_model with mixed gateway IDs and underlying models."""
    settings = Settings()
    settings.free_models_list = ["anthropic/open_router/model1", "open_router/model2"]

    # List has gateway ID, check underlying
    assert _is_free_model("open_router/model1", settings) is True
    assert _is_free_model("anthropic/open_router/model1", settings) is True

    # List has underlying, check gateway
    assert _is_free_model("open_router/model2", settings) is True
    assert _is_free_model("anthropic/open_router/model2", settings) is True
    assert (
        _is_free_model("claude-3-freecc-no-thinking/open_router/model2", settings)
        is True
    )

    # Non-match
    assert _is_free_model("other/model", settings) is False


def test_is_free_model_empty_list():
    """Test _is_free_model with empty free models list."""
    settings = Settings()
    settings.free_models_list = []

    assert _is_free_model("any/model", settings) is False
    assert _is_free_model("model:free", settings) is True  # suffix still works


def test_is_free_model_no_thinking_variants():
    """Test _is_free_model with no-thinking gateway ID variants."""
    settings = Settings()
    settings.free_models_list = ["nvidia_nim/model"]

    # Direct
    assert _is_free_model("nvidia_nim/model", settings) is True

    # Standard gateway
    assert _is_free_model("anthropic/nvidia_nim/model", settings) is True

    # No-thinking gateway
    assert (
        _is_free_model("claude-3-freecc-no-thinking/nvidia_nim/model", settings) is True
    )

    # Reverse: no-thinking in list
    settings_no_think = Settings()
    settings_no_think.free_models_list = [
        "claude-3-freecc-no-thinking/nvidia_nim/model"
    ]
    assert _is_free_model("nvidia_nim/model", settings_no_think) is True
    assert _is_free_model("anthropic/nvidia_nim/model", settings_no_think) is True
    assert (
        _is_free_model(
            "claude-3-freecc-no-thinking/nvidia_nim/model", settings_no_think
        )
        is True
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
