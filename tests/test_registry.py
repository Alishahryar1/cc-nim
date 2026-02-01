"""Tests for provider registry."""

import pytest
from providers.registry import (
    register_provider,
    get_provider_class,
    list_providers,
    create_provider,
    _PROVIDER_REGISTRY,
)
from providers.base import BaseProvider, ProviderConfig


class MockProvider(BaseProvider):
    """Mock provider for testing."""

    async def complete(self, request):
        return {}

    async def stream_response(self, request, input_tokens=0):
        yield "data: test\n\n"

    def convert_response(self, response_json, original_request):
        return {}


class TestProviderRegistry:
    """Tests for provider registry functions."""

    def test_register_and_get_provider(self):
        """Test registering and retrieving a provider."""
        register_provider("mock_test", MockProvider)
        assert get_provider_class("mock_test") == MockProvider

    def test_get_unknown_provider_returns_none(self):
        """Test that unknown provider returns None."""
        assert get_provider_class("nonexistent_provider_xyz") is None

    def test_list_providers_includes_registered(self):
        """Test that list_providers includes registered providers."""
        register_provider("mock_list_test", MockProvider)
        providers = list_providers()
        assert "mock_list_test" in providers

    def test_nvidia_nim_auto_registered(self):
        """Test that nvidia_nim is auto-registered on import."""
        # This import triggers auto-registration
        from providers import NvidiaNimProvider

        assert "nvidia_nim" in list_providers()
        assert get_provider_class("nvidia_nim") == NvidiaNimProvider

    def test_create_provider_success(self):
        """Test creating a provider instance."""
        register_provider("mock_create", MockProvider)
        config = ProviderConfig(api_key="test_key")
        provider = create_provider("mock_create", config)
        assert isinstance(provider, MockProvider)

    def test_create_provider_unknown_raises(self):
        """Test that creating unknown provider raises ValueError."""
        config = ProviderConfig(api_key="test")
        with pytest.raises(ValueError, match="Unknown provider"):
            create_provider("completely_unknown_provider", config)

    def test_create_provider_error_message_lists_available(self):
        """Test that error message lists available providers."""
        config = ProviderConfig(api_key="test")
        try:
            create_provider("unknown_xyz", config)
        except ValueError as e:
            assert "nvidia_nim" in str(e)

    def test_register_overwrites_existing(self):
        """Test that registering same name overwrites."""

        class AnotherMockProvider(BaseProvider):
            async def complete(self, request):
                return {"new": True}

            async def stream_response(self, request, input_tokens=0):
                yield "new"

            def convert_response(self, response_json, original_request):
                return {}

        register_provider("overwrite_test", MockProvider)
        register_provider("overwrite_test", AnotherMockProvider)
        assert get_provider_class("overwrite_test") == AnotherMockProvider
