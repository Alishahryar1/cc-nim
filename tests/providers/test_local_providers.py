from __future__ import annotations

from providers.llamacpp.client import LlamaCppProvider
from providers.lmstudio.client import LMStudioProvider
from providers.ollama.client import OllamaProvider


class FakeConfig:
    def __init__(
        self,
        api_key: str = "",
        base_url: str = "",
        proxy: str = "",
        rate_limit: int = 40,
        rate_window: int = 60,
        max_concurrency: int = 5,
        http_read_timeout: float = 30.0,
        http_connect_timeout: float = 10.0,
        http_write_timeout: float = 30.0,
        log_api_error_tracebacks: bool = False,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.proxy = proxy
        self.rate_limit = rate_limit
        self.rate_window = rate_window
        self.max_concurrency = max_concurrency
        self.http_read_timeout = http_read_timeout
        self.http_connect_timeout = http_connect_timeout
        self.http_write_timeout = http_write_timeout
        self.log_api_error_tracebacks = log_api_error_tracebacks


SMALL_FACTORIES = (
    LMStudioProvider,
    LlamaCppProvider,
    OllamaProvider,
)


def test_local_providers_honor_configured_api_key() -> None:
    for factory in SMALL_FACTORIES:
        provider = factory(FakeConfig(api_key="test-local-key-123"))
        assert provider._api_key == "test-local-key-123"


def test_local_providers_use_empty_api_key_by_default() -> None:
    for factory in SMALL_FACTORIES:
        provider = factory(FakeConfig())
        assert provider._api_key == ""
