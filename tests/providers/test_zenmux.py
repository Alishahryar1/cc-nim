from free_claude_code.providers.base import ProviderConfig
from tests.providers.request_factory import make_messages_request
from tests.providers.support import passthrough_rate_limiter, profiled_provider

def test_zenmux_max_completion_tokens():
    config = ProviderConfig(
        api_key="test_zenmux_key",
        base_url="https://api.zenmux.com/v1",
        rate_limit=10,
        rate_window=60,
    )
    provider = profiled_provider("zenmux", config, rate_limiter=passthrough_rate_limiter())
    req = make_messages_request("gpt-4o")
    body = provider._build_request_body(req)
    assert "max_completion_tokens" in body
    assert "max_tokens" not in body
