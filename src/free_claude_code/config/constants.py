"""Shared defaults used by config models and provider adapters."""

# HTTP client connect timeout (seconds). Keep aligned with README.md and .env.example.
HTTP_CONNECT_TIMEOUT_DEFAULT = 10.0

# Provider admission defaults. One request may start per window while active
# upstream operations are bounded independently by concurrency.
PROVIDER_RATE_LIMIT_DEFAULT = 1
PROVIDER_RATE_WINDOW_DEFAULT = 2
PROVIDER_MAX_CONCURRENCY_DEFAULT = 2

# Anthropic Messages API default when the client omits max_tokens.
ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS = 81920
