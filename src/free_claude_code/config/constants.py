"""Shared defaults used by config models and provider adapters."""

# HTTP client connect timeout (seconds). Keep aligned with README.md and .env.example.
HTTP_CONNECT_TIMEOUT_DEFAULT = 10.0

# Anthropic Messages API default when the client omits max_tokens.
ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS = 81920

PROVIDER_ERROR_BODY_DISPLAY_CAP_BYTES = 4096
NATIVE_MESSAGES_ERROR_BODY_LOG_CAP_BYTES = 4096
