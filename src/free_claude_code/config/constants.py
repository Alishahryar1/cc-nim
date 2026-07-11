"""Shared defaults used by config models and provider adapters."""

# HTTP client connect timeout (seconds). Keep aligned with README.md and .env.example.
HTTP_CONNECT_TIMEOUT_DEFAULT = 10.0

# Anthropic Messages API default when the client omits max_tokens.
ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS = 81920

# Max bytes read from a non-200 native messages response when verbose error logging is on.
NATIVE_MESSAGES_ERROR_BODY_LOG_CAP_BYTES = 4096
