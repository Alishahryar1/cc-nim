"""Errors for OpenAI Chat Completions compatibility.

The OpenAI wire error envelope and the neutral-failure-to-error-type mapping are
shared with the Responses dialect and live in ``core.openai_responses`` (already
treated as shared OpenAI helpers by ``api.request_errors``); this module only
owns the Chat-specific deterministic conversion error.
"""


class ChatCompletionsConversionError(ValueError):
    """Raised when a Chat Completions request cannot be converted deterministically."""
