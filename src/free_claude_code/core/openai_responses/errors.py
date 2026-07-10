"""Errors and error envelopes for OpenAI Responses compatibility."""

from typing import Any

from free_claude_code.core.anthropic.errors import redact_sensitive_error_text


class ResponsesConversionError(ValueError):
    """Raised when a Responses request cannot be converted deterministically."""


def openai_error_payload(*, message: str, error_type: str) -> dict[str, Any]:
    """Return an OpenAI-compatible error envelope."""

    return {
        "error": {
            "message": redact_sensitive_error_text(message),
            "type": error_type,
            "param": None,
            "code": None,
        }
    }
