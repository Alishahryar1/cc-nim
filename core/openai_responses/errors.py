"""Errors and error envelopes for OpenAI Responses compatibility."""

from __future__ import annotations

from typing import Any


class ResponsesConversionError(ValueError):
    """Raised when a Responses request cannot be converted deterministically."""


def openai_error_payload(*, message: str, error_type: str) -> dict[str, Any]:
    """Return an OpenAI-compatible error envelope."""

    return {
        "error": {
            "message": message,
            "type": error_type,
            "param": None,
            "code": None,
        }
    }
