"""Strict recovery for Groq requests above an account TPM ceiling."""

import re
from collections.abc import Mapping
from dataclasses import dataclass

_TPM_LIMIT_HEADER = "x-ratelimit-limit-tokens"
_MAX_DECIMAL_DIGITS = 19
_TPM_CLAUSE = re.compile(
    r"tokens\s+per\s+minute\s*\(\s*tpm\s*\)\s*:\s*"
    r"limit\s+([0-9]+)\s*,\s*requested\s+([0-9]+)(?=\s*(?:[,;]|$))",
    re.IGNORECASE,
)
_OUTPUT_FIELDS = frozenset({"max_completion_tokens", "max_tokens"})


@dataclass(frozen=True, slots=True)
class GroqTpmCorrection:
    """Validated request-local reduction derived from one Groq rejection."""

    body: dict[str, object]
    limit: int
    requested: int
    previous_max_completion_tokens: int
    corrected_max_completion_tokens: int


def correct_tpm_completion_budget(
    error: Exception,
    body: Mapping[str, object],
) -> GroqTpmCorrection | None:
    """Clone ``body`` with Groq's reported TPM overage removed from its cap."""
    if _status_code(error) != 413:
        return None

    detail = _error_detail(getattr(error, "body", None))
    if detail is None:
        return None
    if _normalized_marker(detail.get("type")) != "tokens":
        return None
    if _normalized_marker(detail.get("code")) != "rate_limit_exceeded":
        return None

    message = detail.get("message")
    if not isinstance(message, str):
        return None
    matches = tuple(_TPM_CLAUSE.finditer(message))
    if len(matches) != 1:
        return None
    limit = _ascii_decimal(matches[0].group(1))
    requested = _ascii_decimal(matches[0].group(2))
    if limit is None or requested is None:
        return None
    if limit <= 0 or requested <= limit:
        return None
    if not _header_agrees(error, limit):
        return None

    previous = body.get("max_completion_tokens")
    if not isinstance(previous, int) or isinstance(previous, bool) or previous <= 0:
        return None
    extra_body = body.get("extra_body")
    if isinstance(extra_body, Mapping) and any(
        field in extra_body for field in _OUTPUT_FIELDS
    ):
        return None

    corrected = previous - (requested - limit)
    if corrected <= 0 or corrected >= previous:
        return None

    corrected_body = dict(body)
    corrected_body["max_completion_tokens"] = corrected
    return GroqTpmCorrection(
        body=corrected_body,
        limit=limit,
        requested=requested,
        previous_max_completion_tokens=previous,
        corrected_max_completion_tokens=corrected,
    )


def _status_code(error: Exception) -> int | None:
    status = getattr(error, "status_code", None)
    if isinstance(status, int) and not isinstance(status, bool):
        return status
    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) and not isinstance(status, bool) else None


def _error_detail(value: object) -> Mapping[object, object] | None:
    if not isinstance(value, Mapping):
        return None
    if _has_detail_shape(value):
        return value
    if len(value) != 1:
        return None
    nested = value.get("error")
    if not isinstance(nested, Mapping) or not _has_detail_shape(nested):
        return None
    return nested


def _has_detail_shape(value: Mapping[object, object]) -> bool:
    return all(field in value for field in ("message", "type", "code"))


def _normalized_marker(value: object) -> str | None:
    return value.strip().lower() if isinstance(value, str) else None


def _header_agrees(error: Exception, limit: int) -> bool:
    values = _header_values(error, _TPM_LIMIT_HEADER)
    if values is None:
        return False
    if not values:
        return True
    if len(values) != 1:
        return False
    return _ascii_decimal(values[0].strip()) == limit


def _ascii_decimal(value: str) -> int | None:
    if not value.isascii() or not value.isdecimal() or len(value) > _MAX_DECIMAL_DIGITS:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _header_values(error: Exception, name: str) -> tuple[str, ...] | None:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return ()

    get_list = getattr(headers, "get_list", None)
    if callable(get_list):
        try:
            values = get_list(name)
        except Exception:
            return None
        if isinstance(values, list) and all(isinstance(value, str) for value in values):
            return tuple(values)
        return None

    if not isinstance(headers, Mapping):
        return None
    values = [
        value
        for key, value in headers.items()
        if isinstance(key, str) and key.lower() == name
    ]
    if not all(isinstance(value, str) for value in values):
        return None
    return tuple(values)
