"""Shared retry and recovery policy for Anthropic streams."""

import json
import time
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import httpx
import jsonschema
import openai
from loguru import logger

from core.trace import trace_event

from .transient_errors import retryable_transient_status

EARLY_TRANSPARENT_TOTAL_ATTEMPTS = 5
EARLY_TRANSPARENT_MAX_RETRIES = EARLY_TRANSPARENT_TOTAL_ATTEMPTS - 1
MIDSTREAM_RECOVERY_ATTEMPTS = 5
EARLY_HOLDBACK_SECONDS = 0.75
RECOVERY_BUFFER_MAX_BYTES = 65_536

_RECOVERY_USER_PREFIX = (
    "The previous provider stream was interrupted. Continue the assistant response "
    "exactly where it stopped. Do not repeat text already written."
)
_RECOVERY_THINKING_PREFIX = (
    "The assistant had already emitted this hidden thinking before the interruption:\n"
)


class TruncatedProviderStreamError(RuntimeError):
    """Raised internally when an upstream stream ends without a terminal marker."""


class RecoveryFailureAction(StrEnum):
    """How the stream lifecycle should respond to an upstream failure."""

    EARLY_RETRY = "early_retry"
    MIDSTREAM_RECOVERY = "midstream_recovery"
    FINAL_ERROR = "final_error"


@dataclass(frozen=True, slots=True)
class RecoveryDecision:
    """Failure classification result for one stream exception."""

    action: RecoveryFailureAction
    retryable: bool
    committed: bool
    has_buffered: bool
    early_retry_attempt: int | None = None
    midstream_recovery_attempt: int | None = None


@dataclass(frozen=True, slots=True)
class ToolSchema:
    """Tool schema resolved from the original Anthropic request."""

    name: str
    input_schema: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolRepair:
    """Accepted append-only tool JSON repair."""

    suffix: str
    parsed_input: dict[str, Any]


class RecoveryHoldbackBuffer:
    """Briefly hold downstream SSE so early stream cutoffs can be retried invisibly."""

    def __init__(
        self,
        *,
        holdback_seconds: float = EARLY_HOLDBACK_SECONDS,
        max_bytes: int = RECOVERY_BUFFER_MAX_BYTES,
        now: Callable[[], float] | None = None,
    ) -> None:
        self._holdback_seconds = holdback_seconds
        self._max_bytes = max_bytes
        self._now = now or time.monotonic
        self._events: list[str] = []
        self._bytes = 0
        self._started_at: float | None = None
        self.committed = False

    def push(self, event: str) -> list[str]:
        if self.committed:
            return [event]
        if self._started_at is None:
            self._started_at = self._now()
        self._events.append(event)
        self._bytes += len(event.encode("utf-8", errors="replace"))
        if (
            self._bytes >= self._max_bytes
            or self._now() - self._started_at >= self._holdback_seconds
        ):
            return self.flush()
        return []

    def flush(self) -> list[str]:
        if self.committed:
            return []
        self.committed = True
        events = self._events
        self._events = []
        self._bytes = 0
        self._started_at = None
        return events

    def discard(self) -> None:
        self._events = []
        self._bytes = 0
        self._started_at = None

    @property
    def has_buffered(self) -> bool:
        return bool(self._events)


class RecoveryController:
    """Own holdback and failure classification for one provider stream lifecycle."""

    def __init__(self, *, provider_name: str, request_id: str | None) -> None:
        self._provider_name = provider_name
        self._request_id = request_id
        self._holdback = RecoveryHoldbackBuffer()
        self._early_retry_count = 0
        self._midstream_recovery_count = 0

    @property
    def committed(self) -> bool:
        return self._holdback.committed

    @property
    def has_buffered(self) -> bool:
        return self._holdback.has_buffered

    @property
    def early_retries(self) -> int:
        return self._early_retry_count

    @property
    def midstream_recoveries(self) -> int:
        return self._midstream_recovery_count

    def push(self, event: str) -> list[str]:
        return self._holdback.push(event)

    def flush(self) -> list[str]:
        return self._holdback.flush()

    def discard(self) -> None:
        self._holdback.discard()

    def flush_uncommitted(self, decision: RecoveryDecision) -> list[str]:
        if not decision.committed and decision.has_buffered:
            return self.flush()
        return []

    def advance_failure(
        self,
        error: BaseException,
        *,
        stream_opened: bool,
        generated_output: bool,
        complete_tool_salvageable: bool,
    ) -> RecoveryDecision:
        retryable = is_retryable_stream_error(error)
        committed = self._holdback.committed
        has_buffered = self._holdback.has_buffered

        if (
            retryable
            and stream_opened
            and not committed
            and not complete_tool_salvageable
            and self._early_retry_count < EARLY_TRANSPARENT_MAX_RETRIES
        ):
            self._early_retry_count += 1
            self._holdback.discard()
            self._holdback = RecoveryHoldbackBuffer()
            trace_event(
                stage="provider",
                event="provider.recovery.early_retry",
                source="provider",
                provider=self._provider_name,
                request_id=self._request_id,
                retry_attempt=self._early_retry_count,
                retryable=True,
            )
            return RecoveryDecision(
                action=RecoveryFailureAction.EARLY_RETRY,
                retryable=True,
                committed=False,
                has_buffered=has_buffered,
                early_retry_attempt=self._early_retry_count,
            )

        if (
            retryable
            and generated_output
            and self._midstream_recovery_count < MIDSTREAM_RECOVERY_ATTEMPTS
        ):
            self._midstream_recovery_count += 1
            return RecoveryDecision(
                action=RecoveryFailureAction.MIDSTREAM_RECOVERY,
                retryable=True,
                committed=committed,
                has_buffered=has_buffered,
                midstream_recovery_attempt=self._midstream_recovery_count,
            )

        return RecoveryDecision(
            action=RecoveryFailureAction.FINAL_ERROR,
            retryable=retryable,
            committed=committed,
            has_buffered=has_buffered,
        )


def is_retryable_stream_error(exc: BaseException) -> bool:
    """Return whether a provider stream error can be retried/recovered."""
    if isinstance(exc, TruncatedProviderStreamError):
        return True
    if isinstance(exc, openai.AuthenticationError | openai.BadRequestError):
        return False
    if retryable_transient_status(exc) is not None:
        return True
    return isinstance(
        exc,
        (
            TimeoutError,
            httpx.ReadTimeout,
            httpx.ReadError,
            httpx.RemoteProtocolError,
            httpx.ConnectError,
            httpx.NetworkError,
            openai.APITimeoutError,
            openai.APIConnectionError,
        ),
    )


def tool_schemas_by_name(request: Any) -> dict[str, ToolSchema]:
    """Return Anthropic tool input schemas keyed by tool name."""
    schemas: dict[str, ToolSchema] = {}
    tools = getattr(request, "tools", None)
    if not tools:
        return schemas

    for tool in tools:
        name = _tool_attr(tool, "name")
        if not isinstance(name, str) or not name:
            continue
        schema = _tool_attr(tool, "input_schema")
        if not isinstance(schema, dict):
            schema = {"type": "object"}
        schemas[name] = ToolSchema(name=name, input_schema=deepcopy(schema))
    return schemas


def validate_tool_input(
    tool_name: str, parsed_input: dict[str, Any], schemas: dict[str, ToolSchema]
) -> bool:
    tool_schema = schemas.get(tool_name)
    if tool_schema is None:
        return True
    try:
        validator_cls = jsonschema.validators.validator_for(tool_schema.input_schema)
        validator_cls.check_schema(tool_schema.input_schema)
        validator_cls(tool_schema.input_schema).validate(parsed_input)
    except jsonschema.exceptions.SchemaError as exc:
        logger.warning("Skipping invalid tool schema for {}: {}", tool_name, exc)
        return True
    except jsonschema.exceptions.ValidationError:
        return False
    return True


def parse_complete_tool_input(
    raw_json: str, tool_name: str, schemas: dict[str, ToolSchema]
) -> dict[str, Any] | None:
    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    if not validate_tool_input(tool_name, parsed, schemas):
        return None
    return parsed


def accept_tool_json_repair(
    prefix: str,
    candidate: str,
    *,
    tool_name: str,
    schemas: dict[str, ToolSchema],
) -> ToolRepair | None:
    for suffix in _repair_suffix_candidates(prefix, candidate):
        combined = prefix + suffix
        parsed = parse_complete_tool_input(combined, tool_name, schemas)
        if parsed is not None:
            return ToolRepair(suffix=suffix, parsed_input=parsed)
    return None


def continuation_suffix(existing: str, candidate: str) -> str | None:
    existing = existing or ""
    candidate = candidate or ""
    if not candidate:
        return ""
    if not existing:
        return candidate
    if candidate.startswith(existing):
        return candidate[len(existing) :]

    max_overlap = min(len(existing), len(candidate))
    for size in range(max_overlap, 0, -1):
        if existing.endswith(candidate[:size]):
            return candidate[size:]

    if len(candidate) < max(200, len(existing) // 2):
        return candidate
    return None


def make_text_recovery_body(
    body: dict[str, Any],
    partial_text: str,
    partial_thinking: str = "",
) -> dict[str, Any]:
    """Build a text-only continuation request for either transport family."""
    recovery = deepcopy(body)
    recovery.pop("tools", None)
    recovery.pop("tool_choice", None)
    recovery["stream"] = True
    messages = _copied_messages(recovery)
    if partial_text:
        messages.append({"role": "assistant", "content": partial_text})
    prompt = _RECOVERY_USER_PREFIX
    if partial_thinking:
        prompt = f"{_RECOVERY_THINKING_PREFIX}{partial_thinking}\n\n{prompt}"
    messages.append({"role": "user", "content": prompt})
    recovery["messages"] = messages
    return recovery


def make_tool_repair_body(
    body: dict[str, Any],
    *,
    tool_name: str,
    prefix: str,
    input_schema: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build a text-only request asking for a JSON suffix."""
    recovery = deepcopy(body)
    recovery.pop("tools", None)
    recovery.pop("tool_choice", None)
    recovery["stream"] = True
    messages = _copied_messages(recovery)
    messages.append(
        {
            "role": "user",
            "content": _tool_repair_prompt(
                tool_name=tool_name, prefix=prefix, input_schema=input_schema
            ),
        }
    )
    recovery["messages"] = messages
    return recovery


def _tool_attr(tool: Any, attr: str) -> Any:
    if isinstance(tool, dict):
        return tool.get(attr)
    return getattr(tool, attr, None)


def _copied_messages(body: dict[str, Any]) -> list[Any]:
    messages = body.get("messages")
    return deepcopy(messages) if isinstance(messages, list) else []


def _repair_suffix_candidates(prefix: str, candidate: str) -> list[str]:
    raw = candidate.strip()
    if not raw:
        return []
    candidates: list[str] = []
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    candidates.append(raw)
    if raw.startswith(prefix):
        candidates.append(raw[len(prefix) :])
    return list(dict.fromkeys(candidates))


def _tool_repair_prompt(
    *, tool_name: str, prefix: str, input_schema: dict[str, Any] | None
) -> str:
    schema_text = json.dumps(input_schema or {"type": "object"}, separators=(",", ":"))
    return (
        "A streamed tool call was interrupted while writing JSON arguments.\n"
        f"Tool name: {tool_name}\n"
        f"JSON schema: {schema_text}\n"
        f"Already emitted JSON prefix: {prefix}\n\n"
        "Return only the exact missing JSON suffix needed to complete the same object. "
        "Do not repeat the prefix. Do not include markdown or explanation."
    )
