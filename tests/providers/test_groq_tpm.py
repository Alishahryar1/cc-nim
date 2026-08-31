"""Groq request-local TPM ceiling correction contracts."""

from copy import deepcopy
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import httpx2
import openai
import pytest

from free_claude_code.config.provider_catalog import GROQ_DEFAULT_BASE
from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.core.reasoning import ReasoningEffort, ReasoningPolicy
from free_claude_code.providers.admission import ProviderOperationKind
from free_claude_code.providers.groq import GroqProvider
from free_claude_code.providers.groq.tpm import correct_tpm_completion_budget
from tests.providers.request_factory import make_messages_request
from tests.providers.support import immediate_admission, make_provider_config

_MODEL = "openai/gpt-oss-120b"
_LIMIT = 8_000
_REQUESTED = 26_206
_PREVIOUS = 24_576
_CORRECTED = 6_370
_TPM_HEADER = "x-ratelimit-limit-tokens"
_MESSAGE = (
    "Request too large for model `openai/gpt-oss-120b` in organization "
    "`org_private` service tier `on_demand` on tokens per minute (TPM): "
    "Limit 8000, Requested 26206, please reduce your message size and try again."
)


class _StatusError(Exception):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None,
        body: object,
        response: object | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body
        self.response = response


class _Response:
    def __init__(self, headers: object) -> None:
        self.headers = headers


def _error(
    *,
    status: int = 413,
    message: str = _MESSAGE,
    error_type: str = "tokens",
    code: str = "rate_limit_exceeded",
    wrapped: bool = False,
    headers: list[tuple[str, str]] | None = None,
) -> openai.APIStatusError:
    request = httpx2.Request("POST", f"{GROQ_DEFAULT_BASE}/chat/completions")
    response = httpx2.Response(status, request=request, headers=headers)
    detail = {"message": message, "type": error_type, "code": code}
    body = {"error": detail} if wrapped else detail
    return openai.APIStatusError("Groq rejected request", response=response, body=body)


def _body(**updates: object) -> dict[str, object]:
    body: dict[str, object] = {
        "model": _MODEL,
        "messages": [{"role": "user", "content": "hello"}],
        "max_completion_tokens": _PREVIOUS,
        "reasoning_effort": "high",
        "stream": True,
    }
    body.update(updates)
    return body


def _provider(*, max_attempts: int = 5) -> GroqProvider:
    return GroqProvider(
        make_provider_config(
            api_key="test_groq_key",
            base_url=GROQ_DEFAULT_BASE,
            rate_limit=10,
            rate_window=60,
        ),
        admission=immediate_admission(
            provider_name="GROQ",
            max_attempts=max_attempts,
        ),
    )


def _chunk(*, content: str | None = None, finish_reason: str | None = None):
    return MagicMock(
        choices=[
            MagicMock(
                delta=MagicMock(
                    content=content,
                    reasoning_content=None,
                    tool_calls=None,
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=None,
    )


async def _successful_stream(text: str = "visible"):
    yield _chunk(content=text)
    yield _chunk(finish_reason="stop")


async def _failing_stream():
    if False:
        yield _chunk()
    raise httpx.ReadError("early cutoff")


def test_exact_observed_failure_reduces_only_completion_budget() -> None:
    body = _body()
    original = deepcopy(body)

    correction = correct_tpm_completion_budget(_error(), body)

    assert correction is not None
    assert correction.limit == _LIMIT
    assert correction.requested == _REQUESTED
    assert correction.previous_max_completion_tokens == _PREVIOUS
    assert correction.corrected_max_completion_tokens == _CORRECTED
    assert correction.body == {**body, "max_completion_tokens": _CORRECTED}
    assert correction.body is not body
    assert body == original


@pytest.mark.parametrize("wrapped", [False, True], ids=["sdk_inner", "raw_wrapper"])
def test_accepts_sdk_inner_or_one_raw_error_wrapper(wrapped: bool) -> None:
    correction = correct_tpm_completion_budget(_error(wrapped=wrapped), _body())

    assert correction is not None
    assert correction.corrected_max_completion_tokens == _CORRECTED


def test_accepts_bounded_case_and_whitespace_variation() -> None:
    error = _error(
        message="Request on TOKENS  per minute ( TPM ) : LIMIT 8000 , REQUESTED 26206"
    )

    assert correct_tpm_completion_budget(error, _body()) is not None


@pytest.mark.parametrize(
    "error",
    [
        _error(status=429),
        _error(error_type="requests"),
        _error(code="context_length_exceeded"),
        _error(message="Request requires 26206 tokens but limit is 8000"),
        _error(message="tokens per minute (TPM): Limit 8000, Requested 8000"),
        _error(message="tokens per minute (TPM): Limit 0, Requested 26206"),
        _error(
            message=(
                "tokens per minute (TPM): Limit 8000, Requested 26206; "
                "tokens per minute (TPM): Limit 8000, Requested 26206"
            )
        ),
    ],
    ids=[
        "wrong_status",
        "wrong_type",
        "wrong_code",
        "unbound_numbers",
        "not_over_limit",
        "zero_limit",
        "ambiguous_clauses",
    ],
)
def test_rejects_unrecognized_or_ambiguous_errors(
    error: openai.APIStatusError,
) -> None:
    assert correct_tpm_completion_budget(error, _body()) is None


@pytest.mark.parametrize(
    "body",
    [
        _body(max_completion_tokens=None),
        _body(max_completion_tokens=True),
        _body(max_completion_tokens="24576"),
        _body(max_completion_tokens=1),
        _body(extra_body={"max_completion_tokens": _PREVIOUS}),
        _body(extra_body={"max_tokens": _PREVIOUS}),
    ],
    ids=[
        "missing_integer",
        "boolean",
        "string",
        "nonpositive_correction",
        "extra_completion_override",
        "extra_tokens_override",
    ],
)
def test_rejects_bodies_that_cannot_be_authoritatively_lowered(
    body: dict[str, object],
) -> None:
    assert correct_tpm_completion_budget(_error(), body) is None


@pytest.mark.parametrize(
    "headers,expected",
    [
        (None, True),
        ([("x-ratelimit-limit-tokens", "8000")], True),
        ([("X-RateLimit-Limit-Tokens", "8001")], False),
        ([("x-ratelimit-limit-tokens", "eight-thousand")], False),
        (
            [
                ("x-ratelimit-limit-tokens", "8000"),
                ("x-ratelimit-limit-tokens", "8000"),
            ],
            False,
        ),
    ],
    ids=["absent", "matching", "mismatch", "malformed", "repeated"],
)
def test_rate_limit_header_must_agree_when_present(
    headers: list[tuple[str, str]] | None,
    expected: bool,
) -> None:
    correction = correct_tpm_completion_budget(_error(headers=headers), _body())

    assert (correction is not None) is expected


def test_rejects_present_header_that_cannot_be_validated() -> None:
    error = _StatusError(
        "Groq rejected request",
        status_code=413,
        body={
            "message": _MESSAGE,
            "type": "tokens",
            "code": "rate_limit_exceeded",
        },
        response=_Response({_TPM_HEADER: object()}),
    )

    assert correct_tpm_completion_budget(error, _body()) is None


@pytest.mark.parametrize("source", ["message", "header"])
def test_rejects_oversized_decimal_without_masking_the_413(source: str) -> None:
    oversized = "9" * 5_000
    error = (
        _error(
            message=(
                f"tokens per minute (TPM): Limit {oversized}, Requested {oversized}0"
            )
        )
        if source == "message"
        else _error(headers=[(_TPM_HEADER, oversized)])
    )

    assert correct_tpm_completion_budget(error, _body()) is None


def test_requires_machine_status_not_body_status() -> None:
    error = _StatusError(
        "Groq rejected request",
        status_code=None,
        body={
            "status": 413,
            "message": _MESSAGE,
            "type": "tokens",
            "code": "rate_limit_exceeded",
        },
    )

    assert correct_tpm_completion_budget(error, _body()) is None


@pytest.mark.asyncio
async def test_provider_corrects_once_before_one_downstream_stream() -> None:
    provider = _provider()
    request = make_messages_request(_MODEL, max_tokens=_PREVIOUS)
    create = AsyncMock(side_effect=[_error(), _successful_stream()])

    with (
        patch.object(provider._client.chat.completions, "create", create),
        patch("free_claude_code.providers.admission.asyncio.sleep") as sleep,
        patch("free_claude_code.providers.admission.trace_event") as trace,
    ):
        raw = "".join(
            [
                event
                async for event in provider.stream_messages(
                    request,
                    reasoning=ReasoningPolicy.on(effort=ReasoningEffort.MAX),
                )
            ]
        )

    events = parse_sse_text(raw)
    assert create.await_count == 2
    assert create.await_args_list[0].kwargs["max_completion_tokens"] == _PREVIOUS
    assert create.await_args_list[1].kwargs["max_completion_tokens"] == _CORRECTED
    assert create.await_args_list[0].kwargs["reasoning_effort"] == "high"
    assert create.await_args_list[1].kwargs["reasoning_effort"] == "high"
    assert [event.event for event in events].count("message_start") == 1
    assert [event.event for event in events].count("message_stop") == 1
    sleep.assert_not_awaited()
    assert all(
        call.kwargs.get("event") != "provider.recovery.opened"
        for call in trace.call_args_list
    )


@pytest.mark.asyncio
async def test_second_tpm_rejection_is_terminal_without_a_third_create() -> None:
    provider = _provider()
    body = provider._build_request_body(
        make_messages_request(_MODEL, max_tokens=_PREVIOUS),
        reasoning=ReasoningPolicy.on(effort=ReasoningEffort.MAX),
    )
    create = AsyncMock(side_effect=[_error(), _error()])

    with (
        patch.object(provider._client.chat.completions, "create", create),
        pytest.raises(openai.APIStatusError),
    ):
        await provider._create_stream(
            body,
            provider._admission.start_execution(),
            ProviderOperationKind.GENERATION,
        )

    assert create.await_count == 2


@pytest.mark.asyncio
async def test_precommit_stream_retry_cannot_reset_the_tpm_correction_guard() -> None:
    provider = _provider()
    request = make_messages_request(_MODEL, max_tokens=_PREVIOUS)
    tpm_error = _error(message="tokens per minute (TPM): Limit 8000, Requested 9000")
    create = AsyncMock(
        side_effect=[tpm_error, _failing_stream(), tpm_error, _successful_stream()]
    )

    with (
        patch.object(provider._client.chat.completions, "create", create),
        pytest.raises(ExecutionFailure),
    ):
        _ = [event async for event in provider.stream_messages(request)]

    corrected = _PREVIOUS - 1_000
    assert create.await_count == 3
    assert create.await_args_list[0].kwargs["max_completion_tokens"] == _PREVIOUS
    assert create.await_args_list[1].kwargs["max_completion_tokens"] == corrected
    assert create.await_args_list[2].kwargs["max_completion_tokens"] == corrected


@pytest.mark.asyncio
async def test_single_attempt_budget_does_not_create_corrected_request() -> None:
    provider = _provider(max_attempts=1)
    body = provider._build_request_body(
        make_messages_request(_MODEL, max_tokens=_PREVIOUS),
        reasoning=ReasoningPolicy.on(effort=ReasoningEffort.MAX),
    )
    create = AsyncMock(side_effect=_error())

    with (
        patch.object(provider._client.chat.completions, "create", create),
        pytest.raises(openai.APIStatusError),
    ):
        await provider._create_stream(
            body,
            provider._admission.start_execution(),
            ProviderOperationKind.GENERATION,
        )

    assert create.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tpm_first", [True, False], ids=["tpm_first", "reasoning_first"]
)
async def test_tpm_and_reasoning_corrections_compose(tpm_first: bool) -> None:
    provider = _provider()
    body = provider._build_request_body(
        make_messages_request(_MODEL, max_tokens=_PREVIOUS),
        reasoning=ReasoningPolicy.on(effort=ReasoningEffort.MAX),
    )
    vocabulary_message = (
        "`reasoning_effort` value `high` must be one of `none` or `default`"
    )
    vocabulary_error = _StatusError(
        vocabulary_message,
        status_code=400,
        body={"message": vocabulary_message, "type": "invalid_request_error"},
    )
    errors = [_error(), vocabulary_error] if tpm_first else [vocabulary_error, _error()]
    create = AsyncMock(side_effect=[*errors, _successful_stream()])

    with patch.object(provider._client.chat.completions, "create", create):
        stream, used_body, attempt = await provider._create_stream(
            body,
            provider._admission.start_execution(),
            ProviderOperationKind.GENERATION,
        )
        await stream.aclose()
        await attempt.aclose()

    assert create.await_count == 3
    assert used_body["max_completion_tokens"] == _CORRECTED
    assert used_body["reasoning_effort"] == "default"


@pytest.mark.asyncio
async def test_tpm_limit_is_not_cached_across_requests() -> None:
    provider = _provider()
    request = make_messages_request(_MODEL, max_tokens=_PREVIOUS)
    body = provider._build_request_body(request)
    create = AsyncMock(
        side_effect=[_error(), _successful_stream(), _successful_stream()]
    )

    with patch.object(provider._client.chat.completions, "create", create):
        first_stream, _, first_attempt = await provider._create_stream(
            body,
            provider._admission.start_execution(),
            ProviderOperationKind.GENERATION,
        )
        await first_stream.aclose()
        await first_attempt.aclose()
        second_stream, _, second_attempt = await provider._create_stream(
            provider._build_request_body(request),
            provider._admission.start_execution(),
            ProviderOperationKind.GENERATION,
        )
        await second_stream.aclose()
        await second_attempt.aclose()

    assert create.await_args_list[1].kwargs["max_completion_tokens"] == _CORRECTED
    assert create.await_args_list[2].kwargs["max_completion_tokens"] == _PREVIOUS
