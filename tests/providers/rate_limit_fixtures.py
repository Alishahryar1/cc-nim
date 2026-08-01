"""Documented upstream 429 shapes and the retry hint each one carries.

Header names, header formats, and body shapes follow the providers' current
public documentation:

* Gemini — ``https://generativelanguage.googleapis.com/v1beta/openai/`` relays
  Google's standard error envelope, so a ``429 RESOURCE_EXHAUSTED`` carries no
  ``Retry-After`` header and puts the machine-readable delay in
  ``error.details[] @type=type.googleapis.com/google.rpc.RetryInfo`` as a
  protobuf ``Duration`` string (``"34s"``). Daily-quota exhaustion sometimes
  omits ``RetryInfo`` entirely.
* Groq — documents ``retry-after`` (seconds) as set only on 429, alongside the
  always-present ``x-ratelimit-*`` counters whose reset values use Go duration
  syntax (``"2m59.56s"``) rather than the ``Retry-After`` format.
* OpenRouter — documents ``X-RateLimit-Reset`` (Unix epoch **milliseconds**) on
  platform 429s and ``Retry-After`` only when every attempted upstream returned
  a retry hint.
* DeepSeek — documents no retry header for 429; only a cause and a pacing
  recommendation.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import httpx
import openai

RETRY_INFO_TYPE = "type.googleapis.com/google.rpc.RetryInfo"
QUOTA_FAILURE_TYPE = "type.googleapis.com/google.rpc.QuotaFailure"
HELP_TYPE = "type.googleapis.com/google.rpc.Help"

_HINT_HEADER = "retry-after header"
_HINT_RETRY_INFO = "google.rpc.RetryInfo retryDelay"
_HINT_NONE = "none"


@dataclass(frozen=True, slots=True)
class RateLimitFixture:
    """One documented provider 429 response and its expected parsed retry hint."""

    name: str
    provider: str
    scope: str
    headers: Mapping[str, str]
    body: Mapping[str, Any]
    retry_after_s: float | None
    hint_source: str


def rate_limit_error(fixture: RateLimitFixture) -> openai.RateLimitError:
    """Build the OpenAI-compatible exception a provider raises for one fixture."""
    request = httpx.Request("POST", "https://provider.test/v1/chat/completions")
    response = httpx.Response(
        429,
        request=request,
        headers=dict(fixture.headers),
        json=fixture.body,
    )
    return openai.RateLimitError(
        f"Error code: 429 - {fixture.body}",
        response=response,
        body=fixture.body,
    )


def _google_quota_body(
    *,
    message: str,
    quota_id: str,
    quota_value: int,
    retry_delay: str | None,
) -> dict[str, Any]:
    details: list[dict[str, Any]] = [
        {
            "@type": QUOTA_FAILURE_TYPE,
            "violations": [
                {
                    "quotaMetric": (
                        "generativelanguage.googleapis.com/"
                        "generate_content_free_tier_requests"
                    ),
                    "quotaId": quota_id,
                    "quotaValue": str(quota_value),
                },
            ],
        },
        {
            "@type": HELP_TYPE,
            "links": [
                {
                    "description": "Learn more about Gemini API quotas",
                    "url": "https://ai.google.dev/gemini-api/docs/rate-limits",
                },
            ],
        },
    ]
    if retry_delay is not None:
        details.append({"@type": RETRY_INFO_TYPE, "retryDelay": retry_delay})
    return {
        "error": {
            "code": 429,
            "message": message,
            "status": "RESOURCE_EXHAUSTED",
            "details": details,
        },
    }


GEMINI_RPM = RateLimitFixture(
    name="gemini_rpm",
    provider="GEMINI",
    scope="requests per minute (free tier)",
    headers={"content-type": "application/json"},
    body=_google_quota_body(
        message=(
            "You exceeded your current quota, please check your plan and billing "
            "details. * Quota exceeded for metric: generativelanguage.googleapis"
            ".com/generate_content_free_tier_requests, limit: 10"
        ),
        quota_id="GenerateRequestsPerMinutePerProjectPerModel-FreeTier",
        quota_value=10,
        retry_delay="24.5s",
    ),
    retry_after_s=24.5,
    hint_source=_HINT_RETRY_INFO,
)

GEMINI_RPD = RateLimitFixture(
    name="gemini_rpd",
    provider="GEMINI",
    scope="requests per day (free tier)",
    headers={"content-type": "application/json"},
    body=_google_quota_body(
        message=(
            "You exceeded your current quota, please check your plan and billing "
            "details. * Quota exceeded for metric: generativelanguage.googleapis"
            ".com/generate_content_free_tier_requests, limit: 250"
        ),
        quota_id="GenerateRequestsPerDayPerProjectPerModel-FreeTier",
        quota_value=250,
        retry_delay="34s",
    ),
    retry_after_s=34.0,
    hint_source=_HINT_RETRY_INFO,
)

GEMINI_RPD_WITHOUT_RETRY_INFO = RateLimitFixture(
    name="gemini_rpd_without_retry_info",
    provider="GEMINI",
    scope="requests per day (free tier), no RetryInfo detail",
    headers={"content-type": "application/json"},
    body=_google_quota_body(
        message=(
            "You exceeded your current quota, please check your plan and billing "
            "details."
        ),
        quota_id="GenerateRequestsPerDayPerProjectPerModel-FreeTier",
        quota_value=250,
        retry_delay=None,
    ),
    retry_after_s=None,
    hint_source=_HINT_NONE,
)

GROQ_TPM = RateLimitFixture(
    name="groq_tpm",
    provider="GROQ",
    scope="tokens per minute",
    headers={
        "content-type": "application/json",
        "retry-after": "2",
        "x-ratelimit-limit-requests": "14400",
        "x-ratelimit-limit-tokens": "6000",
        "x-ratelimit-remaining-requests": "14370",
        "x-ratelimit-remaining-tokens": "0",
        "x-ratelimit-reset-requests": "2m59.56s",
        "x-ratelimit-reset-tokens": "7.66s",
    },
    body={
        "error": {
            "message": (
                "Rate limit reached for model `llama-3.3-70b-versatile` in "
                "organization `org_test` service tier `on_demand` on tokens per "
                "minute (TPM): Limit 6000, Used 6000, Requested ~100."
            ),
            "type": "tokens",
            "code": "rate_limit_exceeded",
        },
    },
    retry_after_s=2.0,
    hint_source=_HINT_HEADER,
)

GROQ_RPD = RateLimitFixture(
    name="groq_rpd",
    provider="GROQ",
    scope="requests per day",
    headers={
        "content-type": "application/json",
        "retry-after": "7200",
        "x-ratelimit-limit-requests": "14400",
        "x-ratelimit-remaining-requests": "0",
        "x-ratelimit-reset-requests": "2h0m0s",
    },
    body={
        "error": {
            "message": (
                "Rate limit reached for model `llama-3.3-70b-versatile` in "
                "organization `org_test` service tier `on_demand` on requests per "
                "day (RPD): Limit 14400, Used 14400, Requested 1."
            ),
            "type": "requests",
            "code": "rate_limit_exceeded",
        },
    },
    retry_after_s=7200.0,
    hint_source=_HINT_HEADER,
)

OPENROUTER_PLATFORM = RateLimitFixture(
    name="openrouter_platform",
    provider="OPEN_ROUTER",
    scope="platform limit, reset header only",
    headers={
        "content-type": "application/json",
        "x-ratelimit-limit": "50",
        "x-ratelimit-remaining": "0",
        "x-ratelimit-reset": "1777420800000",
    },
    body={
        "error": {
            "code": 429,
            "message": "Rate limit exceeded",
            "metadata": {"error_type": "rate_limit_exceeded"},
        },
    },
    retry_after_s=None,
    hint_source=_HINT_NONE,
)

OPENROUTER_UPSTREAM_HINT = RateLimitFixture(
    name="openrouter_upstream_hint",
    provider="OPEN_ROUTER",
    scope="every upstream returned a retry hint",
    headers={
        "content-type": "application/json",
        "retry-after": "30",
        "x-ratelimit-limit": "50",
        "x-ratelimit-remaining": "0",
        "x-ratelimit-reset": "1777420800000",
    },
    body={
        "error": {
            "code": 429,
            "message": "Provider returned error",
            "metadata": {"error_type": "rate_limit_exceeded"},
        },
    },
    retry_after_s=30.0,
    hint_source=_HINT_HEADER,
)

DEEPSEEK = RateLimitFixture(
    name="deepseek",
    provider="DEEPSEEK",
    scope="requests sent too quickly",
    headers={"content-type": "application/json"},
    body={
        "error": {
            "message": (
                "You are sending requests too quickly. Please pace your requests "
                "reasonably."
            ),
            "type": "rate_limit_reached",
            "code": "rate_limit_reached",
        },
    },
    retry_after_s=None,
    hint_source=_HINT_NONE,
)

DOCUMENTED_RATE_LIMIT_FIXTURES: tuple[RateLimitFixture, ...] = (
    GEMINI_RPM,
    GEMINI_RPD,
    GEMINI_RPD_WITHOUT_RETRY_INFO,
    GROQ_TPM,
    GROQ_RPD,
    OPENROUTER_PLATFORM,
    OPENROUTER_UPSTREAM_HINT,
    DEEPSEEK,
)
