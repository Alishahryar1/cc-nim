"""Tests for multi-credential parsing, rotation state, and the rotating wrapper."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from free_claude_code.config.admin.manifest import FIELD_BY_KEY
from free_claude_code.config.provider_catalog import PROVIDER_CATALOG
from free_claude_code.config.settings import Settings
from free_claude_code.core.anthropic.models import Message, MessagesRequest
from free_claude_code.core.reasoning import DEFAULT_REASONING_POLICY, ReasoningPolicy
from free_claude_code.providers.base import BaseProvider, ProviderConfig
from free_claude_code.providers.credential_rotation import (
    CredentialRotationState,
    error_justifies_rotation,
)
from free_claude_code.providers.runtime.config import (
    build_provider_config,
    credential_rotation_policy,
    parse_credential_keys,
)
from free_claude_code.providers.runtime.rotating import RotatingProvider


class _RetryableError(Exception):
    status_code = 429


class _InvalidRequestError(Exception):
    status_code = 400


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


def _request() -> MessagesRequest:
    return MessagesRequest(
        model="test-model",
        messages=[Message(role="user", content="hi")],
    )


def test_parse_credential_keys_splits_and_strips():
    assert parse_credential_keys("k1, k2 ,k3") == ("k1", "k2", "k3")
    assert parse_credential_keys("solo") == ("solo",)
    assert parse_credential_keys("") == ()


def test_credential_rotation_policy_defaults_to_single():
    descriptor = PROVIDER_CATALOG["nvidia_nim"]
    assert credential_rotation_policy(descriptor, _settings()) == "single"


def test_credential_rotation_policy_reads_process_env(monkeypatch):
    descriptor = PROVIDER_CATALOG["nvidia_nim"]
    monkeypatch.setenv("NVIDIA_NIM_API_KEY_ROTATION", "round_robin")
    assert credential_rotation_policy(descriptor, _settings()) == "round_robin"


def test_credential_rotation_policy_ignores_unknown_values(monkeypatch):
    descriptor = PROVIDER_CATALOG["nvidia_nim"]
    monkeypatch.setenv("NVIDIA_NIM_API_KEY_ROTATION", "bogus")
    assert credential_rotation_policy(descriptor, _settings()) == "single"


def test_build_provider_config_parses_multiple_keys():
    descriptor = PROVIDER_CATALOG["nvidia_nim"]
    config = build_provider_config(
        descriptor, _settings(nvidia_nim_api_key="k1,k2 , k3")
    )
    assert config.api_keys == ("k1", "k2", "k3")
    assert config.api_key == "k1"
    assert config.credential_rotation == "single"


@pytest.mark.asyncio
async def test_round_robin_state_advances():
    state = CredentialRotationState(3, "round_robin")
    assert await state.acquire() == 0
    assert await state.acquire() == 1
    assert await state.acquire() == 2
    assert await state.acquire() == 0


@pytest.mark.asyncio
async def test_on_error_state_sticks_then_fails_over():
    state = CredentialRotationState(2, "on_error")
    assert await state.acquire() == 0
    assert await state.acquire() == 0
    rotate = await state.report_failure(0, _RetryableError())
    assert rotate is True
    assert await state.acquire() == 1


@pytest.mark.asyncio
async def test_backed_off_keys_are_skipped_in_round_robin():
    state = CredentialRotationState(3, "round_robin")
    await state.report_failure(1, _RetryableError())
    assert await state.acquire() == 0
    assert await state.acquire() == 2
    assert await state.acquire() == 0


@pytest.mark.asyncio
async def test_least_used_picks_least_requested_healthy_key():
    state = CredentialRotationState(3, "least_used")
    assert await state.acquire() == 0
    assert await state.acquire() == 1
    assert await state.acquire() == 2
    # All used once; key 0 was used longest ago
    assert await state.acquire() == 0
    # Bench key 0; least-used must skip it
    await state.report_failure(0, _RetryableError())
    assert await state.acquire() == 1


@pytest.mark.asyncio
async def test_failover_sticks_to_first_healthy_key():
    state = CredentialRotationState(3, "failover")
    assert await state.acquire() == 0
    assert await state.acquire() == 0
    await state.report_failure(0, _RetryableError())
    assert await state.acquire() == 1
    assert await state.acquire() == 1


@pytest.mark.asyncio
async def test_cooldown_tiers_escalate_on_repeated_failures():
    state = CredentialRotationState(1, "failover")
    await state.report_failure(0, _RetryableError())
    metrics = state.get_metrics()[0]
    assert metrics["state"] == "COOLDOWN"
    assert metrics["tier"] == 1
    first = metrics["cooldown_remaining"]
    assert 9.0 < first <= 10.0

    await state.report_failure(0, _RetryableError())
    metrics = state.get_metrics()[0]
    assert metrics["tier"] == 2
    assert 29.0 < metrics["cooldown_remaining"] <= 30.0


@pytest.mark.asyncio
async def test_circuit_opens_after_three_consecutive_failures():
    state = CredentialRotationState(1, "failover")
    for _ in range(3):
        await state.report_failure(0, Exception("boom"))
    assert state.get_metrics()[0]["state"] == "CIRCUIT_OPEN"


@pytest.mark.asyncio
async def test_auth_failures_escalate_lockout_tiers():
    state = CredentialRotationState(2, "failover")

    class _AuthError(Exception):
        status_code = 401

    await state.report_failure(0, _AuthError())
    metrics = state.get_metrics()[0]
    assert metrics["state"] == "LOCKED_OUT"
    assert 290.0 < metrics["lockout_remaining"] <= 300.0

    await state.report_failure(0, _AuthError())
    metrics = state.get_metrics()[0]
    assert 3500.0 < metrics["lockout_remaining"] <= 3600.0

    await state.report_failure(0, _AuthError())
    metrics = state.get_metrics()[0]
    assert 86300.0 < metrics["lockout_remaining"] <= 86400.0


@pytest.mark.asyncio
async def test_acquire_returns_minus_one_when_all_keys_benched():
    state = CredentialRotationState(2, "round_robin")
    await state.report_failure(0, _RetryableError())
    await state.report_failure(1, _RetryableError())
    assert await state.acquire() == -1
    wait = await state.shortest_cooldown_remaining()
    assert 0 < wait <= 10.0


@pytest.mark.asyncio
async def test_report_success_restores_health():
    state = CredentialRotationState(1, "failover")
    await state.report_failure(0, _RetryableError())
    await state.report_success(0)
    metrics = state.get_metrics()[0]
    assert metrics["state"] == "HEALTHY"
    assert metrics["tier"] == 0
    assert await state.acquire() == 0


def test_error_justifies_rotation():
    assert error_justifies_rotation(_RetryableError()) is True
    assert error_justifies_rotation(_InvalidRequestError()) is False


class _FakeProvider(BaseProvider):
    """Provider double yielding canned chunks with optional failure points."""

    def __init__(
        self,
        *,
        chunks: tuple[str, ...] = ("chunk",),
        fail_before_first: Exception | None = None,
        fail_after_first: Exception | None = None,
    ) -> None:
        super().__init__(ProviderConfig(api_key="k", base_url="http://x"))
        self._chunks = chunks
        self._fail_before_first = fail_before_first
        self._fail_after_first = fail_after_first
        self.calls = 0

    def preflight_stream(
        self,
        request: MessagesRequest,
        *,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
    ) -> None:
        return None

    async def cleanup(self) -> None:
        return None

    async def list_model_ids(self) -> frozenset[str]:
        return frozenset({"test-model"})

    def stream_response(
        self,
        request: MessagesRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
    ) -> AsyncIterator[str]:
        self.calls += 1
        chunks = self._chunks
        fail_before = self._fail_before_first
        fail_after = self._fail_after_first

        async def _gen() -> AsyncIterator[str]:
            if fail_before is not None:
                raise fail_before
            first = True
            for chunk in chunks:
                yield chunk
                if first and fail_after is not None:
                    raise fail_after
                first = False

        return _gen()


def _rotating(providers: list[_FakeProvider], policy: str) -> RotatingProvider:
    config = ProviderConfig(
        api_key="k1",
        base_url="http://x",
        api_keys=tuple(f"k{i + 1}" for i in range(len(providers))),
        credential_rotation=policy,
    )
    state = CredentialRotationState(len(providers), policy)
    return RotatingProvider(config, providers, state)


@pytest.mark.asyncio
async def test_rotating_provider_round_robin_across_requests():
    first = _FakeProvider(chunks=("a",))
    second = _FakeProvider(chunks=("b",))
    provider = _rotating([first, second], "round_robin")

    assert [c async for c in provider.stream_response(_request())] == ["a"]
    assert [c async for c in provider.stream_response(_request())] == ["b"]
    assert first.calls == 1
    assert second.calls == 1


@pytest.mark.asyncio
async def test_rotating_provider_fails_over_before_first_chunk():
    first = _FakeProvider(fail_before_first=_RetryableError())
    second = _FakeProvider(chunks=("ok",))
    provider = _rotating([first, second], "on_error")

    assert [c async for c in provider.stream_response(_request())] == ["ok"]
    assert first.calls == 1
    assert second.calls == 1


@pytest.mark.asyncio
async def test_rotating_provider_does_not_rotate_non_rotatable_errors():
    first = _FakeProvider(fail_before_first=_InvalidRequestError())
    second = _FakeProvider(chunks=("ok",))
    provider = _rotating([first, second], "on_error")

    with pytest.raises(_InvalidRequestError):
        [c async for c in provider.stream_response(_request())]
    assert second.calls == 0


@pytest.mark.asyncio
async def test_rotating_provider_does_not_retry_after_output_started():
    first = _FakeProvider(chunks=("partial",), fail_after_first=_RetryableError())
    second = _FakeProvider(chunks=("ok",))
    provider = _rotating([first, second], "on_error")

    chunks: list[str] = []
    with pytest.raises(_RetryableError):
        async for chunk in provider.stream_response(_request()):
            chunks.append(chunk)
    assert chunks == ["partial"]
    assert second.calls == 0


def test_admin_manifest_exposes_rotation_select_for_nvidia_nim():
    field = FIELD_BY_KEY.get("NVIDIA_NIM_API_KEY_ROTATION")
    assert field is not None
    assert field.field_type == "select"
    assert set(field.options) == {"single", "round_robin", "least_used", "failover"}
    assert field.restart_required is True
