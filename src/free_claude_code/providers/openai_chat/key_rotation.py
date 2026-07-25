"""Key rotation and fallback for OpenAI-compatible providers."""

import asyncio
import random
import threading
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Any, TypeVar

import httpx
import openai
from loguru import logger
from openai import AsyncOpenAI

from free_claude_code.config.credentials import CredentialStrategy, get_credential_strategy
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.failure_policy import (
    is_retryable_provider_error,
    underlying_provider_error,
)

T = TypeVar("T")


@dataclass(slots=True)
class _KeyState:
    failures: int = 0
    cooldown_until: float = 0.0


@dataclass(frozen=True, slots=True)
class CredentialLease:
    index: int
    api_key: str


class ProviderCredentialPool:
    """Choose among multiple API keys and temporarily cool down bad ones."""

    def __init__(
        self,
        api_keys: Sequence[str],
        *,
        strategy: CredentialStrategy,
    ) -> None:
        keys = tuple(key.strip() for key in api_keys if key and key.strip())
        if not keys:
            raise ValueError("ProviderCredentialPool requires at least one API key.")
        self._keys = keys
        self._strategy = strategy
        self._states = [_KeyState() for _ in self._keys]
        self._round_robin_cursor = 0
        self._lock = threading.Lock()
        self._rng = random.Random()

    @property
    def size(self) -> int:
        return len(self._keys)

    def select(self) -> CredentialLease:
        """Pick the next healthy key, preferring the configured strategy."""

        now = time.monotonic()
        with self._lock:
            healthy = [
                index
                for index, state in enumerate(self._states)
                if state.cooldown_until <= now
            ]
            if not healthy:
                index = min(
                    range(len(self._keys)),
                    key=lambda candidate: self._states[candidate].cooldown_until,
                )
                return CredentialLease(index=index, api_key=self._keys[index])

            if self._strategy is CredentialStrategy.SEQUENTIAL:
                index = healthy[0]
            elif self._strategy is CredentialStrategy.ROUND_ROBIN:
                index = self._select_round_robin(healthy)
            else:
                index = self._rng.choice(healthy)
            return CredentialLease(index=index, api_key=self._keys[index])

    def record_success(self, index: int) -> None:
        """Mark a key as healthy again after a successful request."""

        with self._lock:
            state = self._states[index]
            state.failures = 0
            state.cooldown_until = 0.0

    def record_failure(self, index: int, error: BaseException) -> None:
        """Apply a temporary cooldown after a key-specific failure."""

        cooldown = _cooldown_seconds(error, failures=self._states[index].failures + 1)
        with self._lock:
            state = self._states[index]
            state.failures += 1
            state.cooldown_until = max(state.cooldown_until, time.monotonic() + cooldown)

    def _select_round_robin(self, healthy: list[int]) -> int:
        """Return the next healthy key in circular order."""

        key_count = len(self._keys)
        start = self._round_robin_cursor % key_count
        for offset in range(key_count):
            index = (start + offset) % key_count
            if index in healthy:
                self._round_robin_cursor = (index + 1) % key_count
                return index
        return healthy[0]


class _ModelsEndpoint:
    def __init__(self, owner: "RotatingOpenAIClient") -> None:
        self._owner = owner

    async def list(self, *args: Any, **kwargs: Any) -> Any:
        return await self._owner._call_with_fallback(
            "models.list",
            lambda client: client.models.list(*args, **kwargs),
        )


class _ChatCompletionsEndpoint:
    def __init__(self, owner: "RotatingOpenAIClient") -> None:
        self._owner = owner

    async def create(self, *args: Any, **kwargs: Any) -> Any:
        return await self._owner._call_with_fallback(
            "chat.completions.create",
            lambda client: client.chat.completions.create(*args, **kwargs),
        )


class _ChatEndpoint:
    def __init__(self, owner: "RotatingOpenAIClient") -> None:
        self.completions = _ChatCompletionsEndpoint(owner)


class RotatingOpenAIClient:
    """Small AsyncOpenAI-compatible facade that rotates between multiple keys."""

    def __init__(
        self,
        *,
        provider_name: str,
        config: ProviderConfig,
        base_url: str,
        default_headers: Mapping[str, str] | None = None,
    ) -> None:
        self._provider_name = provider_name
        self._config = config
        self._base_url = base_url.rstrip("/")
        self._default_headers = dict(default_headers) if default_headers else None
        
        raw_keys = getattr(config, "api_keys", None)
        if not raw_keys:
            single = getattr(config, "api_key", None)
            raw_keys = [single] if single else []
            
        self._pool = ProviderCredentialPool(
            raw_keys,
            strategy=get_credential_strategy(getattr(config, "credential_strategy", "round_robin")),
        )
        self._clients: dict[str, AsyncOpenAI] = {}
        self._clients_lock = threading.Lock()
        self.models = _ModelsEndpoint(self)
        self.chat = _ChatEndpoint(self)

    async def close(self) -> None:
        """Close every cached SDK client and its underlying HTTP client."""

        with self._clients_lock:
            clients = list(self._clients.values())
            self._clients.clear()
        for client in clients:
            await client.close()

    def _client_for_key(self, api_key: str) -> AsyncOpenAI:
        with self._clients_lock:
            client = self._clients.get(api_key)
            if client is not None:
                return client

            read_timeout = getattr(self._config, "http_read_timeout", 60.0)
            connect_timeout = getattr(self._config, "http_connect_timeout", 10.0)
            write_timeout = getattr(self._config, "http_write_timeout", 60.0)
            proxy = getattr(self._config, "proxy", None)

            timeout = httpx.Timeout(
                read_timeout,
                connect=connect_timeout,
                read=read_timeout,
                write=write_timeout,
            )
            http_client = (
                httpx.AsyncClient(proxy=proxy, timeout=timeout)
                if proxy
                else None
            )
            client = AsyncOpenAI(
                api_key=api_key,
                base_url=self._base_url,
                max_retries=0,
                default_headers=self._default_headers,
                timeout=timeout,
                http_client=http_client,
            )
            self._clients[api_key] = client
            return client

    async def _call_with_fallback(
        self,
        operation_name: str,
        operation: Callable[[AsyncOpenAI], Awaitable[T]],
    ) -> T:
        last_error: Exception | None = None
        for _ in range(self._pool.size):
            lease = self._pool.select()
            client = self._client_for_key(lease.api_key)
            try:
                result = await operation(client)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                last_error = error
                if not is_credential_fallback_error(error):
                    raise
                self._pool.record_failure(lease.index, error)
                logger.warning(
                    "{} credential fallback in {} after {} (key #{})",
                    self._provider_name,
                    operation_name,
                    type(error).__name__,
                    lease.index + 1,
                )
                continue
            self._pool.record_success(lease.index)
            return result

        if last_error is not None:
            raise last_error
        raise RuntimeError(f"{self._provider_name} exhausted all configured API keys.")


def is_credential_fallback_error(error: BaseException) -> bool:
    """Return whether another key should be tried for this upstream failure."""

    error = underlying_provider_error(error)
    return isinstance(error, openai.AuthenticationError) or is_retryable_provider_error(
        error
    )


def _cooldown_seconds(error: BaseException, *, failures: int) -> float:
    """Compute a brief cooldown to avoid hammering a bad key."""

    error = underlying_provider_error(error)
    retry_after = _retry_after_seconds(error)
    if retry_after is not None:
        return max(1.0, retry_after)

    if isinstance(error, openai.AuthenticationError):
        return 300.0

    if is_retryable_provider_error(error):
        return min(60.0, max(1.0, 2.0 ** max(0, failures - 1)))

    return 5.0


def _retry_after_seconds(error: BaseException) -> float | None:
    response = getattr(error, "response", None)
    if response is None:
        return None

    retry_after = getattr(response, "headers", {}).get("retry-after")
    if not retry_after:
        return None

    try:
        return float(retry_after)
    except (TypeError, ValueError):
        pass

    try:
        retry_after_dt = parsedate_to_datetime(retry_after)
    except (TypeError, ValueError, OverflowError):
        return None

    if retry_after_dt.tzinfo is None:
        retry_after_dt = retry_after_dt.replace(tzinfo=datetime.now().astimezone().tzinfo)
    delta = retry_after_dt - datetime.now(retry_after_dt.tzinfo)
    return max(0.0, delta.total_seconds())
