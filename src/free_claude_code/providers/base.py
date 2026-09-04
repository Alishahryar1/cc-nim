"""Base provider interface - extend this to implement your own provider."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass

from loguru import logger

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.diagnostics import (
    exception_cause_types,
    redacted_exception_traceback,
)
from free_claude_code.core.openai_responses import OpenAIResponsesRequest
from free_claude_code.core.reasoning import DEFAULT_REASONING_POLICY, ReasoningPolicy
from free_claude_code.core.trace import trace_event


@dataclass(frozen=True, slots=True)
class ProviderConfig:
    """Resolved immutable configuration for one provider instance.

    Base fields apply to all providers. Provider-specific parameters
    (e.g. NIM temperature, top_p) are passed by the provider constructor.
    """

    api_keys: list[str] | None
    base_url: str
    rate_limit: int
    rate_window: int
    max_concurrency: int
    http_read_timeout: float
    http_write_timeout: float
    http_connect_timeout: float
    proxy: str | None
    log_raw_sse_events: bool
    log_api_error_tracebacks: bool


class BaseProvider(ABC):
    """Base class for all providers. Extend this to add your own."""

    def __init__(self, config: ProviderConfig):
        self._config = config
        # Initialize key rotation state
        self._current_key_index = 0
        self._key_failure_count: dict[int, int] = {}  # Track failures per key index

    def _get_current_api_key(self) -> str | None:
        """Get the current API key for requests.

        Returns:
            The current API key string, or None if no keys are configured.
        """
        api_keys = self._config.api_keys
        if not api_keys:
            return None
        # Ensure index is within bounds
        if self._current_key_index >= len(api_keys):
            self._current_key_index = 0
        return api_keys[self._current_key_index]

    def _rotate_api_key(self) -> None:
        """Rotate to the next available API key."""
        api_keys = self._config.api_keys
        if not api_keys or len(api_keys) <= 1:
            return

        self._current_key_index = (self._current_key_index + 1) % len(api_keys)
        # Reset failure count for the new key to give it a fresh start
        if self._current_key_index in self._key_failure_count:
            del self._key_failure_count[self._current_key_index]

    def _mark_key_failed(self) -> None:
        """Mark the current key as failed and increment its failure count."""
        api_keys = self._config.api_keys
        if not api_keys:
            return

        self._key_failure_count[self._current_key_index] = (
            self._key_failure_count.get(self._current_key_index, 0) + 1
        )
        # Optionally rotate immediately on failure - comment out if you want to keep trying the same key
        # self._rotate_api_key()

    def _is_key_exhausted(self, key_index: int, max_failures: int = 3) -> bool:
        """Check if a key has exceeded its failure threshold.

        Args:
            key_index: The index of the key to check
            max_failures: Maximum allowed failures before considering a key exhausted

        Returns:
            True if the key has failed too many times, False otherwise
        """
        return self._key_failure_count.get(key_index, 0) >= max_failures

    def _get_next_available_key_index(self, max_failures: int = 3) -> int | None:
        """Get the index of the next available key that hasn't failed too many times.

        Args:
            max_failures: Maximum allowed failures before considering a key exhausted

        Returns:
            Index of next available key, or None if all keys are exhausted
        """
        api_keys = self._config.api_keys
        if not api_keys:
            return None

        # Check each key starting from current index
        for i in range(len(api_keys)):
            index = (self._current_key_index + i) % len(api_keys)
            if not self._is_key_exhausted(index, max_failures):
                return index
        return None

    def _attempt_with_key_rotation(self, max_attempts: int | None = None) -> bool:
        """Attempt to find a usable API key through rotation.

        Args:
            max_attempts: Maximum number of keys to try (None for all keys)

        Returns:
            True if a usable key was found and set as current, False otherwise
        """
        api_keys = self._config.api_keys
        if not api_keys:
            return False

        if max_attempts is None:
            max_attempts = len(api_keys)
        else:
            max_attempts = min(max_attempts, len(api_keys))

        # Try up to max_attempts keys
        for _attempt in range(max_attempts):
            # Check if current key is usable
            if not self._is_key_exhausted(self._current_key_index):
                return True
            # Try next key
            self._rotate_api_key()

        return False

    @abstractmethod
    def preflight_messages(
        self,
        request: MessagesRequest,
        *,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
    ) -> None:
        """Validate a Messages request before opening its SSE stream."""

    @abstractmethod
    def preflight_responses(
        self,
        request: OpenAIResponsesRequest,
        *,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
    ) -> None:
        """Validate a Responses request before opening its SSE stream."""

    def _log_stream_transport_error(
        self,
        tag: str,
        req_tag: str,
        error: Exception,
        *,
        request_id: str | None = None,
    ) -> None:
        """Log streaming transport failures (metadata-only unless verbose is enabled)."""
        response = getattr(error, "response", None)
        http_status = (
            getattr(response, "status_code", None) if response is not None else None
        )
        cause_types = exception_cause_types(error)
        trace_event(
            stage="provider",
            event="provider.response.transport_error",
            source="provider",
            provider=tag,
            request_id=request_id,
            exc_type=type(error).__name__,
            http_status=http_status,
            cause_types=cause_types,
        )

        if self._config.log_api_error_tracebacks:
            logger.error(
                "{}_ERROR:{} exc_type={}\n{}",
                tag,
                req_tag,
                type(error).__name__,
                redacted_exception_traceback(error),
            )
            return
        logger.error(
            "{}_ERROR:{} exc_type={} http_status={} cause_types={}",
            tag,
            req_tag,
            type(error).__name__,
            http_status,
            ",".join(cause_types) if cause_types else None,
        )

    @abstractmethod
    async def cleanup(self) -> None:
        """Release any resources held by this provider."""

    @abstractmethod
    async def list_model_infos(self) -> frozenset[ProviderModelInfo]:
        """Return the model metadata currently advertised by this provider."""

    @abstractmethod
    def stream_messages(
        self,
        request: MessagesRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
    ) -> AsyncIterator[str]:
        """Stream response in Anthropic SSE format."""

    @abstractmethod
    def stream_responses(
        self,
        request: OpenAIResponsesRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
    ) -> AsyncIterator[str]:
        """Stream response in OpenAI Responses SSE format."""
