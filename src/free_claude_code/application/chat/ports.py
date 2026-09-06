"""Narrow ports owned by the local Chat Sessions application."""

from collections.abc import AsyncIterator
from typing import Protocol

from .models import (
    ChatActiveOperation,
    ChatCompaction,
    ChatContextEstimate,
    ChatGeneration,
    ChatModelOption,
    ChatOperationAcknowledgement,
    ChatPreferences,
    ChatPublishedEvent,
    ChatReasoning,
    ChatSegment,
    ChatSession,
    ChatSessionDetail,
    ChatSessionPage,
    ChatSessionSummary,
    ChatTranscript,
    ChatTurn,
    GenerationStatus,
)


class ChatStorePort(Protocol):
    """Transactions required by Chat use cases, independent of SQLite."""

    async def start(self) -> None: ...

    async def close(self) -> None: ...

    async def repair_retired_model_selections(
        self,
        *,
        retired_provider_ids: frozenset[str],
        default_model: str,
        disable_reasoning: bool,
    ) -> int: ...

    async def load_preferences(self) -> ChatPreferences: ...

    async def save_system_prompt(self, system_prompt: str) -> ChatPreferences: ...

    async def create_session(
        self, *, session_id: str, model: str, reasoning: ChatReasoning
    ) -> ChatSession: ...

    async def list_sessions(
        self,
        *,
        query: str,
        cursor: tuple[int, str] | None,
        limit: int,
    ) -> ChatSessionPage: ...

    async def get_session_summary(self, session_id: str) -> ChatSessionSummary: ...

    async def get_session(self, session_id: str) -> ChatSession: ...

    async def update_session(
        self,
        session_id: str,
        *,
        expected_revision: int,
        title: str | None,
        model: str | None,
        reasoning: ChatReasoning | None,
    ) -> ChatSession: ...

    async def delete_session(
        self, session_id: str, *, expected_revision: int
    ) -> None: ...

    async def get_transcript(self, session_id: str) -> ChatTranscript: ...

    async def get_turn_page(
        self,
        session_id: str,
        *,
        before_sequence: int | None,
        limit: int,
    ) -> tuple[tuple[ChatTurn, ...], int | None, ChatCompaction | None]: ...

    async def generation_start_committed(
        self,
        session_id: str,
        *,
        generation_id: str,
        staged: bool,
    ) -> bool: ...

    async def begin_send(
        self,
        session_id: str,
        *,
        expected_revision: int,
        turn_id: str,
        generation_id: str,
        operation_id: str,
        user_text: str,
        requested_model: str,
        reasoning: ChatReasoning,
        effective_output_limit: int,
    ) -> ChatTurn: ...

    async def begin_retry(
        self,
        session_id: str,
        *,
        expected_revision: int,
        requested_model: str,
        reasoning: ChatReasoning,
        effective_output_limit: int,
    ) -> ChatGeneration: ...

    async def begin_regenerate(
        self,
        session_id: str,
        *,
        expected_revision: int,
        generation_id: str,
        requested_model: str,
        reasoning: ChatReasoning,
        effective_output_limit: int,
    ) -> tuple[ChatTurn, ChatGeneration]: ...

    async def set_generation_actual_model(
        self, generation_id: str, actual_model: str
    ) -> None: ...

    async def replace_generation_segments(
        self, generation_id: str, segments: tuple[ChatSegment, ...]
    ) -> None: ...

    async def finish_generation(
        self,
        generation_id: str,
        *,
        status: GenerationStatus,
        stop_reason: str | None,
        error_code: str | None,
        error_message: str | None,
    ) -> ChatSession: ...

    async def discard_generation(self, generation_id: str) -> None: ...

    async def finish_regeneration(
        self,
        generation_id: str,
        *,
        status: GenerationStatus,
        stop_reason: str | None,
        error_code: str | None,
        error_message: str | None,
    ) -> ChatSession: ...

    async def upsert_compaction(
        self,
        session_id: str,
        *,
        covered_through_sequence: int,
        summary: str,
        estimated_tokens: int,
        requested_model: str,
        actual_model: str,
    ) -> ChatCompaction: ...


class ChatEventSubscriptionPort(Protocol):
    """Read-only process-local Chat event subscription."""

    cursor: int

    def __aiter__(self) -> AsyncIterator[ChatPublishedEvent]: ...

    async def aclose(self) -> None: ...


class ChatApplicationPort(Protocol):
    """Complete Chat capability consumed by the local HTTP adapter."""

    def availability(self) -> tuple[bool, str | None]: ...

    def models(self) -> tuple[ChatModelOption, ...]: ...

    async def subscribe(
        self,
    ) -> tuple[ChatEventSubscriptionPort, tuple[ChatActiveOperation, ...]]: ...

    async def preferences(self) -> ChatPreferences: ...

    async def save_system_prompt(self, value: str) -> ChatPreferences: ...

    async def reset_system_prompt(self) -> ChatPreferences: ...

    async def create_session(self) -> ChatSession: ...

    async def list_sessions(
        self,
        *,
        query: str,
        cursor: tuple[int, str] | None,
        limit: int,
    ) -> ChatSessionPage: ...

    async def get_session(self, session_id: str) -> ChatSession: ...

    async def get_detail(self, session_id: str) -> ChatSessionDetail: ...

    async def update_session(
        self,
        session_id: str,
        *,
        expected_revision: int,
        title: str | None,
        model: str | None,
        reasoning: ChatReasoning | None,
    ) -> ChatSession: ...

    async def delete_session(
        self, session_id: str, *, expected_revision: int
    ) -> None: ...

    async def get_turn_page(
        self,
        session_id: str,
        *,
        before_sequence: int | None,
        limit: int,
    ) -> tuple[tuple[ChatTurn, ...], int | None, ChatCompaction | None]: ...

    async def estimate(self, session_id: str, *, draft: str) -> ChatContextEstimate: ...

    async def send(
        self,
        session_id: str,
        *,
        expected_revision: int,
        operation_id: str,
        text: str,
    ) -> ChatOperationAcknowledgement: ...

    async def retry(
        self,
        session_id: str,
        *,
        expected_revision: int,
        operation_id: str,
    ) -> ChatOperationAcknowledgement: ...

    async def regenerate(
        self,
        session_id: str,
        *,
        expected_revision: int,
        operation_id: str,
    ) -> ChatOperationAcknowledgement: ...

    async def compact(
        self,
        session_id: str,
        *,
        expected_revision: int,
        operation_id: str,
    ) -> ChatOperationAcknowledgement: ...

    async def stop(self, session_id: str, *, operation_id: str) -> bool: ...
