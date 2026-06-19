"""Abstract base class for messaging platforms."""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import (
    Any,
    Protocol,
    runtime_checkable,
)

from ..models import IncomingMessage


@runtime_checkable
class ManagedClaudeSessionProtocol(Protocol):
    """Protocol for managed Claude sessions - avoid circular imports."""

    def start_task(
        self, prompt: str, session_id: str | None = None, fork_session: bool = False
    ) -> AsyncGenerator[dict, Any]: ...

    @property
    def is_busy(self) -> bool: ...


@runtime_checkable
class ManagedClaudeSessionManagerProtocol(Protocol):
    """
    Protocol for managed Claude session managers to avoid tight coupling.
    """

    async def get_or_create_session(
        self, session_id: str | None = None
    ) -> tuple[ManagedClaudeSessionProtocol, str, bool]:
        """
        Get an existing session or create a new one.

        Returns: Tuple of (session, session_id, is_new_session)
        """
        ...

    async def register_real_session_id(
        self, temp_id: str, real_session_id: str
    ) -> bool:
        """Register the real session ID from CLI output."""
        ...

    async def stop_all(self) -> None:
        """Stop all sessions."""
        ...

    async def remove_session(self, session_id: str) -> bool:
        """Remove a session from the manager."""
        ...

    def get_stats(self) -> dict:
        """Get session statistics."""
        ...


class MessagingPlatform(ABC):
    """
    Base class for all messaging platform adapters.

    Implement this to add support for Telegram, Discord, Slack, etc.
    """

    name: str = "base"

    @abstractmethod
    async def start(self) -> None:
        """Initialize and connect to the messaging platform."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Disconnect and cleanup resources."""
        pass

    @abstractmethod
    async def send_message(
        self,
        chat_id: str,
        text: str,
        reply_to: str | None = None,
        parse_mode: str | None = None,
        message_thread_id: str | None = None,
    ) -> str:
        """
        Send a message to a chat.

        Args:
            chat_id: The chat/channel ID to send to
            text: Message content
            reply_to: Optional message ID to reply to
            parse_mode: Optional formatting mode ("markdown", "html")
            message_thread_id: Optional thread or topic id for threaded channels
                (e.g. forum topics); unused on platforms that do not support it.

        Returns:
            The message ID of the sent message
        """
        pass

    @abstractmethod
    async def edit_message(
        self,
        chat_id: str,
        message_id: str,
        text: str,
        parse_mode: str | None = None,
    ) -> None:
        """
        Edit an existing message.

        Args:
            chat_id: The chat/channel ID
            message_id: The message ID to edit
            text: New message content
            parse_mode: Optional formatting mode
        """
        pass

    @abstractmethod
    async def delete_message(
        self,
        chat_id: str,
        message_id: str,
    ) -> None:
        """
        Delete a message from a chat.

        Args:
            chat_id: The chat/channel ID
            message_id: The message ID to delete
        """
        pass

    @abstractmethod
    async def queue_send_message(
        self,
        chat_id: str,
        text: str,
        reply_to: str | None = None,
        parse_mode: str | None = None,
        fire_and_forget: bool = True,
        message_thread_id: str | None = None,
    ) -> str | None:
        """
        Enqueue a message to be sent.

        If fire_and_forget is True, returns None immediately.
        Otherwise, waits for the rate limiter and returns message ID.
        """
        pass

    @abstractmethod
    async def queue_edit_message(
        self,
        chat_id: str,
        message_id: str,
        text: str,
        parse_mode: str | None = None,
        fire_and_forget: bool = True,
    ) -> None:
        """
        Enqueue a message edit.

        If fire_and_forget is True, returns immediately.
        Otherwise, waits for the rate limiter.
        """
        pass

    @abstractmethod
    async def queue_delete_message(
        self,
        chat_id: str,
        message_id: str,
        fire_and_forget: bool = True,
    ) -> None:
        """
        Enqueue a message deletion.

        If fire_and_forget is True, returns immediately.
        Otherwise, waits for the rate limiter.
        """
        pass

    async def queue_delete_messages(
        self,
        chat_id: str,
        message_ids: list[str],
        *,
        fire_and_forget: bool = True,
    ) -> None:
        """Delete many messages; default loops :meth:`queue_delete_message`.

        Adapters with native bulk delete should override.
        """
        for mid in message_ids:
            await self.queue_delete_message(
                chat_id, mid, fire_and_forget=fire_and_forget
            )

    @abstractmethod
    def on_message(
        self,
        handler: Callable[[IncomingMessage], Awaitable[None]],
    ) -> None:
        """
        Register a message handler callback.

        The handler will be called for each incoming message.

        Args:
            handler: Async function that processes incoming messages
        """
        pass

    @abstractmethod
    def fire_and_forget(self, task: Awaitable[Any]) -> None:
        """Execute a coroutine without awaiting it."""
        pass

    @property
    def is_connected(self) -> bool:
        """Check if the platform is connected."""
        return False
