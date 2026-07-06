"""Messaging platform ports used by the customer-facing workflow."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from ..models import IncomingMessage

InboundMessageHandler = Callable[[IncomingMessage], Awaitable[None]]


@runtime_checkable
class MessagingRuntime(Protocol):
    """Owns inbound SDK lifecycle for one messaging platform."""

    @property
    def name(self) -> str: ...

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    def on_message(self, handler: InboundMessageHandler) -> None: ...

    @property
    def is_connected(self) -> bool: ...


@runtime_checkable
class OutboundMessenger(Protocol):
    """Owns queued outbound platform delivery."""

    async def queue_send_message(
        self,
        chat_id: str,
        text: str,
        reply_to: str | None = None,
        parse_mode: str | None = None,
        fire_and_forget: bool = True,
        message_thread_id: str | None = None,
    ) -> str | None: ...

    async def queue_edit_message(
        self,
        chat_id: str,
        message_id: str,
        text: str,
        parse_mode: str | None = None,
        fire_and_forget: bool = True,
    ) -> None: ...

    async def queue_delete_messages(
        self,
        chat_id: str,
        message_ids: list[str],
        fire_and_forget: bool = True,
    ) -> None: ...

    def fire_and_forget(self, task: Awaitable[Any]) -> None: ...


@runtime_checkable
class VoiceCancellation(Protocol):
    """Optional voice-note cancellation port used by /clear replies."""

    async def cancel_pending_voice(
        self, chat_id: str, reply_id: str
    ) -> tuple[str, str] | None: ...


@dataclass(frozen=True, slots=True)
class MessagingPlatformComponents:
    """Runtime/outbound bundle for one configured messaging platform."""

    name: str
    runtime: MessagingRuntime
    outbound: OutboundMessenger
    voice_cancellation: VoiceCancellation | None = None
