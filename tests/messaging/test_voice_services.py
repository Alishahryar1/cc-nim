import pytest

from free_claude_code.messaging.models import MessageScope
from free_claude_code.messaging.voice import (
    PendingVoiceRegistry,
    VoiceCancellationResult,
)

TELEGRAM_CHAT = MessageScope(platform="telegram", chat_id="chat")
DISCORD_CHAT = MessageScope(platform="discord", chat_id="chat")


@pytest.mark.asyncio
async def test_cancel_before_status_binding_rejects_late_flow() -> None:
    registry = PendingVoiceRegistry()
    claim = await registry.reserve(TELEGRAM_CHAT, "voice-1")

    assert claim is not None
    assert await registry.cancel(TELEGRAM_CHAT, "voice-1") == (
        VoiceCancellationResult(
            voice_message_id="voice-1",
            status_message_id=None,
        )
    )
    assert await registry.bind_status(claim, "status-1") is False
    assert await registry.claim_for_handoff(claim) is False


@pytest.mark.asyncio
async def test_pending_voice_registry_cancel_and_handoff_are_exclusive() -> None:
    registry = PendingVoiceRegistry()
    cancelled_claim = await registry.reserve(TELEGRAM_CHAT, "voice-1")

    assert cancelled_claim is not None
    assert await registry.bind_status(cancelled_claim, "status-1") is True
    assert await registry.cancel(TELEGRAM_CHAT, "status-1") == (
        VoiceCancellationResult(
            voice_message_id="voice-1",
            status_message_id="status-1",
        )
    )
    assert await registry.claim_for_handoff(cancelled_claim) is False

    handed_off_claim = await registry.reserve(TELEGRAM_CHAT, "voice-2")
    assert handed_off_claim is not None
    assert await registry.bind_status(handed_off_claim, "status-2") is True
    assert await registry.claim_for_handoff(handed_off_claim) is True
    assert await registry.cancel(TELEGRAM_CHAT, "voice-2") is None


@pytest.mark.asyncio
async def test_pending_voice_registry_isolates_platform_scopes() -> None:
    registry = PendingVoiceRegistry()
    discord_claim = await registry.reserve(DISCORD_CHAT, "voice-1")
    telegram_claim = await registry.reserve(TELEGRAM_CHAT, "voice-1")

    assert discord_claim is not None
    assert telegram_claim is not None
    assert await registry.bind_status(discord_claim, "status-1") is True
    assert await registry.bind_status(telegram_claim, "status-1") is True
    assert await registry.cancel(TELEGRAM_CHAT, "voice-1") == (
        VoiceCancellationResult(
            voice_message_id="voice-1",
            status_message_id="status-1",
        )
    )
    assert await registry.cancel(DISCORD_CHAT, "voice-1") == (
        VoiceCancellationResult(
            voice_message_id="voice-1",
            status_message_id="status-1",
        )
    )


@pytest.mark.asyncio
async def test_stale_claim_cannot_mutate_reused_voice_id() -> None:
    registry = PendingVoiceRegistry()
    stale_claim = await registry.reserve(TELEGRAM_CHAT, "voice-1")

    assert stale_claim is not None
    assert await registry.cancel(TELEGRAM_CHAT, "voice-1") is not None

    current_claim = await registry.reserve(TELEGRAM_CHAT, "voice-1")
    assert current_claim is not None
    assert current_claim != stale_claim
    assert await registry.bind_status(current_claim, "status-current") is True

    assert await registry.bind_status(stale_claim, "status-stale") is False
    assert await registry.claim_for_handoff(stale_claim) is False
    assert await registry.discard(stale_claim) is False
    assert await registry.cancel(TELEGRAM_CHAT, "status-current") == (
        VoiceCancellationResult(
            voice_message_id="voice-1",
            status_message_id="status-current",
        )
    )


@pytest.mark.asyncio
async def test_pending_voice_registry_rejects_duplicate_and_unbound_handoff() -> None:
    registry = PendingVoiceRegistry()
    claim = await registry.reserve(TELEGRAM_CHAT, "voice-1")

    assert claim is not None
    assert await registry.reserve(TELEGRAM_CHAT, "voice-1") is None
    assert await registry.claim_for_handoff(claim) is False
    assert await registry.cancel(TELEGRAM_CHAT, "voice-1") is not None
