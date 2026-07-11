"""Messaging workflow coordinator for Discord and Telegram prompts."""

import asyncio
from collections.abc import Coroutine
from typing import Any

from loguru import logger

from free_claude_code.core.trace import trace_event

from .command_context import ReplyClearResult
from .managed_protocols import ManagedClaudeSessionManagerProtocol
from .models import IncomingMessage, MessageScope
from .node_runner import MessagingNodeRunner
from .platforms.ports import OutboundMessenger, VoiceCancellation
from .rendering.profiles import build_rendering_profile
from .safe_diagnostics import format_exception_for_log
from .session import SessionStore
from .transcript import RenderCtx
from .trees import (
    CancellationReason,
    CancellationResult,
    CancellationUiOwner,
    ConversationSnapshot,
    FailureResult,
    NodeUiTarget,
    QueueDecision,
    ReplyTarget,
    TreeQueueManager,
)
from .turn_intake import MessagingTurnIntake
from .voice import VoiceCancellationResult


async def _finish_owned_operation[T](
    operation: Coroutine[Any, Any, T],
    *,
    name: str,
) -> T:
    """Finish owned work, preserving its failures before caller cancellation."""
    task = asyncio.create_task(operation, name=name)
    current = asyncio.current_task()
    cancellation: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            if current is not None and current.cancelling():
                cancellation = cancellation or exc
        except Exception:
            break

    # Task.result() raises the owned operation's failure. Caller cancellation
    # is restored only after successful completion.
    result = task.result()
    if cancellation is not None:
        raise cancellation
    return result


class MessagingWorkflow:
    """Own messaging state transitions and their external side effects."""

    def __init__(
        self,
        outbound: OutboundMessenger,
        cli_manager: ManagedClaudeSessionManagerProtocol,
        session_store: SessionStore,
        *,
        platform_name: str | None = None,
        voice_cancellation: VoiceCancellation | None = None,
        debug_platform_edits: bool = False,
        debug_subagent_stack: bool = False,
        log_raw_cli_diagnostics: bool = False,
        log_messaging_error_details: bool = False,
    ) -> None:
        self.platform_name = platform_name or "messaging"
        self.outbound = outbound
        self.voice_cancellation = voice_cancellation
        self.cli_manager = cli_manager
        self.session_store = session_store
        self._log_messaging_error_details = log_messaging_error_details
        self._rendering_profile = build_rendering_profile(self.platform_name)
        self._state_lock = asyncio.Lock()
        self._admission_epoch = 0
        self._pending_restored_status_targets: tuple[NodeUiTarget, ...] = ()

        self._tree_queue: TreeQueueManager
        self.node_runner = MessagingNodeRunner(
            platform_name=self.platform_name,
            outbound=outbound,
            cli_manager=cli_manager,
            session_store=session_store,
            get_tree_queue=lambda: self._tree_queue,
            format_status=self.format_status,
            get_parse_mode=self._parse_mode,
            get_render_ctx=self.get_render_ctx,
            get_limit_chars=self._get_limit_chars,
            debug_platform_edits=debug_platform_edits,
            debug_subagent_stack=debug_subagent_stack,
            log_raw_cli_diagnostics=log_raw_cli_diagnostics,
            log_messaging_error_details=log_messaging_error_details,
        )
        self.turn_intake = MessagingTurnIntake(
            platform_name=self.platform_name,
            outbound=outbound,
            session_store=session_store,
            command_context=self,
            resolve_reply=self.resolve_reply,
            admit_turn=self._admit_turn_if_current,
            format_status=self.format_status,
            get_parse_mode=self._parse_mode,
            record_outgoing_message=self.record_outgoing_message,
            log_messaging_error_details=log_messaging_error_details,
        )
        self._tree_queue = self._build_tree_queue()

    def _build_tree_queue(
        self, snapshot: ConversationSnapshot | None = None
    ) -> TreeQueueManager:
        if snapshot is None:
            return TreeQueueManager(
                self.node_runner.process_node,
                queue_update_callback=self.turn_intake.update_queue_positions,
                node_started_callback=self.turn_intake.mark_node_processing,
                unexpected_failure_callback=self._apply_unexpected_failure,
                log_messaging_error_details=self._log_messaging_error_details,
            )
        return TreeQueueManager.from_snapshot(
            snapshot,
            self.node_runner.process_node,
            queue_update_callback=self.turn_intake.update_queue_positions,
            node_started_callback=self.turn_intake.mark_node_processing,
            unexpected_failure_callback=self._apply_unexpected_failure,
            log_messaging_error_details=self._log_messaging_error_details,
        )

    def format_status(self, emoji: str, label: str, suffix: str | None = None) -> str:
        return self._rendering_profile.format_status(emoji, label, suffix)

    def _parse_mode(self) -> str | None:
        return self._rendering_profile.parse_mode

    def get_render_ctx(self) -> RenderCtx:
        return self._rendering_profile.render_ctx

    def _get_limit_chars(self) -> int:
        return self._rendering_profile.limit_chars

    @property
    def tree_queue(self) -> TreeQueueManager:
        """Expose the manager facade for diagnostics and smoke tests."""
        return self._tree_queue

    def restore(self) -> None:
        """Restore and reconcile persisted conversations before platform start."""
        snapshot = self.session_store.load_conversation_snapshot()
        if snapshot.is_empty:
            return
        logger.info("Restoring {} conversation trees...", len(snapshot.trees))
        self._tree_queue = self._build_tree_queue(snapshot)
        normalized = self._tree_queue.restored_snapshot
        if normalized is not None and normalized != snapshot:
            self.session_store.save_conversation_snapshot(normalized)
        self._pending_restored_status_targets = self._tree_queue.restored_stale_targets

    async def repair_restored_statuses(self) -> None:
        """Replace stale queued/processing UI after delivery becomes available."""
        targets = self._pending_restored_status_targets
        self._pending_restored_status_targets = ()
        for target in targets:
            if self.platform_name != "messaging" and (
                target.scope.platform != self.platform_name
            ):
                continue
            try:
                await self.outbound.queue_edit_message(
                    target.scope.chat_id,
                    target.status_message_id,
                    self.format_status("❌", "Interrupted by server restart"),
                    parse_mode=self._parse_mode(),
                    fire_and_forget=False,
                )
            except Exception as exc:
                logger.debug(
                    "Failed to repair restored status for node {}: {}",
                    target.node_id,
                    type(exc).__name__,
                )

    async def close(self) -> None:
        """Finish every owned task and durable write before releasing delivery."""
        await self.stop_all_tasks()
        await self._tree_queue.wait_idle()
        self.session_store.flush_pending_save()

    async def handle_message(self, incoming: IncomingMessage) -> None:
        """Handle one platform message."""
        trace_event(
            stage="ingress",
            event="turn.received",
            source=self.platform_name,
            chat_id=incoming.chat_id,
            platform_message_id=incoming.message_id,
            reply_to_message_id=incoming.reply_to_message_id,
            thread_id=incoming.message_thread_id,
            message_text=incoming.text or "",
        )
        with logger.contextualize(
            chat_id=incoming.chat_id,
            node_id=incoming.message_id,
        ):
            async with self._state_lock:
                admission_epoch = self._admission_epoch
            await self.turn_intake.handle_message(
                incoming,
                admission_epoch=admission_epoch,
            )

    async def resolve_reply(
        self,
        scope: MessageScope,
        reference_id: str,
    ) -> ReplyTarget | None:
        return await self._tree_queue.resolve_reply(scope, reference_id)

    async def _admit_turn_if_current(
        self,
        incoming: IncomingMessage,
        status_message_id: str,
        parent_node_id: str | None,
        admission_epoch: int,
    ) -> QueueDecision | None:
        return await _finish_owned_operation(
            self._admit_if_current(
                incoming,
                status_message_id,
                parent_node_id,
                admission_epoch,
            ),
            name=f"messaging-admit-{incoming.message_id}",
        )

    async def _admit_if_current(
        self,
        incoming: IncomingMessage,
        status_message_id: str,
        parent_node_id: str | None,
        admission_epoch: int,
    ) -> QueueDecision | None:
        """Commit admission and its exact snapshot as one owned transaction."""
        async with self._state_lock:
            if admission_epoch != self._admission_epoch:
                return None
            decision = await self._tree_queue.admit(
                incoming,
                status_message_id,
                parent_node_id=parent_node_id,
            )
            if decision.snapshot is not None:
                self.session_store.save_tree_snapshot(decision.snapshot)
            return decision

    def get_tree_count(self) -> int:
        return self._tree_queue.get_tree_count()

    async def stop_reply(
        self,
        scope: MessageScope,
        reply_id: str,
    ) -> int | None:
        """Stop the exact voice/tree owner of one replied-to message."""
        return await _finish_owned_operation(
            self._stop_reply(scope, reply_id),
            name=f"messaging-stop-reply-{reply_id}",
        )

    async def _stop_reply(
        self,
        scope: MessageScope,
        reply_id: str,
    ) -> int | None:
        voice_result = await self._cancel_pending_voice(scope, reply_id)
        if voice_result is not None:
            self.render_voice_stopped(voice_result)

        async with self._state_lock:
            node_id = await self._tree_queue.resolve_node_id(scope, reply_id)
            if node_id is None:
                return 1 if voice_result is not None else None
            result = await self._tree_queue.cancel_node(
                scope,
                node_id,
                reason=CancellationReason.STOP,
            )
            self._apply_cancellation_result(result)

        owners = {(effect.node.scope, effect.node.node_id) for effect in result.effects}
        if voice_result is not None:
            owners.add((voice_result.scope, voice_result.voice_message_id))
        return len(owners)

    async def clear_reply(
        self,
        scope: MessageScope,
        reply_id: str,
    ) -> ReplyClearResult | None:
        """Clear the exact voice/tree owner of one replied-to message."""
        return await _finish_owned_operation(
            self._clear_reply(scope, reply_id),
            name=f"messaging-clear-reply-{reply_id}",
        )

    async def _clear_reply(
        self,
        scope: MessageScope,
        reply_id: str,
    ) -> ReplyClearResult | None:
        voice_result = await self._cancel_pending_voice(scope, reply_id)
        if voice_result is not None:
            self.render_voice_stopped(voice_result)

        async with self._state_lock:
            node_id = await self._tree_queue.resolve_node_id(scope, reply_id)
            if node_id is None:
                if voice_result is None:
                    return None
                return ReplyClearResult(
                    message_ids=voice_result.message_ids,
                    tree_cleared=False,
                )

            branch = await self._tree_queue.remove_branch(scope, node_id)
            self._apply_cancellation_result(branch.cancellation)
            if branch.removed_tree_identity is not None:
                self.session_store.remove_tree_snapshot(branch.removed_tree_identity)

        message_ids = set(branch.message_ids)
        if voice_result is not None:
            message_ids.update(voice_result.message_ids)
        return ReplyClearResult(
            message_ids=frozenset(message_ids),
            tree_cleared=True,
        )

    async def stop_all_tasks(self) -> int:
        """Stop every pending and active messaging task."""
        return await _finish_owned_operation(
            self._stop_all_tasks(),
            name="messaging-stop-all",
        )

    async def _stop_all_tasks(self) -> int:
        voice_results = await self._cancel_all_pending_voices()
        for voice in voice_results:
            self.render_voice_stopped(voice)
        async with self._state_lock:
            self._admission_epoch += 1
            logger.info("Cancelling tree queue tasks...")
            result = await self._tree_queue.cancel_all(reason=CancellationReason.STOP)
            logger.info("Cancelled {} nodes", len(result.effects))
            self._apply_cancellation_result(result)
            logger.info("Stopping all CLI sessions...")
            await self.cli_manager.stop_all()
            tree_keys = {
                (effect.node.scope, effect.node.node_id) for effect in result.effects
            }
            voice_keys = {
                (voice.scope, voice.voice_message_id) for voice in voice_results
            }
            return len(tree_keys | voice_keys)

    async def clear_all_state(self, platform: str, chat_id: str) -> frozenset[str]:
        """Clear FCC state atomically with respect to later turn admission."""
        return await _finish_owned_operation(
            self._clear_all_state(platform, chat_id),
            name="messaging-clear-all",
        )

    async def _clear_all_state(
        self,
        platform: str,
        chat_id: str,
    ) -> frozenset[str]:
        """Globally reset FCC state; return deletion IDs for the invoking chat."""
        voice_results = await self._cancel_all_pending_voices()
        for voice in voice_results:
            self.render_voice_stopped(voice)
        async with self._state_lock:
            message_ids: set[str] = set()
            clear_scope = MessageScope(platform=platform, chat_id=chat_id)
            for voice in voice_results:
                if voice.scope == clear_scope:
                    message_ids.update(voice.message_ids)
            try:
                message_ids.update(
                    str(message_id)
                    for message_id in self.session_store.get_message_ids_for_chat(
                        platform,
                        chat_id,
                    )
                    if message_id is not None
                )
            except Exception as exc:
                logger.debug(
                    "Failed to read message log for /clear: {}", type(exc).__name__
                )

            message_ids.update(
                await self._tree_queue.get_message_ids_for_chat(platform, chat_id)
            )
            # All fallible/cancellable reads precede the commit boundary. Once
            # the epoch advances, the following synchronous wipe and the
            # manager's cancellation-safe detach complete as one-way work.
            self._admission_epoch += 1
            try:
                self.session_store.clear_all()
            except Exception as exc:
                logger.warning("Failed to clear session store: {}", type(exc).__name__)

            failures: list[Exception] = []
            result = CancellationResult()
            try:
                result = await self._tree_queue.clear_all(
                    reason=CancellationReason.STOP
                )
            except Exception as exc:
                failures.append(exc)
            # A runner may terminalize during task draining after the early
            # store clear. The detached manager now rejects later writes; clear
            # this final pre-detach snapshot without erasing newer message logs.
            try:
                self.session_store.clear_conversation_snapshot()
            except Exception as exc:
                failures.append(exc)
                logger.warning(
                    "Failed to persist final cleared session state: {}",
                    type(exc).__name__,
                )
            try:
                self._apply_cancellation_result(result)
            except Exception as exc:
                failures.append(exc)
            try:
                await self.cli_manager.stop_all()
            except Exception as exc:
                failures.append(exc)
            if len(failures) == 1:
                raise failures[0]
            if failures:
                raise ExceptionGroup("Global clear failed", failures)
            return frozenset(message_ids)

    async def _cancel_all_pending_voices(
        self,
    ) -> tuple[VoiceCancellationResult, ...]:
        cancellation = self.voice_cancellation
        if cancellation is None:
            return ()
        return await cancellation.cancel_all_pending_voices()

    async def _cancel_pending_voice(
        self,
        scope: MessageScope,
        reply_id: str,
    ) -> VoiceCancellationResult | None:
        cancellation = self.voice_cancellation
        if cancellation is None:
            return None
        return await cancellation.cancel_pending_voice(scope, reply_id)

    def render_voice_stopped(self, result: VoiceCancellationResult) -> None:
        """Publish terminal UI for a voice cancelled before tree ownership."""
        if result.status_message_id is None:
            return
        self.outbound.fire_and_forget(
            self.outbound.queue_edit_message(
                result.scope.chat_id,
                result.status_message_id,
                self.format_status("⏹", "Stopped."),
                parse_mode=self._parse_mode(),
            )
        )

    def forget_message_ids(
        self,
        platform: str,
        chat_id: str,
        message_ids: set[str],
    ) -> None:
        try:
            self.session_store.forget_message_ids(platform, chat_id, message_ids)
        except Exception as exc:
            logger.warning(
                "Failed to update session store after branch clear: {}",
                type(exc).__name__,
            )

    def record_outgoing_message(
        self,
        platform: str,
        chat_id: str,
        msg_id: str | None,
        kind: str,
    ) -> None:
        """Record an outgoing message ID for /clear, best effort."""
        if not msg_id:
            return
        try:
            self.session_store.record_message_id(
                platform,
                chat_id,
                str(msg_id),
                direction="out",
                kind=kind,
            )
        except Exception as exc:
            logger.debug(
                "Failed to record message_id: {}",
                format_exception_for_log(
                    exc,
                    log_full_message=self._log_messaging_error_details,
                ),
            )

    def _apply_cancellation_result(self, result: CancellationResult) -> None:
        """Apply detached UI and persistence effects from one transition."""
        for effect in result.effects:
            if effect.ui_owner is CancellationUiOwner.WORKFLOW:
                self.outbound.fire_and_forget(
                    self.outbound.queue_edit_message(
                        effect.node.scope.chat_id,
                        effect.node.status_message_id,
                        self.format_status("⏹", "Stopped."),
                        parse_mode=self._parse_mode(),
                    )
                )
        for snapshot in result.snapshots:
            self.session_store.save_tree_snapshot(snapshot)

    def _apply_unexpected_failure(self, result: FailureResult) -> None:
        """Persist and render a failure that escaped the total node runner."""
        if result.snapshot is not None:
            self.session_store.save_tree_snapshot(result.snapshot)
        for target in result.affected:
            self.outbound.fire_and_forget(
                self.outbound.queue_edit_message(
                    target.scope.chat_id,
                    target.status_message_id,
                    self.format_status("💥", "Task Failed"),
                    parse_mode=self._parse_mode(),
                )
            )


__all__ = ["MessagingWorkflow"]
