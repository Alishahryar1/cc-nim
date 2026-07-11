import asyncio
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from free_claude_code.messaging.models import MessageScope
from free_claude_code.messaging.session import SessionStore
from free_claude_code.messaging.trees import (
    CancellationReason,
    CancellationResult,
    CancellationUiOwner,
    FailureResult,
    MessageState,
    NodeClaim,
    NodeUiTarget,
    QueueEntry,
    ReplyTarget,
    TreeIdentity,
    TreeSnapshot,
)
from free_claude_code.messaging.trees.transitions import CancellationEffect
from free_claude_code.messaging.voice import VoiceCancellationResult
from free_claude_code.messaging.workflow import MessagingWorkflow

_SCOPE = MessageScope(platform="telegram", chat_id="chat_1")


async def _event_stream(events):
    for event in events:
        await asyncio.sleep(0)
        yield event


def _claim(
    node_id: str = "node_1",
    *,
    prompt: str = "hello",
    parent_session_id: str | None = None,
) -> NodeClaim:
    return NodeClaim(
        identity=TreeIdentity(scope=_SCOPE, root_id="root_1"),
        claim_id="claim_1",
        node=NodeUiTarget(
            scope=_SCOPE,
            node_id=node_id,
            status_message_id="status_1",
        ),
        prompt=prompt,
        parent_session_id=parent_session_id,
    )


def _snapshot(root_id: str = "root_1") -> TreeSnapshot:
    return TreeSnapshot(scope=_SCOPE, root_id=root_id, nodes={})


def _session(events) -> MagicMock:
    session = MagicMock()
    session.start_task.return_value = _event_stream(events)
    return session


async def _wait_for_idle(workflow: MessagingWorkflow) -> None:
    for _ in range(200):
        if workflow.tree_queue.task_count() == 0:
            await asyncio.sleep(0)
            return
        await asyncio.sleep(0.01)
    raise AssertionError("messaging workflow did not become idle")


@pytest.fixture
def handler(mock_platform, mock_cli_manager, mock_session_store):
    default_session = _session([{"type": "exit", "code": 0}])
    mock_cli_manager.get_or_create_session.return_value = (
        default_session,
        "session_1",
        False,
    )
    return MessagingWorkflow(
        mock_platform,
        mock_cli_manager,
        mock_session_store,
        platform_name="telegram",
        voice_cancellation=mock_platform,
    )


@pytest.mark.asyncio
async def test_handle_message_turn_trace_always_includes_full_message_text(
    mock_platform,
    mock_cli_manager,
    mock_session_store,
    incoming_message_factory,
):
    text = "user-message-content-visible-in-trace"
    workflow = MessagingWorkflow(
        mock_platform,
        mock_cli_manager,
        mock_session_store,
    )
    incoming = incoming_message_factory(text=text)
    with (
        patch.object(workflow.turn_intake, "handle_message", new_callable=AsyncMock),
        patch("free_claude_code.messaging.workflow.trace_event") as trace_mock,
    ):
        await workflow.handle_message(incoming)

    assert trace_mock.call_args.kwargs["event"] == "turn.received"
    assert trace_mock.call_args.kwargs["message_text"] == text


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        (None, "Launching"),
        (ReplyTarget(node_id="parent", queue_position=None), "Continuing"),
        (ReplyTarget(node_id="parent", queue_position=3), "position 3"),
    ],
)
def test_initial_status_uses_immutable_reply_advice(handler, target, expected):
    assert expected in handler.turn_intake._get_initial_status(target)


@pytest.mark.asyncio
async def test_handle_message_stop_command(
    handler, mock_platform, incoming_message_factory
):
    incoming = incoming_message_factory(text="/stop")
    handler.stop_all_tasks = AsyncMock(return_value=5)

    await handler.handle_message(incoming)

    handler.stop_all_tasks.assert_awaited_once()
    mock_platform.queue_send_message.assert_awaited_once_with(
        incoming.chat_id,
        "⏹ *Stopped\\.* Cancelled 5 pending or active requests\\.",
        fire_and_forget=False,
        message_thread_id=None,
    )


@pytest.mark.asyncio
async def test_reply_stop_resolves_and_stops_only_target(
    handler, mock_platform, mock_cli_manager, incoming_message_factory
):
    handler.resolve_node_id = AsyncMock(return_value="root_msg")
    handler.stop_task = AsyncMock(return_value=1)
    handler.stop_all_tasks = AsyncMock(return_value=999)
    incoming = incoming_message_factory(
        text="/stop",
        message_id="stop_msg",
        reply_to_message_id="status_root",
    )

    await handler.handle_message(incoming)

    handler.resolve_node_id.assert_awaited_once_with(incoming.scope, "status_root")
    handler.stop_task.assert_awaited_once_with(incoming.scope, "root_msg")
    handler.stop_all_tasks.assert_not_awaited()
    mock_cli_manager.stop_all.assert_not_awaited()
    assert "Cancelled 1 request" in mock_platform.queue_send_message.call_args.args[1]


@pytest.mark.asyncio
async def test_reply_stop_unknown_does_not_stop_all(
    handler, mock_platform, mock_cli_manager, incoming_message_factory
):
    handler.resolve_node_id = AsyncMock(return_value=None)
    handler.stop_all_tasks = AsyncMock(return_value=5)
    incoming = incoming_message_factory(
        text="/stop",
        message_id="stop_msg",
        reply_to_message_id="unknown_msg",
    )

    await handler.handle_message(incoming)

    handler.stop_all_tasks.assert_not_awaited()
    mock_cli_manager.stop_all.assert_not_awaited()
    assert (
        "Nothing to stop for that message"
        in mock_platform.queue_send_message.call_args.args[1]
    )


@pytest.mark.asyncio
async def test_stats_command_reports_cli_and_tree_counts(
    handler, mock_platform, mock_cli_manager, incoming_message_factory
):
    mock_cli_manager.get_stats.return_value = {"active_sessions": 2}

    await handler.handle_message(incoming_message_factory(text="/stats"))

    text = mock_platform.queue_send_message.call_args.args[1]
    assert "Active CLI: 2" in text
    assert "Message Trees: 0" in text
    assert mock_platform.queue_send_message.call_args.kwargs["fire_and_forget"] is False


@pytest.mark.asyncio
async def test_status_echo_is_filtered(
    handler, mock_platform, incoming_message_factory
):
    await handler.handle_message(incoming_message_factory(text="⏳ Thinking..."))

    mock_platform.queue_send_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_new_turn_uses_public_admission_and_persists_exact_snapshot(
    handler,
    mock_platform,
    mock_session_store,
    incoming_message_factory,
):
    incoming = incoming_message_factory(text="hello", message_id="node_1")
    mock_platform.queue_send_message.return_value = "status_123"

    await handler.handle_message(incoming)
    await _wait_for_idle(handler)

    assert "Launching" in mock_platform.queue_send_message.call_args.args[1]
    assert mock_session_store.save_tree_snapshot.call_count >= 2
    view = await handler.tree_queue.get_node(incoming.scope, "node_1")
    assert view is not None
    assert view.state is MessageState.COMPLETED


@pytest.mark.asyncio
async def test_duplicate_delivery_removes_its_provisional_status(
    handler,
    mock_platform,
    mock_session_store,
    incoming_message_factory,
):
    incoming = incoming_message_factory(text="hello", message_id="duplicate")
    mock_platform.queue_send_message.side_effect = ["status-first", "status-rejected"]

    await handler.handle_message(incoming)
    await _wait_for_idle(handler)
    await handler.handle_message(incoming)

    mock_platform.queue_delete_messages.assert_awaited_once_with(
        incoming.chat_id,
        ["status-rejected"],
        fire_and_forget=False,
    )
    mock_session_store.forget_message_ids.assert_called_once_with(
        incoming.platform,
        incoming.chat_id,
        {"status-rejected"},
    )


@pytest.mark.asyncio
async def test_pre_sent_status_is_edited_in_place(
    handler, mock_platform, incoming_message_factory
):
    incoming = incoming_message_factory(
        text="hello",
        message_id="node_1",
        status_message_id="existing_status",
    )

    await handler.handle_message(incoming)
    await _wait_for_idle(handler)

    first_edit = mock_platform.queue_edit_message.call_args_list[0]
    assert first_edit.args[1] == "existing_status"
    assert "Launching" in first_edit.args[2]
    assert first_edit.kwargs["fire_and_forget"] is False
    mock_platform.queue_send_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_busy_reply_is_rendered_with_atomic_queue_position(
    handler, mock_platform, mock_cli_manager, incoming_message_factory
):
    started = asyncio.Event()

    async def blocking_start(*args, **kwargs):
        started.set()
        await asyncio.sleep(60)
        if False:
            yield {}

    session = MagicMock()
    session.start_task = blocking_start
    mock_cli_manager.get_or_create_session.return_value = (
        session,
        "session_1",
        False,
    )
    mock_platform.queue_send_message.side_effect = ["status_root", "status_child"]

    root = incoming_message_factory(text="root", message_id="root")
    await handler.handle_message(root)
    await started.wait()
    child = incoming_message_factory(
        text="child",
        message_id="child",
        reply_to_message_id="status_root",
    )
    await handler.handle_message(child)

    assert "position 1" in mock_platform.queue_send_message.call_args.args[1]
    queued_edit = mock_platform.queue_edit_message.call_args_list[-1]
    assert queued_edit.args[1] == "status_child"
    assert "position 1" in queued_edit.args[2]

    await handler.stop_all_tasks()


@pytest.mark.asyncio
async def test_queue_position_callback_consumes_immutable_entries(
    handler, mock_platform
):
    queue = (
        QueueEntry(
            node=NodeUiTarget(
                scope=_SCOPE,
                node_id="child_1",
                status_message_id="status_1",
            ),
            position=1,
        ),
        QueueEntry(
            node=NodeUiTarget(
                scope=_SCOPE,
                node_id="child_2",
                status_message_id="status_2",
            ),
            position=2,
        ),
    )

    await handler.turn_intake.update_queue_positions(queue)
    await asyncio.sleep(0)

    calls = mock_platform.queue_edit_message.call_args_list
    assert [call.args[1] for call in calls] == ["status_1", "status_2"]
    assert "position 1" in calls[0].args[2]
    assert "position 2" in calls[1].args[2]


@pytest.mark.asyncio
async def test_claim_started_callback_renders_processing(handler, mock_platform):
    claim = _claim()

    await handler.turn_intake.mark_node_processing(claim)
    await asyncio.sleep(0)

    args, kwargs = mock_platform.queue_edit_message.call_args
    assert args[0:2] == ("chat_1", "status_1")
    assert "Processing" in args[2]
    assert kwargs["parse_mode"] == "MarkdownV2"


@pytest.mark.asyncio
async def test_stop_all_applies_immutable_ui_ownership_and_snapshots(
    handler, mock_cli_manager, mock_platform, mock_session_store
):
    workflow_owned = CancellationEffect(
        node=NodeUiTarget(
            scope=_SCOPE,
            node_id="queued",
            status_message_id="status_queued",
        ),
        ui_owner=CancellationUiOwner.WORKFLOW,
    )
    runner_owned = CancellationEffect(
        node=NodeUiTarget(
            scope=_SCOPE,
            node_id="active",
            status_message_id="status_active",
        ),
        ui_owner=CancellationUiOwner.RUNNER,
    )
    snapshot = _snapshot()
    result = CancellationResult(
        effects=(workflow_owned, runner_owned),
        snapshots=(snapshot,),
    )
    with patch.object(
        handler.tree_queue,
        "cancel_all",
        AsyncMock(return_value=result),
    ) as cancel_all:
        count = await handler.stop_all_tasks()
    await asyncio.sleep(0)

    assert count == 2
    cancel_all.assert_awaited_once_with(reason=CancellationReason.STOP)
    mock_cli_manager.stop_all.assert_awaited_once()
    assert mock_platform.fire_and_forget.call_count == 1
    assert mock_platform.queue_edit_message.call_args.args[1] == "status_queued"
    mock_session_store.save_tree_snapshot.assert_called_once_with(snapshot)


@pytest.mark.asyncio
async def test_stop_all_persists_committed_transition_before_cli_shutdown(
    handler,
    mock_cli_manager,
    mock_session_store,
):
    shutdown_started = asyncio.Event()
    snapshot = _snapshot()
    result = CancellationResult(snapshots=(snapshot,))

    async def block_shutdown() -> None:
        shutdown_started.set()
        await asyncio.Event().wait()

    mock_cli_manager.stop_all.side_effect = block_shutdown
    with patch.object(
        handler.tree_queue,
        "cancel_all",
        AsyncMock(return_value=result),
    ):
        stop_task = asyncio.create_task(handler.stop_all_tasks())
        await shutdown_started.wait()

        mock_session_store.save_tree_snapshot.assert_called_once_with(snapshot)
        stop_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await stop_task


@pytest.mark.asyncio
async def test_node_runner_success_uses_claim_and_semantic_completion(
    handler, mock_cli_manager, mock_platform, mock_session_store
):
    claim = _claim(prompt="say hello")
    session = _session(
        [
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "thinking", "thinking": "Let me think"},
                        {"type": "text", "text": "Hello world"},
                    ]
                },
            },
            {"type": "exit", "code": 0},
        ]
    )
    mock_cli_manager.get_or_create_session.return_value = (
        session,
        "session_1",
        False,
    )
    snapshot = _snapshot()
    with patch.object(
        handler.tree_queue,
        "complete_claim",
        AsyncMock(return_value=snapshot),
    ) as complete_claim:
        await handler.node_runner.process_node(claim)

    complete_claim.assert_awaited_once_with(claim, "session_1")
    mock_session_store.save_tree_snapshot.assert_called_once_with(snapshot)
    rendered = mock_platform.queue_edit_message.call_args_list[-1].args[2]
    assert "✅ *Complete*" in rendered
    assert "Hello world" in rendered
    mock_cli_manager.get_or_create_session.assert_awaited_once_with(session_id=None)
    assert session.start_task.call_args.args == ("say hello",)
    assert session.start_task.call_args.kwargs == {
        "session_id": None,
        "fork_session": False,
    }


@pytest.mark.asyncio
async def test_node_runner_uses_claim_parent_session_for_fork(
    handler, mock_cli_manager
):
    claim = _claim(parent_session_id="parent_session")
    session = _session([{"type": "exit", "code": 0}])
    mock_cli_manager.get_or_create_session.return_value = (
        session,
        "child_session",
        False,
    )

    await handler.node_runner.process_node(claim)

    mock_cli_manager.get_or_create_session.assert_awaited_once_with(
        session_id="parent_session"
    )
    assert session.start_task.call_args.kwargs == {
        "session_id": "parent_session",
        "fork_session": True,
    }


@pytest.mark.asyncio
async def test_session_info_records_real_session_through_manager(
    handler, mock_cli_manager, mock_session_store
):
    claim = _claim()
    session = _session(
        [
            {"type": "session_info", "session_id": "real_session"},
            {"type": "exit", "code": 0},
        ]
    )
    mock_cli_manager.get_or_create_session.return_value = (
        session,
        "temporary_session",
        True,
    )
    record_snapshot = _snapshot("record")
    complete_snapshot = _snapshot("complete")
    with (
        patch.object(
            handler.tree_queue,
            "record_session",
            AsyncMock(return_value=record_snapshot),
        ) as record_session,
        patch.object(
            handler.tree_queue,
            "complete_claim",
            AsyncMock(return_value=complete_snapshot),
        ) as complete_claim,
    ):
        await handler.node_runner.process_node(claim)

    mock_cli_manager.register_real_session_id.assert_awaited_once_with(
        "temporary_session", "real_session"
    )
    record_session.assert_awaited_once_with(claim, "real_session")
    complete_claim.assert_awaited_once_with(claim, "real_session")
    assert mock_session_store.save_tree_snapshot.call_args_list == [
        ((record_snapshot,), {}),
        ((complete_snapshot,), {}),
    ]


@pytest.mark.asyncio
async def test_session_limit_failure_uses_non_propagating_claim_failure(
    handler, mock_cli_manager, mock_platform
):
    claim = _claim()
    mock_cli_manager.get_or_create_session.side_effect = RuntimeError("session limit")
    result = FailureResult(
        affected=(claim.node,),
        queue_update=None,
        snapshot=_snapshot(),
    )
    with patch.object(
        handler.tree_queue,
        "fail_claim",
        AsyncMock(return_value=result),
    ) as fail_claim:
        await handler.node_runner.process_node(claim)

    fail_claim.assert_awaited_once_with(claim, propagate=False)
    rendered = mock_platform.queue_edit_message.call_args_list[-1].args[2]
    assert "Session limit reached" in rendered


@pytest.mark.asyncio
async def test_non_exit_error_defers_child_failure_until_stream_ends(
    handler, mock_cli_manager, mock_platform, mock_session_store
):
    claim = _claim()
    session = _session([{"type": "error", "error": {"message": "CLI crashed"}}])
    mock_cli_manager.get_or_create_session.return_value = (
        session,
        "session_1",
        False,
    )
    child = NodeUiTarget(
        scope=_SCOPE,
        node_id="child",
        status_message_id="status_child",
    )
    snapshot = _snapshot()
    result = FailureResult(
        affected=(claim.node, child),
        queue_update=None,
        snapshot=snapshot,
    )
    with patch.object(
        handler.tree_queue,
        "fail_claim",
        AsyncMock(return_value=result),
    ) as fail_claim:
        await handler.node_runner.process_node(claim)
    await asyncio.sleep(0)

    assert fail_claim.await_args_list == [
        call(claim, propagate=False),
        call(claim, propagate=True),
    ]
    assert mock_session_store.save_tree_snapshot.call_args_list == [
        call(snapshot),
        call(snapshot),
    ]
    rendered = "\n".join(
        call.args[2] for call in mock_platform.queue_edit_message.call_args_list
    )
    assert "❌ *Error*" in rendered
    assert "CLI crashed" in rendered
    assert "Parent task failed" in rendered


@pytest.mark.asyncio
async def test_provider_error_exit_does_not_mask_or_complete(
    handler, mock_cli_manager, mock_platform
):
    claim = _claim()
    provider_error = "API Error: Request rejected (429)\nProvider rate limit reached."
    session = _session(
        [
            {"type": "error", "error": {"message": provider_error}},
            {"type": "exit", "code": 1},
        ]
    )
    mock_cli_manager.get_or_create_session.return_value = (
        session,
        "session_1",
        False,
    )
    failure = FailureResult(
        affected=(claim.node,),
        queue_update=None,
        snapshot=_snapshot(),
    )
    with (
        patch.object(
            handler.tree_queue,
            "fail_claim",
            AsyncMock(return_value=failure),
        ) as fail_claim,
        patch.object(
            handler.tree_queue,
            "complete_claim",
            AsyncMock(),
        ) as complete_claim,
    ):
        await handler.node_runner.process_node(claim)

    assert fail_claim.await_args_list == [
        call(claim, propagate=False),
        call(claim, propagate=True),
    ]
    complete_claim.assert_not_awaited()
    rendered = mock_platform.queue_edit_message.call_args_list[-1].args[2]
    assert "API Error: Request rejected" in rendered
    assert "Process exited with code" not in rendered
    assert "✅ *Complete*" not in rendered


@pytest.mark.asyncio
async def test_success_exit_still_renders_complete_after_non_exit_error(
    handler, mock_cli_manager, mock_platform
):
    claim = _claim()
    session = _session(
        [
            {"type": "error", "error": {"message": "recoverable warning"}},
            {"type": "exit", "code": 0},
        ]
    )
    mock_cli_manager.get_or_create_session.return_value = (
        session,
        "session_1",
        False,
    )
    failure = FailureResult(
        affected=(claim.node,),
        queue_update=None,
        snapshot=_snapshot(),
    )
    with (
        patch.object(
            handler.tree_queue,
            "fail_claim",
            AsyncMock(return_value=failure),
        ) as fail_claim,
        patch.object(
            handler.tree_queue,
            "complete_claim",
            AsyncMock(return_value=_snapshot()),
        ) as complete_claim,
    ):
        await handler.node_runner.process_node(claim)

    fail_claim.assert_awaited_once_with(
        claim,
        propagate=False,
    )
    complete_claim.assert_awaited_once_with(claim, "session_1")
    assert (
        "✅ *Complete*" in mock_platform.queue_edit_message.call_args_list[-1].args[2]
    )


@pytest.mark.asyncio
async def test_unexpected_runner_exception_uses_detailed_task_failed_ui(
    handler, mock_cli_manager, mock_platform
):
    claim = _claim()

    async def failing_start(*args, **kwargs):
        raise ValueError("runner exploded")
        if False:
            yield {}

    session = MagicMock()
    session.start_task = failing_start
    mock_cli_manager.get_or_create_session.return_value = (
        session,
        "session_1",
        False,
    )
    failure = FailureResult(
        affected=(claim.node,),
        queue_update=None,
        snapshot=_snapshot(),
    )
    with patch.object(
        handler.tree_queue,
        "fail_claim",
        AsyncMock(return_value=failure),
    ) as fail_claim:
        await handler.node_runner.process_node(claim)

    fail_claim.assert_awaited_once_with(
        claim,
        propagate=True,
    )
    rendered = mock_platform.queue_edit_message.call_args_list[-1].args[2]
    assert "Task Failed" in rendered
    assert "runner exploded" in rendered


@pytest.mark.asyncio
async def test_stop_cancellation_preserves_partial_transcript(
    handler, mock_cli_manager, mock_platform
):
    claim = _claim(prompt="work")
    started = asyncio.Event()

    async def start_task(*args, **kwargs):
        yield {
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": "partial answer"}]},
        }
        started.set()
        await asyncio.sleep(60)

    session = MagicMock()
    session.start_task = start_task
    mock_cli_manager.get_or_create_session.return_value = (
        session,
        "session_1",
        False,
    )
    failure = FailureResult(
        affected=(claim.node,),
        queue_update=None,
        snapshot=_snapshot(),
    )
    with patch.object(
        handler.tree_queue,
        "fail_claim",
        AsyncMock(return_value=failure),
    ) as fail_claim:
        task = asyncio.create_task(handler.node_runner.process_node(claim))
        await started.wait()
        task.cancel(CancellationReason.STOP)
        await task

    fail_claim.assert_awaited_once_with(
        claim,
        propagate=False,
    )
    rendered = mock_platform.queue_edit_message.call_args_list[-1].args[2]
    assert "partial answer" in rendered
    assert "⏹ *Stopped\\.*" in rendered
    assert rendered.index("partial answer") < rendered.index("⏹ *Stopped\\.*")


@pytest.mark.asyncio
async def test_global_clear_command_deletes_returned_ids(
    handler, mock_platform, incoming_message_factory
):
    handler.clear_all_state = AsyncMock(return_value=frozenset({"100", "101"}))
    incoming = incoming_message_factory(
        text="/clear",
        chat_id="chat_1",
        message_id="150",
    )

    await handler.handle_message(incoming)

    handler.clear_all_state.assert_awaited_once_with("telegram", "chat_1")
    mock_platform.queue_delete_messages.assert_awaited_once_with(
        "chat_1",
        ["150", "101", "100"],
        fire_and_forget=False,
    )
    mock_platform.queue_send_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_clear_all_state_is_chat_scoped_for_deletes_and_global_for_fcc_state(
    handler, mock_cli_manager, mock_session_store, incoming_message_factory
):
    root_1 = incoming_message_factory(
        text="one",
        chat_id="chat_1",
        message_id="100",
    )
    root_2 = incoming_message_factory(
        text="two",
        chat_id="chat_2",
        message_id="200",
    )
    await handler.tree_queue.admit(root_1, "101")
    await handler.tree_queue.admit(root_2, "201")
    await _wait_for_idle(handler)
    mock_session_store.get_message_ids_for_chat.return_value = ["42"]
    mock_session_store.reset_mock()
    mock_session_store.get_message_ids_for_chat.return_value = ["42"]

    message_ids = await handler.clear_all_state("telegram", "chat_1")

    assert message_ids == frozenset({"42", "100", "101"})
    assert "200" not in message_ids
    assert handler.get_tree_count() == 0
    mock_cli_manager.stop_all.assert_awaited_once()
    mock_session_store.clear_all.assert_called_once()


@pytest.mark.asyncio
async def test_cancelled_global_clear_before_commit_preserves_tree_and_store(
    handler,
    mock_cli_manager,
    mock_session_store,
    incoming_message_factory,
):
    root = incoming_message_factory(text="work", message_id="100")
    await handler.tree_queue.admit(root, "101")
    await _wait_for_idle(handler)
    initial_epoch = handler._admission_epoch
    id_read_started = asyncio.Event()

    async def block_id_read(platform: str, chat_id: str) -> set[str]:
        id_read_started.set()
        await asyncio.Future()
        raise AssertionError("unreachable")

    mock_session_store.reset_mock()
    with patch.object(
        handler.tree_queue,
        "get_message_ids_for_chat",
        new=block_id_read,
    ):
        clear_task = asyncio.create_task(
            handler.clear_all_state("telegram", root.chat_id)
        )
        await id_read_started.wait()
        clear_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await clear_task

    assert handler._admission_epoch == initial_epoch
    assert handler.get_tree_count() == 1
    mock_session_store.clear_all.assert_not_called()
    mock_session_store.clear_conversation_snapshot.assert_not_called()
    mock_cli_manager.stop_all.assert_not_awaited()


@pytest.mark.asyncio
async def test_clear_with_mention_uses_same_global_command(
    handler, mock_platform, incoming_message_factory
):
    handler.clear_all_state = AsyncMock(return_value=frozenset())
    incoming = incoming_message_factory(
        text="/clear@MyBot",
        chat_id="chat_1",
        message_id="10",
    )

    await handler.handle_message(incoming)

    handler.clear_all_state.assert_awaited_once_with("telegram", "chat_1")
    mock_platform.queue_delete_messages.assert_awaited_once_with(
        "chat_1",
        ["10"],
        fire_and_forget=False,
    )


@pytest.mark.asyncio
async def test_clear_continues_after_platform_delete_failure(
    handler, mock_platform, incoming_message_factory
):
    handler.clear_all_state = AsyncMock(return_value=frozenset({"41", "42"}))
    mock_platform.queue_delete_messages.side_effect = RuntimeError(
        "platform rejected delete"
    )

    await handler.handle_message(
        incoming_message_factory(text="/clear", message_id="150")
    )

    handler.clear_all_state.assert_awaited_once()
    mock_platform.queue_delete_messages.assert_awaited_once()


@pytest.mark.asyncio
async def test_reply_clear_removes_only_branch_and_persists_remaining_tree(
    handler,
    mock_platform,
    mock_session_store,
    incoming_message_factory,
):
    root = incoming_message_factory(text="root", message_id="100")
    child = incoming_message_factory(
        text="child",
        message_id="102",
        reply_to_message_id="100",
    )
    await handler.tree_queue.admit(root, "101")
    await _wait_for_idle(handler)
    await handler.tree_queue.admit(child, "103", parent_node_id="100")
    await _wait_for_idle(handler)
    mock_session_store.reset_mock()
    deleted_ids: list[str] = []

    async def capture_delete(chat_id, message_ids, fire_and_forget=True):
        deleted_ids.extend(message_ids)

    mock_platform.queue_delete_messages.side_effect = capture_delete
    await handler.handle_message(
        incoming_message_factory(
            text="/clear",
            message_id="150",
            reply_to_message_id="103",
        )
    )

    assert set(deleted_ids) == {"102", "103", "150"}
    assert "100" not in deleted_ids
    assert "101" not in deleted_ids
    assert await handler.tree_queue.get_node(root.scope, "102") is None
    assert await handler.tree_queue.get_node(root.scope, "100") is not None
    mock_session_store.save_tree_snapshot.assert_called_once()
    mock_session_store.forget_message_ids.assert_called_once_with(
        "telegram",
        "chat_1",
        {"102", "103", "150"},
    )


@pytest.mark.asyncio
async def test_reply_clear_unknown_reports_nothing_to_clear(
    handler, mock_platform, mock_session_store, incoming_message_factory
):
    incoming = incoming_message_factory(
        text="/clear",
        message_id="150",
        reply_to_message_id="999",
    )

    await handler.handle_message(incoming)

    assert "Nothing to clear" in mock_platform.queue_send_message.call_args.args[1]
    mock_session_store.clear_all.assert_not_called()


@pytest.mark.asyncio
async def test_reply_clear_root_removes_tree_snapshot(
    handler,
    mock_platform,
    mock_session_store,
    incoming_message_factory,
):
    root = incoming_message_factory(text="root", message_id="100")
    await handler.tree_queue.admit(root, "101")
    await _wait_for_idle(handler)
    mock_session_store.reset_mock()
    deleted_ids: list[str] = []

    async def capture_delete(chat_id, message_ids, fire_and_forget=True):
        deleted_ids.extend(message_ids)

    mock_platform.queue_delete_messages.side_effect = capture_delete
    await handler.handle_message(
        incoming_message_factory(
            text="/clear",
            message_id="150",
            reply_to_message_id="100",
        )
    )

    assert set(deleted_ids) == {"100", "101", "150"}
    mock_session_store.remove_tree_snapshot.assert_called_once_with(
        TreeIdentity(scope=root.scope, root_id="100")
    )
    assert handler.get_tree_count() == 0


@pytest.mark.asyncio
async def test_late_cancelled_runner_cannot_save_after_global_clear(
    handler, mock_cli_manager, mock_session_store, incoming_message_factory
):
    started = asyncio.Event()

    async def blocking_start(*args, **kwargs):
        started.set()
        await asyncio.sleep(60)
        if False:
            yield {}

    session = MagicMock()
    session.start_task = blocking_start
    mock_cli_manager.get_or_create_session.return_value = (
        session,
        "pending_1",
        True,
    )
    await handler.handle_message(
        incoming_message_factory(text="work", message_id="100")
    )
    await started.wait()
    mock_session_store.reset_mock()

    await handler.clear_all_state("telegram", "chat_1")

    mock_session_store.save_tree_snapshot.assert_not_called()
    mock_session_store.clear_all.assert_called_once()
    assert handler.get_tree_count() == 0


@pytest.mark.asyncio
async def test_global_clear_removes_snapshot_saved_during_detach_window(
    tmp_path,
    mock_platform,
    mock_cli_manager,
    incoming_message_factory,
):
    runner_started = asyncio.Event()
    release_runner = asyncio.Event()
    id_read_started = asyncio.Event()
    release_id_read = asyncio.Event()

    async def finish_after_release(*args, **kwargs):
        runner_started.set()
        await release_runner.wait()
        yield {"type": "exit", "code": 0}

    session = MagicMock()
    session.start_task = finish_after_release
    mock_cli_manager.get_or_create_session.return_value = (
        session,
        "session_1",
        False,
    )
    mock_platform.queue_send_message.return_value = "status-new"
    store_path = tmp_path / "sessions.json"
    store = SessionStore(storage_path=str(store_path))
    workflow = MessagingWorkflow(
        mock_platform,
        mock_cli_manager,
        store,
        platform_name="telegram",
    )
    incoming = incoming_message_factory(text="work", message_id="new")
    await workflow.handle_message(incoming)
    await runner_started.wait()

    get_ids = workflow.tree_queue.get_message_ids_for_chat

    async def block_id_read(platform: str, chat_id: str) -> set[str]:
        id_read_started.set()
        await release_id_read.wait()
        return await get_ids(platform, chat_id)

    try:
        with patch.object(
            workflow.tree_queue,
            "get_message_ids_for_chat",
            new=block_id_read,
        ):
            clear_task = asyncio.create_task(
                workflow.clear_all_state("telegram", incoming.chat_id)
            )
            await id_read_started.wait()
            release_runner.set()
            await _wait_for_idle(workflow)

            assert not store.load_conversation_snapshot().is_empty
            store.flush_pending_save()
            assert (
                not SessionStore(storage_path=str(store_path))
                .load_conversation_snapshot()
                .is_empty
            )
            release_id_read.set()
            await clear_task

        assert store.load_conversation_snapshot().is_empty
        assert (
            SessionStore(storage_path=str(store_path))
            .load_conversation_snapshot()
            .is_empty
        )
    finally:
        release_runner.set()
        release_id_read.set()
        workflow.close()


@pytest.mark.asyncio
async def test_global_clear_invalidates_inflight_prompt_without_waiting_for_status(
    handler,
    mock_platform,
    incoming_message_factory,
):
    status_send_started = asyncio.Event()
    release_status_send = asyncio.Event()

    async def block_status_send(*args, **kwargs):
        status_send_started.set()
        await release_status_send.wait()
        return "status-new"

    mock_platform.queue_send_message.side_effect = block_status_send
    prompt_task = asyncio.create_task(
        handler.handle_message(
            incoming_message_factory(text="new prompt", message_id="new")
        )
    )
    await status_send_started.wait()
    clear_task = asyncio.create_task(handler.clear_all_state("telegram", "chat_1"))

    try:
        await asyncio.wait_for(clear_task, timeout=1)
        release_status_send.set()
        await prompt_task
        assert handler.get_tree_count() == 0
        mock_platform.queue_delete_messages.assert_awaited_once_with(
            "chat_1",
            ["status-new"],
            fire_and_forget=False,
        )
    finally:
        release_status_send.set()
        if not prompt_task.done():
            prompt_task.cancel()
        if not clear_task.done():
            clear_task.cancel()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_message_id", "expected_deleted_ids"),
    [
        ("101", {"100", "101", "150"}),
        (None, {"100", "150"}),
    ],
)
async def test_reply_clear_pending_voice_cancels_and_reports(
    handler,
    mock_platform,
    incoming_message_factory,
    status_message_id,
    expected_deleted_ids,
) -> None:
    mock_platform.cancel_pending_voice.return_value = VoiceCancellationResult(
        voice_message_id="100",
        status_message_id=status_message_id,
    )
    deleted_ids: list[str] = []

    async def capture_delete(chat_id, message_ids, fire_and_forget=True):
        deleted_ids.extend(message_ids)

    mock_platform.queue_delete_messages.side_effect = capture_delete
    incoming = incoming_message_factory(
        text="/clear",
        message_id="150",
        reply_to_message_id="100",
    )

    await handler.handle_message(incoming)

    mock_platform.cancel_pending_voice.assert_awaited_once_with(incoming.scope, "100")
    assert set(deleted_ids) == expected_deleted_ids
    assert "Voice note cancelled" in mock_platform.queue_send_message.call_args.args[1]
