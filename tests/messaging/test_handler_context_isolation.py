from unittest.mock import AsyncMock, MagicMock

import pytest

from messaging.trees import MessageState
from messaging.workflow import MessagingWorkflow


async def _gen_session(events):
    for e in events:
        yield e


@pytest.fixture
def handler(mock_platform, mock_cli_manager, mock_session_store):
    return MessagingWorkflow(mock_platform, mock_cli_manager, mock_session_store)


@pytest.mark.asyncio
async def test_sibling_replies_fork_from_parent_session_id(
    handler, mock_cli_manager, incoming_message_factory
):
    # Root node A with a known session_id.
    root_incoming = incoming_message_factory(text="A", message_id="A")
    tree = await handler.tree_queue.create_tree(
        node_id="A", incoming=root_incoming, status_message_id="status_A"
    )
    await tree.update_state("A", MessageState.COMPLETED, session_id="sess_A")

    # Add two sibling replies R1 and R2 under A.
    r1_incoming = incoming_message_factory(
        text="R1", message_id="R1", reply_to_message_id="A"
    )
    r2_incoming = incoming_message_factory(
        text="R2", message_id="R2", reply_to_message_id="A"
    )
    _, r1_node = await handler.tree_queue.add_to_tree(
        "A", "R1", r1_incoming, "status_R1"
    )
    _, r2_node = await handler.tree_queue.add_to_tree(
        "A", "R2", r2_incoming, "status_R2"
    )

    # Mock a fresh cli_session per node.
    calls = []

    async def _get_or_create_session(session_id=None):
        cli_session = MagicMock()

        async def _start_task(prompt, session_id=None, fork_session=False):
            calls.append((prompt, session_id, fork_session))
            child_sid = f"sess_{prompt}"
            async for ev in _gen_session(
                [
                    {"type": "session_info", "session_id": child_sid},
                    {"type": "exit", "code": 0, "stderr": None},
                ]
            ):
                yield ev

        cli_session.start_task = _start_task
        return cli_session, f"pending_{len(calls) + 1}", True

    mock_cli_manager.get_or_create_session = AsyncMock(
        side_effect=_get_or_create_session
    )

    await handler.node_runner.process_node("R1", r1_node)
    await handler.node_runner.process_node("R2", r2_node)

    # Both siblings must resume from the same parent session and fork.
    assert calls[0][0] == "R1"
    assert calls[0][1] == "sess_A"
    assert calls[0][2] is True

    assert calls[1][0] == "R2"
    assert calls[1][1] == "sess_A"
    assert calls[1][2] is True


@pytest.mark.asyncio
async def test_grandchild_reply_forks_from_branch_session(
    handler, mock_cli_manager, incoming_message_factory
):
    root_incoming = incoming_message_factory(text="A", message_id="A")
    tree = await handler.tree_queue.create_tree(
        node_id="A", incoming=root_incoming, status_message_id="status_A"
    )
    await tree.update_state("A", MessageState.COMPLETED, session_id="sess_A")

    r1_incoming = incoming_message_factory(
        text="R1", message_id="R1", reply_to_message_id="A"
    )
    _, r1_node = await handler.tree_queue.add_to_tree(
        "A", "R1", r1_incoming, "status_R1"
    )

    calls = []

    async def _get_or_create_session(session_id=None):
        cli_session = MagicMock()

        async def _start_task(prompt, session_id=None, fork_session=False):
            calls.append((prompt, session_id, fork_session))
            # R1 gets its own forked session id.
            child_sid = "sess_R1"
            async for ev in _gen_session(
                [
                    {"type": "session_info", "session_id": child_sid},
                    {"type": "exit", "code": 0, "stderr": None},
                ]
            ):
                yield ev

        cli_session.start_task = _start_task
        return cli_session, "pending_R1", True

    mock_cli_manager.get_or_create_session = AsyncMock(
        side_effect=_get_or_create_session
    )

    await handler.node_runner.process_node("R1", r1_node)
    assert r1_node.session_id == "sess_R1"

    # Grandchild C1 replies to R1 and must fork from sess_R1, not sess_A.
    c1_incoming = incoming_message_factory(
        text="C1", message_id="C1", reply_to_message_id="R1"
    )
    _, c1_node = await handler.tree_queue.add_to_tree(
        "R1", "C1", c1_incoming, "status_C1"
    )

    async def _get_or_create_session_c1(session_id=None):
        cli_session = MagicMock()

        async def _start_task(prompt, session_id=None, fork_session=False):
            calls.append((prompt, session_id, fork_session))
            async for ev in _gen_session(
                [
                    {"type": "session_info", "session_id": "sess_C1"},
                    {"type": "exit", "code": 0, "stderr": None},
                ]
            ):
                yield ev

        cli_session.start_task = _start_task
        return cli_session, "pending_C1", True

    mock_cli_manager.get_or_create_session = AsyncMock(
        side_effect=_get_or_create_session_c1
    )

    await handler.node_runner.process_node("C1", c1_node)

    # The last call should be for C1 and must resume from sess_R1.
    assert calls[-1][0] == "C1"
    assert calls[-1][1] == "sess_R1"
    assert calls[-1][2] is True
