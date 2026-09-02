"""Pure normalization and bounded projection for Codex Work activity."""

import hashlib
import json
from dataclasses import dataclass, replace

from free_claude_code.core.json_types import JsonObject, JsonValue

from .codex import CodexInteractionKind, CodexInteractionRequest, CodexNotification
from .models import (
    WorkInteraction,
    WorkInteractionKind,
    WorkStatus,
    WorkTimelineItem,
    WorkTurnPage,
)

_DELTA_METHOD_FIELDS = {
    "item/agentMessage/delta": "text",
    "item/commandExecution/outputDelta": "aggregatedOutput",
    "item/fileChange/outputDelta": "output",
    "item/mcpToolCall/progress": "progress",
    "item/plan/delta": "text",
    "item/reasoning/summaryTextDelta": "summary",
    "item/reasoning/textDelta": "content",
}


@dataclass(frozen=True, slots=True)
class ProjectionState:
    """At most one active/completed turn not yet confirmed in native history."""

    items: tuple[WorkTimelineItem, ...] = ()
    turn_id: str | None = None
    completed: bool = False


def history_page(thread_id: str, turns: tuple[JsonObject, ...]) -> WorkTurnPage:
    """Convert Codex's newest-first turns into chronological browser items."""

    items: list[WorkTimelineItem] = []
    for turn in reversed(turns):
        turn_id = optional_string(turn.get("id"))
        native_items = turn.get("items")
        if turn_id is None or not isinstance(native_items, list):
            continue
        items.extend(
            timeline_item(thread_id, turn_id, native_item)
            for native_item in native_items
            if isinstance(native_item, dict)
        )
    return WorkTurnPage(items=tuple(items))


def history_contains_turn(turns: tuple[JsonObject, ...], turn_id: str) -> bool:
    return any(optional_string(turn.get("id")) == turn_id for turn in turns)


def without_persisted(
    state: ProjectionState,
    persisted: WorkTurnPage,
) -> tuple[WorkTimelineItem, ...]:
    persisted_keys = {
        (item.thread_id, item.turn_id, item.item_id) for item in persisted.items
    }
    return tuple(
        item
        for item in state.items
        if (item.thread_id, item.turn_id, item.item_id) not in persisted_keys
    )


def begin_turn(state: ProjectionState, turn_id: str) -> ProjectionState:
    if state.turn_id == turn_id:
        return replace(state, completed=False)
    if state.items:
        raise RuntimeError("A new Work turn began before projection handoff.")
    return ProjectionState(turn_id=turn_id)


def complete_turn(state: ProjectionState, turn_id: str) -> ProjectionState:
    if state.turn_id not in {None, turn_id}:
        return state
    return replace(state, turn_id=turn_id, completed=True)


def clear_completed(state: ProjectionState, turn_id: str) -> ProjectionState:
    if state.turn_id != turn_id or not state.completed:
        return state
    return ProjectionState()


def apply_notification(
    state: ProjectionState,
    thread_id: str,
    event: CodexNotification,
) -> tuple[ProjectionState, WorkTimelineItem | None]:
    params = as_object(event.params)
    method = event.method
    if method in {"item/started", "item/completed"}:
        turn_id = optional_string(params.get("turnId"))
        item = as_object(params.get("item"))
        if turn_id is None or not item:
            return state, None
        projected = timeline_item(thread_id, turn_id, item)
        return _upsert(state, projected), projected
    if method in _DELTA_METHOD_FIELDS:
        return _apply_delta(state, thread_id, method, params)
    if method == "item/fileChange/patchUpdated":
        return _replace_item_field(state, thread_id, params, "changes")
    if method == "turn/diff/updated":
        return _upsert_synthetic(
            state,
            thread_id,
            params,
            item_id="turn-diff",
            kind="diff",
            text=optional_string(params.get("diff")),
        )
    if method == "turn/plan/updated":
        return _upsert_synthetic(
            state,
            thread_id,
            params,
            item_id="turn-plan",
            kind="plan",
            text=json_text(params.get("plan")),
        )
    turn_id = optional_string(params.get("turnId"))
    if turn_id is None or method.startswith(("thread/", "turn/")):
        return state, None
    item_id = (
        optional_string(params.get("itemId"))
        or _digest(
            method,
            json.dumps(params, sort_keys=True, separators=(",", ":"), default=str),
        )[:24]
    )
    item = WorkTimelineItem(
        thread_id=thread_id,
        turn_id=turn_id,
        item_id=f"event-{item_id}",
        kind="codexActivity",
        status=None,
        text=method,
        payload={"method": method},
    )
    return _upsert(state, item), item


def public_interaction(
    interaction_id: str,
    request: CodexInteractionRequest,
) -> WorkInteraction:
    kind = {
        CodexInteractionKind.COMMAND_APPROVAL: WorkInteractionKind.COMMAND_APPROVAL,
        CodexInteractionKind.FILE_CHANGE_APPROVAL: WorkInteractionKind.FILE_CHANGE_APPROVAL,
        CodexInteractionKind.PERMISSION_APPROVAL: WorkInteractionKind.PERMISSION_APPROVAL,
        CodexInteractionKind.USER_INPUT: WorkInteractionKind.USER_INPUT,
    }[request.kind]
    return WorkInteraction(
        interaction_id=interaction_id,
        thread_id=request.thread_id,
        turn_id=request.turn_id,
        kind=kind,
        title={
            WorkInteractionKind.COMMAND_APPROVAL: "Command approval",
            WorkInteractionKind.FILE_CHANGE_APPROVAL: "File change approval",
            WorkInteractionKind.PERMISSION_APPROVAL: "Additional permission",
            WorkInteractionKind.USER_INPUT: "Codex needs your input",
        }[kind],
        payload=_public_interaction_payload(request),
    )


def notification_thread_id(event: CodexNotification) -> str | None:
    params = as_object(event.params)
    thread_id = optional_string(params.get("threadId"))
    if thread_id is not None:
        return thread_id
    if event.method == "thread/started":
        return optional_string(as_object(params.get("thread")).get("id"))
    return None


def native_status(value: JsonValue) -> WorkStatus:
    status = as_object(value)
    kind = status.get("type")
    flags = status.get("activeFlags")
    if kind == "active" and isinstance(flags, list):
        if "waitingOnUserInput" in flags:
            return WorkStatus.WAITING_FOR_INPUT
        if "waitingOnApproval" in flags:
            return WorkStatus.WAITING_FOR_APPROVAL
        return WorkStatus.WORKING
    if kind == "systemError":
        return WorkStatus.FAILED
    return WorkStatus.READY


def turn_status(turn: JsonObject) -> WorkStatus:
    status = optional_string(turn.get("status"))
    if status in {"interrupted", "cancelled", "canceled"}:
        return WorkStatus.INTERRUPTED
    if status in {"failed", "error"}:
        return WorkStatus.FAILED
    return WorkStatus.COMPLETED


def turn_error(turn: JsonObject) -> str:
    error = turn.get("error")
    if isinstance(error, dict):
        message = optional_string(error.get("message"))
        if message:
            return message
    return "Codex could not complete this turn."


def timeline_item(
    thread_id: str,
    turn_id: str,
    item: JsonObject,
) -> WorkTimelineItem:
    item_id = (
        optional_string(item.get("id"))
        or _digest(
            thread_id,
            turn_id,
            json.dumps(item, sort_keys=True, separators=(",", ":"), default=str),
        )[:24]
    )
    kind = optional_string(item.get("type")) or "codexActivity"
    return WorkTimelineItem(
        thread_id=thread_id,
        turn_id=turn_id,
        item_id=item_id,
        kind=kind,
        status=optional_string(item.get("status")),
        text=_item_text(kind, item),
        payload=dict(item),
    )


def as_object(value: JsonValue) -> JsonObject:
    return dict(value) if isinstance(value, dict) else {}


def optional_string(value: JsonValue) -> str | None:
    return value if isinstance(value, str) and value else None


def json_text(value: JsonValue) -> str | None:
    if isinstance(value, str):
        return value
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _upsert(state: ProjectionState, item: WorkTimelineItem) -> ProjectionState:
    retained = tuple(
        candidate
        for candidate in state.items
        if not (
            candidate.thread_id == item.thread_id
            and candidate.turn_id == item.turn_id
            and candidate.item_id == item.item_id
        )
    )
    turn_id = state.turn_id or item.turn_id
    if turn_id != item.turn_id:
        raise RuntimeError("Work projection received overlapping turns.")
    return ProjectionState(
        items=(*retained, item),
        turn_id=turn_id,
        completed=state.completed,
    )


def _apply_delta(
    state: ProjectionState,
    thread_id: str,
    method: str,
    params: JsonObject,
) -> tuple[ProjectionState, WorkTimelineItem | None]:
    turn_id = optional_string(params.get("turnId"))
    item_id = optional_string(params.get("itemId"))
    delta = optional_string(params.get("delta"))
    if turn_id is None or item_id is None or delta is None:
        return state, None
    current = next(
        (
            item
            for item in state.items
            if item.turn_id == turn_id and item.item_id == item_id
        ),
        None,
    )
    kind = _kind_from_delta(method)
    if current is None:
        current = WorkTimelineItem(
            thread_id=thread_id,
            turn_id=turn_id,
            item_id=item_id,
            kind=kind,
            status="inProgress",
            text="",
            payload={"id": item_id, "type": kind},
        )
    field_name = _DELTA_METHOD_FIELDS[method]
    payload = dict(current.payload)
    payload[field_name] = (optional_string(payload.get(field_name)) or "") + delta
    updated = replace(current, text=(current.text or "") + delta, payload=payload)
    return _upsert(state, updated), updated


def _replace_item_field(
    state: ProjectionState,
    thread_id: str,
    params: JsonObject,
    field_name: str,
) -> tuple[ProjectionState, WorkTimelineItem | None]:
    turn_id = optional_string(params.get("turnId"))
    item_id = optional_string(params.get("itemId"))
    if turn_id is None or item_id is None:
        return state, None
    current = next(
        (
            item
            for item in state.items
            if item.turn_id == turn_id and item.item_id == item_id
        ),
        WorkTimelineItem(
            thread_id=thread_id,
            turn_id=turn_id,
            item_id=item_id,
            kind="fileChange",
            status="inProgress",
            text=None,
            payload={"id": item_id, "type": "fileChange"},
        ),
    )
    payload = dict(current.payload)
    payload[field_name] = params.get(field_name)
    updated = replace(current, payload=payload)
    return _upsert(state, updated), updated


def _upsert_synthetic(
    state: ProjectionState,
    thread_id: str,
    params: JsonObject,
    *,
    item_id: str,
    kind: str,
    text: str | None,
) -> tuple[ProjectionState, WorkTimelineItem | None]:
    turn_id = optional_string(params.get("turnId"))
    if turn_id is None:
        return state, None
    item = WorkTimelineItem(
        thread_id=thread_id,
        turn_id=turn_id,
        item_id=item_id,
        kind=kind,
        status="inProgress",
        text=text,
        payload=dict(params),
    )
    return _upsert(state, item), item


def _item_text(kind: str, item: JsonObject) -> str | None:
    direct_fields = {
        "agentMessage": "text",
        "plan": "text",
        "commandExecution": "aggregatedOutput",
        "webSearch": "query",
        "imageView": "path",
    }
    field_name = direct_fields.get(kind)
    if field_name is not None and (direct := optional_string(item.get(field_name))):
        return direct
    if kind == "userMessage":
        content = item.get("content")
        if isinstance(content, list):
            values: list[str] = []
            for entry in content:
                if not isinstance(entry, dict):
                    continue
                if text := optional_string(entry.get("text")):
                    values.append(text)
                elif entry.get("type") in {"image", "inputImage"}:
                    values.append("[Image]")
            return "\n".join(values) or None
    if kind == "reasoning":
        return json_text(item.get("summary")) or json_text(item.get("content"))
    if kind == "commandExecution":
        command = item.get("command")
        if isinstance(command, list):
            return " ".join(value for value in command if isinstance(value, str))
        return optional_string(command)
    if kind in {"mcpToolCall", "dynamicToolCall"}:
        return optional_string(item.get("tool"))
    if kind == "fileChange":
        return json_text(item.get("changes"))
    return None


def _public_interaction_payload(request: CodexInteractionRequest) -> JsonObject:
    params = request.params
    if request.kind is CodexInteractionKind.COMMAND_APPROVAL:
        return {
            key: params.get(key)
            for key in (
                "itemId",
                "startedAtMs",
                "approvalId",
                "environmentId",
                "reason",
                "networkApprovalContext",
                "command",
                "cwd",
                "commandActions",
                "proposedExecpolicyAmendment",
                "proposedNetworkPolicyAmendments",
                "availableDecisions",
            )
            if key in params
        }
    if request.kind is CodexInteractionKind.FILE_CHANGE_APPROVAL:
        return {
            key: params.get(key)
            for key in ("itemId", "startedAtMs", "reason", "grantRoot")
            if key in params
        }
    if request.kind is CodexInteractionKind.PERMISSION_APPROVAL:
        return {
            key: params.get(key)
            for key in (
                "itemId",
                "startedAtMs",
                "environmentId",
                "cwd",
                "reason",
                "permissions",
            )
            if key in params
        }
    return {
        key: params.get(key)
        for key in ("itemId", "questions", "isBlocking", "autoResolutionMs")
        if key in params
    }


def _kind_from_delta(method: str) -> str:
    if "agentMessage" in method:
        return "agentMessage"
    if "commandExecution" in method:
        return "commandExecution"
    if "fileChange" in method:
        return "fileChange"
    if "reasoning" in method:
        return "reasoning"
    if "plan" in method:
        return "plan"
    return "mcpToolCall"


def _digest(*values: str) -> str:
    payload = json.dumps(values, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()
