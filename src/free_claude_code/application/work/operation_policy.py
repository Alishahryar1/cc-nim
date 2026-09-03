"""Pure lifecycle policy for durable Work operations."""

from dataclasses import dataclass
from enum import StrEnum

from .models import (
    WorkInteraction,
    WorkInteractionKind,
    WorkOperation,
    WorkOperationKind,
    WorkOperationState,
    WorkStatus,
)

ACTIVE_OPERATION_STATES = frozenset(
    {
        WorkOperationState.ACCEPTED,
        WorkOperationState.EXECUTING,
        WorkOperationState.UNKNOWN,
    }
)

LEGAL_OPERATION_TRANSITIONS: dict[WorkOperationState, frozenset[WorkOperationState]] = {
    WorkOperationState.ACCEPTED: frozenset(
        {WorkOperationState.EXECUTING, WorkOperationState.FAILED}
    ),
    WorkOperationState.EXECUTING: frozenset(
        {
            WorkOperationState.UNKNOWN,
            WorkOperationState.SUCCEEDED,
            WorkOperationState.FAILED,
        }
    ),
    WorkOperationState.UNKNOWN: frozenset(
        {
            WorkOperationState.SUCCEEDED,
            WorkOperationState.FAILED,
            WorkOperationState.ABANDONED,
        }
    ),
}


class AdmissionAction(StrEnum):
    ADMIT = "admit"
    RETURN_EXISTING = "return_existing"
    CONFLICT = "conflict"


@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    action: AdmissionAction
    existing_operation_id: str | None = None
    message: str | None = None


def decide_operation_admission(
    kind: WorkOperationKind,
    active_operations: tuple[WorkOperation, ...],
) -> AdmissionDecision:
    """Decide one mutation from a complete active-operation snapshot."""

    active = tuple(
        operation
        for operation in active_operations
        if operation.state in ACTIVE_OPERATION_STATES
    )
    if kind is WorkOperationKind.CREATE:
        if active:
            return _conflict("A Work session is already being created.")
        return _admit()

    if any(operation.state is WorkOperationState.UNKNOWN for operation in active):
        return _conflict(
            "Resolve this Work session's uncertain operation before continuing."
        )

    deleting = _first(active, WorkOperationKind.DELETE)
    if deleting is not None:
        if kind is WorkOperationKind.DELETE:
            return _existing(deleting)
        return _conflict("This Work session is being deleted.")

    stopping = _first(active, WorkOperationKind.STOP)
    if stopping is not None:
        if kind is WorkOperationKind.STOP:
            return _existing(stopping)
        if kind is WorkOperationKind.RESPOND:
            return _admit()
        return _conflict("Wait for the current Stop operation to finish.")

    sending = _first(active, WorkOperationKind.SEND)
    responding = _first(active, WorkOperationKind.RESPOND)
    if sending is not None:
        if kind in {WorkOperationKind.STOP, WorkOperationKind.RESPOND}:
            return _admit()
        if kind is WorkOperationKind.SEND:
            return _conflict("This Work session already has an active turn.")
        if kind is WorkOperationKind.DELETE:
            return _conflict("Stop the active turn before deleting this session.")
    elif responding is not None:
        if kind in {WorkOperationKind.STOP, WorkOperationKind.RESPOND}:
            return _admit()
        if kind is WorkOperationKind.SEND:
            return _conflict("Answer or stop the pending Codex request first.")
        if kind is WorkOperationKind.DELETE:
            return _conflict(
                "Answer or stop the pending Codex request before deleting."
            )

    return _admit()


def settings_conflict(active_operations: tuple[WorkOperation, ...]) -> str | None:
    """Return why settings cannot change for the active durable facts, if any."""

    active = tuple(
        operation
        for operation in active_operations
        if operation.state in ACTIVE_OPERATION_STATES
    )
    if any(operation.state is WorkOperationState.UNKNOWN for operation in active):
        return (
            "Resolve this Work session's uncertain operation before changing settings."
        )
    if _first(active, WorkOperationKind.DELETE) is not None:
        return "This Work session is being deleted."
    if _first(active, WorkOperationKind.STOP) is not None:
        return "Wait for the current Stop operation before changing settings."
    return None


def derive_work_status(
    operations: tuple[WorkOperation, ...],
    interactions: tuple[WorkInteraction, ...],
    *,
    native_status: WorkStatus,
    disconnected: bool,
) -> WorkStatus:
    """Derive one public status from durable and observed session facts."""

    active = tuple(
        operation
        for operation in operations
        if operation.state in ACTIVE_OPERATION_STATES
    )
    if any(operation.state is WorkOperationState.UNKNOWN for operation in active):
        return WorkStatus.NEEDS_ATTENTION
    if _first(active, WorkOperationKind.DELETE) is not None:
        return WorkStatus.DELETING
    if _first(active, WorkOperationKind.STOP) is not None:
        return WorkStatus.STOPPING
    if interactions:
        return (
            WorkStatus.WAITING_FOR_INPUT
            if any(
                interaction.kind is WorkInteractionKind.USER_INPUT
                for interaction in interactions
            )
            else WorkStatus.WAITING_FOR_APPROVAL
        )
    if _first(active, WorkOperationKind.SEND) is not None:
        return WorkStatus.WORKING
    if disconnected:
        return WorkStatus.DISCONNECTED
    return native_status


def _first(
    operations: tuple[WorkOperation, ...], kind: WorkOperationKind
) -> WorkOperation | None:
    return next((operation for operation in operations if operation.kind is kind), None)


def _admit() -> AdmissionDecision:
    return AdmissionDecision(AdmissionAction.ADMIT)


def _existing(operation: WorkOperation) -> AdmissionDecision:
    return AdmissionDecision(
        AdmissionAction.RETURN_EXISTING,
        existing_operation_id=operation.operation_id,
    )


def _conflict(message: str) -> AdmissionDecision:
    return AdmissionDecision(AdmissionAction.CONFLICT, message=message)
