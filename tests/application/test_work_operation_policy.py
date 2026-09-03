from dataclasses import replace

from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule

from free_claude_code.application.work.models import (
    WorkOperation,
    WorkOperationKind,
    WorkOperationState,
)
from free_claude_code.application.work.operation_policy import (
    ACTIVE_OPERATION_STATES,
    AdmissionAction,
    decide_operation_admission,
)


def _operation(
    number: int,
    kind: WorkOperationKind,
    state: WorkOperationState = WorkOperationState.ACCEPTED,
) -> WorkOperation:
    return WorkOperation(
        operation_id=f"00000000-0000-4000-8000-{number:012d}",
        kind=kind,
        session_id=None if kind is WorkOperationKind.CREATE else "thread-1",
        interaction_id=(
            f"interaction-{number}" if kind is WorkOperationKind.RESPOND else None
        ),
        intent_digest=f"{number:064x}",
        payload={}
        if state in {WorkOperationState.ACCEPTED, WorkOperationState.EXECUTING}
        else None,
        state=state,
        expected_revision=None,
        captured_model=None,
        captured_reasoning_effort=None,
        native_thread_id=None,
        native_turn_id=None,
        native_connection_id=None,
        error_code=None,
        error_message=None,
        created_at_ms=number,
        updated_at_ms=number,
    )


def _reference_action(
    kind: WorkOperationKind, active: tuple[WorkOperation, ...]
) -> tuple[AdmissionAction, WorkOperationKind | None]:
    if kind is WorkOperationKind.CREATE:
        return (
            (AdmissionAction.CONFLICT, None)
            if active
            else (AdmissionAction.ADMIT, None)
        )
    if any(operation.state is WorkOperationState.UNKNOWN for operation in active):
        return AdmissionAction.CONFLICT, None
    kinds = {operation.kind for operation in active}
    if WorkOperationKind.DELETE in kinds:
        return (
            (AdmissionAction.RETURN_EXISTING, WorkOperationKind.DELETE)
            if kind is WorkOperationKind.DELETE
            else (AdmissionAction.CONFLICT, None)
        )
    if WorkOperationKind.STOP in kinds:
        if kind is WorkOperationKind.STOP:
            return AdmissionAction.RETURN_EXISTING, WorkOperationKind.STOP
        if kind is WorkOperationKind.RESPOND:
            return AdmissionAction.ADMIT, None
        return AdmissionAction.CONFLICT, None
    if WorkOperationKind.SEND in kinds:
        return (
            (AdmissionAction.ADMIT, None)
            if kind in {WorkOperationKind.STOP, WorkOperationKind.RESPOND}
            else (AdmissionAction.CONFLICT, None)
        )
    if WorkOperationKind.RESPOND in kinds:
        return (
            (AdmissionAction.ADMIT, None)
            if kind in {WorkOperationKind.STOP, WorkOperationKind.RESPOND}
            else (AdmissionAction.CONFLICT, None)
        )
    return AdmissionAction.ADMIT, None


class WorkOperationPolicyMachine(RuleBasedStateMachine):
    def __init__(self) -> None:
        super().__init__()
        self.operations: dict[str, WorkOperation] = {}
        self.next_number = 1

    def _active_for(self, kind: WorkOperationKind) -> tuple[WorkOperation, ...]:
        return tuple(
            operation
            for operation in self.operations.values()
            if operation.state in ACTIVE_OPERATION_STATES
            and (
                operation.kind is WorkOperationKind.CREATE
                if kind is WorkOperationKind.CREATE
                else operation.kind is not WorkOperationKind.CREATE
            )
        )

    @rule(kind=st.sampled_from(tuple(WorkOperationKind)))
    def admit(self, kind: WorkOperationKind) -> None:
        active = self._active_for(kind)
        expected_action, expected_kind = _reference_action(kind, active)
        actual = decide_operation_admission(kind, active)
        assert actual.action is expected_action
        if expected_kind is not None:
            existing = self.operations[actual.existing_operation_id or ""]
            assert existing.kind is expected_kind
        if actual.action is AdmissionAction.ADMIT:
            operation = _operation(self.next_number, kind)
            self.next_number += 1
            self.operations[operation.operation_id] = operation

    @rule(data=st.data())
    def transition_one(self, data: st.DataObject) -> None:
        active = [
            operation
            for operation in self.operations.values()
            if operation.state in ACTIVE_OPERATION_STATES
        ]
        if not active:
            return
        operation = data.draw(st.sampled_from(active))
        if operation.state is WorkOperationState.ACCEPTED:
            target = WorkOperationState.EXECUTING
        elif operation.state is WorkOperationState.EXECUTING:
            target = data.draw(
                st.sampled_from(
                    (
                        WorkOperationState.UNKNOWN,
                        WorkOperationState.SUCCEEDED,
                        WorkOperationState.FAILED,
                    )
                )
            )
        else:
            target = data.draw(
                st.sampled_from(
                    (
                        WorkOperationState.SUCCEEDED,
                        WorkOperationState.FAILED,
                        WorkOperationState.ABANDONED,
                    )
                )
            )
        self.operations[operation.operation_id] = replace(
            operation,
            state=target,
            payload=(
                operation.payload
                if target in {WorkOperationState.ACCEPTED, WorkOperationState.EXECUTING}
                else None
            ),
        )

    @invariant()
    def admitted_state_respects_exclusion_rules(self) -> None:
        active = [
            operation
            for operation in self.operations.values()
            if operation.state in ACTIVE_OPERATION_STATES
        ]
        creates = [
            operation
            for operation in active
            if operation.kind is WorkOperationKind.CREATE
        ]
        session = [
            operation
            for operation in active
            if operation.kind is not WorkOperationKind.CREATE
        ]
        assert len(creates) <= 1
        assert (
            sum(operation.kind is WorkOperationKind.SEND for operation in session) <= 1
        )
        assert (
            sum(operation.kind is WorkOperationKind.STOP for operation in session) <= 1
        )
        assert (
            sum(operation.kind is WorkOperationKind.DELETE for operation in session)
            <= 1
        )
        if any(operation.kind is WorkOperationKind.DELETE for operation in session):
            assert len(session) == 1
        if any(operation.state is WorkOperationState.UNKNOWN for operation in session):
            for kind in (
                WorkOperationKind.SEND,
                WorkOperationKind.STOP,
                WorkOperationKind.DELETE,
                WorkOperationKind.RESPOND,
            ):
                assert (
                    decide_operation_admission(kind, tuple(session)).action
                    is AdmissionAction.CONFLICT
                )


TestWorkOperationPolicy = WorkOperationPolicyMachine.TestCase
