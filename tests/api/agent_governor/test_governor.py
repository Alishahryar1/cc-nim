"""End-to-end behaviour tests for the AgentGovernor verdict pipeline."""

from __future__ import annotations

from api.models.anthropic import (
    ContentBlockText,
    ContentBlockToolResult,
    ContentBlockToolUse,
    Message,
    MessagesRequest,
    Tool,
)
from api.agent_governor import AgentGovernor, GovernorAction
from api.agent_governor.config import GovernorConfig


def _request(
    *,
    model: str = "qwen/qwen3-next-80b-a3b-instruct",
    messages: list[Message] | None = None,
    tools: list[Tool] | None = None,
    system: str | None = None,
) -> MessagesRequest:
    return MessagesRequest(
        model=model,
        max_tokens=4096,
        messages=messages
        or [Message(role="user", content=[ContentBlockText(type="text", text="hi")])],
        tools=tools,
        system=system,
    )


def _tool_use_message(name: str, args: dict, tool_id: str) -> Message:
    return Message(
        role="assistant",
        content=[ContentBlockToolUse(type="tool_use", id=tool_id, name=name, input=args)],
    )


def _tool_result_message(tool_id: str) -> Message:
    return Message(
        role="user",
        content=[
            ContentBlockToolResult(
                type="tool_result", tool_use_id=tool_id, content="ok"
            )
        ],
    )


def _disabled_governor() -> AgentGovernor:
    return AgentGovernor(GovernorConfig(enabled=False))


def _default_weak_governor() -> AgentGovernor:
    return AgentGovernor(GovernorConfig())


def test_disabled_governor_passes_through():
    gov = _disabled_governor()
    verdict = gov.review(_request(), request_id="req_x", provider_id="nvidia_nim")
    assert verdict.action == GovernorAction.PASS
    assert verdict.augmented_request is None


def test_strong_model_bypasses_governor():
    gov = AgentGovernor(GovernorConfig())
    verdict = gov.review(
        _request(model="claude-sonnet-4-6"),
        request_id="req_x",
        provider_id="anthropic",
    )
    assert verdict.action == GovernorAction.PASS


def test_total_budget_aborts():
    cfg = GovernorConfig(max_total_tool_calls=3)
    gov = AgentGovernor(cfg)
    msgs = []
    for i in range(3):
        msgs.append(_tool_use_message("read", {"path": f"f{i}"}, f"t{i}"))
        msgs.append(_tool_result_message(f"t{i}"))
    verdict = gov.review(
        _request(messages=msgs), request_id="req_x", provider_id="nvidia_nim"
    )
    assert verdict.action == GovernorAction.ABORT
    assert verdict.short_circuit_response is not None
    assert verdict.short_circuit_response.stop_reason == "end_turn"
    assert "total_tool_budget_exceeded" in verdict.short_circuit_response.content[0].text


def test_loop_signature_aborts_before_budget():
    cfg = GovernorConfig(max_total_tool_calls=1000, loop_repeat_threshold=3)
    gov = AgentGovernor(cfg)
    msgs = []
    for i in range(3):
        msgs.append(_tool_use_message("read", {"path": "/etc/hosts"}, f"t{i}"))
        msgs.append(_tool_result_message(f"t{i}"))
    verdict = gov.review(
        _request(messages=msgs), request_id="req_x", provider_id="nvidia_nim"
    )
    assert verdict.action == GovernorAction.ABORT
    assert "loop_signature_detected" in verdict.short_circuit_response.content[0].text


def test_consecutive_cap_aborts():
    cfg = GovernorConfig(
        max_total_tool_calls=1000,
        max_consecutive_tool_calls=2,
        loop_repeat_threshold=99,
    )
    gov = AgentGovernor(cfg)
    msgs = []
    for i in range(2):
        msgs.append(_tool_use_message("read", {"path": f"f{i}"}, f"t{i}"))
        msgs.append(_tool_result_message(f"t{i}"))
    verdict = gov.review(
        _request(messages=msgs), request_id="req_x", provider_id="nvidia_nim"
    )
    assert verdict.action == GovernorAction.ABORT
    assert "consecutive_tool_calls_exceeded" in verdict.short_circuit_response.content[0].text


def test_augment_culls_tools_when_over_threshold():
    cfg = GovernorConfig(
        tool_cull_max=5,
        plan_mode_for_weak=False,
        termination_hint_for_weak=False,
    )
    gov = AgentGovernor(cfg)
    tools = [Tool(name=f"tool_{i}", description=f"thing {i}") for i in range(20)]
    verdict = gov.review(
        _request(tools=tools), request_id="req_x", provider_id="nvidia_nim"
    )
    assert verdict.action == GovernorAction.AUGMENT
    assert verdict.augmented_request is not None
    assert len(verdict.augmented_request.tools) <= 5


def test_augment_appends_plan_mode_for_weak_models():
    cfg = GovernorConfig(
        tool_cull_enabled=False,
        plan_mode_for_weak=True,
        termination_hint_for_weak=False,
    )
    gov = AgentGovernor(cfg)
    verdict = gov.review(
        _request(system="You are helpful."),
        request_id="req_x",
        provider_id="nvidia_nim",
    )
    assert verdict.action == GovernorAction.AUGMENT
    assert "[Operating mode: plan-then-execute]" in verdict.augmented_request.system


def test_augment_appends_termination_hint_for_weak_models():
    cfg = GovernorConfig(
        tool_cull_enabled=False,
        plan_mode_for_weak=False,
        termination_hint_for_weak=True,
    )
    gov = AgentGovernor(cfg)
    verdict = gov.review(
        _request(system="You are helpful."),
        request_id="req_x",
        provider_id="nvidia_nim",
    )
    assert verdict.action == GovernorAction.AUGMENT
    assert "[Termination policy]" in verdict.augmented_request.system


def test_strong_model_keeps_all_tools_and_no_preamble():
    cfg = GovernorConfig(tool_cull_max=5)
    gov = AgentGovernor(cfg)
    tools = [Tool(name=f"tool_{i}", description=f"thing {i}") for i in range(20)]
    verdict = gov.review(
        _request(model="claude-sonnet-4-6", tools=tools, system="You are helpful."),
        request_id="req_x",
        provider_id="anthropic",
    )
    assert verdict.action == GovernorAction.PASS


def test_clean_simple_request_passes():
    gov = _default_weak_governor()
    verdict = gov.review(_request(), request_id="req_x", provider_id="nvidia_nim")
    # No tools, no history → only the weak-model preambles get added.
    # plan_mode_for_weak + termination_hint_for_weak both default to True,
    # so we expect AUGMENT here, not PASS.
    assert verdict.action == GovernorAction.AUGMENT
    assert "[Operating mode: plan-then-execute]" in verdict.augmented_request.system
    assert "[Termination policy]" in verdict.augmented_request.system
