from typing import Any

from pydantic import BaseModel

from api.routes import _sanitize_breaking_tools


class MockMessageRequest(BaseModel):
    tools: list[Any] | None = None
    tool_choice: Any | None = None
    system: Any | None = ""
    thinking: Any | None = {"type": "enabled", "budget": 2048}


def test_sanitize_drops_only_unsupported_tools():
    req = MockMessageRequest(
        tools=[
            {"name": "advisor_code_review"},
            {"name": "web_search_tool"},
            {"name": "web_search_20250305"},
            {"name": "get_weather"},
        ]
    )
    res = _sanitize_breaking_tools(req)
    names = [t["name"] for t in res.tools]
    assert "advisor_code_review" not in names
    assert "web_search_tool" not in names
    assert "web_search_20250305" in names
    assert "get_weather" in names


def test_sanitize_resolves_named_tool_choice_mismatch_with_remaining_tools():
    req = MockMessageRequest(
        tools=[{"name": "computer_action"}, {"name": "fetch_url"}],
        tool_choice={"type": "tool", "name": "computer_action"},
    )
    res = _sanitize_breaking_tools(req)
    assert res.tool_choice is None


def test_sanitize_preserves_valid_named_tool_choice():
    req = MockMessageRequest(
        tools=[{"name": "web_search_2026"}, {"name": "calculator"}],
        tool_choice={"type": "tool", "name": "web_search_2026"},
    )
    res = _sanitize_breaking_tools(req)
    assert res.tool_choice == {"type": "tool", "name": "web_search_2026"}


def test_sanitize_preserves_legitimate_beta_web_search():
    req = MockMessageRequest(tools=[{"name": "web_search_20251201"}])
    res = _sanitize_breaking_tools(req)
    assert len(res.tools) == 1
    assert res.tools[0]["name"] == "web_search_20251201"


def test_sanitize_wipes_all_when_no_tools_remain():
    req = MockMessageRequest(
        tools=[{"name": "advisor_shell"}, {"name": "web_search"}],
        tool_choice={"type": "tool", "name": "web_search"},
    )
    res = _sanitize_breaking_tools(req)
    assert res.tools is None
    assert res.tool_choice is None
