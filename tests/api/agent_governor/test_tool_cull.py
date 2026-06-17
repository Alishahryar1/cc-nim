"""Tests for keyword-relevance tool culling."""

from __future__ import annotations

from api.models.anthropic import (
    ContentBlockText,
    Message,
    Tool,
)
from api.agent_governor.tool_cull import apply_cull, cull_tools


def _tool(name: str, description: str = "") -> Tool:
    return Tool(name=name, description=description)


def _user(text: str) -> Message:
    return Message(role="user", content=[ContentBlockText(type="text", text=text)])


def test_culling_a_short_list_returns_unchanged():
    tools = [_tool("read"), _tool("write")]
    msgs = [_user("hello")]
    result = cull_tools(tools, msgs, system="", max_keep=10)
    assert result.dropped_count == 0
    assert apply_cull(tools, result) == tools


def test_always_keep_pinned_even_when_irrelevant():
    tools = [
        _tool("read", "read a file"),
        _tool("write", "write a file"),
        _tool("bash", "run a shell command"),
        _tool("grep", "search files"),
        _tool("glob", "list files matching a pattern"),
        _tool("fancy_postgres_explainer", "analyse pg query plans"),
        _tool("kubernetes_audit", "audit a kubernetes cluster"),
        _tool("slack_post", "send a message to slack"),
        _tool("jira_create", "open a jira ticket"),
        _tool("salesforce_query", "query SFDC"),
        _tool("ad_creative_generator", "generate ads"),
        _tool("airtable_read", "read airtable"),
        _tool("notion_write", "write to notion"),
    ]
    msgs = [_user("can you tell me a joke")]
    result = cull_tools(tools, msgs, system="", max_keep=5)
    kept_names = [tools[i].name for i in result.kept_indices]
    # All ALWAYS_KEEP tools survive.
    for must_keep in ("read", "write", "bash", "grep", "glob"):
        assert must_keep in kept_names, f"{must_keep} should be pinned"


def test_culling_keeps_relevant_tools_first():
    tools = [
        _tool("read"),
        _tool("write"),
        _tool("bash"),
        _tool("grep"),
        _tool("glob"),
        _tool("postgres_query", "execute postgres SQL queries"),
        _tool("kubernetes_audit", "audit kubernetes cluster"),
        _tool("slack_post", "send slack messages"),
    ]
    msgs = [_user("can you check the postgres database tables for me")]
    result = cull_tools(tools, msgs, system="", max_keep=6)
    kept_names = [tools[i].name for i in result.kept_indices]
    # postgres_query scored highest non-pinned, must survive.
    assert "postgres_query" in kept_names
    # 5 always-keep + postgres_query = 6 kept; nothing else.
    assert len(kept_names) == 6
