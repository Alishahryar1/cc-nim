"""build_claude_proxy_env must always win over inherited Anthropic env vars,
including on Windows where environment variable names are case-insensitive.
"""

from free_claude_code.cli.claude_env import build_claude_proxy_env


def test_proxy_env_overrides_matching_case_anthropic_vars() -> None:
    base_env = {
        "ANTHROPIC_BASE_URL": "https://stale.example.com",
        "ANTHROPIC_API_KEY": "sk-stale",
        "PATH": "/usr/bin",
    }

    env = build_claude_proxy_env(
        proxy_root_url="http://127.0.0.1:8082",
        auth_token="fcc-token",
        base_env=base_env,
    )

    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8082"
    assert env["ANTHROPIC_AUTH_TOKEN"] == "fcc-token"
    assert "ANTHROPIC_API_KEY" not in env
    assert env["PATH"] == "/usr/bin"


def test_proxy_env_overrides_differently_cased_anthropic_vars() -> None:
    """Regression test for #1043: on Windows, a leftover `Anthropic_Base_Url`
    (e.g. set via the System Properties GUI, or by another installer) is a
    distinct key from `ANTHROPIC_BASE_URL` in this Python dict, but the two
    are the same variable to the Windows OS and to a case-insensitive child
    runtime - so it must be filtered out here rather than left to coexist
    with the value FCC is trying to set.
    """
    base_env = {
        "Anthropic_Base_Url": "https://stale.example.com",
        "anthropic_api_key": "sk-stale",
        "Claude_Code_Disable_Nonessential_Traffic": "1",
    }

    env = build_claude_proxy_env(
        proxy_root_url="http://127.0.0.1:8082",
        auth_token="fcc-token",
        base_env=base_env,
    )

    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8082"
    assert not any(
        key.casefold() == "anthropic_base_url" and key != "ANTHROPIC_BASE_URL"
        for key in env
    )
    assert not any(key.casefold() == "anthropic_api_key" for key in env)
    assert not any(
        key.casefold() == "claude_code_disable_nonessential_traffic" for key in env
    )


def test_proxy_env_sets_expected_claude_code_flags() -> None:
    env = build_claude_proxy_env(
        proxy_root_url="http://127.0.0.1:8082",
        auth_token="fcc-token",
        base_env={},
    )

    assert env["CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY"] == "1"
    assert env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] == "190000"
    assert env["DISABLE_AUTOUPDATER"] == "1"
    assert env["DISABLE_FEEDBACK_COMMAND"] == "1"
    assert env["DISABLE_ERROR_REPORTING"] == "1"
