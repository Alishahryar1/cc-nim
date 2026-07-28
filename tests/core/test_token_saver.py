from free_claude_code.core.token_saver import (
    TokenSaver,
    _fit_capped,
    _stringify,
)


def test_none_mode_returns_original():
    saver = TokenSaver("none")
    assert saver.save_result("hello " * 1000, tool_name="Bash") == "hello " * 1000


def test_medium_mode_truncates_large_output():
    saver = TokenSaver("medium")
    result = saver.save_result("x" * 20000, tool_name="Bash")
    assert len(result) < 20000
    assert "[token-saver]" in result


def test_never_worse_preserves_short_output():
    saver = TokenSaver("max")
    assert saver.save_result("short", tool_name="Read") == "short"


def test_write_content_is_elided():
    saver = TokenSaver("medium")
    result = saver.save_result("x" * 100000, tool_name="Write")
    assert result == "[Write status preserved — user content elided]"


def test_error_bang_guarded():
    saver = TokenSaver("medium")
    result = saver.save_result("!!! CRITICAL " + "x" * 1000, is_error=True)
    assert "!!!" in result
    assert result.startswith("!!!")


def test_fit_truncates_pre_serialized_string():
    saver = TokenSaver("medium")
    result = saver.fit("x" * 10000)
    assert len(result) < 10000
    assert "[token-saver]" in result


def test_fit_none_mode_is_identity():
    ts = TokenSaver("none")
    assert ts.fit("hello") == "hello"


def test_invalid_mode_defaults_to_medium():
    ts = TokenSaver("invalid")
    assert ts.mode == "medium"


def test_bash_strips_ansi_in_high_mode():
    ts = TokenSaver("high")
    result = ts.save_result("\x1b[31m" + "y" * 3000 + "\x1b[0m", tool_name="Bash")
    assert "\x1b" not in result


def test_never_worse_for_small_text():
    ts = TokenSaver("low")
    original = "small text"
    assert ts.fit(original) == original


def test_todo_compact_in_high_mode():
    ts = TokenSaver("high")
    content = "- [in_progress] Task A\n- [x] done\norphan line\n- [pending] Task B"
    result = ts.save_result(content, tool_name="TodoWrite")
    assert len(result) < len(content)
    assert "orphan" not in result


def test_stringify_handles_all_types():
    assert _stringify(None) == ""
    assert _stringify("hello") == "hello"
    assert "foo" in _stringify({"type": "text", "text": "foo"})
    assert _stringify(42) == "42"


def test_mode_caps_strictly_decreasing():
    low = TokenSaver("low")
    high = TokenSaver("high")
    max_saver = TokenSaver("max")

    low_r = low.fit("z" * 20000)
    high_r = high.fit("z" * 20000)
    max_r = max_saver.fit("z" * 20000)

    assert len(max_r) < len(high_r) <= len(low_r)


def test_set_mode_works():
    ts = TokenSaver("low")
    ts.set_mode("high")
    assert ts.mode == "high"


def test_singleton_reapplies_mode():
    a = TokenSaver.singleton("low")
    b = TokenSaver.singleton("high")
    assert a is b
    assert a.mode == "high"


def test_fit_capped_never_worse():
    result = _fit_capped("abc", cap=2)
    assert result == "abc"


def test_fit_capped_zero_cap():
    result = _fit_capped("abc", 0)
    assert result == "abc"


def test_fit_capped_exact_boundary():
    result = _fit_capped("abcdef", 6)
    assert result == "abcdef"


def test_fit_capped_small_cap_guard():
    result = _fit_capped("hello!", cap=1)
    assert result == "hello!"


def test_save_result_fallback_generic():
    saver = TokenSaver("medium")
    result = saver.save_result("x" * 20000)
    assert len(result) < 20000
    assert "[token-saver]" in result


def test_error_truncation_skip():
    saver = TokenSaver("medium")
    result = saver.save_result("ERROR: disk full " + "x" * 9000, is_error=True)
    assert result.startswith("ERROR:")


def test_bash_max_mode_uses_line_truncation():
    saver = TokenSaver("max")
    lines = "\n".join("line " + str(i) for i in range(5000))
    result = saver.save_result(lines, tool_name="Bash")
    assert "lines elided" in result
