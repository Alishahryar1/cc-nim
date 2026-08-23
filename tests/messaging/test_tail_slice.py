"""Cutting already-rendered markup must never produce markup a platform rejects."""

from free_claude_code.messaging.rendering.discord_markdown import discord_tail_slice
from free_claude_code.messaging.rendering.tail_slice import (
    safe_tail,
    standalone_cut_points,
)
from free_claude_code.messaging.rendering.telegram_markdown import (
    escape_md_v2,
    mdv2_tail_slice,
)
from free_claude_code.messaging.transcript.context import keep_whole_or_drop

MDV2 = ("*", "_", "~")


def _unescaped(text: str, character: str) -> bool:
    """Whether ``character`` appears outside a backslash escape pair."""
    index = 0
    while index < len(text):
        if text[index] == "\\":
            index += 2
            continue
        if text[index] == character:
            return True
        index += 1
    return False


def test_whole_text_returned_when_it_fits():
    assert safe_tail("abc", 10, MDV2) == "abc"
    assert safe_tail("abc", 3, MDV2) == "abc"


def test_non_positive_budget_returns_empty():
    assert safe_tail("abc", 0, MDV2) == ""
    assert safe_tail("abc", -1, MDV2) == ""


def test_tail_never_splits_a_backslash_escape_pair():
    rendered = escape_md_v2("step-1.done " * 40)
    for budget in range(4, 200):
        tail = mdv2_tail_slice(rendered, budget)
        assert len(tail) <= budget
        assert not tail.startswith("\\") or tail[:2] in {"\\\\", *_escape_pairs()}
        # A stranded escaped character would appear bare in the tail.
        for character in (".", "-"):
            assert not _unescaped(tail, character)


def _escape_pairs() -> set[str]:
    return {f"\\{character}" for character in "\\_*[]()~`>#+-=|{}.!"}


def test_tail_never_orphans_a_fenced_code_block():
    rendered = "intro\n```\n" + ("x" * 200) + "\n```\ntrailer"
    for budget in range(4, 260):
        tail = mdv2_tail_slice(rendered, budget)
        assert tail.count("```") % 2 == 0


def test_tail_never_leaves_an_inline_entity_open():
    rendered = "*bold* plain _italic_ ~strike~ " * 10
    for budget in range(4, 200):
        tail = mdv2_tail_slice(rendered, budget)
        for delimiter in MDV2:
            assert tail.count(delimiter) % 2 == 0


def test_tail_keeps_links_atomic():
    rendered = "before [label](https://e\\.com/a) after"
    for budget in range(4, len(rendered)):
        tail = mdv2_tail_slice(rendered, budget)
        assert "](" not in tail or tail.count("[") == tail.count("](")


def test_tail_prefers_the_longest_valid_suffix():
    rendered = escape_md_v2("abcdefghij" * 5)
    tail = mdv2_tail_slice(rendered, 20)
    assert len(tail) == 20
    assert rendered.endswith(tail)


def test_tail_returns_empty_when_no_cut_point_fits():
    # A single fenced block offers no interior cut point.
    rendered = "```\n" + ("x" * 100) + "\n```"
    assert mdv2_tail_slice(rendered, 20) == ""


def test_discord_treats_double_delimiters_as_one_marker():
    rendered = "**bold** plain __under__ " * 8
    for budget in range(4, 150):
        tail = discord_tail_slice(rendered, budget)
        assert tail.count("**") % 2 == 0
        assert tail.count("__") % 2 == 0


def test_discord_link_destination_may_contain_balanced_parens():
    """Discord escapes neither parenthesis, so link nesting must be matched."""
    rendered = "prefix [label](https://example.test/a_(b)) trailing content"

    for budget in range(4, len(rendered)):
        tail = discord_tail_slice(rendered, budget)
        # A ')' belonging to the omitted link wrapper must never lead the tail.
        assert not tail.startswith(")")
        assert tail.count("](") == tail.count("[")


def test_telegram_link_destination_may_contain_an_open_paren():
    """Telegram escapes ')' but not '(', so nesting must NOT be matched."""
    rendered = "pre [a](http://x\\.io/a_(b\\)) tail here"

    for budget in range(4, len(rendered)):
        tail = mdv2_tail_slice(rendered, budget)
        assert not tail.startswith(")")
        assert tail.count("](") == tail.count("[")


def test_cut_points_are_ascending_and_in_range():
    rendered = "*a* `b` [c](d) e"
    points = standalone_cut_points(rendered, MDV2)
    assert list(points) == sorted(points)
    assert all(0 <= point <= len(rendered) for point in points)


def test_unterminated_fence_offers_no_cut_point():
    # Already-malformed markup: every suffix would carry the unterminated
    # fence, so the segment must be dropped rather than sliced.
    rendered = "safe ```\nnever closed"
    assert standalone_cut_points(rendered, MDV2) == ()
    assert mdv2_tail_slice(rendered, 8) == ""


def test_default_context_slicer_refuses_partial_tails():
    assert keep_whole_or_drop("abcdef", 10) == "abcdef"
    assert keep_whole_or_drop("abcdef", 3) == ""
