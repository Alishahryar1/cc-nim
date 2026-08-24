"""Rendering context used by transcript segments."""

from collections.abc import Callable
from dataclasses import dataclass


def keep_whole_or_drop(text: str, max_chars: int) -> str:
    """Conservative default tail slicer: never cut inside rendered markup.

    Only the platform renderer knows where its markup can be split safely, so a
    context without a platform slicer refuses partial tails rather than risk
    emitting markup the platform would reject.
    """
    return text if len(text) <= max_chars else ""


@dataclass
class RenderCtx:
    bold: Callable[[str], str]
    code_inline: Callable[[str], str]
    escape_code: Callable[[str], str]
    escape_text: Callable[[str], str]
    render_markdown: Callable[[str], str]
    # Returns a suffix of already-rendered markup that fits and stands alone.
    tail_slice: Callable[[str, int], str] = keep_whole_or_drop

    thinking_tail_max: int | None = 1000
    tool_input_tail_max: int | None = 1200
    tool_output_tail_max: int | None = 1600
    text_tail_max: int | None = 2000
