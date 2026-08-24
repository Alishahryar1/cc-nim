"""Cut already-rendered platform markup without breaking it.

Transcript truncation drops whole segments first, then needs a suffix of one
oversized rendered segment. That segment is finished markup, not source text:
slicing it at an arbitrary offset can split a backslash escape pair, orphan the
opening fence of a code block, or leave an inline entity unclosed. Telegram then
rejects the send with ``can't parse entities``, and the adapter's retry falls
back to ``parse_mode=None`` -- delivering the update with every escape
backslash visible and all formatting lost.

A cut is only safe where the remainder stands alone: no escape pair straddles
it, and it is not inside a code span, fenced block, link, or open inline entity.
This module finds those points; platform modules supply their delimiter syntax.
"""


def standalone_cut_points(
    text: str,
    delimiters: tuple[str, ...],
    *,
    balanced_link_parens: bool = False,
) -> tuple[int, ...]:
    """Return ascending indexes where ``text[index:]`` is self-contained markup.

    ``delimiters`` are the paired inline markers for one platform, longest
    first, so that a two-character marker is not read as two one-character ones.

    ``balanced_link_parens`` selects how a link destination ends. Platforms that
    escape ``)`` inside destinations (Telegram) end at the first unescaped one;
    platforms that escape neither parenthesis (Discord) must match nesting, or a
    destination such as ``/a_(b)`` ends the link early and strands its ``)``.
    """
    points: list[int] = []
    open_inline: list[str] = []
    index = 0
    length = len(text)

    while index < length:
        if not open_inline:
            points.append(index)

        character = text[index]

        if character == "\\":
            # An escape pair is atomic; a cut between the two halves would
            # either strand the backslash or unescape a reserved character.
            index += 2
            continue

        if text.startswith("```", index):
            closing = text.find("```", index + 3)
            if closing == -1:
                # The markup is already malformed: every suffix would carry the
                # unterminated fence, so offer no cut point and let the caller
                # drop the segment instead.
                return ()
            index = closing + 3
            continue

        if character == "`":
            closing = _closing_code_span(text, index + 1)
            if closing is None:
                return ()
            index = closing + 1
            continue

        if character == "[":
            link_end = _link_end(text, index, balanced_parens=balanced_link_parens)
            if link_end is not None:
                index = link_end
                continue
            index += 1
            continue

        delimiter = _matched_delimiter(text, index, delimiters)
        if delimiter is not None:
            if open_inline and open_inline[-1] == delimiter:
                open_inline.pop()
            else:
                open_inline.append(delimiter)
            index += len(delimiter)
            continue

        index += 1

    if not open_inline:
        points.append(length)
    return tuple(points)


def safe_tail(
    text: str,
    max_chars: int,
    delimiters: tuple[str, ...],
    *,
    balanced_link_parens: bool = False,
) -> str:
    """Return the longest suffix of ``text`` within ``max_chars`` that is valid.

    Returns ``""`` when no cut point yields a short enough standalone suffix, so
    callers can fall back to dropping the segment instead of emitting markup the
    platform would reject.
    """
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text

    earliest = len(text) - max_chars
    for index in standalone_cut_points(
        text, delimiters, balanced_link_parens=balanced_link_parens
    ):
        if index >= earliest:
            return text[index:]
    return ""


def _matched_delimiter(
    text: str, index: int, delimiters: tuple[str, ...]
) -> str | None:
    for delimiter in delimiters:
        if text.startswith(delimiter, index):
            return delimiter
    return None


def _closing_code_span(text: str, start: int) -> int | None:
    index = start
    length = len(text)
    while index < length:
        if text[index] == "\\":
            index += 2
            continue
        if text[index] == "`":
            return index
        index += 1
    return None


def _link_end(text: str, start: int, *, balanced_parens: bool) -> int | None:
    """Return the index just past a ``[label](destination)`` run, if one starts here."""
    index = start + 1
    length = len(text)
    while index < length:
        if text[index] == "\\":
            index += 2
            continue
        if text[index] == "]":
            break
        index += 1
    else:
        return None
    if index + 1 >= length or text[index + 1] != "(":
        return None
    index += 2
    depth = 1
    while index < length:
        if text[index] == "\\":
            index += 2
            continue
        if balanced_parens and text[index] == "(":
            depth += 1
        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                return index + 1
        index += 1
    return None
