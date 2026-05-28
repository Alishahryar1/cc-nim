1.  **Refactor `text_content`, `thinking_content`, and `has_tool_use` in `core/anthropic/stream_contracts.py`**
    *   Currently, these functions blindly call `.get()` on the `.data` dict (often multiple times) for *every* event in a stream, creating new empty dictionaries `{}`, making unnecessary lookups, and checking properties even when the event type makes those properties impossible (e.g. looking for `delta` in a `content_block_start` event).
    *   By checking `event.event` first (which is a fast string comparison), we can skip dictionary lookups entirely for irrelevant events.
    *   For `text_content`: Only check `content_block` if `event.event == "content_block_start"`. Only check `delta` if `event.event == "content_block_delta"`.
    *   For `thinking_content`: Only check `delta` if `event.event == "content_block_delta"`.
    *   For `has_tool_use`: Only check `content_block` if `event.event == "content_block_start"`.

2.  **Add a note to `bolt.md`**
    *   Record the learning about avoiding dictionary lookups and empty dictionary creations on irrelevant SSE event types during high-frequency stream processing.

3.  **Run format, lint, type checks, and tests**
    *   Run `uv run ruff format`, `uv run ruff check`, `uv run ty check`, and `uv run pytest`.

4.  **Complete pre commit steps**
    *   Complete pre-commit steps to ensure proper testing, verification, review, and reflection are done.

5.  **Create PR**
    *   Title: `⚡ Bolt: Optimize SSE event parsing in stream contracts`
    *   Description with What, Why, Impact, and Measurement.
