import re

from playwright.sync_api import Page, expect


def _new_chat(page: Page, admin_base_url: str) -> None:
    page.goto(f"{admin_base_url}/admin")
    page.get_by_role("button", name="Chat Sessions").click()
    expect(page).to_have_url(f"{admin_base_url}/admin/chat")
    page.get_by_role("button", name="New chat").click()
    expect(page).to_have_url(re.compile(r"/admin/chat/[0-9a-f-]{36}$"))
    expect(page.get_by_role("textbox", name="Message", exact=True)).to_be_visible()


def test_chat_navigation_create_and_browser_history(
    page: Page,
    admin_base_url: str,
) -> None:
    _new_chat(page, admin_base_url)

    expect(page.locator(".action-bar")).to_be_hidden()
    page.get_by_role("button", name="Chats", exact=False).click()
    expect(page).to_have_url(f"{admin_base_url}/admin/chat")
    expect(page.locator(".chat-session-card")).to_have_count(1)
    page.go_back()
    expect(page.get_by_role("textbox", name="Message", exact=True)).to_be_visible()
    page.go_forward()
    expect(page.get_by_role("button", name="New chat", exact=True)).to_be_visible()


def test_chat_streams_thinking_and_persists_answer(
    page: Page,
    admin_base_url: str,
) -> None:
    _new_chat(page, admin_base_url)

    page.get_by_role("textbox", name="Message", exact=True).fill("hello")
    page.get_by_role("button", name="Send").click()
    expect(page.get_by_text("E2E answer")).to_be_visible()
    expect(page.locator(".chat-thinking summary", has_text="Thinking")).to_be_visible()
    expect(page.get_by_role("button", name="Regenerate")).to_be_visible()
    expect(page.locator("#chatContextMeter")).to_contain_text("Context:")

    page.reload()
    expect(page.get_by_text("E2E answer")).to_be_visible()
    expect(page.get_by_text("hello", exact=True)).to_be_visible()


def test_rejected_send_preserves_draft_after_stale_revision(
    page: Page,
    admin_base_url: str,
) -> None:
    _new_chat(page, admin_base_url)
    session_id = page.url.rsplit("/", 1)[-1]
    renamed = page.request.patch(
        f"{admin_base_url}/admin/api/chat/sessions/{session_id}",
        data={"expected_revision": 1, "title": "Changed elsewhere"},
    )
    assert renamed.ok

    message = page.get_by_role("textbox", name="Message", exact=True)
    message.fill("do not discard this draft")
    page.get_by_role("button", name="Send").click()

    expect(message).to_have_value("do not discard this draft")
    expect(page.locator("#chatNotice")).to_contain_text("changed in another tab")


def test_chat_stop_then_retry_uses_one_operation_owner(
    page: Page,
    admin_base_url: str,
) -> None:
    _new_chat(page, admin_base_url)

    page.get_by_role("textbox", name="Message", exact=True).fill("[slow] please answer")
    page.get_by_role("button", name="Send").click()
    expect(page.get_by_text("E2E answer")).to_be_visible()
    page.get_by_role("button", name="Stop").click()
    expect(page.get_by_role("button", name="Retry")).to_be_visible()

    page.get_by_role("button", name="Retry").click()
    expect(page.get_by_role("button", name="Regenerate")).to_be_visible()
    expect(page.get_by_text("E2E answer")).to_be_visible()


def test_chat_opened_in_another_tab_recovers_when_operation_stops(
    page: Page,
    admin_base_url: str,
) -> None:
    _new_chat(page, admin_base_url)
    page.get_by_role("textbox", name="Message", exact=True).fill("[slow] wait")
    page.get_by_role("button", name="Send").click()
    expect(page.get_by_role("button", name="Stop")).to_be_visible()

    other = page.context.new_page()
    try:
        other.goto(page.url)
        expect(other.locator("#chatComposerStatus")).to_contain_text(
            "running in another tab"
        )
        page.get_by_role("button", name="Stop").click()
        expect(page.get_by_role("button", name="Retry")).to_be_visible()
        expect(other.get_by_role("button", name="Retry")).to_be_visible(timeout=3_000)
    finally:
        other.close()


def test_regeneration_is_visible_and_recovers_in_another_tab(
    page: Page,
    admin_base_url: str,
) -> None:
    _new_chat(page, admin_base_url)
    page.get_by_role("textbox", name="Message", exact=True).fill(
        "[slow-regenerate] answer twice"
    )
    page.get_by_role("button", name="Send").click()
    expect(page.get_by_role("button", name="Regenerate")).to_be_visible()

    page.get_by_role("button", name="Regenerate").click()
    expect(page.get_by_role("button", name="Stop")).to_be_visible()
    other = page.context.new_page()
    try:
        other.goto(page.url)
        expect(other.locator("#chatComposerStatus")).to_contain_text(
            "running in another tab"
        )
        expect(other.get_by_label("Selected model")).to_be_disabled()

        page.get_by_role("button", name="Stop").click()

        expect(other.get_by_label("Selected model")).to_be_enabled(timeout=3_000)
        expect(other.get_by_role("button", name="Regenerate")).to_be_visible()
    finally:
        other.close()


def test_manual_compaction_is_visible_and_recovers_in_another_tab(
    page: Page,
    admin_base_url: str,
) -> None:
    _new_chat(page, admin_base_url)
    message = page.get_by_role("textbox", name="Message", exact=True)
    message.fill("[slow-compaction] first")
    page.get_by_role("button", name="Send").click()
    expect(page.get_by_role("button", name="Regenerate")).to_be_visible()
    message.fill("second")
    page.get_by_role("button", name="Send").click()
    expect(page.get_by_role("button", name="Regenerate")).to_be_visible()

    page.get_by_role("button", name="Compact now").click()
    expect(page.get_by_role("button", name="Stop")).to_be_visible()
    other = page.context.new_page()
    try:
        other.goto(page.url)
        expect(other.locator("#chatComposerStatus")).to_contain_text(
            "running in another tab"
        )
        expect(other.get_by_label("Thinking")).to_be_disabled()

        page.get_by_role("button", name="Stop").click()

        expect(other.get_by_label("Thinking")).to_be_enabled(timeout=3_000)
    finally:
        other.close()


def test_terminal_refresh_preserves_reader_scroll_position(
    page: Page,
    admin_base_url: str,
) -> None:
    _new_chat(page, admin_base_url)
    long_message = "[slow]\n" + "\n".join(
        f"line {index}: keep reading here" for index in range(100)
    )
    page.get_by_role("textbox", name="Message", exact=True).fill(long_message)
    page.get_by_role("button", name="Send").click()
    expect(page.get_by_role("button", name="Stop")).to_be_visible()
    scroller = page.locator("#chatTranscript")
    assert scroller.evaluate("node => node.scrollHeight > node.clientHeight")
    scroller.evaluate("node => { node.scrollTop = 0; }")

    page.get_by_role("button", name="Stop").click()

    expect(page.get_by_role("button", name="Retry")).to_be_visible()
    assert scroller.evaluate("node => node.scrollTop") < 10
    expect(page.get_by_role("button", name="Jump to latest")).to_be_visible()


def test_chat_rename_prompt_and_delete(
    page: Page,
    admin_base_url: str,
) -> None:
    _new_chat(page, admin_base_url)

    title = page.get_by_label("Chat title")
    title.fill("Release notes")
    title.press("Enter")
    expect(page.get_by_label("Chat title")).to_have_value("Release notes")

    page.get_by_role("button", name="System prompt").click()
    prompt = page.get_by_role("dialog").get_by_label("System prompt")
    prompt.fill("Be concise.")
    page.get_by_role("dialog").get_by_role("button", name="Save").click()
    page.get_by_role("button", name="System prompt").click()
    expect(page.get_by_role("dialog").get_by_label("System prompt")).to_have_value(
        "Be concise."
    )
    page.get_by_role("dialog").get_by_role("button", name="Reset to default").click()

    page.on("dialog", lambda dialog: dialog.accept())
    page.get_by_role("button", name="Delete").click()
    expect(page).to_have_url(f"{admin_base_url}/admin/chat")
    expect(page.locator(".chat-session-card")).to_have_count(0)


def test_delayed_title_save_cannot_restore_chat_after_back_navigation(
    page: Page,
    admin_base_url: str,
) -> None:
    _new_chat(page, admin_base_url)
    title = page.get_by_label("Chat title")
    title.fill("[delay-title-save] Release notes")
    with page.expect_request(
        lambda request: (
            request.method == "PATCH" and "/admin/api/chat/sessions/" in request.url
        )
    ):
        title.press("Enter")

    page.get_by_role("button", name="Chats", exact=False).click()
    expect(page).to_have_url(f"{admin_base_url}/admin/chat")
    expect(page.get_by_role("button", name="New chat", exact=True)).to_be_visible()
    page.wait_for_timeout(1_000)
    expect(page.get_by_role("button", name="New chat", exact=True)).to_be_visible()
    expect(page).to_have_url(f"{admin_base_url}/admin/chat")


def test_reset_system_prompt_refreshes_context_and_unblocks_send(
    page: Page,
    admin_base_url: str,
) -> None:
    _new_chat(page, admin_base_url)
    page.get_by_label("Selected model").select_option(
        "open_router/vendor/small-context"
    )
    message = page.get_by_role("textbox", name="Message", exact=True)
    message.fill("send after reset")

    page.get_by_role("button", name="System prompt").click()
    prompt = page.get_by_role("dialog").get_by_label("System prompt")
    prompt.fill("oversized prompt " * 5_000)
    page.get_by_role("dialog").get_by_role("button", name="Save").click()
    expect(page.get_by_role("button", name="Send")).to_be_disabled()
    expect(page.locator("#chatComposerStatus")).to_contain_text(
        "exceeds the model context"
    )
    message.fill("[delay-first-estimate] send after reset")
    page.wait_for_timeout(350)

    page.get_by_role("button", name="System prompt").click()
    page.get_by_role("dialog").get_by_role("button", name="Reset to default").click()

    expect(page.get_by_role("button", name="Send")).to_be_enabled(timeout=3_000)
    page.wait_for_timeout(800)
    expect(page.get_by_role("button", name="Send")).to_be_enabled()


def test_estimate_from_before_operation_cannot_overwrite_terminal_context(
    page: Page,
    admin_base_url: str,
) -> None:
    _new_chat(page, admin_base_url)
    message = page.get_by_role("textbox", name="Message", exact=True)
    message.fill("first")
    page.get_by_role("button", name="Send").click()
    expect(page.locator(".assistant-message")).to_have_count(1)

    message.fill("[delay-first-estimate] second")
    page.wait_for_timeout(350)
    page.get_by_role("button", name="Send").click()

    expect(page.locator(".assistant-message")).to_have_count(2)
    compact = page.get_by_role("button", name="Compact now")
    expect(compact).to_be_enabled(timeout=3_000)
    page.wait_for_timeout(800)
    expect(compact).to_be_enabled()


def test_chat_remains_usable_at_narrow_viewport(
    page: Page,
    admin_base_url: str,
) -> None:
    page.set_viewport_size({"width": 390, "height": 844})
    _new_chat(page, admin_base_url)

    expect(page.get_by_label("Selected model")).to_be_visible()
    expect(page.get_by_label("Thinking")).to_be_visible()
    expect(page.get_by_role("textbox", name="Message", exact=True)).to_be_in_viewport()
