import json
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


def test_model_refresh_updates_chat_bootstrap(
    page: Page,
    admin_base_url: str,
) -> None:
    bootstrap = page.request.get(f"{admin_base_url}/admin/api/chat/bootstrap").json()
    target = "open_router/vendor/model-b"
    stale_bootstrap = {
        **bootstrap,
        "models": [
            option for option in bootstrap["models"] if option["model_ref"] != target
        ],
    }
    refreshed = False

    def serve_bootstrap(route) -> None:
        payload = bootstrap if refreshed else stale_bootstrap
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(payload),
        )

    page.route("**/admin/api/chat/bootstrap", serve_bootstrap)
    page.goto(f"{admin_base_url}/admin")
    expect(page.locator("#messageArea")).to_have_text("")
    session = page.request.post(
        f"{admin_base_url}/admin/api/chat/sessions",
        data={},
    ).json()
    updated = page.request.patch(
        f"{admin_base_url}/admin/api/chat/sessions/{session['id']}",
        data={"expected_revision": session["revision"], "model": target},
    )
    assert updated.ok

    refreshed = True
    card = page.locator('[data-provider="open_router"]')
    card.get_by_role("button", name="Refresh models", exact=True).click()
    expect(card.locator(".provider-check-result")).to_have_text("3 models available")
    page.get_by_role("button", name="Chat Sessions").click()
    page.locator(".chat-session-card").click()

    expect(page.get_by_label("Selected model")).to_have_value(target)
    expect(page.get_by_label("Selected model").locator("option:checked")).to_have_text(
        "vendor/model-b"
    )
    page.get_by_role("textbox", name="Message", exact=True).fill("still available")
    expect(page.get_by_role("button", name="Send")).to_be_enabled()


def test_delayed_older_page_cannot_cross_into_another_chat(
    page: Page,
    admin_base_url: str,
) -> None:
    _new_chat(page, admin_base_url)
    message = page.get_by_role("textbox", name="Message", exact=True)
    message.fill("A OLD LEAK")
    page.get_by_role("button", name="Send").click()
    expect(page.locator(".assistant-message")).to_have_count(1)
    message.fill("A latest")
    page.get_by_role("button", name="Send").click()
    expect(message).to_be_enabled()
    title = page.get_by_label("Chat title")
    title.fill("[delay-older-page] Chat A")
    with page.expect_response(
        lambda response: (
            response.request.method == "PATCH"
            and "/admin/api/chat/sessions/" in response.url
        )
    ):
        title.press("Enter")

    page.get_by_role("button", name="Chats", exact=False).click()
    page.get_by_role("button", name="New chat", exact=True).click()
    message = page.get_by_role("textbox", name="Message", exact=True)
    message.fill("B only")
    page.get_by_role("button", name="Send").click()
    expect(page.get_by_text("B only", exact=True)).to_be_visible()

    page.get_by_role("button", name="Chats", exact=False).click()
    page.locator(".chat-session-card", has_text="[delay-older-page] Chat A").click()
    load_older = page.get_by_role("button", name="Load older messages")
    expect(load_older).to_be_visible()
    with page.expect_request(lambda request: "/turns?before=" in request.url):
        load_older.click()

    page.get_by_role("button", name="Chats", exact=False).click()
    page.locator(".chat-session-card", has_text="B only").click()
    expect(page.get_by_text("B only", exact=True)).to_be_visible()
    page.wait_for_timeout(1_000)
    expect(page.get_by_text("A OLD LEAK", exact=True)).to_have_count(0)


def test_out_of_order_library_search_keeps_latest_results(
    page: Page,
    admin_base_url: str,
) -> None:
    page.goto(f"{admin_base_url}/admin")
    for title in ("race-old result", "race-new result"):
        session = page.request.post(
            f"{admin_base_url}/admin/api/chat/sessions",
            data={},
        ).json()
        renamed = page.request.patch(
            f"{admin_base_url}/admin/api/chat/sessions/{session['id']}",
            data={"expected_revision": session["revision"], "title": title},
        )
        assert renamed.ok
    page.get_by_role("button", name="Chat Sessions").click()
    expect(page.locator(".chat-session-card")).to_have_count(2)

    search = page.get_by_role("searchbox", name="Search chats")
    with page.expect_request(lambda request: "query=race-old" in request.url):
        search.fill("race-old")
    with page.expect_response(lambda response: "query=race-new" in response.url):
        search.fill("race-new")

    expect(
        page.locator(".chat-session-card", has_text="race-new result")
    ).to_be_visible()
    page.wait_for_timeout(1_000)
    expect(
        page.locator(".chat-session-card", has_text="race-new result")
    ).to_be_visible()
    expect(
        page.locator(".chat-session-card", has_text="race-old result")
    ).to_have_count(0)


def test_double_load_more_appends_each_session_once(
    page: Page,
    admin_base_url: str,
) -> None:
    page.goto(f"{admin_base_url}/admin")
    for _index in range(26):
        created = page.request.post(
            f"{admin_base_url}/admin/api/chat/sessions",
            data={},
        )
        assert created.ok
    page.get_by_role("button", name="Chat Sessions").click()
    expect(page.locator(".chat-session-card")).to_have_count(25)
    more = page.get_by_role("button", name="Load more")
    expect(more).to_be_visible()

    more.evaluate("button => { button.click(); button.click(); }")

    expect(page.locator(".chat-session-card")).to_have_count(26, timeout=3_000)
    page.wait_for_timeout(750)
    expect(page.locator(".chat-session-card")).to_have_count(26)


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


def test_fragmented_stream_does_not_rebuild_transcript_per_delta(
    page: Page,
    admin_base_url: str,
) -> None:
    _new_chat(page, admin_base_url)
    page.evaluate(
        """
        () => {
          const original = Element.prototype.replaceChildren;
          window.__chatTranscriptRenderCount = 0;
          Element.prototype.replaceChildren = function (...children) {
            if (this.id === "chatTranscript") {
              window.__chatTranscriptRenderCount += 1;
            }
            return original.apply(this, children);
          };
        }
        """
    )

    page.get_by_role("textbox", name="Message", exact=True).fill("[fragmented]")
    page.get_by_role("button", name="Send").click()

    expect(page.get_by_role("button", name="Regenerate")).to_be_visible(timeout=10_000)
    expect(page.locator(".assistant-message .chat-markdown").last).to_contain_text(
        "abcd" * 20
    )
    render_count = page.evaluate("window.__chatTranscriptRenderCount")
    assert isinstance(render_count, int)
    assert render_count < 100


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


def test_committed_send_does_not_restore_draft_when_stream_ack_is_lost(
    page: Page,
    admin_base_url: str,
) -> None:
    _new_chat(page, admin_base_url)
    session_id = page.url.rsplit("/", 1)[-1]
    message = page.get_by_role("textbox", name="Message", exact=True)
    message.fill("[delay-send-ack] keep one draft")
    page.get_by_role("button", name="Send").click()

    turn_url = f"{admin_base_url}/admin/api/chat/sessions/{session_id}"
    for _attempt in range(50):
        detail = page.request.get(turn_url).json()
        if detail["turns"]:
            break
        page.wait_for_timeout(20)
    else:
        raise AssertionError("The delayed send did not commit its turn.")

    page.get_by_role("button", name="Stop").click()
    expect(page.get_by_role("button", name="Retry")).to_be_visible()

    expect(page.get_by_role("textbox", name="Message", exact=True)).to_have_value("")
    expect(
        page.get_by_text("[delay-send-ack] keep one draft", exact=True)
    ).to_be_visible()


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


def test_active_chat_deleted_in_another_tab_returns_to_library(
    page: Page,
    admin_base_url: str,
) -> None:
    _new_chat(page, admin_base_url)
    session_url = page.url
    page.get_by_role("textbox", name="Message", exact=True).fill("[slow] delete me")
    page.get_by_role("button", name="Send").click()
    expect(page.get_by_role("button", name="Stop")).to_be_visible()

    other = page.context.new_page()
    try:
        other.goto(session_url)
        expect(other.locator("#chatComposerStatus")).to_contain_text(
            "running in another tab"
        )
        other.on("dialog", lambda dialog: dialog.accept())
        other.get_by_role("button", name="Delete").click()
        expect(other).to_have_url(f"{admin_base_url}/admin/chat")

        expect(page).to_have_url(f"{admin_base_url}/admin/chat", timeout=3_000)
        expect(page.get_by_role("button", name="New chat", exact=True)).to_be_visible()
        expect(page.get_by_role("button", name="Stop")).to_have_count(0)
    finally:
        other.close()


def test_deleted_chat_cannot_render_from_an_inflight_detail_request(
    page: Page,
    admin_base_url: str,
) -> None:
    _new_chat(page, admin_base_url)
    session_url = page.url
    title = page.get_by_label("Chat title")
    title.fill("[delay-detail] delete while loading")
    with page.expect_response(
        lambda response: (
            response.request.method == "PATCH"
            and "/admin/api/chat/sessions/" in response.url
        )
    ):
        title.press("Enter")

    other = page.context.new_page()
    try:
        with other.expect_request(
            lambda request: (
                request.method == "GET" and "/admin/api/chat/sessions/" in request.url
            )
        ):
            other.goto(session_url)

        page.on("dialog", lambda dialog: dialog.accept())
        page.get_by_role("button", name="Delete").click()
        expect(page).to_have_url(f"{admin_base_url}/admin/chat")

        expect(other).to_have_url(f"{admin_base_url}/admin/chat", timeout=3_000)
        expect(other.get_by_role("button", name="New chat", exact=True)).to_be_visible()
        expect(other.get_by_role("textbox", name="Message", exact=True)).to_have_count(
            0
        )
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


def test_manual_compaction_failure_remains_visible(
    page: Page,
    admin_base_url: str,
) -> None:
    _new_chat(page, admin_base_url)
    message = page.get_by_role("textbox", name="Message", exact=True)
    message.fill("[fail-compaction] first")
    page.get_by_role("button", name="Send").click()
    expect(page.get_by_role("button", name="Regenerate")).to_be_visible()
    message.fill("second")
    page.get_by_role("button", name="Send").click()
    expect(page.get_by_role("button", name="Regenerate")).to_be_visible()

    page.get_by_role("button", name="Compact now").click()

    notice = page.locator("#chatNotice")
    expect(notice).to_be_visible()
    expect(notice).to_have_text("summary provider failed")


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
