import json
import re

from playwright.sync_api import Page, Request, expect


def _new_chat(page: Page, admin_base_url: str) -> None:
    page.goto(f"{admin_base_url}/admin")
    page.get_by_role("button", name="Chat Sessions").click()
    expect(page).to_have_url(f"{admin_base_url}/admin/chat")
    page.get_by_role("button", name="New chat").click()
    expect(page).to_have_url(re.compile(r"/admin/chat/[0-9a-f-]{36}$"))
    expect(page.get_by_role("textbox", name="Message", exact=True)).to_be_visible()


def _hold_next_chat_operation(page: Page, action: str) -> None:
    page.evaluate(
        """
        action => {
          const originalFetch = window.fetch.bind(window);
          window.fetch = (...args) => {
            if (!String(args[0]).endsWith(`/${action}`)) {
              return originalFetch(...args);
            }
            window.fetch = originalFetch;
            return new Promise((resolve, reject) => {
              window.__releaseHeldChatRequest = () => {
                originalFetch(...args).then(resolve, reject);
              };
            });
          };
        }
        """,
        action,
    )


def test_stale_chat_selection_reloads_as_default(page: Page, admin_base_url: str):
    session = page.request.post(
        f"{admin_base_url}/admin/api/chat/sessions", data={}
    ).json()
    response = page.request.patch(
        f"{admin_base_url}/admin/api/chat/sessions/{session['id']}",
        data={
            "expected_revision": session["revision"],
            "model": "github_models/openai/old",
        },
    )
    assert response.ok
    page.goto(f"{admin_base_url}/admin/chat/{session['id']}")
    expect(page.get_by_role("combobox", name="Selected model")).to_have_value(
        "open_router/e2e-default"
    )
    expect(page.get_by_role("textbox", name="Message", exact=True)).to_be_visible()


def _select_model(page: Page, model_ref: str) -> None:
    model = page.get_by_role("combobox", name="Selected model")
    model.click()
    model.fill(model_ref)
    expect(page.get_by_role("option", name=model_ref, exact=True)).to_be_visible()
    with page.expect_response(
        lambda response: (
            response.request.method == "PATCH"
            and "/admin/api/chat/sessions/" in response.url
        )
    ):
        model.press("Enter")
    expect(page.get_by_role("combobox", name="Selected model")).to_have_value(model_ref)


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


def test_chat_routes_render_while_provider_health_is_still_loading(
    page: Page,
    admin_base_url: str,
) -> None:
    session = page.request.post(
        f"{admin_base_url}/admin/api/chat/sessions",
        data={},
    ).json()
    routes = (
        ("/admin/chat", page.get_by_role("button", name="New chat", exact=True)),
        (
            f"/admin/chat/{session['id']}",
            page.get_by_role("textbox", name="Message", exact=True),
        ),
    )
    page.add_init_script(
        """
        const originalFetch = window.fetch.bind(window);
        window.__localProviderStatusRequested = false;
        window.fetch = (...args) => {
          const input = args[0];
          const value = typeof input === "string" ? input : input.url;
          const url = new URL(value, window.location.href);
          if (url.pathname === "/admin/api/providers/local-status") {
            window.__localProviderStatusRequested = true;
            return new Promise(() => {});
          }
          return originalFetch(...args);
        };
        """
    )

    for path, ready in routes:
        page.goto(f"{admin_base_url}{path}")
        expect(ready).to_be_visible()
        assert page.evaluate("window.__localProviderStatusRequested === true")


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

    expect(page.get_by_role("combobox", name="Selected model")).to_have_value(target)
    page.get_by_role("textbox", name="Message", exact=True).fill("still available")
    expect(page.get_by_role("button", name="Send")).to_be_enabled()


def test_chat_model_picker_searches_and_selects_in_one_control(
    page: Page,
    admin_base_url: str,
) -> None:
    _new_chat(page, admin_base_url)
    model_patches: list[str] = []
    page.on(
        "request",
        lambda request: (
            model_patches.append(request.url)
            if request.method == "PATCH" and "/admin/api/chat/sessions/" in request.url
            else None
        ),
    )

    model = page.get_by_role("combobox", name="Selected model")
    expect(model).to_have_value("open_router/e2e-default")
    expect(page.get_by_role("searchbox", name="Filter models")).to_have_count(0)
    expect(page.locator("select#chatModel")).to_have_count(0)
    expect(page.locator("#chatNotice")).to_be_hidden()

    page.reload()
    expect(model).to_have_value("open_router/e2e-default")
    model.click()
    options = page.get_by_role("listbox").get_by_role("option")
    expect(options).to_have_count(1)
    expect(options).to_have_text("open_router/e2e-default")
    model.press("Escape")

    model.fill("not-a-catalog-model")
    expect(page.get_by_text("No matching models.", exact=True)).to_be_visible()
    page.get_by_label("Thinking").click()
    expect(model).to_have_value("open_router/e2e-default")

    _select_model(page, "open_router/vendor/small-context")
    expect(page.get_by_role("listbox")).to_be_hidden()
    page.wait_for_timeout(100)
    assert len(model_patches) == 1


def test_chat_context_meter_shows_used_over_advertised_context_window(
    page: Page,
    admin_base_url: str,
) -> None:
    _new_chat(page, admin_base_url)
    _select_model(page, "open_router/vendor/model-b")
    with page.expect_response(
        lambda response: (
            response.request.method == "PATCH"
            and "/admin/api/chat/sessions/" in response.url
        )
    ):
        page.get_by_label("Thinking").select_option("high")
    page.get_by_role("textbox", name="Message", exact=True).fill("hello")

    meter = page.locator("#chatContextMeter")
    expect(meter).to_have_text(re.compile(r"^Context: \d+% · \d+ / 100K$"))


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
    expect(page.locator(".assistant-message:not(.live-message)")).to_have_count(2)
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


def test_generation_status_occupies_the_answer_slot(
    page: Page,
    admin_base_url: str,
) -> None:
    _new_chat(page, admin_base_url)
    message = page.get_by_role("textbox", name="Message", exact=True)

    _hold_next_chat_operation(page, "send")
    message.fill("answer in place")
    page.get_by_role("button", name="Send").click()

    assistant = page.locator(".assistant-message")
    expect(assistant).to_have_count(1)
    expect(assistant.get_by_text("Thinking…", exact=True)).to_be_visible()
    expect(page.locator("#chatComposerStatus")).to_be_empty()

    page.evaluate("() => { window.__releaseHeldChatRequest(); }")
    expect(page.get_by_role("button", name="Regenerate")).to_be_visible()

    _hold_next_chat_operation(page, "regenerate")
    page.get_by_role("button", name="Regenerate").click()

    expect(assistant).to_have_count(1)
    expect(assistant.get_by_text("Thinking…", exact=True)).to_be_visible()
    expect(assistant).not_to_contain_text("E2E answer")
    expect(page.locator("#chatComposerStatus")).to_be_empty()

    page.evaluate("() => { window.__releaseHeldChatRequest(); }")
    expect(page.get_by_role("button", name="Regenerate")).to_be_visible()
    expect(assistant).to_contain_text("E2E answer")


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
    _hold_next_chat_operation(page, "send")
    message = page.get_by_role("textbox", name="Message", exact=True)
    message.fill("do not discard this draft")
    page.get_by_role("button", name="Send").click()

    renamed = page.request.patch(
        f"{admin_base_url}/admin/api/chat/sessions/{session_id}",
        data={"expected_revision": 1, "title": "Changed elsewhere"},
    )
    assert renamed.ok
    page.evaluate("() => { window.__releaseHeldChatRequest(); }")

    expect(message).to_have_value("do not discard this draft")
    expect(page.locator("#chatNotice")).to_contain_text("changed in another tab")


def test_provider_failure_appears_only_in_assistant_reply(
    page: Page,
    admin_base_url: str,
) -> None:
    _new_chat(page, admin_base_url)
    page.get_by_role("textbox", name="Message", exact=True).fill("[fail-turn]")
    page.get_by_role("button", name="Send").click()

    expect(page.get_by_role("button", name="Retry")).to_be_visible()
    expect(
        page.locator(".assistant-message .chat-generation-status.failed")
    ).to_have_text("E2E provider failed")
    expect(page.locator("#chatNotice")).to_be_hidden()
    expect(page.get_by_text("E2E provider failed", exact=True)).to_have_count(1)


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


def test_send_keeps_composer_ready_for_the_next_draft(
    page: Page,
    admin_base_url: str,
) -> None:
    _new_chat(page, admin_base_url)
    message = page.get_by_role("textbox", name="Message", exact=True)

    message.fill("first")
    message.press("Enter")
    expect(page.get_by_role("button", name="Regenerate")).to_be_visible()
    expect(message).to_be_focused()

    message.fill("[slow] second")
    message.press("Enter")
    expect(page.locator(".live-message")).to_contain_text("E2E answer")
    expect(message).to_be_enabled()
    expect(message).to_be_focused()
    message.fill("next draft")

    page.get_by_role("button", name="Stop").click()
    expect(page.get_by_role("button", name="Retry")).to_be_visible()
    expect(message).to_have_value("next draft")


def test_chat_stop_then_retry_uses_one_operation_owner(
    page: Page,
    admin_base_url: str,
) -> None:
    _new_chat(page, admin_base_url)

    page.get_by_role("textbox", name="Message", exact=True).fill("[slow] please answer")
    page.get_by_role("button", name="Send").click()
    expect(page.get_by_text("E2E answer")).to_be_visible()
    stop = page.get_by_role("button", name="Stop")
    expect(stop).to_be_visible()
    send_is_hidden = page.locator("#chatSend").is_hidden()
    stop.click()
    expect(page.get_by_role("button", name="Retry")).to_be_visible()
    assert send_is_hidden

    page.get_by_role("button", name="Retry").click()
    expect(page.get_by_role("button", name="Regenerate")).to_be_visible()
    expect(page.get_by_text("E2E answer")).to_be_visible()


def test_chat_opened_in_another_tab_shares_live_operation_and_stop(
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
        expect(other.get_by_text("E2E answer", exact=True)).to_be_visible()
        expect(other.get_by_role("button", name="Stop")).to_be_visible()
        other.get_by_role("button", name="Stop").click()
        expect(page.get_by_role("button", name="Retry")).to_be_visible()
        expect(other.get_by_role("button", name="Retry")).to_be_visible(timeout=3_000)
    finally:
        other.close()


def test_refresh_keeps_one_live_operation_and_reconstructs_its_answer(
    page: Page,
    admin_base_url: str,
) -> None:
    _new_chat(page, admin_base_url)
    message = page.get_by_role("textbox", name="Message", exact=True)
    message.fill("[slow] survive refresh")
    message.press("Enter")
    expect(page.get_by_text("E2E answer", exact=True)).to_be_visible()
    expect(page.get_by_role("button", name="Stop")).to_be_visible()

    page.reload()

    expect(page.get_by_text("[slow] survive refresh", exact=True)).to_have_count(1)
    expect(page.locator(".assistant-message")).to_have_count(1)
    expect(page.get_by_text("E2E answer", exact=True)).to_be_visible()
    page.get_by_role("button", name="Stop").click()
    expect(page.get_by_role("button", name="Retry")).to_be_visible()


def test_terminal_event_cannot_be_undone_by_an_older_detail_response(
    page: Page,
    admin_base_url: str,
) -> None:
    _new_chat(page, admin_base_url)
    title = page.get_by_label("Chat title")
    title.fill("[delay-detail] stale operation snapshot")
    with page.expect_response(lambda response: response.request.method == "PATCH"):
        title.press("Enter")
    page.get_by_role("textbox", name="Message", exact=True).fill("[slow] finish once")
    page.get_by_role("button", name="Send").click()
    expect(page.get_by_role("button", name="Stop")).to_be_visible()

    other = page.context.new_page()
    try:
        with other.expect_request(
            lambda request: (
                request.method == "GET"
                and request.url.endswith(page.url.rsplit("/", 1)[-1])
            )
        ):
            other.goto(page.url, wait_until="commit")
        page.wait_for_timeout(150)
        page.get_by_role("button", name="Stop").click()
        expect(page.get_by_role("button", name="Retry")).to_be_visible()

        expect(other.get_by_role("button", name="Retry")).to_be_visible(timeout=3_000)
        expect(other.get_by_role("button", name="Stop")).to_have_count(0)
        expect(other.locator("#chatComposerStatus")).not_to_have_text("Thinking…")
    finally:
        other.close()


def test_two_sessions_run_concurrently_across_navigation(
    page: Page,
    admin_base_url: str,
) -> None:
    _new_chat(page, admin_base_url)
    first_url = page.url
    message = page.get_by_role("textbox", name="Message", exact=True)
    message.fill("[slow] first active chat")
    message.press("Enter")
    expect(page.get_by_role("button", name="Stop")).to_be_visible()

    page.get_by_role("button", name="Chats", exact=False).click()
    page.get_by_role("button", name="New chat", exact=True).click()
    expect(page).to_have_url(re.compile(r"/admin/chat/[0-9a-f-]{36}$"))
    second_url = page.url
    message = page.get_by_role("textbox", name="Message", exact=True)
    message.fill("[slow] second active chat")
    message.press("Enter")
    expect(page.get_by_role("button", name="Stop")).to_be_visible()

    page.get_by_role("button", name="Chats", exact=False).click()
    expect(page.locator(".chat-session-status", has_text="Thinking…")).to_have_count(2)

    page.goto(first_url)
    expect(page.get_by_text("E2E answer", exact=True)).to_be_visible()
    page.get_by_role("button", name="Stop").click()
    expect(page.get_by_role("button", name="Retry")).to_be_visible()

    page.goto(second_url)
    expect(page.get_by_text("E2E answer", exact=True)).to_be_visible()
    page.get_by_role("button", name="Stop").click()
    expect(page.get_by_role("button", name="Retry")).to_be_visible()


def test_library_card_tracks_background_operation_and_durable_summary(
    page: Page,
    admin_base_url: str,
) -> None:
    _new_chat(page, admin_base_url)
    title = page.get_by_label("Chat title")
    title.fill("Library observer")
    with page.expect_response(lambda response: response.request.method == "PATCH"):
        title.press("Enter")
    other = page.context.new_page()
    try:
        other.goto(f"{admin_base_url}/admin/chat")
        message = page.get_by_role("textbox", name="Message", exact=True)
        message.fill("[slow] live library preview")
        message.press("Enter")

        card = other.locator(".chat-session-card", has_text="Library observer")
        expect(card).to_be_visible(timeout=3_000)
        expect(card.locator("p")).to_have_text("[slow] live library preview")
        expect(card.locator(".chat-session-status")).to_have_text("Thinking…")

        page.get_by_role("button", name="Stop").click()
        expect(page.get_by_role("button", name="Retry")).to_be_visible()
        expect(card.locator(".chat-session-status")).to_have_count(0, timeout=3_000)
        expect(card.locator("p")).to_have_text("[slow] live library preview")
        expect(card.locator("span").first).to_contain_text("just now")
    finally:
        other.close()


def test_chat_operation_continues_while_providers_view_is_open(
    page: Page,
    admin_base_url: str,
) -> None:
    _new_chat(page, admin_base_url)
    session_url = page.url
    message = page.get_by_role("textbox", name="Message", exact=True)
    message.fill("[slow] keep running outside chat")
    message.press("Enter")
    expect(page.get_by_text("E2E answer", exact=True)).to_be_visible()

    page.get_by_role("button", name="Providers", exact=True).click()
    expect(page.locator("#pageTitle")).to_have_text("Providers")
    page.goto(session_url)

    expect(
        page.get_by_text("[slow] keep running outside chat", exact=True)
    ).to_have_count(1)
    expect(page.get_by_text("E2E answer", exact=True)).to_be_visible()
    expect(page.get_by_role("button", name="Stop")).to_be_visible()
    page.get_by_role("button", name="Stop").click()
    expect(page.get_by_role("button", name="Retry")).to_be_visible()


def test_event_feed_reconnect_preserves_transcript_and_draft(
    page: Page,
    admin_base_url: str,
) -> None:
    _new_chat(page, admin_base_url)
    message = page.get_by_role("textbox", name="Message", exact=True)
    message.fill("before reconnect")
    message.press("Enter")
    expect(page.get_by_role("button", name="Regenerate")).to_be_visible()
    message.fill("draft during reconnect")

    page.context.set_offline(True)
    expect(page.locator("#chatComposerStatus")).to_have_text("Reconnecting…")
    expect(page.get_by_text("E2E answer", exact=True)).to_be_visible()
    expect(message).to_have_value("draft during reconnect")
    expect(page.get_by_role("button", name="Send")).to_be_disabled()

    page.context.set_offline(False)
    expect(page.locator("#chatComposerStatus")).to_be_empty(timeout=10_000)
    expect(message).to_have_value("draft during reconnect")
    expect(page.get_by_role("button", name="Send")).to_be_enabled()
    expect(page.get_by_text("E2E answer", exact=True)).to_have_count(1)


def test_drafts_are_session_scoped_and_private_to_each_tab(
    page: Page,
    admin_base_url: str,
) -> None:
    _new_chat(page, admin_base_url)
    first_url = page.url
    message = page.get_by_role("textbox", name="Message", exact=True)
    message.fill("first private draft")
    page.reload()
    expect(message).to_have_value("first private draft")

    other = page.context.new_page()
    try:
        other.goto(first_url)
        expect(other.get_by_role("textbox", name="Message", exact=True)).to_be_empty()
    finally:
        other.close()

    page.get_by_role("button", name="Chats", exact=False).click()
    page.get_by_role("button", name="New chat", exact=True).click()
    page.get_by_role("textbox", name="Message", exact=True).fill("second private draft")
    page.goto(first_url)
    expect(page.get_by_role("textbox", name="Message", exact=True)).to_have_value(
        "first private draft"
    )


def test_session_settings_sync_without_disturbing_another_tab_draft(
    page: Page,
    admin_base_url: str,
) -> None:
    _new_chat(page, admin_base_url)
    other = page.context.new_page()
    try:
        other.goto(page.url)
        other_patches: list[str] = []

        def remember_other_patch(request: Request) -> None:
            if request.method == "PATCH":
                other_patches.append(request.post_data or "")

        other.on("request", remember_other_patch)
        other_message = other.get_by_role("textbox", name="Message", exact=True)
        other_message.fill("private draft")
        other_title = other.get_by_label("Chat title")
        other_title.fill("Local title draft")

        title = page.get_by_label("Chat title")
        title.fill("Shared title")
        with page.expect_response(lambda response: response.request.method == "PATCH"):
            title.press("Enter")
        expect(other_title).to_have_value("Local title draft")
        other.wait_for_timeout(50)
        assert other_patches == []
        with other.expect_response(lambda response: response.request.method == "PATCH"):
            other_title.press("Enter")
        assert len(other_patches) == 1
        expect(title).to_have_value("Local title draft")

        _select_model(page, "open_router/vendor/model-b")
        expect(other.get_by_label("Selected model")).to_have_value(
            "open_router/vendor/model-b"
        )
        with page.expect_response(lambda response: response.request.method == "PATCH"):
            page.get_by_label("Thinking").select_option("high")
        expect(other.get_by_label("Thinking")).to_have_value("high")
        expect(other_message).to_have_value("private draft")
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
        expect(other.get_by_text("E2E answer", exact=True)).to_be_visible()
        expect(other.get_by_role("button", name="Stop")).to_be_visible()
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
        expect(other.get_by_text("E2E answer", exact=True)).to_be_visible()
        expect(other.get_by_role("button", name="Stop")).to_be_visible()
        expect(other.get_by_label("Selected model")).to_be_disabled()

        other.get_by_role("button", name="Stop").click()

        expect(other.get_by_label("Selected model")).to_be_enabled(timeout=3_000)
        expect(other.get_by_role("button", name="Regenerate")).to_be_visible()
    finally:
        other.close()


def test_failed_regeneration_replaces_the_reply_without_a_page_notice(
    page: Page,
    admin_base_url: str,
) -> None:
    _new_chat(page, admin_base_url)
    page.get_by_role("textbox", name="Message", exact=True).fill("[fail-regenerate]")
    page.get_by_role("button", name="Send").click()
    expect(page.get_by_text("E2E answer", exact=True)).to_be_visible()

    page.get_by_role("button", name="Regenerate").click()

    expect(page.get_by_role("button", name="Retry")).to_be_visible()
    expect(page.get_by_text("E2E answer", exact=True)).to_have_count(0)
    expect(page.get_by_text("Partial replacement", exact=True)).to_be_visible()
    expect(
        page.locator(".assistant-message .chat-generation-status.failed")
    ).to_have_text("E2E provider failed")
    expect(page.locator("#chatNotice")).to_be_hidden()
    expect(page.get_by_text("E2E provider failed", exact=True)).to_have_count(1)


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
        expect(other.locator("#chatComposerStatus")).to_have_text("Compacting…")
        expect(other.get_by_role("button", name="Stop")).to_be_visible()
        expect(other.get_by_label("Thinking")).to_be_disabled()

        other.get_by_role("button", name="Stop").click()

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


def test_long_transcript_keeps_composer_visible_and_preserves_reader_scroll_position(
    page: Page,
    admin_base_url: str,
) -> None:
    _new_chat(page, admin_base_url)
    long_message = "[slow]\n" + "\n".join(
        f"line {index}: keep reading here" for index in range(100)
    )
    message = page.get_by_role("textbox", name="Message", exact=True)
    message.fill(long_message)
    page.get_by_role("button", name="Send").click()
    expect(page.get_by_role("button", name="Stop")).to_be_visible()
    scroller = page.locator("#chatTranscript")
    page.wait_for_function(
        """() => {
          const node = document.getElementById('chatTranscript');
          return node && node.scrollHeight > node.clientHeight;
        }"""
    )
    composer_is_fully_visible = message.evaluate(
        "node => { const box = node.getBoundingClientRect(); "
        "return box.top >= 0 && box.bottom <= window.innerHeight; }"
    )
    scroller.evaluate("node => { node.scrollTop = 0; }")

    page.get_by_role("button", name="Stop").click()

    expect(page.get_by_role("button", name="Retry")).to_be_visible()
    assert composer_is_fully_visible
    assert scroller.evaluate("node => node.scrollTop") < 10


def test_chat_composer_is_one_compact_surface_and_grows_to_six_lines(
    page: Page,
    admin_base_url: str,
) -> None:
    page.set_viewport_size({"width": 1_258, "height": 566})
    _new_chat(page, admin_base_url)
    composer = page.locator(".chat-composer")
    message = page.get_by_role("textbox", name="Message", exact=True)

    surface = composer.evaluate(
        """node => {
            const box = node.getBoundingClientRect();
            const textarea = node.querySelector("textarea").getBoundingClientRect();
            const send = node.querySelector("#chatSend").getBoundingClientRect();
            const surfaceStyle = getComputedStyle(node);
            const textareaStyle = getComputedStyle(node.querySelector("textarea"));
            return {
                contained:
                    textarea.left >= box.left && textarea.right <= box.right &&
                    textarea.top >= box.top && textarea.bottom <= box.bottom &&
                    send.left >= box.left && send.right <= box.right &&
                    send.top >= box.top && send.bottom <= box.bottom,
                surfaceBorder: surfaceStyle.borderLeftWidth,
                textareaBorder: textareaStyle.borderLeftWidth,
                bottomGap: window.innerHeight - box.bottom,
            };
        }"""
    )
    assert surface == {
        "contained": True,
        "surfaceBorder": "1px",
        "textareaBorder": "0px",
        "bottomGap": 16,
    }

    two_line_height = message.evaluate("node => node.getBoundingClientRect().height")
    message.fill("one\ntwo\nthree\nfour")
    four_line_height = message.evaluate("node => node.getBoundingClientRect().height")
    message.fill("\n".join(f"line {index}" for index in range(6)))
    six_line_height = message.evaluate("node => node.getBoundingClientRect().height")
    six_line_overflow = message.evaluate("node => getComputedStyle(node).overflowY")
    message.fill("\n".join(f"line {index}" for index in range(8)))
    capped_height = message.evaluate("node => node.getBoundingClientRect().height")
    overflow = message.evaluate("node => getComputedStyle(node).overflowY")
    message.fill("short again")
    reset_height = message.evaluate("node => node.getBoundingClientRect().height")

    assert four_line_height > two_line_height
    assert six_line_height > four_line_height
    assert six_line_overflow == "hidden"
    assert capped_height == six_line_height
    assert overflow == "auto"
    assert reset_height == two_line_height


def test_chat_uses_desktop_width_with_one_responsive_gutter(
    page: Page,
    admin_base_url: str,
) -> None:
    page.set_viewport_size({"width": 1_600, "height": 900})
    _new_chat(page, admin_base_url)
    page.get_by_role("textbox", name="Message", exact=True).fill("hello")
    page.get_by_role("button", name="Send").click()
    expect(page.get_by_text("E2E answer")).to_be_visible()
    page.reload()
    expect(page.get_by_text("E2E answer")).to_be_visible()

    layout = page.locator(".chat-session-shell").evaluate(
        """shell => {
            const main = document.querySelector(".main").getBoundingClientRect();
            const shellBox = shell.getBoundingClientRect();
            const composer = shell.querySelector(".chat-composer").getBoundingClientRect();
            const transcript = shell.querySelector(".chat-transcript").getBoundingClientRect();
            const user = shell.querySelector(".user-message").getBoundingClientRect();
            const assistant = shell.querySelector(".assistant-message").getBoundingClientRect();
            return {
                shellShare: shellBox.width / main.width,
                composerShare: composer.width / main.width,
                leftGutter: composer.left - shellBox.left,
                rightGutter: shellBox.right - composer.right,
                userRightGap: transcript.right - user.right,
                assistantLeftGap: assistant.left - transcript.left,
            };
        }"""
    )

    assert layout["shellShare"] > 0.9
    assert layout["composerShare"] > 0.9
    assert layout["leftGutter"] < 0.5
    assert layout["rightGutter"] < 0.5
    assert abs(layout["leftGutter"] - layout["rightGutter"]) < 0.5
    assert abs(layout["userRightGap"]) < 0.5
    assert abs(layout["assistantLeftGap"]) < 0.5


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
    _select_model(page, "open_router/vendor/small-context")
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

    expect(page.get_by_role("combobox", name="Selected model")).to_be_visible()
    expect(page.get_by_label("Thinking")).to_be_visible()
    expect(page.get_by_role("textbox", name="Message", exact=True)).to_be_in_viewport()
