import re
import uuid

import pytest
from playwright.sync_api import expect

from e2e.test_chat_sessions import _new_chat
from free_claude_code.application.code_sessions.models import (
    HarnessEvent,
    PromptRequest,
)


def create_session(page, base_url, directory):
    page.goto(f"{base_url}/admin/code")
    page.get_by_role("button", name="New code session", exact=True).click()
    page.get_by_role("textbox", name="Folder", exact=True).fill(str(directory))
    expect(page.get_by_role("combobox", name="Harness", exact=True)).to_have_value(
        "codex"
    )
    page.get_by_role("button", name="Create session", exact=True).click()
    expect(page).to_have_url(re.compile(r"/admin/code/[0-9a-f-]+$"))
    expect(page.get_by_role("textbox", name="Message", exact=True)).to_be_enabled()
    return page.url


def send(page, text):
    page.get_by_role("textbox", name="Message", exact=True).fill(text)
    page.get_by_role("button", name="Send", exact=True).click()
    expect(page.get_by_role("button", name="Stop", exact=True)).to_be_visible()


def test_code_streams_survive_refresh_and_all_viewers_leaving(
    page, context, admin_base_url, tmp_path, code_control
):
    url = create_session(page, admin_base_url, tmp_path)
    send(page, "Inspect this project")
    connection = code_control.connection()
    code_control.run(
        connection.text(
            "turn-1", "reason", "Reading the files", complete=True, kind="reasoning"
        )
    )
    code_control.run(
        connection.text(
            "turn-1", "command", "directory output", complete=True, kind="tool"
        )
    )
    code_control.run(connection.text("turn-1", "reply", "First finding", complete=True))
    expect(page.get_by_text("First finding", exact=True)).to_be_visible()
    page.locator("summary").filter(has_text="Thinking").click()
    expect(page.get_by_text("Reading the files", exact=True)).to_be_visible()
    expect(page.get_by_role("button", name="Regenerate", exact=True)).to_have_count(0)
    expect(page.get_by_role("button", name="Edit message", exact=True)).to_have_count(0)
    second = context.new_page()
    try:
        second.goto(url)
        expect(second.get_by_text("First finding", exact=True)).to_be_visible()
        page.goto("about:blank")
        second.close()
        code_control.run(
            connection.text("turn-1", "reply", "Final finding", complete=True)
        )
        code_control.run(connection.finish("turn-1"))
        page.goto(url)
        expect(page.get_by_text("Final finding", exact=True)).to_be_visible()
        page.get_by_role("textbox", name="Message", exact=True).fill("Follow up")
        expect(page.get_by_role("button", name="Send", exact=True)).to_be_enabled()
        page.reload()
        expect(page.get_by_text("Final finding", exact=True)).to_be_visible()
        assert len(connection.inputs) == 1
    finally:
        if not second.is_closed():
            second.close()


def test_prompt_claim_syncs_tabs_and_stop_keeps_output(
    page, context, admin_base_url, tmp_path, code_control
):
    url = create_session(page, admin_base_url, tmp_path)
    send(page, "Run a command")
    connection = code_control.connection()
    code_control.run(
        connection.text("turn-1", "reply", "Output before approval", complete=True)
    )
    code_control.run(connection.prompt(0))
    second = context.new_page()
    try:
        second.goto(url)
        expect(second.get_by_role("button", name="Allow", exact=True)).to_be_enabled()
        page.get_by_role("button", name="Allow", exact=True).click()
        expect(second.get_by_role("button", name="Allow", exact=True)).to_be_disabled()
        page.get_by_role("button", name="Stop", exact=True).click()
        page.get_by_role("textbox", name="Message", exact=True).fill("Follow up")
        expect(page.get_by_role("button", name="Send", exact=True)).to_be_enabled()
        expect(second.get_by_text("Output before approval", exact=True)).to_be_visible()
        assert len(connection.answers) == 1
    finally:
        second.close()


def test_old_detail_cannot_replace_streamed_output(
    page, admin_base_url, tmp_path, code_control
):
    url = create_session(page, admin_base_url, tmp_path)
    send(page, "Keep the latest output")
    connection = code_control.connection()
    code_control.run(connection.text("turn-1", "reply", "Old output", complete=True))
    code_control.run(connection.prompt(0))
    expect(page.get_by_text("Old output", exact=True)).to_be_visible()
    page.add_init_script("""(() => {
      const original = window.fetch;
      window.fetch = async (...args) => {
        const result = await original(...args);
        if (/\\/api\\/code\\/sessions\\/[0-9a-f-]+$/.test(String(args[0]))) {
          window.detailCaptured = true;
          await new Promise(resolve => { window.releaseDetail = resolve; });
        }
        return result;
      };
    })();""")
    page.goto(url)
    page.wait_for_function("window.detailCaptured === true")
    code_control.run(connection.text("turn-1", "reply", "Newest output", complete=True))
    code_control.run(connection.resolve(0))
    code_control.run(connection.finish("turn-1"))
    page.evaluate("window.releaseDetail()")
    expect(page.get_by_text("Newest output", exact=True)).to_be_visible()
    expect(page.get_by_text("Old output", exact=True)).to_have_count(0)
    expect(page.get_by_role("button", name="Send", exact=True)).to_be_visible()
    expect(page.get_by_role("button", name="Allow", exact=True)).to_be_disabled()


def test_competing_tabs_keep_the_rejected_draft(
    page, context, admin_base_url, tmp_path, code_control
):
    url = create_session(page, admin_base_url, tmp_path)
    second = context.new_page()
    try:
        second.goto(url)
        first_input = page.get_by_role("textbox", name="Message", exact=True)
        other_input = second.get_by_role("textbox", name="Message", exact=True)
        first_input.fill("First tab")
        other_input.fill("Second tab draft")
        expect(second.get_by_role("button", name="Send", exact=True)).to_be_enabled()
        code_control.run(code_control.hold_send())
        page.get_by_role("button", name="Send", exact=True).click()
        second.get_by_role("button", name="Send", exact=True).click()
        code_control.run(code_control.release_send())
        expect(page.get_by_role("button", name="Stop", exact=True)).to_be_visible()
        expect(second.get_by_role("button", name="Stop", exact=True)).to_be_visible()
        expect(first_input).to_have_value("")
        expect(other_input).to_have_value("Second tab draft")
        second.reload()
        expect(second.get_by_role("textbox", name="Message", exact=True)).to_have_value(
            "Second tab draft"
        )
        assert (
            sum(
                len(connection.inputs)
                for connection in code_control.harness.connections
            )
            == 1
        )
    finally:
        code_control.run(code_control.release_send())
        second.close()


def test_lost_send_response_keeps_later_typing_and_does_not_replay(
    page, admin_base_url, tmp_path, code_control
):
    create_session(page, admin_base_url, tmp_path)
    page.evaluate("""() => {
      const original = window.fetch;
      window.fetch = async (...args) => {
        const result = await original(...args);
        if (String(args[0]).endsWith('/turns')) {
          window.sendCaptured = true;
          await new Promise((resolve, reject) => { window.loseSend = () => reject(new TypeError('Connection lost')); });
        }
        return result;
      };
    }""")
    send(page, "Run once")
    page.wait_for_function("window.sendCaptured === true")
    draft = page.get_by_role("textbox", name="Message", exact=True)
    draft.fill("Keep my next message")
    page.evaluate("window.loseSend()")
    connection = code_control.connection()
    code_control.run(connection.finish("turn-1"))
    page.reload()
    expect(page.get_by_role("textbox", name="Message", exact=True)).to_have_value(
        "Keep my next message"
    )
    expect(page.get_by_role("button", name="Send", exact=True)).to_be_enabled()
    assert len(connection.inputs) == 1


def test_rename_and_delete_sync_library_without_touching_project(
    page, context, admin_base_url, tmp_path, code_control
):
    url = create_session(page, admin_base_url, tmp_path)
    project = tmp_path / "project.txt"
    project.write_text("unchanged")
    second = context.new_page()
    try:
        second.goto(url)
        page.get_by_role("textbox", name="Code title", exact=True).fill("My project")
        page.get_by_role("textbox", name="Code title", exact=True).press("Enter")
        expect(
            second.get_by_role("textbox", name="Code title", exact=True)
        ).to_have_value("My project")
        page.get_by_role("button", name="Delete", exact=True).click()
        page.get_by_role("button", name="Delete session", exact=True).click()
        expect(page).to_have_url(f"{admin_base_url}/admin/code")
        expect(second).to_have_url(f"{admin_base_url}/admin/code")
        expect(second.locator(".session-card")).to_have_count(0)
        assert project.read_text() == "unchanged"
    finally:
        second.close()


def test_question_input_survives_streaming_and_secret_answer_is_not_stored(
    page, admin_base_url, tmp_path, code_control
):
    control_feed(page)
    create_session(page, admin_base_url, tmp_path)
    send(page, "Ask me a question")
    connection = code_control.connection()
    prompt = PromptRequest(
        7,
        "questions",
        {
            "title": "Codex needs your input",
            "questions": [
                {
                    "id": "destination",
                    "label": "Which destination?",
                    "header": "Destination",
                    "options": [
                        {
                            "label": "Default",
                            "description": "Use the current destination",
                        }
                    ],
                    "allow_other": True,
                    "secret": True,
                }
            ],
        },
        {"question": "destination"},
        "turn-1",
    )

    async def ask():
        connection.requests[7] = prompt
        await connection.sink(
            HarnessEvent(
                connection.generation,
                connection.thread_id,
                "prompt",
                turn_id="turn-1",
                prompt=prompt,
            )
        )

    code_control.run(ask())
    secret = page.get_by_label("Your answer", exact=True)
    secret.fill("private-answer")
    expect(secret).to_have_attribute("type", "password")
    code_control.run(
        connection.text("turn-1", "stream", "Still checking", complete=True)
    )
    expect(page.get_by_text("Still checking", exact=True)).to_be_visible()
    expect(secret).to_have_value("private-answer")
    page.evaluate("window.codeFeed.onerror(new Event('error'))")
    expect(
        page.get_by_role("button", name="Submit answers", exact=True)
    ).to_be_enabled()
    expect(secret).to_have_value("private-answer")
    page.get_by_role("button", name="Submit answers", exact=True).click()
    expect(
        page.get_by_role("button", name="Submit answers", exact=True)
    ).to_be_disabled()
    code_control.run(code_control.harness.answered.wait())
    assert connection.answers == [(7, {"answers": {"destination": ["private-answer"]}})]
    assert "private-answer" not in page.evaluate("JSON.stringify(sessionStorage)")
    detail = code_control.run(
        code_control.service.get_detail(page.url.rsplit("/", 1)[1])
    )
    assert all(
        "private-answer" not in value.model_dump_json() for value in detail.prompts
    )


def control_feed(page):
    page.add_init_script("""(() => {
      const Native = window.EventSource;
      window.EventSource = class extends Native {
        constructor(...args) {
          super(...args);
          this.isCode = String(args[0]).includes('/api/code/events');
          if (this.isCode) window.codeFeed = this;
          if (String(args[0]).includes('/api/chat/events')) window.chatFeed = this;
        }
        addEventListener(type, listener, ...options) {
          return super.addEventListener(type, event => {
            if (this.isCode && window.dropCodeEvents?.includes(type)) return;
            if (this.isCode && type === 'feed.ready')
              window.replayCodeReady = () => listener(event);
            listener(event);
            if (this.isCode && type === 'run.updated')
              window.lastCodeRun = JSON.parse(event.data).run;
          }, ...options);
        }
      };
    })();""")


def hold_code_reads(page):
    page.evaluate("""() => {
      const original = window.fetch;
      window.codeReadHolds = [];
      window.holdCodeReads = true;
      window.fetch = async (...args) => {
        const response = await original(...args);
        const path = new URL(String(args[0]), location.origin).pathname;
        if (window.holdCodeReads && path.startsWith('/admin/api/code/') &&
            (!args[1]?.method || args[1].method === 'GET'))
          await new Promise(resolve => window.codeReadHolds.push({path, resolve}));
        return response;
      };
      window.releaseCodeReads = () => {
        window.holdCodeReads = false;
        for (const held of window.codeReadHolds.splice(0)) held.resolve();
      };
    }""")


def send_from_other_client(page, url, text):
    endpoint = url.replace("/admin/code/", "/admin/api/code/sessions/")
    detail = page.request.get(endpoint).json()
    response = page.request.post(
        f"{endpoint}/turns",
        data={
            "operation_id": str(uuid.uuid4()),
            "expected_revision": detail["session"]["revision"],
            "expected_epoch": detail["epoch"],
            "text": text,
        },
    )
    assert response.ok, response.text()
    return response.json()["id"]


def test_hidden_chat_cannot_change_code_control_state(
    page, admin_base_url, tmp_path, code_control
):
    control_feed(page)
    _new_chat(page, admin_base_url)
    chat_id = page.url.rsplit("/", 1)[1]
    expect(page.locator("#chatModel")).to_be_enabled()
    page.get_by_role("button", name="Code sessions", exact=True).click()
    page.get_by_role("button", name="New code session", exact=True).click()
    page.get_by_role("textbox", name="Folder", exact=True).fill(str(tmp_path))
    page.get_by_role("button", name="Create session", exact=True).click()
    send(page, "Keep working")
    connection = code_control.connection()
    controls = page.locator(
        "#codeRoot .session-model-control input, "
        "#codeRoot .session-model-control button, #codeReasoning, #codeDelete"
    )
    assert controls.count() == 4
    for control in controls.all():
        expect(control).to_be_disabled()
    endpoint = f"{admin_base_url}/admin/api/chat/sessions/{chat_id}"
    session = page.request.get(endpoint).json()["session"]
    response = page.request.patch(
        endpoint,
        data={"expected_revision": session["revision"], "title": "Hidden update"},
    )
    assert response.ok
    expect(page.locator("#chatRoot .session-title")).to_have_value("Hidden update")
    for control in controls.all():
        expect(control).to_be_disabled()
    expect(page.locator("#codeRoot .session-title")).to_be_enabled()

    code_control.run(connection.finish("turn-1"))
    for control in controls.all():
        expect(control).to_be_enabled()
    page.evaluate("window.chatFeed.dispatchEvent(new Event('error'))")
    expect(page.locator("#chatModel")).to_be_disabled()
    for control in controls.all():
        expect(control).to_be_enabled()
    expect(page.locator("#codeRoot .session-title")).to_be_enabled()


def test_model_default_clears_effort_when_all_choices_disappear(
    page, context, admin_base_url, tmp_path, code_control
):
    url = create_session(page, admin_base_url, tmp_path)
    effort = page.locator("#codeReasoning")
    effort.select_option("high")
    expect(effort).to_be_enabled()
    page.locator("#codeComposer").fill("Preserve this draft")
    code_control.harness.efforts = ()
    code_control.harness.default_effort = None
    page.evaluate("window.CodeSessions.refresh()")
    expect(page.locator("#codeComposerStatus")).to_contain_text("unavailable")
    expect(effort).to_have_value("high")
    expect(effort.locator("option:checked")).to_have_text("high (unavailable)")
    expect(page.locator("#codeSend")).to_be_disabled()
    expect(effort).to_be_enabled()
    patches = []
    page.on(
        "request",
        lambda request: (
            patches.append(request.post_data_json)
            if request.method == "PATCH" and "/api/code/sessions/" in request.url
            else None
        ),
    )
    with page.expect_response(
        lambda response: (
            response.request.method == "PATCH" and "/api/code/sessions/" in response.url
        )
    ) as reset:
        effort.select_option("")
    assert reset.value.json()["reasoning_effort"] is None
    assert len(patches) == 1 and patches[0]["reasoning_effort"] is None
    expect(effort).to_be_disabled()
    expect(page.locator("#codeComposerStatus")).not_to_contain_text("unavailable")
    expect(page.locator("#codeSend")).to_be_enabled()
    page.reload()
    expect(effort).to_have_value("")
    expect(effort).to_be_disabled()
    expect(page.locator("#codeComposer")).to_have_value("Preserve this draft")
    second = context.new_page()
    try:
        second.goto(url)
        expect(second.locator("#codeReasoning")).to_have_value("")
        expect(second.locator("#codeModel")).to_have_value("provider/model")
        page.locator("#codeSend").click()
        connection = code_control.connection()
        assert connection.efforts == [None]
        assert connection.inputs[0][1:] == ("Preserve this draft", "provider/model")
    finally:
        second.close()


@pytest.mark.parametrize("status", ["completed", "failed"])
def test_library_reconnect_retires_missed_completion_and_loads_real_outcome(
    page, admin_base_url, tmp_path, code_control, status
):
    control_feed(page)
    create_session(page, admin_base_url, tmp_path)
    send(page, "Cached task")
    connection = code_control.connection()
    code_control.run(
        connection.text("turn-1", "reply", "Preserved output", complete=True)
    )
    expect(page.get_by_text("Preserved output", exact=True)).to_be_visible()
    page.locator("#codeComposer").fill("Next draft")
    page.get_by_role("button", name="← Code sessions", exact=True).click()
    card = page.locator("#codeLibrary .session-card")
    expect(card).to_contain_text("Running")
    page.evaluate("window.dropCodeEvents = ['run.updated', 'session.updated']")
    code_control.run(
        connection.finish(
            "turn-1", status, "Actual failure" if status == "failed" else None
        )
    )
    page.evaluate(
        "window.dropCodeEvents = []; void window.codeFeed.onerror(new Event('error'))"
    )
    expect(page.locator("#codeNew")).to_be_enabled()
    expect(card).not_to_contain_text("Running")
    hold_code_reads(page)
    card.click()
    page.wait_for_function(
        "window.codeReadHolds.some(held => /\\/sessions\\/[0-9a-f-]+$/.test(held.path))"
    )
    expect(page.locator("#codeSend")).to_be_disabled()
    expect(page.locator("#codeModel")).to_be_disabled()
    expect(page.get_by_text("Preserved output", exact=True)).to_be_visible()
    expect(page.locator("#codeComposer")).to_have_value("Next draft")
    page.evaluate("window.releaseCodeReads()")
    expect(page.locator("#codeSend")).to_be_enabled()
    expect(page.locator(".code-run")).to_have_count(1)
    outcome = page.locator(".code-outcome")
    if status == "failed":
        expect(outcome.locator(".session-generation-status")).to_have_text(
            "Actual failure"
        )
        expect(outcome).to_be_visible()
    else:
        expect(outcome).to_be_hidden()


@pytest.mark.parametrize("snapshot_active", [False, True])
def test_new_run_survives_delayed_reconnect_reads(
    page, admin_base_url, tmp_path, code_control, snapshot_active
):
    control_feed(page)
    url = create_session(page, admin_base_url, tmp_path)
    send(page, "First run")
    connection = code_control.connection()
    page.get_by_role("button", name="← Code sessions", exact=True).click()
    expect(page.locator("#codeLibrary .session-card")).to_contain_text("Running")
    if not snapshot_active:
        page.evaluate("window.dropCodeEvents = ['run.updated', 'session.updated']")
        code_control.run(connection.finish("turn-1"))
    hold_code_reads(page)
    page.evaluate(
        "window.dropCodeEvents = []; void window.codeFeed.onerror(new Event('error'))"
    )
    page.wait_for_function(
        "window.codeReadHolds.some(held => held.path === '/admin/api/code/sessions')"
    )
    if snapshot_active:
        expect(page.locator("#codeLibrary .session-card")).to_contain_text("Running")
        code_control.run(connection.finish("turn-1"))
    run_id = send_from_other_client(page, url, "Newer run")
    code_control.run(code_control.harness.wait_inputs(2))
    page.wait_for_function(
        "id => window.lastCodeRun?.id === id && window.lastCodeRun.status === 'running'",
        arg=run_id,
    )
    expect(page.locator("#codeLibrary .session-card")).to_contain_text("Running")
    page.evaluate("window.releaseCodeReads()")
    expect(page.locator("#codeNew")).to_be_enabled()
    expect(page.locator("#codeLibrary .session-card")).to_contain_text("Running")


@pytest.mark.parametrize("source", ["sse", "detail"])
def test_stale_ready_snapshot_preserves_newer_activity(
    page, admin_base_url, tmp_path, code_control, source
):
    control_feed(page)
    url = create_session(page, admin_base_url, tmp_path)
    send(page, "First run")
    connection = code_control.connection()
    code_control.run(connection.finish("turn-1"))
    expect(page.locator("#codeStop")).to_be_hidden()
    page.get_by_role("button", name="← Code sessions", exact=True).click()
    page.evaluate("window.codeFeed.onerror(new Event('error'))")
    expect(page.locator("#codeNew")).to_be_enabled()
    if source == "detail":
        page.evaluate(
            "window.dropCodeEvents = ['run.updated', 'session.updated', 'item.updated']"
        )
    run_id = send_from_other_client(page, url, "Newer run")
    code_control.run(code_control.harness.wait_inputs(2))
    if source == "detail":
        page.locator("#codeLibrary .session-card").click()
        expect(page.locator("#codeStop")).to_be_visible()
        expect(page.get_by_text("Newer run", exact=True)).to_be_visible()
        page.get_by_role("button", name="← Code sessions", exact=True).click()
    else:
        page.wait_for_function(
            "id => window.lastCodeRun?.id === id && window.lastCodeRun.status === 'running'",
            arg=run_id,
        )
    expect(page.locator("#codeLibrary .session-card")).to_contain_text("Running")
    hold_code_reads(page)
    page.evaluate("window.replayCodeReady()")
    page.wait_for_function("window.codeReadHolds.length >= 2")
    expect(page.locator("#codeLibrary .session-card")).to_contain_text("Running")
    page.evaluate("window.releaseCodeReads()")
    expect(page.locator("#codeNew")).to_be_enabled()
    expect(page.locator("#codeLibrary .session-card")).to_contain_text("Running")


@pytest.mark.parametrize("endpoint", ["sessions", "bootstrap"])
def test_first_http_sync_failure_recovers_without_reloading(
    page, admin_base_url, endpoint
):
    page.add_init_script(
        """(endpoint => {
      const original = window.fetch;
      window.fetch = async (...args) => {
        if (!window.failedCodeLoad && new URL(String(args[0]), location.origin).pathname === `/admin/api/code/${endpoint}`) {
          window.failedCodeLoad = true;
          throw new TypeError('Initial snapshot failed');
        }
        return original(...args);
      };
    })("""
        + repr(endpoint)
        + ");"
    )
    page.goto(f"{admin_base_url}/admin/code")
    expect(
        page.get_by_role("button", name="New code session", exact=True)
    ).to_be_enabled()
    assert page.evaluate("window.failedCodeLoad")
    expect(page.locator("#codeNotice")).to_be_hidden()


def test_unavailable_feed_shows_reason_and_clears_it_on_recovery(
    page, admin_base_url, code_control
):
    async def availability(available):
        code_control.service._accepting = available
        code_control.service._message = (
            None if available else "Code storage is unavailable"
        )

    code_control.run(availability(False))
    page.goto(f"{admin_base_url}/admin/code")
    expect(page.locator("#codeNotice")).to_contain_text("Code storage is unavailable")
    expect(page.locator("#codeNew")).to_be_disabled()
    code_control.run(availability(True))
    expect(page.locator("#codeNew")).to_be_enabled()
    expect(page.locator("#codeNotice")).to_be_hidden()


def test_reconnect_retires_out_of_page_prompt_without_discarding_live_form_input(
    page, admin_base_url, tmp_path, code_control
):
    control_feed(page)
    create_session(page, admin_base_url, tmp_path)
    send(page, "Work")
    connection = code_control.connection()
    code_control.run(connection.prompt(0, turn_id=None))
    expect(page.get_by_role("button", name="Allow", exact=True)).to_be_enabled()
    page.evaluate("window.dropCodeEvents = ['prompt.updated']")
    code_control.run(connection.resolve(0))
    for index in range(55):
        code_control.run(
            connection.text("turn-1", str(index), f"Line {index}", complete=True)
        )
    code_control.run(connection.finish("turn-1"))
    page.get_by_role("textbox", name="Message", exact=True).fill("Keep draft")
    page.evaluate(
        "window.dropCodeEvents = []; void window.codeFeed.onerror(new Event('error'))"
    )
    expect(page.get_by_role("button", name="Send", exact=True)).to_be_enabled()
    expect(page.get_by_role("button", name="Allow", exact=True)).to_be_disabled()
    expect(page.get_by_text("No longer active", exact=True)).to_be_visible()
    expect(page.get_by_role("textbox", name="Message", exact=True)).to_have_value(
        "Keep draft"
    )


def test_failed_reply_stays_after_its_output_once_and_not_in_composer(
    page, admin_base_url, tmp_path, code_control
):
    url = create_session(page, admin_base_url, tmp_path)
    send(page, "First")
    connection = code_control.connection()
    code_control.run(
        connection.text(
            "turn-1", "reasoning", "Reasoning so far", kind="reasoning", complete=True
        )
    )
    code_control.run(
        connection.text("turn-1", "tool", "Tool so far", kind="tool", complete=True)
    )
    code_control.run(connection.text("turn-1", "reply", "Partial reply", complete=True))
    code_control.run(
        connection.finish("turn-1", "failed", "Provider refused the request")
    )
    expect(
        page.locator(".code-outcome").filter(has_text="Provider refused the request")
    ).to_have_count(1)
    expect(page.locator("#codeComposerStatus")).not_to_contain_text("Provider refused")
    expect(page.locator("#codeNotice")).to_be_hidden()
    send(page, "Second")
    code_control.run(connection.finish("turn-2"))
    page.goto(url)
    expect(page.locator(".code-run")).to_have_count(2)
    expect(page.locator(".code-run").first).to_contain_text("Partial reply")
    expect(page.locator(".code-run").first.locator(".code-outcome")).to_contain_text(
        "Provider refused"
    )
    expect(page.locator(".code-run").last).to_contain_text("Second")


@pytest.mark.parametrize("refresh_first", [True, False])
def test_model_effort_picker_syncs_tabs_and_keeps_missing_selection(
    page, context, admin_base_url, tmp_path, code_control, refresh_first
):
    code_control.harness.configurations["provider/other"] = "other"
    url = create_session(page, admin_base_url, tmp_path)
    second = context.new_page()
    try:
        second.goto(url)
        page.get_by_role("combobox", name="Selected model", exact=True).fill(
            "provider/other"
        )
        page.get_by_role("option", name="provider/other", exact=True).click()
        expect(second.locator("#codeModel")).to_have_value("provider/other")
        page.locator("#codeReasoning").select_option("high")
        expect(second.locator("#codeReasoning")).to_have_value("high")
        send(page, "Use choice")
        connection = code_control.connection()
        assert connection.inputs[0][2] == "provider/other"
        assert connection.efforts == ["high"]
        expect(page.locator("#codeModel")).to_be_disabled()
        expect(second.locator("#codeReasoning")).to_be_disabled()
        code_control.run(connection.finish("turn-1"))
        page.locator("#codeReasoning").select_option("")
        expect(second.locator("#codeReasoning")).to_have_value("")
        page.get_by_role("textbox", name="Message", exact=True).fill("Preserve me")
        code_control.harness.configurations.pop("provider/other")
        if refresh_first:
            page.evaluate("window.CodeSessions.refresh()")
        else:
            page.locator("#codeSend").click()
            expect(page.locator("#codeNotice")).to_contain_text("unavailable")
        expect(page.locator("#codeModel")).to_have_value("provider/other")
        expect(page.locator("#codeSend")).to_be_disabled()
        expect(page.locator("#codeComposer")).to_have_value("Preserve me")
        expect(page.locator("#codeComposerStatus")).to_contain_text("unavailable")
    finally:
        second.close()


def test_library_reconnect_removes_missed_deletion_and_searches_beyond_first_page(
    page, admin_base_url, tmp_path, code_control
):
    control_feed(page)

    async def seed():
        first = await code_control.service.create_session(
            str(uuid.uuid4()), str(tmp_path)
        )
        await code_control.service.update_settings(
            first.id, first.revision, {"title": "Needle project"}
        )
        for _ in range(26):
            await code_control.service.create_session(str(uuid.uuid4()), str(tmp_path))
        return first.id

    # Start the isolated service before seeding it.
    page.goto(f"{admin_base_url}/admin/code")
    expect(page.locator("#codeNew")).to_be_enabled()
    session_id = code_control.run(seed())
    page.get_by_role("searchbox", name="Search titles and folders").fill("Needle")
    expect(page.locator(".session-card")).to_have_count(1)
    expect(page.locator(".session-card")).to_contain_text("Needle project")
    page.evaluate("window.dropCodeEvents = ['session.deleted']")

    async def remove_seed():
        detail = await code_control.service.get_detail(session_id)
        await code_control.service.delete_session(session_id, detail.session.revision)
        await code_control.service.wait_idle(session_id)

    code_control.run(remove_seed())
    page.evaluate(
        "window.dropCodeEvents = []; void window.codeFeed.onerror(new Event('error'))"
    )
    expect(page.locator(".session-card")).to_have_count(0)
    expect(page.get_by_text("No matching sessions.", exact=True)).to_be_visible()


@pytest.mark.parametrize("width,height", [(1440, 1000), (390, 844)])
def test_code_composer_uses_chat_geometry_and_preserves_focus(
    page, admin_base_url, tmp_path, code_control, width, height
):
    page.set_viewport_size({"width": width, "height": height})
    create_session(page, admin_base_url, tmp_path)
    composer = page.locator("#codeComposer")
    composer.fill("a draft\nwith a second line")
    composer.evaluate("input => input.setSelectionRange(2, 6)")
    expected = composer.evaluate(
        "input => { const s=getComputedStyle(input); return [s.minHeight,s.maxHeight,s.padding,s.lineHeight,s.resize]; }"
    )
    page.locator("#codeModel").click()
    page.keyboard.press("Escape")
    composer.focus()
    composer.evaluate("input => input.setSelectionRange(2, 6)")
    page.evaluate("window.CodeSessions.refresh()")
    expect(composer).to_be_focused()
    assert composer.evaluate("input => [input.selectionStart,input.selectionEnd]") == [
        2,
        6,
    ]
    assert (
        page.locator("#codeComposer").bounding_box()["y"]
        + page.locator("#codeComposer").bounding_box()["height"]
        <= height
    )
    page.goto(f"{admin_base_url}/admin/chat")
    page.get_by_role("button", name="New chat", exact=True).click()
    chat = page.locator("#chatComposer")
    expect(chat).to_be_visible()
    actual = chat.evaluate(
        "input => { const s=getComputedStyle(input); return [s.minHeight,s.maxHeight,s.padding,s.lineHeight,s.resize]; }"
    )
    assert actual == expected
