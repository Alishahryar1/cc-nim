import re

from playwright.sync_api import expect

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
    page.get_by_text("Thinking", exact=True).click()
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
    code_control.run(connection.finish("turn-1"))
    page.evaluate("window.releaseDetail()")
    expect(page.get_by_text("Newest output", exact=True)).to_be_visible()
    expect(page.get_by_text("Old output", exact=True)).to_have_count(0)
    expect(page.get_by_role("button", name="Send", exact=True)).to_be_visible()


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
        page.get_by_role("button", name="Rename", exact=True).click()
        page.get_by_role("textbox", name="Title", exact=True).fill("My project")
        page.get_by_role("button", name="Save title", exact=True).click()
        expect(
            second.get_by_role("heading", name="My project", exact=True)
        ).to_be_visible()
        page.get_by_role("button", name="Delete", exact=True).click()
        page.get_by_role("button", name="Delete session", exact=True).click()
        expect(page).to_have_url(f"{admin_base_url}/admin/code")
        expect(second).to_have_url(f"{admin_base_url}/admin/code")
        expect(second.locator(".code-session-card")).to_have_count(0)
        assert project.read_text() == "unchanged"
    finally:
        second.close()


def test_question_input_survives_streaming_and_secret_answer_is_not_stored(
    page, admin_base_url, tmp_path, code_control
):
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
