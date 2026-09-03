import re
from pathlib import Path

from playwright.sync_api import Page, expect


def _open_work(page: Page, admin_base_url: str) -> None:
    page.goto(f"{admin_base_url}/admin")
    page.get_by_role("button", name="Work Sessions").click()
    expect(page).to_have_url(f"{admin_base_url}/admin/work")
    expect(page.get_by_text("Start your first Work session.")).to_be_visible()


def test_work_session_runs_codex_across_refresh_and_two_tabs(
    page: Page,
    admin_base_url: str,
    tmp_path: Path,
) -> None:
    _open_work(page, admin_base_url)
    page.get_by_role("button", name="New session").click()
    page.get_by_label("Absolute project folder path").fill(str(tmp_path))
    page.get_by_role("button", name="Create", exact=True).click()

    expect(page).to_have_url(re.compile(r"/admin/work/thread-1$"))
    composer = page.get_by_role("textbox", name="Work message")
    expect(composer).to_be_visible()
    expect(page.get_by_role("heading", name="Fixture Work")).to_be_visible()
    expect(page.get_by_role("combobox", name="Model", exact=True)).to_have_value(
        "model-1"
    )
    expect(page.get_by_role("combobox", name="Mode", exact=True)).to_have_count(0)
    expect(page.get_by_role("combobox", name="Permissions", exact=True)).to_have_count(
        0
    )
    expect(page.get_by_role("button", name="Load older activity")).to_have_count(0)

    other = page.context.new_page()
    try:
        other.goto(page.url)
        expect(other.get_by_role("textbox", name="Work message")).to_be_visible()

        composer.fill("Inspect this repository")
        composer.press("Enter")
        expect(composer).to_be_focused()
        expect(page.locator(".work-user-message")).to_have_text(
            "Inspect this repository"
        )
        expect(other.get_by_role("button", name="Accept", exact=True)).to_be_visible()

        page.reload()
        expect(page.get_by_role("button", name="Accept", exact=True)).to_be_visible()
        page.get_by_role("button", name="Accept", exact=True).click()

        expect(
            page.locator(".work-agent-message").filter(has_text="Fixture answer")
        ).to_have_count(1)
        expect(
            other.locator(".work-agent-message").filter(has_text="Fixture answer")
        ).to_have_count(1)
        expect(page.locator("#workStatus")).to_have_text("Completed")
        expect(other.locator("#workStatus")).to_have_text("Completed")

        page.get_by_role("button", name="← Work", exact=True).click()
        expect(page.locator(".work-session-card")).to_have_count(1)
        expect(page.locator(".work-session-card")).to_contain_text("Fixture Work")
    finally:
        other.close()


def test_work_send_reuses_browser_operation_after_lost_acknowledgement(
    page: Page,
    admin_base_url: str,
    tmp_path: Path,
) -> None:
    _open_work(page, admin_base_url)
    page.get_by_role("button", name="New session").click()
    page.get_by_label("Absolute project folder path").fill(str(tmp_path))
    page.get_by_role("button", name="Create", exact=True).click()
    expect(page).to_have_url(re.compile(r"/admin/work/thread-1$"))

    attempts: list[dict[str, object]] = []

    def intercept_turn(route) -> None:
        attempts.append(route.request.post_data_json)
        if len(attempts) == 1:
            route.abort("connectionrefused")
        else:
            route.continue_()

    page.route("**/admin/api/work/sessions/*/turns", intercept_turn)
    composer = page.get_by_role("textbox", name="Work message")
    composer.fill("Recover the same send")
    composer.press("Enter")
    page.wait_for_function(
        "JSON.parse(sessionStorage.getItem('fcc.work.operations.v1') || '[]').length === 1"
    )
    stored = page.evaluate(
        "JSON.parse(sessionStorage.getItem('fcc.work.operations.v1'))[0]"
    )
    assert stored["payload"]["text"] == "Recover the same send"

    page.reload()
    expect(page.get_by_role("button", name="Accept", exact=True)).to_be_visible()
    assert len(attempts) == 2
    assert attempts[0]["operation_id"] == attempts[1]["operation_id"]
    assert attempts[1]["operation_id"] == stored["operation_id"]

    page.get_by_role("button", name="Accept", exact=True).click()
    expect(
        page.locator(".work-agent-message").filter(has_text="Fixture answer")
    ).to_have_count(1)
    page.wait_for_function(
        "JSON.parse(sessionStorage.getItem('fcc.work.operations.v1') || '[]').length === 0"
    )


def test_work_create_is_latched_and_does_not_navigate_another_tab(
    page: Page,
    admin_base_url: str,
    tmp_path: Path,
) -> None:
    _open_work(page, admin_base_url)
    other = page.context.new_page()
    try:
        other.goto(f"{admin_base_url}/admin/work")
        expect(other.get_by_role("button", name="New session")).to_be_visible()
        create_requests: list[str] = []
        page.on(
            "request",
            lambda request: (
                create_requests.append(request.url)
                if request.method == "POST"
                and request.url.endswith("/admin/api/work/sessions")
                else None
            ),
        )

        page.get_by_role("button", name="New session").click()
        page.get_by_label("Absolute project folder path").fill(str(tmp_path))
        page.get_by_role("button", name="Create", exact=True).dblclick()

        expect(page).to_have_url(re.compile(r"/admin/work/thread-1$"))
        expect(other).to_have_url(f"{admin_base_url}/admin/work")
        expect(other.locator(".work-session-card")).to_have_count(1)
        assert len(create_requests) == 1
    finally:
        other.close()


def test_failed_pre_turn_send_restores_text_without_overwriting_a_newer_draft(
    page: Page,
    admin_base_url: str,
    tmp_path: Path,
) -> None:
    _open_work(page, admin_base_url)
    page.get_by_role("button", name="New session").click()
    page.get_by_label("Absolute project folder path").fill(str(tmp_path))
    page.get_by_role("button", name="Create", exact=True).click()
    expect(page).to_have_url(re.compile(r"/admin/work/thread-1$"))

    composer = page.get_by_role("textbox", name="Work message")
    composer.fill("[reject-turn] restore this draft")
    composer.press("Enter")
    composer.fill("newer private draft")

    expect(composer).to_have_value("newer private draft")
    restore = page.get_by_role("button", name="Restore", exact=True)
    expect(restore).to_be_visible()
    restore.click()
    expect(composer).to_have_value("[reject-turn] restore this draft")
    page.wait_for_function(
        "JSON.parse(sessionStorage.getItem('fcc.work.operations.v1') || '[]').length === 0"
    )
