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
    expect(page.get_by_role("textbox", name="Work session name")).to_have_value(
        "Fixture Work"
    )
    expect(page.get_by_role("combobox", name="Model", exact=True)).to_have_value(
        "model-1"
    )

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
