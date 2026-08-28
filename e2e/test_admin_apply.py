"""Rendered one-step Admin configuration workflow regressions."""

from playwright.sync_api import Page, expect


def test_apply_is_the_only_config_action_and_retains_invalid_edits(
    page: Page,
    admin_base_url: str,
) -> None:
    config_mutations: list[tuple[str, str]] = []

    def record_config_mutation(method: str, url: str) -> None:
        if "/admin/api/config/" in url:
            config_mutations.append((method, url.rsplit("/", maxsplit=1)[-1]))

    page.on(
        "request",
        lambda request: record_config_mutation(request.method, request.url),
    )
    page.emulate_media(reduced_motion="reduce")
    page.goto(f"{admin_base_url}/admin")
    expect(page.locator("#messageArea")).to_have_text("")

    expect(page.get_by_role("button", name="Validate", exact=True)).to_have_count(0)
    apply_button = page.get_by_role("button", name="Apply", exact=True)
    expect(apply_button).to_be_disabled()

    runtime_section = page.locator("#section-runtime")
    runtime_section.get_by_role("button", name="Show advanced", exact=True).click()
    timeout_input = runtime_section.locator("#field-PROVIDER_PROGRESS_TIMEOUT")
    timeout_input.fill("0")
    expect(page.locator("#dirtyState")).to_have_text("1 unsaved change")
    expect(apply_button).to_be_enabled()

    apply_button.click()

    expect(page.locator("#messageArea")).to_contain_text("PROVIDER_PROGRESS_TIMEOUT")
    expect(timeout_input).to_have_value("0")
    expect(page.locator("#dirtyState")).to_have_text("1 unsaved change")
    expect(apply_button).to_be_enabled()
    assert config_mutations == [("POST", "apply")]
