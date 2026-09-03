"""Rendered contracts for Admin asset generation consistency."""

from urllib.parse import urlsplit

from playwright.sync_api import Error, Page, Request, expect

from free_claude_code.core.version import package_version


def test_admin_loads_current_release_assets_before_rendering_dynamic_content(
    page: Page,
    admin_base_url: str,
) -> None:
    requested_paths: list[str] = []
    page_errors: list[str] = []

    def record_request(request: Request) -> None:
        requested_paths.append(urlsplit(request.url).path)

    def record_page_error(error: Error) -> None:
        page_errors.append(str(error))

    page.on("request", record_request)
    page.on("pageerror", record_page_error)
    page.goto(f"{admin_base_url}/admin")

    expect(page.locator('[data-provider="nvidia_nim"]')).to_be_visible()
    expect(page.locator(".brand p")).to_have_text(
        f"Server Control · v{package_version()}"
    )
    logo_link = page.get_by_role("link", name="Open Free Claude Code on GitHub")
    expect(logo_link).to_be_visible()
    expect(logo_link).to_have_attribute(
        "href", "https://github.com/Alishahryar1/free-claude-code"
    )
    expect(logo_link).to_have_attribute("target", "_blank")

    versioned_root = f"/admin/assets/{package_version()}"
    assert f"{versioned_root}/app-icon.svg" in requested_paths
    assert f"{versioned_root}/admin.css" in requested_paths
    assert f"{versioned_root}/chat_sessions.css" in requested_paths
    assert f"{versioned_root}/model_combobox.js" in requested_paths
    assert f"{versioned_root}/chat_sessions.js" in requested_paths
    assert f"{versioned_root}/admin.js" in requested_paths
    assert "/admin/assets/admin.css" not in requested_paths
    assert "/admin/assets/chat_sessions.css" not in requested_paths
    assert "/admin/assets/model_combobox.js" not in requested_paths
    assert "/admin/assets/chat_sessions.js" not in requested_paths
    assert "/admin/assets/admin.js" not in requested_paths
    assert page_errors == []
