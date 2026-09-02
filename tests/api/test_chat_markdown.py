from free_claude_code.api.admin_markdown import render_admin_markdown


def test_chat_markdown_escapes_html_and_never_loads_remote_images():
    rendered = render_admin_markdown(
        '<script>alert("x")</script>\n\n![diagram](https://example.com/image.png)'
    )

    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered
    assert "<img" not in rendered
    assert 'href="https://example.com/image.png"' in rendered
    assert 'rel="noopener noreferrer"' in rendered


def test_chat_markdown_disables_non_http_links():
    rendered = render_admin_markdown("[unsafe](javascript:alert(1))")

    assert "href=" not in rendered
    assert "unsafe" in rendered
