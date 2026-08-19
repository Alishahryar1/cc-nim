"""HTML parsing for web_search / web_fetch."""

import html
import re
from html.parser import HTMLParser
from urllib.parse import parse_qs, unquote, urlparse

from pydantic import BaseModel

_URL_PATTERN = re.compile(r"https?://[^\s<>\"']+", flags=re.IGNORECASE)
_URL_TRAILING_PUNCTUATION = ").,;:!?]}"


class SearchResultParser(HTMLParser):
    """DuckDuckGo lite HTML: extract result links and titles."""

    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict[str, str]] = []
        self._href: str | None = None
        self._title_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        href = dict(attrs).get("href")
        if not href or "uddg=" not in href:
            return
        parsed = urlparse(href)
        query = parse_qs(parsed.query)
        uddg = query.get("uddg", [""])[0]
        if not uddg:
            return
        self._href = unquote(uddg)
        self._title_parts = []

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            self._title_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag != "a" or self._href is None:
            return
        title = " ".join("".join(self._title_parts).split())
        if title and not any(result["url"] == self._href for result in self.results):
            self.results.append({"title": html.unescape(title), "url": self._href})
        self._href = None
        self._title_parts = []


class HTMLTextParser(HTMLParser):
    """Strip scripts/styles and collect visible text + title for fetch previews."""

    def __init__(self) -> None:
        super().__init__()
        self.title = ""
        self.text_parts: list[str] = []
        self._in_title = False
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1
        elif tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        text = " ".join(data.split())
        if not text:
            return
        if self._in_title:
            self.title = f"{self.title} {text}".strip()
        elif not self._skip_depth:
            self.text_parts.append(text)


def collect_urls(value: object) -> set[str]:
    """Collect literal HTTP(S) URLs from one trusted transcript value."""
    if isinstance(value, str):
        return {
            match.group(0).rstrip(_URL_TRAILING_PUNCTUATION)
            for match in _URL_PATTERN.finditer(value)
        }
    if isinstance(value, BaseModel):
        return collect_urls(value.model_dump())
    if isinstance(value, dict):
        return set().union(*(collect_urls(item) for item in value.values()))
    if isinstance(value, list | tuple):
        return set().union(*(collect_urls(item) for item in value))
    return set()
