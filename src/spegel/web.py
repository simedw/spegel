from __future__ import annotations

import re
from typing import Optional, List
import requests
from bs4 import BeautifulSoup

"""Web fetching and HTML cleaning utilities for Spegel.

This module centralises network I/O so the UI layer can remain async and testable.
"""


__all__ = ["fetch_url", "extract_clean_text"]


HEADERS = {"User-Agent": "Spegel/1.0 (Terminal Browser)"}


def fetch_url(url: str, timeout: int = 10) -> Optional[str]:
    """Blocking HTTP GET returning the raw HTML text or None on error."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()

        # Handle encoding more robustly to fix Unicode issues, but only when Requests clearly does not know
        if not resp.encoding or resp.encoding.lower() in ("iso-8859-1", "ascii"):
            resp.encoding = resp.apparent_encoding

        return resp.text
    except requests.RequestException:
        return None


# ---------------------------------------------------------------------------
# Cleaning helpers (copied from previous main.py implementation)
# ---------------------------------------------------------------------------


def _extract_table_content(table) -> List[str]:
    content: List[str] = []
    for row in table.find_all("tr"):
        cells = row.find_all(["td", "th"])
        if cells:
            row_content: List[str] = []
            for cell in cells:
                links = cell.find_all("a")
                if links:
                    for link in links:
                        text = link.get_text().strip()
                        href = link.get("href", "")
                        if text and len(text) > 3:
                            if href and not href.startswith("#"):
                                row_content.append(f"[{text}]({href})")
                            else:
                                row_content.append(text)
                else:
                    cell_text = cell.get_text().strip()
                    if cell_text and len(cell_text) > 2:
                        row_content.append(cell_text)
            if row_content:
                combined = " | ".join(row_content)
                if len(combined) > 10:
                    content.append(combined)
    return content


def extract_clean_text(
    html: str,
    url: str | None = None,
    *,
    max_chars: int | None = None,
) -> str:
    """Return cleaned markdown using a content-first approach with targeted noise removal.

    Parameters
    ----------
    html : str
        Raw HTML source.
    url : str | None, optional
        Source URL (used only for header display).
    max_chars : int | None, optional
        If set, output will be truncated to this many characters with a
        "…(truncated)" marker.  If None (default) the text is returned in full.  The
        browser passes a limit to stay within LLM token budgets; the CLI leaves
        it unlimited.
    """

    soup = BeautifulSoup(html, "lxml")

    # 1️⃣ Try to find main content area first (content-first approach)
    main_content = None
    content_selectors = [
        "main",
        "article",
        ".post-content",
        ".entry-content",
        ".content",
        "[role='main']",
        ".post",
        ".hrecipe",  # Recipe-specific
        "[itemtype*='Recipe']",  # Schema.org Recipe markup
    ]

    for selector in content_selectors:
        elements = soup.select(selector)
        if elements:
            # Use the element with the most text content
            best_element = max(elements, key=lambda e: len(e.get_text(strip=True)))
            content_text = best_element.get_text(strip=True)
            if len(content_text) > 500:  # Substantial content
                main_content = best_element
                break

    # 2️⃣ If we found main content, use that. Otherwise, clean the whole page more conservatively
    if main_content:
        # Work with just the main content area
        working_soup = BeautifulSoup(str(main_content), "lxml")

        # Remove only the most obvious noise from within the content
        minimal_noise_selectors = [
            "script",
            "style",
            "noscript",
            ".advertisement",
            ".ads",
            ".ad-container",
            ".social-share",
            ".share-buttons",
            "[aria-hidden='true']",
        ]
        for sel in minimal_noise_selectors:
            for node in working_soup.select(sel):
                node.decompose()
    else:
        # Fallback: clean the whole page but be more conservative
        working_soup = soup

        # More targeted noise removal (avoid overly broad selectors)
        conservative_noise_selectors = [
            "script",
            "style",
            "noscript",
            "iframe",
            "embed",
            "object",
            "nav",
            "header",
            "footer",
            "[role='navigation']",
            "[role='banner']",
            "[role='contentinfo']",
            "[aria-hidden='true']",
            ".advertisement",
            ".ads",
            ".ad-container",
            ".sidebar-ads",
            ".social-share",
            ".share-buttons",
            ".social-media",
            ".cookie-notice",
            ".newsletter-signup",
        ]
        for sel in conservative_noise_selectors:
            for node in working_soup.select(sel):
                node.decompose()

    # 3️⃣ Convert remaining HTML to Markdown via html2text
    try:
        import html2text  # lazy import; already dependency of project
    except ImportError:
        html2text = None  # type: ignore

    if html2text is None:
        cleaned_markdown = working_soup.get_text("\n", strip=True)
    else:
        h = html2text.HTML2Text()
        h.ignore_links = False
        h.ignore_images = True
        h.body_width = None
        h.wrap_links = False
        h.protect_links = True
        cleaned_markdown = h.handle(str(working_soup))
        # html2text wraps URLs in <...>. Remove the angle brackets for cleaner markdown.
        cleaned_markdown = re.sub(r"\]\(<([^>]+)>\)", r"](\1)", cleaned_markdown)

    # Optional truncation for token safety when used inside the browser
    if max_chars is not None and len(cleaned_markdown) > max_chars:
        cleaned_markdown = cleaned_markdown[:max_chars] + "\n...(truncated)"

    title_tag = soup.find("title")
    title_text = title_tag.get_text().strip() if title_tag else "No Title"

    header = f"Title: {title_text}\nURL: {url or ''}\n\n"
    return header + cleaned_markdown


# ---------------------------------------------------------------------------
# Full HTML → Markdown (no aggressive cleaning) used for the Raw view
# ---------------------------------------------------------------------------


def html_to_markdown(html: str, base_url: str | None = None) -> str:
    """Convert full HTML document to terminal-friendly markdown."""
    try:
        import html2text  # already dependency

        h = html2text.HTML2Text()
        h.ignore_links = False
        h.ignore_images = False
        h.ignore_emphasis = False
        h.body_width = None  # disable line wrapping to avoid broken URLs
        h.wrap_links = False
        h.unicode_snob = True
        h.skip_internal_links = True
        h.inline_links = True
        h.protect_links = True
        h.mark_code = True

        markdown_content = h.handle(html)

        # 1) Remove angle brackets around URLs
        markdown_content = re.sub(r"\]\(<([^>]+)>\)", r"](\1)", markdown_content)

        # 2) Collapse whitespace inside URL parentheses which can appear when html2text wraps long links
        def _fix(m):
            url = re.sub(r"\s+", "", m.group(1))
            return f"]({url})"

        markdown_content = re.sub(r"\]\(([^)]+)\)", _fix, markdown_content)

        soup = BeautifulSoup(html, "lxml")
        title = soup.find("title")
        title_text = title.get_text().strip() if title else "No Title"

        header = [f"# {title_text}", "", f"**URL:** `{base_url or ''}`", "", "---", ""]
        return "\n".join(header) + markdown_content
    except Exception as exc:
        return f"## ❌ Error parsing HTML\n\n```\n{exc}\n```"


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Fetch a URL and print Spegel's cleaned text representation."
    )
    parser.add_argument("url", help="URL to fetch")
    args = parser.parse_args()

    html = fetch_url(args.url)
    if html is None:
        print("Error: Failed to fetch URL", file=sys.stderr)
        sys.exit(1)

    print(extract_clean_text(html, args.url))
