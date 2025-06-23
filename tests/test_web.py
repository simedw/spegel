import re
import sys
from pathlib import Path

# Add project 'src' directory to sys.path so tests work without editable install
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pytest

from spegel.web import html_to_markdown, extract_clean_text


SIMPLE_HTML = """
<html>
  <head><title>Title</title></head>
  <body>
    <p>A link to <a href="https://example.com">Example</a>.</p>
  </body>
</html>
"""


def test_html_to_markdown_link_cleanup():
    """html_to_markdown should output markdown with clean link syntax."""
    md = html_to_markdown(SIMPLE_HTML, base_url="https://example.com")

    # Should contain a normal markdown link without angle brackets or whitespace
    assert "[Example](https://example.com)" in md

    # Ensure angle brackets have been stripped out everywhere
    assert "<https://" not in md

    # The URL inside parentheses must not contain spaces
    for m in re.finditer(r"\]\(([^)]+)\)", md):
        assert " " not in m.group(1)


def test_extract_clean_text_link_cleanup():
    """extract_clean_text should also clean up links the same way."""
    md = extract_clean_text(SIMPLE_HTML, url="https://example.com")
    assert "[Example](https://example.com)" in md
    assert "<https://" not in md


def test_extract_clean_text_truncation():
    """When max_chars is set, output should be truncated with marker."""
    huge_html = (
        "<html><head><title>Big</title></head><body>" + ("x" * 10_000) + "</body></html>"
    )
    md = extract_clean_text(huge_html, url="https://example.com", max_chars=100)

    assert md.endswith("...(truncated)")
    # Length should not exceed max_chars + len("...(truncated)") + a small header allowance
    assert len(md) <= 100 + len("...(truncated)") + 50  # header lines allow some slack 