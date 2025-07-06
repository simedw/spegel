import re

from spegel.web import extract_clean_text, html_to_markdown

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
        "<html><head><title>Big</title></head><body>"
        + ("x" * 10_000)
        + "</body></html>"
    )
    md = extract_clean_text(huge_html, url="https://example.com", max_chars=100)

    assert md.endswith("...(truncated)")
    # Length should not exceed max_chars + len("...(truncated)") + a small header allowance
    assert len(md) <= 100 + len("...(truncated)") + 50  # header lines allow some slack


class TestWebErrorScenarios:
    """Test error scenarios for web functionality."""

    def test_html_to_markdown_malformed_html(self):
        """Test handling of malformed HTML."""
        malformed_html = (
            "<html><head><title>Test</head><body><p>Unclosed paragraph</body></html>"
        )

        # Should not raise exception and return some content
        result = html_to_markdown(malformed_html)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_html_to_markdown_empty_html(self):
        """Test handling of empty HTML."""
        empty_html = ""

        result = html_to_markdown(empty_html)
        assert isinstance(result, str)

    def test_html_to_markdown_invalid_encoding(self):
        """Test handling of HTML with invalid characters."""
        # HTML with invalid UTF-8 sequences
        html_with_invalid_chars = (
            "<html><body>Valid text \udcff invalid char</body></html>"
        )

        # Should handle gracefully without crashing
        result = html_to_markdown(html_with_invalid_chars)
        assert isinstance(result, str)
        # The function should return an error message when it can't process the content
        assert "Error parsing HTML" in result or "Valid text" in result

    def test_extract_clean_text_no_body(self):
        """Test extraction from HTML without body tag."""
        html_no_body = "<html><head><title>Title Only</title></head></html>"

        result = extract_clean_text(html_no_body)
        assert isinstance(result, str)
        assert "Title Only" in result

    def test_extract_clean_text_only_whitespace(self):
        """Test extraction from HTML with only whitespace content."""
        whitespace_html = "<html><body>   \n\t   </body></html>"

        result = extract_clean_text(whitespace_html)
        assert isinstance(result, str)
        # Should handle whitespace-only content gracefully

    def test_html_to_markdown_very_nested_html(self):
        """Test handling of deeply nested HTML structures."""
        # Create deeply nested HTML
        nested_content = "Content"
        for _ in range(100):  # Very deep nesting
            nested_content = f"<div>{nested_content}</div>"

        nested_html = f"<html><body>{nested_content}</body></html>"

        # Should handle without stack overflow
        result = html_to_markdown(nested_html)
        assert "Content" in result

    def test_extract_clean_text_with_scripts_and_styles(self):
        """Test that scripts and styles are properly filtered out."""
        html_with_scripts = """
        <html>
        <head>
            <style>body { color: red; }</style>
            <script>alert('malicious');</script>
        </head>
        <body>
            <p>Visible content</p>
            <script>more_js();</script>
            <style>.hidden { display: none; }</style>
        </body>
        </html>
        """

        result = extract_clean_text(html_with_scripts)

        # Should contain visible content but not scripts/styles
        assert "Visible content" in result
        assert "alert" not in result
        assert "color: red" not in result
        assert "more_js" not in result

    def test_html_to_markdown_with_broken_links(self):
        """Test handling of broken or malformed links."""
        html_with_broken_links = """
        <html><body>
            <a href="">Empty link</a>
            <a>Link without href</a>
            <a href="javascript:void(0)">JS link</a>
            <a href="mailto:">Empty mailto</a>
            <a href="https://valid.com">Valid link</a>
        </body></html>
        """

        result = html_to_markdown(html_with_broken_links)

        # Should handle broken links gracefully
        assert "Valid link" in result
        assert "https://valid.com" in result
        # Should not crash on malformed links
        assert isinstance(result, str)

    def test_extract_clean_text_max_chars_edge_cases(self):
        """Test max_chars parameter edge cases."""
        html_content = "<html><body>" + ("A" * 1000) + "</body></html>"

        # Test with very small max_chars
        result_small = extract_clean_text(html_content, max_chars=10)
        assert len(result_small) <= 10 + len("...(truncated)") + 50

        # Test with zero max_chars
        result_zero = extract_clean_text(html_content, max_chars=0)
        assert isinstance(result_zero, str)

        # Test with negative max_chars
        result_negative = extract_clean_text(html_content, max_chars=-1)
        assert isinstance(result_negative, str)
