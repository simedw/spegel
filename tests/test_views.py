from unittest.mock import Mock

import pytest

from spegel.config import View
from spegel.views import stream_view

SAMPLE_HTML = """
<html>
<head><title>Test Page</title></head>
<body>
    <h1>Main Title</h1>
    <p>This is a test paragraph with <a href="https://example.com">a link</a>.</p>
    <div>Some content in a div.</div>
</body>
</html>
"""


class TestStreamView:
    """Test the stream_view function."""

    @pytest.mark.asyncio
    async def test_stream_raw_view(self):
        """Raw view should yield complete markdown immediately."""
        view = View(id="raw", name="Raw", hotkey="r", prompt="")

        chunks = []
        async for chunk in stream_view(view, SAMPLE_HTML, None, "https://test.com"):
            chunks.append(chunk)

        # Should get one complete chunk
        assert len(chunks) == 1
        result = chunks[0]
        assert "# Test Page" in result
        assert "[a link](https://example.com)" in result

    @pytest.mark.asyncio
    async def test_stream_view_no_llm(self):
        """Non-raw view without LLM should yield error message."""
        view = View(id="summary", name="Summary", hotkey="s", prompt="Summarize this")

        chunks = []
        async for chunk in stream_view(view, SAMPLE_HTML, None, "https://test.com"):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert "LLM not available" in chunks[0]

    @pytest.mark.asyncio
    async def test_stream_view_with_llm(self):
        """Non-raw view with LLM should yield chunks as they arrive."""
        view = View(id="summary", name="Summary", hotkey="s", prompt="Summarize this")

        # Mock LLM client that yields chunks with newlines
        mock_client = Mock()

        async def mock_stream(prompt, content):
            yield "First line\n"
            yield "Second line\n"
            yield "Partial"
            yield " line\n"
            yield "Final chunk"

        mock_client.stream = mock_stream

        chunks = []
        async for chunk in stream_view(
            view, SAMPLE_HTML, mock_client, "https://test.com"
        ):
            chunks.append(chunk)

        # Should yield complete lines plus final buffer
        expected_chunks = [
            "First line\n",
            "Second line\n",
            "Partial line\n",
            "Final chunk",
        ]
        assert chunks == expected_chunks

    @pytest.mark.asyncio
    async def test_stream_view_line_buffering(self):
        """Stream view should buffer partial lines correctly."""
        view = View(id="test", name="Test", hotkey="t", prompt="Test prompt")

        mock_client = Mock()

        async def mock_stream(prompt, content):
            # Simulate chunks that don't align with line boundaries
            yield "Start of"
            yield " first line\nSecond"
            yield " line\nThird line\nPartial"

        mock_client.stream = mock_stream

        chunks = []
        async for chunk in stream_view(
            view, SAMPLE_HTML, mock_client, "https://test.com"
        ):
            chunks.append(chunk)

        expected = [
            "Start of first line\n",
            "Second line\n",
            "Third line\n",
            "Partial",  # Final buffer without newline
        ]
        assert chunks == expected

    @pytest.mark.asyncio
    async def test_stream_view_empty_chunks(self):
        """Stream view should handle empty chunks gracefully."""
        view = View(id="test", name="Test", hotkey="t", prompt="Test prompt")

        mock_client = Mock()

        async def mock_stream(prompt, content):
            yield ""
            yield "Real content\n"
            yield ""
            yield "More content"
            yield ""

        mock_client.stream = mock_stream

        chunks = []
        async for chunk in stream_view(
            view, SAMPLE_HTML, mock_client, "https://test.com"
        ):
            chunks.append(chunk)

        # Empty chunks should be skipped
        assert chunks == ["Real content\n", "More content"]


class TestViewIntegration:
    """Integration tests for view processing."""

    @pytest.mark.asyncio
    async def test_view_prompt_construction(self):
        """Test that view prompt is properly combined with content."""
        view = View(
            id="test", name="Test", hotkey="t", prompt="Please analyze this webpage:"
        )

        mock_client = Mock()
        received_prompts = []

        async def mock_stream(prompt, content):
            received_prompts.append(prompt)
            yield "Analysis complete"

        mock_client.stream = mock_stream

        chunks = []
        async for chunk in stream_view(
            view, SAMPLE_HTML, mock_client, "https://test.com"
        ):
            chunks.append(chunk)

        # Should have received the combined prompt
        assert len(received_prompts) == 1
        full_prompt = received_prompts[0]
        assert "Please analyze this webpage:" in full_prompt
        assert "Webpage content:" in full_prompt
        assert "Main Title" in full_prompt  # Content from cleaned HTML
        assert chunks == ["Analysis complete"]
