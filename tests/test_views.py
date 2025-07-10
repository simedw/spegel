import pytest
from unittest.mock import Mock, AsyncMock, patch

from spegel.views import stream_view, _get_view_llm_client
from spegel.config import View
from spegel.llm import LLMAuthenticationError


class TestStreamView:
    """Test the stream_view function."""

    @pytest.mark.asyncio
    async def test_stream_view_raw(self):
        """Test streaming raw HTML view."""
        view = View(id="raw", name="Raw", hotkey="1", description="Raw HTML", prompt="")
        raw_html = "<html><body>Test</body></html>"

        # Mock html_to_markdown
        with patch("spegel.views.html_to_markdown") as mock_html_to_markdown:
            mock_html_to_markdown.return_value = "# Test Content"

            chunks = []
            async for chunk in stream_view(view, raw_html, None, "http://test.com"):
                chunks.append(chunk)

            assert chunks == ["# Test Content"]
            mock_html_to_markdown.assert_called_once_with(raw_html, "http://test.com")

    @pytest.mark.asyncio
    async def test_stream_view_no_client(self):
        """Test streaming when no LLM client is available."""
        view = View(
            id="summary",
            name="Summary",
            hotkey="2",
            description="Summary view",
            prompt="Summarize this",
        )
        raw_html = "<html><body>Test</body></html>"

        chunks = []
        async for chunk in stream_view(view, raw_html, None, "http://test.com"):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert "LLM not available" in chunks[0]
        assert "OPENAI_API_KEY" in chunks[0]

    @pytest.mark.asyncio
    async def test_stream_view_with_client(self):
        """Test streaming with a working LLM client."""
        view = View(
            id="summary",
            name="Summary",
            hotkey="2",
            description="Summary view",
            prompt="Summarize this",
        )
        raw_html = "<html><body>Test content</body></html>"

        # Mock client
        mock_client = AsyncMock()

        # Mock the stream method to return chunks
        async def mock_stream(prompt, content):
            yield "This is a "
            yield "test response\n"
            yield "with multiple chunks"

        mock_client.stream = mock_stream

        # Mock extract_clean_text
        with patch("spegel.views.extract_clean_text") as mock_extract:
            mock_extract.return_value = "Clean text content"

            chunks = []
            async for chunk in stream_view(
                view, raw_html, mock_client, "http://test.com"
            ):
                chunks.append(chunk)

            # Should get all chunks
            assert "This is a test response\n" in "".join(chunks)
            assert "with multiple chunks" in "".join(chunks)

    @pytest.mark.asyncio
    async def test_stream_view_authentication_error(self):
        """Test streaming with authentication error."""
        view = View(
            id="summary",
            name="Summary",
            hotkey="2",
            description="Summary view",
            prompt="Summarize this",
        )
        raw_html = "<html><body>Test content</body></html>"

        # Mock client that raises authentication error
        mock_client = AsyncMock()

        # Create mock original error
        mock_original_error = Exception("API key not valid")

        async def mock_stream_with_auth_error(prompt, content):
            raise LLMAuthenticationError("openai/gpt-4", "openai", mock_original_error)
            yield  # Make it an async generator (unreachable)

        mock_client.stream = mock_stream_with_auth_error

        # Mock extract_clean_text
        with patch("spegel.views.extract_clean_text") as mock_extract:
            mock_extract.return_value = "Clean text content"

            chunks = []
            async for chunk in stream_view(
                view, raw_html, mock_client, "http://test.com"
            ):
                chunks.append(chunk)

            # Should get authentication error message
            assert len(chunks) == 1
            error_content = chunks[0]
            assert "🔐 Authentication Error" in error_content
            assert "Authentication failed for model 'openai/gpt-4'" in error_content
            assert "Please set a valid API key for openai" in error_content
            assert "Quick Setup:" in error_content
            assert "Create a `.env` file" in error_content
            assert "SPEGEL_API_KEY=your_api_key_here" in error_content

    @pytest.mark.asyncio
    async def test_stream_view_general_runtime_error(self):
        """Test streaming with general runtime error."""
        view = View(
            id="summary",
            name="Summary",
            hotkey="2",
            description="Summary view",
            prompt="Summarize this",
        )
        raw_html = "<html><body>Test content</body></html>"

        # Mock client that raises general runtime error
        mock_client = AsyncMock()

        async def mock_stream_with_error(prompt, content):
            raise RuntimeError("Some general error occurred")
            yield  # Make it an async generator (unreachable)

        mock_client.stream = mock_stream_with_error

        # Mock extract_clean_text
        with patch("spegel.views.extract_clean_text") as mock_extract:
            mock_extract.return_value = "Clean text content"

            chunks = []
            async for chunk in stream_view(
                view, raw_html, mock_client, "http://test.com"
            ):
                chunks.append(chunk)

            # Should get general error message
            assert len(chunks) == 1
            error_content = chunks[0]
            assert "❌ Error" in error_content
            assert "Some general error occurred" in error_content

    @pytest.mark.asyncio
    async def test_stream_view_unexpected_error(self):
        """Test streaming with unexpected error."""
        view = View(
            id="summary",
            name="Summary",
            hotkey="2",
            description="Summary view",
            prompt="Summarize this",
        )
        raw_html = "<html><body>Test content</body></html>"

        # Mock client that raises unexpected error
        mock_client = AsyncMock()

        async def mock_stream_with_unexpected_error(prompt, content):
            raise ValueError("Unexpected error type")
            yield  # Make it an async generator (unreachable)

        mock_client.stream = mock_stream_with_unexpected_error

        # Mock extract_clean_text
        with patch("spegel.views.extract_clean_text") as mock_extract:
            mock_extract.return_value = "Clean text content"

            chunks = []
            async for chunk in stream_view(
                view, raw_html, mock_client, "http://test.com"
            ):
                chunks.append(chunk)

            # Should get unexpected error message
            assert len(chunks) == 1
            error_content = chunks[0]
            assert "❌ Unexpected Error" in error_content
            assert "Unexpected error type" in error_content

    @pytest.mark.asyncio
    async def test_stream_view_with_buffering(self):
        """Test streaming with content that requires buffering."""
        view = View(
            id="summary",
            name="Summary",
            hotkey="2",
            description="Summary view",
            prompt="Summarize this",
        )
        raw_html = "<html><body>Test content</body></html>"

        # Mock client
        mock_client = AsyncMock()

        # Mock the stream method to return chunks with and without newlines
        async def mock_stream(prompt, content):
            yield "Line 1\nLine 2\n"
            yield "Line 3"
            yield "\nLine 4"
            yield " remaining"

        mock_client.stream = mock_stream

        # Mock extract_clean_text
        with patch("spegel.views.extract_clean_text") as mock_extract:
            mock_extract.return_value = "Clean text content"

            chunks = []
            async for chunk in stream_view(
                view, raw_html, mock_client, "http://test.com"
            ):
                chunks.append(chunk)

            # Should properly handle buffering
            result = "".join(chunks)
            assert "Line 1\n" in result
            assert "Line 2\n" in result
            assert "Line 3\n" in result
            assert "Line 4 remaining" in result


class TestGetViewLLMClient:
    """Test the _get_view_llm_client function."""

    def test_get_view_llm_client_no_model_override(self):
        """Test getting client when view has no model override."""
        view = View(
            id="summary",
            name="Summary",
            hotkey="2",
            description="Summary view",
            prompt="Summarize this",
        )
        default_client = Mock()

        result = _get_view_llm_client(view, default_client)

        assert result == default_client

    def test_get_view_llm_client_with_model_override(self):
        """Test getting client when view has model override."""
        view = View(
            id="summary",
            name="Summary",
            hotkey="2",
            description="Summary view",
            prompt="Summarize this",
            model="gpt-4",
        )
        default_client = Mock()

        with patch("spegel.views.create_client") as mock_create_client:
            mock_view_client = Mock()
            mock_create_client.return_value = mock_view_client

            result = _get_view_llm_client(view, default_client)

            assert result == mock_view_client
            mock_create_client.assert_called_once_with(model="gpt-4")

    def test_get_view_llm_client_with_model_override_fails(self):
        """Test getting client when view model override fails."""
        view = View(
            id="summary",
            name="Summary",
            hotkey="2",
            description="Summary view",
            prompt="Summarize this",
            model="invalid-model",
        )
        default_client = Mock()

        with patch("spegel.views.create_client") as mock_create_client:
            mock_create_client.return_value = None  # Simulate failure

            result = _get_view_llm_client(view, default_client)

            # Should fall back to default client
            assert result == default_client
            mock_create_client.assert_called_once_with(model="invalid-model")

    def test_get_view_llm_client_with_empty_model_override(self):
        """Test getting client when view has empty model override."""
        view = View(
            id="summary",
            name="Summary",
            hotkey="2",
            description="Summary view",
            prompt="Summarize this",
            model="  ",
        )
        default_client = Mock()

        with patch("spegel.views.create_client") as mock_create_client:
            result = _get_view_llm_client(view, default_client)

            # Should not call create_client for empty model
            assert result == default_client
            mock_create_client.assert_not_called()
