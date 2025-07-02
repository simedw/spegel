"""Tests specific to the Gemini LLM provider."""

from unittest.mock import Mock, patch

import pytest

from spegel.llm.gemini import DEFAULT_MODEL, GeminiClient, is_available


class TestGeminiAvailability:
    """Test Gemini provider availability checks."""

    def test_is_available_when_imported(self):
        """Test availability check when google-genai is available."""
        with patch("spegel.llm.gemini.genai") as mock_genai:
            mock_genai.Client = Mock()
            # Import happens during module loading, so we need to test the function directly
            available = is_available()
            assert isinstance(available, bool)

    def test_is_available_when_not_imported(self):
        """Test availability check when google-genai is not available."""
        with patch("spegel.llm.gemini.GEMINI_AVAILABLE", False):
            available = is_available()
            assert available is False


class TestGeminiClient:
    """Test GeminiClient functionality."""

    def test_init_without_genai(self):
        """Should raise RuntimeError if genai is not available."""
        with patch("spegel.llm.gemini.GEMINI_AVAILABLE", False):
            with pytest.raises(RuntimeError, match="google-genai not installed but GeminiClient requested"):
                GeminiClient("test-key")

    def test_init_with_genai(self):
        """Should initialize successfully with genai available."""
        with patch("spegel.llm.gemini.genai") as mock_genai:
            mock_client = Mock()
            mock_genai.Client.return_value = mock_client

            client = GeminiClient("test-key", "test-model")
            assert client._client == mock_client
            assert client.model_name == "test-model"  # Test backwards compatibility property
            assert client.model == "test-model"
            mock_genai.Client.assert_called_once_with(api_key="test-key")

    def test_default_model(self):
        """Test default model assignment."""
        with patch("spegel.llm.gemini.genai") as mock_genai:
            mock_client = Mock()
            mock_genai.Client.return_value = mock_client

            client = GeminiClient("test-key")
            assert "gemini" in client.model.lower()

    def test_custom_parameters(self):
        """Test initialization with custom parameters."""
        with patch("spegel.llm.gemini.genai") as mock_genai:
            mock_client = Mock()
            mock_genai.Client.return_value = mock_client

            client = GeminiClient(api_key="test-key", model="custom-model", temperature=0.8, max_tokens=4096)

            assert client.model == "custom-model"
            assert client.temperature == 0.8
            assert client.max_tokens == 4096

    @pytest.mark.asyncio
    async def test_stream_basic(self):
        """Test basic streaming functionality."""
        with patch("spegel.llm.gemini.genai") as mock_genai:
            # Mock the client and streaming response
            mock_client = Mock()
            mock_genai.Client.return_value = mock_client

            # Mock the streaming response
            mock_chunk = Mock()
            mock_chunk.candidates = [Mock()]
            mock_chunk.candidates[0].content.parts = [Mock()]
            mock_chunk.candidates[0].content.parts[0].text = "test response"

            async def mock_stream():
                yield mock_chunk

            async def mock_generate_content_stream(*args, **kwargs):
                return mock_stream()

            mock_client.aio.models.generate_content_stream = mock_generate_content_stream

            client = GeminiClient("test-key")

            # Collect streamed chunks
            chunks = []
            async for chunk in client.stream("test prompt", "test content"):
                chunks.append(chunk)

            assert chunks == ["test response"]

    @pytest.mark.asyncio
    async def test_stream_with_empty_content(self):
        """Test streaming with empty content."""
        with patch("spegel.llm.gemini.genai") as mock_genai:
            mock_client = Mock()
            mock_genai.Client.return_value = mock_client

            # Mock empty streaming response
            async def mock_stream():
                return
                yield  # unreachable

            async def mock_generate_content_stream(*args, **kwargs):
                return mock_stream()

            mock_client.aio.models.generate_content_stream = mock_generate_content_stream

            client = GeminiClient("test-key")

            chunks = []
            async for chunk in client.stream("test prompt", ""):
                chunks.append(chunk)

            assert chunks == []

    @pytest.mark.asyncio
    async def test_stream_handles_exceptions(self):
        """Test that streaming handles chunk parsing exceptions gracefully."""
        with patch("spegel.llm.gemini.genai") as mock_genai:
            mock_client = Mock()
            mock_genai.Client.return_value = mock_client

            # Mock chunks with some that will raise exceptions
            good_chunk = Mock()
            good_chunk.candidates = [Mock()]
            good_chunk.candidates[0].content.parts = [Mock()]
            good_chunk.candidates[0].content.parts[0].text = "good"

            bad_chunk = Mock()
            bad_chunk.candidates = []  # This will cause IndexError

            async def mock_stream():
                yield good_chunk
                yield bad_chunk
                yield good_chunk

            async def mock_generate_content_stream(*args, **kwargs):
                return mock_stream()

            mock_client.aio.models.generate_content_stream = mock_generate_content_stream

            client = GeminiClient("test-key")

            chunks = []
            async for chunk in client.stream("test prompt", "test content"):
                chunks.append(chunk)

            # Should only get the good chunks, bad ones are skipped
            assert chunks == ["good", "good"]

    @pytest.mark.asyncio
    async def test_stream_network_error(self):
        """Test handling of network errors during streaming."""
        with patch("spegel.llm.gemini.genai") as mock_genai:
            mock_client = Mock()
            mock_genai.Client.return_value = mock_client

            # Mock network error during streaming
            async def mock_generate_content_stream(*args, **kwargs):
                raise Exception("Network error")

            mock_client.aio.models.generate_content_stream = mock_generate_content_stream

            client = GeminiClient("test-key")

            # Should propagate the error
            with pytest.raises(Exception, match="Network error"):
                async for chunk in client.stream("test prompt", "test content"):
                    pass

    @pytest.mark.asyncio
    async def test_stream_partial_failure(self):
        """Test handling of partial failures during streaming."""
        with patch("spegel.llm.gemini.genai") as mock_genai:
            mock_client = Mock()
            mock_genai.Client.return_value = mock_client

            # Mock chunks with some failures
            async def mock_stream():
                # First chunk succeeds
                chunk = Mock()
                chunk.candidates = [Mock()]
                chunk.candidates[0].content.parts = [Mock()]
                chunk.candidates[0].content.parts[0].text = "Success"
                yield chunk

            async def mock_generate_content_stream(*args, **kwargs):
                return mock_stream()

            mock_client.aio.models.generate_content_stream = mock_generate_content_stream

            client = GeminiClient("test-key")

            chunks = []
            async for chunk in client.stream("test prompt", "test content"):
                chunks.append(chunk)

            # Should get the successful chunk
            assert chunks == ["Success"]

    @pytest.mark.asyncio
    async def test_stream_malformed_response(self):
        """Test handling of malformed API responses."""
        with patch("spegel.llm.gemini.genai") as mock_genai:
            mock_client = Mock()
            mock_genai.Client.return_value = mock_client

            # Mock malformed response chunks
            async def mock_stream():
                # Chunk with missing structure
                chunk1 = Mock()
                chunk1.candidates = []  # Empty candidates
                yield chunk1

                # Chunk with None text
                chunk2 = Mock()
                chunk2.candidates = [Mock()]
                chunk2.candidates[0].content.parts = [Mock()]
                chunk2.candidates[0].content.parts[0].text = None
                yield chunk2

                # Valid chunk
                chunk3 = Mock()
                chunk3.candidates = [Mock()]
                chunk3.candidates[0].content.parts = [Mock()]
                chunk3.candidates[0].content.parts[0].text = "Valid"
                yield chunk3

            async def mock_generate_content_stream(*args, **kwargs):
                return mock_stream()

            mock_client.aio.models.generate_content_stream = mock_generate_content_stream

            client = GeminiClient("test-key")

            chunks = []
            async for chunk in client.stream("test prompt", "test content"):
                chunks.append(chunk)

            # Should only get the valid chunk
            assert chunks == ["Valid"]

    @pytest.mark.asyncio
    async def test_stream_empty_prompt(self):
        """Test streaming with empty prompt."""
        with patch("spegel.llm.gemini.genai") as mock_genai:
            mock_client = Mock()
            mock_genai.Client.return_value = mock_client

            # Mock empty response for empty prompt
            async def mock_stream():
                return
                yield  # unreachable

            async def mock_generate_content_stream(*args, **kwargs):
                return mock_stream()

            mock_client.aio.models.generate_content_stream = mock_generate_content_stream

            client = GeminiClient("test-key")

            chunks = []
            async for chunk in client.stream("", ""):  # Empty prompt and content
                chunks.append(chunk)

            assert chunks == []

    def test_gemini_client_logging_error_handling(self):
        """Test that logging errors don't affect functionality."""
        with patch("spegel.llm.gemini.genai") as mock_genai:
            mock_client = Mock()
            mock_genai.Client.return_value = mock_client

            # Mock logging to raise exception during client creation
            with patch("spegel.llm.logger.info", side_effect=Exception("Logging failed")):
                # Should still create client despite logging errors
                client = GeminiClient("test-key")
                assert client._client == mock_client
                assert client.model == DEFAULT_MODEL

    def test_model_name_property_backwards_compatibility(self):
        """Test that model_name property works for backwards compatibility."""
        with patch("spegel.llm.gemini.genai") as mock_genai:
            mock_client = Mock()
            mock_genai.Client.return_value = mock_client

            client = GeminiClient("test-key", "test-model")

            # Both should work
            assert client.model == "test-model"
            assert client.model_name == "test-model"
