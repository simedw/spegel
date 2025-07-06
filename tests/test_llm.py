import sys
from pathlib import Path
import os
from unittest.mock import Mock, patch

# Add project 'src' directory to sys.path so tests work without editable install
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pytest

from spegel.llm import get_default_client, GeminiClient, LLMClient


def test_get_default_client_no_api_key():
    """When no API key is set, should return None client."""
    with patch.dict(os.environ, {}, clear=True):
        client: LLMClient | None = get_default_client()
        assert client is None


def test_get_default_client_with_api_key():
    """When API key is set, should return GeminiClient."""
    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
        with patch("spegel.llm.genai") as mock_genai:
            mock_genai.Client.return_value = Mock()
            client: LLMClient | None = get_default_client()
            assert isinstance(client, GeminiClient)


def test_get_default_client_no_genai_module():
    """When genai module is not available, should return None."""
    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
        with patch("spegel.llm.genai", None):
            client: LLMClient | None = get_default_client()
            assert client is None


class TestGeminiClient:
    """Test GeminiClient functionality."""

    def test_init_without_genai(self):
        """Should raise RuntimeError if genai is not available."""
        with patch("spegel.llm.genai", None):
            with pytest.raises(RuntimeError, match="google-genai not installed"):
                GeminiClient("test-key")

    def test_init_with_genai(self):
        """Should initialize successfully with genai available."""
        with patch("spegel.llm.genai") as mock_genai:
            mock_client = Mock()
            mock_genai.Client.return_value = mock_client

            client = GeminiClient("test-key", "test-model")
            assert client._client == mock_client
            assert client.model_name == "test-model"
            mock_genai.Client.assert_called_once_with(api_key="test-key")

    @pytest.mark.asyncio
    async def test_stream_basic(self):
        """Test basic streaming functionality."""
        with patch("spegel.llm.genai") as mock_genai:
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

            mock_client.aio.models.generate_content_stream = (
                mock_generate_content_stream
            )

            client = GeminiClient("test-key")

            # Collect streamed chunks
            chunks = []
            async for chunk in client.stream("test prompt", "test content"):
                chunks.append(chunk)

            assert chunks == ["test response"]

    @pytest.mark.asyncio
    async def test_stream_with_empty_content(self):
        """Test streaming with empty content."""
        with patch("spegel.llm.genai") as mock_genai:
            mock_client = Mock()
            mock_genai.Client.return_value = mock_client

            # Mock empty streaming response
            async def mock_stream():
                return
                yield  # unreachable

            async def mock_generate_content_stream(*args, **kwargs):
                return mock_stream()

            mock_client.aio.models.generate_content_stream = (
                mock_generate_content_stream
            )

            client = GeminiClient("test-key")

            chunks = []
            async for chunk in client.stream("test prompt", ""):
                chunks.append(chunk)

            assert chunks == []

    @pytest.mark.asyncio
    async def test_stream_handles_exceptions(self):
        """Test that streaming handles chunk parsing exceptions gracefully."""
        with patch("spegel.llm.genai") as mock_genai:
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

            mock_client.aio.models.generate_content_stream = (
                mock_generate_content_stream
            )

            client = GeminiClient("test-key")

            chunks = []
            async for chunk in client.stream("test prompt", "test content"):
                chunks.append(chunk)

            # Should only get the good chunks, bad ones are skipped
            assert chunks == ["good", "good"]


class TestLLMClient:
    """Test the abstract LLMClient interface."""

    @pytest.mark.asyncio
    async def test_abstract_stream_method(self):
        """LLMClient.stream should raise NotImplementedError."""
        client = LLMClient()

        with pytest.raises(NotImplementedError):
            async for chunk in client.stream("test", "test"):
                pass


class TestLLMErrorScenarios:
    """Test error scenarios for LLM functionality."""

    def test_get_default_client_missing_environment(self):
        """Test client creation with missing environment variables."""
        with patch.dict(os.environ, {}, clear=True):
            client: LLMClient | None = get_default_client()
            assert client is None

    def test_get_default_client_invalid_api_key(self):
        """Test client creation with invalid API key format."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}):  # Empty key
            client: LLMClient | None = get_default_client()
            assert client is None

    def test_get_default_client_import_error(self):
        """Test graceful handling when google-genai is not installed."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "valid-key"}):
            with patch("spegel.llm.genai", None):
                client: LLMClient | None = get_default_client()
                assert client is None

    @pytest.mark.asyncio
    async def test_gemini_client_stream_network_error(self):
        """Test handling of network errors during streaming."""
        with patch("spegel.llm.genai") as mock_genai:
            mock_client = Mock()
            mock_genai.Client.return_value = mock_client

            # Mock network error during streaming
            async def mock_generate_content_stream(*args, **kwargs):
                raise Exception("Network error")

            mock_client.aio.models.generate_content_stream = (
                mock_generate_content_stream
            )

            client = GeminiClient("test-key")

            # Should handle error gracefully
            chunks = []
            try:
                async for chunk in client.stream("test prompt", "test content"):
                    chunks.append(chunk)  # pragma: no cover
            except Exception:
                # Exception is expected and handled by the test
                pass

            # Should not have collected any chunks due to error
            assert chunks == []

    @pytest.mark.asyncio
    async def test_gemini_client_stream_partial_failure(self):
        """Test handling of partial failures during streaming."""
        with patch("spegel.llm.genai") as mock_genai:
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

            mock_client.aio.models.generate_content_stream = (
                mock_generate_content_stream
            )

            client = GeminiClient("test-key")

            chunks = []
            async for chunk in client.stream("test prompt", "test content"):
                chunks.append(chunk)

            # Should get the successful chunk
            assert chunks == ["Success"]

    @pytest.mark.asyncio
    async def test_gemini_client_stream_malformed_response(self):
        """Test handling of malformed API responses."""
        with patch("spegel.llm.genai") as mock_genai:
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

            mock_client.aio.models.generate_content_stream = (
                mock_generate_content_stream
            )

            client = GeminiClient("test-key")

            chunks = []
            async for chunk in client.stream("test prompt", "test content"):
                chunks.append(chunk)

            # Should only get the valid chunk
            assert chunks == ["Valid"]

    @pytest.mark.asyncio
    async def test_gemini_client_stream_empty_prompt(self):
        """Test streaming with empty prompt."""
        with patch("spegel.llm.genai") as mock_genai:
            mock_client = Mock()
            mock_genai.Client.return_value = mock_client

            # Mock empty response for empty prompt
            async def mock_stream():
                return
                yield  # unreachable

            async def mock_generate_content_stream(*args, **kwargs):
                return mock_stream()

            mock_client.aio.models.generate_content_stream = (
                mock_generate_content_stream
            )

            client = GeminiClient("test-key")

            chunks = []
            async for chunk in client.stream("", ""):  # Empty prompt and content
                chunks.append(chunk)

            assert chunks == []

    def test_gemini_client_logging_error_handling(self):
        """Test that logging errors don't affect functionality."""
        with patch("spegel.llm.genai") as mock_genai:
            mock_client = Mock()
            mock_genai.Client.return_value = mock_client

            # Mock logging to raise exception during client creation
            with patch(
                "spegel.llm.logger.info", side_effect=Exception("Logging failed")
            ):
                # Should still create client despite logging errors
                client = GeminiClient("test-key")
                assert client._client == mock_client
                assert client.model_name == "gemini-2.5-flash-lite-preview-06-17"

    def test_enable_llm_logging_error_handling(self):
        """Test error handling in logging configuration."""
        # Test with invalid logging level
        from spegel.llm import enable_llm_logging

        # Should handle gracefully
        try:
            enable_llm_logging(level=999999)  # Invalid level
        except Exception as e:
            # Should not raise unhandled exceptions
            assert False, f"Logging setup should handle errors: {e}"
