import os
from unittest.mock import Mock, patch

import pytest

from spegel.llm import LiteLLMClient, LLMClient, create_client


def test_create_client_no_api_key():
    """When no API key is set, should return None client."""
    with patch.dict(os.environ, {}, clear=True):
        with patch("os.path.exists", return_value=False):  # No Ollama binary
            # Mock LiteLLMClient to raise exception (simulating no API key)
            with patch("spegel.llm.LiteLLMClient", side_effect=Exception("No API key")):
                client = create_client("test-model")
                assert client is None


def test_create_client_with_specific_model():
    """When specific model is provided, should return LiteLLMClient with that model."""
    with patch("spegel.llm.litellm"):
        client = create_client("gpt-4o-mini")
        assert isinstance(client, LiteLLMClient)
        assert client.model == "gpt-4o-mini"


def test_create_client_with_custom_model():
    """When custom model is set, should return LiteLLMClient with custom config."""
    with patch.dict(
        os.environ, {"SPEGEL_MODEL": "custom-model", "SPEGEL_API_KEY": "test-key"}
    ):
        with patch("spegel.llm.litellm") as mock_litellm:
            client = create_client("should-be-overridden")
            assert isinstance(client, LiteLLMClient)
            assert client.model == "custom-model"


def test_create_client_no_litellm_module():
    """When litellm module is not available, should return None."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        with patch("spegel.llm.litellm", None):
            client = create_client("test-model")
            assert client is None


class TestLiteLLMClient:
    """Test LiteLLMClient functionality."""

    def test_init_without_litellm(self):
        """Should raise RuntimeError if litellm is not available."""
        with patch("spegel.llm.litellm", None):
            with pytest.raises(RuntimeError, match="litellm not installed"):
                LiteLLMClient("test-model")

    def test_init_with_litellm(self):
        """Should initialize successfully with litellm available."""
        with patch("spegel.llm.litellm") as mock_litellm:
            client = LiteLLMClient("test-model", "test-key")
            assert client.model == "test-model"
            assert client.api_key == "test-key"

    @pytest.mark.asyncio
    async def test_stream_basic(self):
        """Test basic streaming functionality."""
        with patch("spegel.llm.litellm") as mock_litellm:
            # Mock the streaming response
            mock_chunk = Mock()
            mock_chunk.choices = [Mock()]
            mock_chunk.choices[0].delta = Mock()
            mock_chunk.choices[0].delta.content = "test response"

            async def mock_stream():
                yield mock_chunk

            # acompletion should return a coroutine that resolves to an async generator
            async def async_completion_mock(**kwargs):
                return mock_stream()

            mock_litellm.acompletion = async_completion_mock

            client = LiteLLMClient("test-model", "test-key")

            # Collect streamed chunks
            chunks = []
            async for chunk in client.stream("test prompt", "test content"):
                chunks.append(chunk)

            assert chunks == ["test response"]

    @pytest.mark.asyncio
    async def test_stream_with_empty_content(self):
        """Test streaming with empty content."""
        with patch("spegel.llm.litellm") as mock_litellm:
            # Mock empty streaming response
            async def mock_stream():
                return
                yield  # unreachable

            async def async_completion_mock(**kwargs):
                return mock_stream()

            mock_litellm.acompletion = async_completion_mock

            client = LiteLLMClient("test-model", "test-key")

            chunks = []
            async for chunk in client.stream("test prompt", ""):
                chunks.append(chunk)

            assert chunks == []

    @pytest.mark.asyncio
    async def test_stream_without_litellm(self):
        """Test streaming when litellm is not available."""
        with patch("spegel.llm.litellm") as mock_litellm:
            client = LiteLLMClient("test-model", "test-key")

            # Patch litellm to None after client creation to simulate unavailability
            with patch("spegel.llm.litellm", None):
                with pytest.raises(RuntimeError, match="litellm not available"):
                    async for chunk in client.stream("test prompt", "test content"):
                        pass

    @pytest.mark.asyncio
    async def test_stream_completion_error(self):
        """Test handling of completion errors."""
        with patch("spegel.llm.litellm") as mock_litellm:
            mock_litellm.acompletion.side_effect = Exception("API Error")

            client = LiteLLMClient("test-model", "test-key")

            with pytest.raises(Exception, match="API Error"):
                async for chunk in client.stream("test prompt", "test content"):
                    pass

    @pytest.mark.asyncio
    async def test_stream_malformed_response(self):
        """Test handling of malformed API responses."""
        with patch("spegel.llm.litellm") as mock_litellm:
            # Mock malformed response chunks
            async def mock_stream():
                # Chunk with empty choices
                chunk1 = Mock()
                chunk1.choices = []
                yield chunk1

                # Chunk with None content
                chunk2 = Mock()
                chunk2.choices = [Mock()]
                chunk2.choices[0].delta = Mock()
                chunk2.choices[0].delta.content = None
                yield chunk2

                # Valid chunk
                chunk3 = Mock()
                chunk3.choices = [Mock()]
                chunk3.choices[0].delta = Mock()
                chunk3.choices[0].delta.content = "Valid"
                yield chunk3

            async def async_completion_mock(**kwargs):
                return mock_stream()

            mock_litellm.acompletion = async_completion_mock

            client = LiteLLMClient("test-model", "test-key")

            chunks = []
            async for chunk in client.stream("test prompt", "test content"):
                chunks.append(chunk)

            # Should only get the valid chunk
            assert chunks == ["Valid"]


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

    def test_create_client_missing_environment(self):
        """Test client creation with missing environment variables."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("os.path.exists", return_value=False):  # No Ollama binary
                # Mock LiteLLMClient to raise exception (simulating no API key)
                with patch(
                    "spegel.llm.LiteLLMClient", side_effect=Exception("No API key")
                ):
                    client = create_client("test-model")
                    assert client is None

    @pytest.mark.asyncio
    async def test_litellm_client_stream_network_error(self):
        """Test handling of network errors during streaming."""
        with patch("spegel.llm.litellm") as mock_litellm:
            mock_litellm.acompletion.side_effect = Exception("Network error")

            client = LiteLLMClient("test-model", "test-key")

            # Should raise the exception
            with pytest.raises(Exception, match="Network error"):
                async for chunk in client.stream("test prompt", "test content"):
                    pass

    @pytest.mark.asyncio
    async def test_litellm_client_stream_partial_failure(self):
        """Test handling of partial failures during streaming."""
        with patch("spegel.llm.litellm") as mock_litellm:
            # Mock chunks with some failures
            async def mock_stream():
                # First chunk succeeds
                chunk = Mock()
                chunk.choices = [Mock()]
                chunk.choices[0].delta = Mock()
                chunk.choices[0].delta.content = "Success"
                yield chunk

                # Second chunk fails in processing
                chunk2 = Mock()
                chunk2.choices = [Mock()]
                chunk2.choices[0].delta = Mock()
                chunk2.choices[0].delta.content = None  # This will be skipped
                yield chunk2

            async def async_completion_mock(**kwargs):
                return mock_stream()

            mock_litellm.acompletion = async_completion_mock

            client = LiteLLMClient("test-model", "test-key")

            chunks = []
            async for chunk in client.stream("test prompt", "test content"):
                chunks.append(chunk)

            # Should get only the successful chunk
            assert chunks == ["Success"]

    @pytest.mark.asyncio
    async def test_litellm_client_stream_empty_prompt(self):
        """Test streaming with empty prompt."""
        with patch("spegel.llm.litellm") as mock_litellm:
            # Mock empty response for empty prompt
            async def mock_stream():
                return
                yield  # unreachable

            async def async_completion_mock(**kwargs):
                return mock_stream()

            mock_litellm.acompletion = async_completion_mock

            client = LiteLLMClient("test-model", "test-key")

            chunks = []
            async for chunk in client.stream("", ""):  # Empty prompt and content
                chunks.append(chunk)

            assert chunks == []

    def test_litellm_client_logging_error_handling(self):
        """Test that logging errors don't affect functionality."""
        with patch("spegel.llm.litellm") as mock_litellm:
            # Mock logging to raise exception during client creation
            with patch(
                "spegel.llm.logger.info", side_effect=Exception("Logging failed")
            ):
                # Should still create client despite logging errors
                client = LiteLLMClient("test-model", "test-key")
                assert client.model == "test-model"
                assert client.api_key == "test-key"

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
