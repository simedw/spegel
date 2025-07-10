import os
from unittest.mock import Mock, patch

import pytest

from spegel.llm import LiteLLMClient, LLMClient, create_client, LLMAuthenticationError


@pytest.fixture
def mock_litellm():
    """Provides a properly configured litellm mock for testing."""
    with patch("spegel.llm.litellm") as mock_litellm:
        # Set up the authentication error class once
        class MockAuthenticationError(Exception):
            pass

        mock_litellm.AuthenticationError = MockAuthenticationError
        yield mock_litellm


def test_create_client_no_api_key():
    """When no API key is set, should return None client."""
    with patch.dict(os.environ, {}, clear=True):
        with patch("os.path.exists", return_value=False):  # No Ollama binary
            # Mock LiteLLMClient to raise exception (simulating no API key)
            with patch("spegel.llm.LiteLLMClient", side_effect=Exception("No API key")):
                client = create_client("test-model")
                assert client is None


def test_create_client_with_specific_model(mock_litellm):
    """When specific model is provided, should return LiteLLMClient with that model."""
    client = create_client("gpt-4o-mini")
    assert isinstance(client, LiteLLMClient)
    assert client.model == "gpt-4o-mini"


def test_create_client_with_custom_model():
    """When SPEGEL_MODEL is set, should use it with custom config."""
    with patch.dict(
        os.environ, {"SPEGEL_MODEL": "custom-model", "SPEGEL_API_KEY": "test-key"}
    ):
        with patch("spegel.llm.LiteLLMClient") as mock_client:
            mock_instance = Mock()
            mock_client.return_value = mock_instance

            client = create_client("default-model")

            # Should use the custom model from environment
            mock_client.assert_called_once_with(
                model="custom-model", api_key="test-key", api_base=None
            )
            assert client == mock_instance


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

    def test_init_with_litellm(self, mock_litellm):
        """Should initialize successfully with litellm available."""
        client = LiteLLMClient("test-model", "test-key")
        assert client.model == "test-model"
        assert client.api_key == "test-key"

    @pytest.mark.asyncio
    async def test_stream_basic(self, mock_litellm):
        """Test basic streaming functionality."""
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
    async def test_stream_authentication_error(self, mock_litellm):
        """Test handling of authentication errors with user-friendly message."""
        # Set up the mock exception
        mock_auth_error = mock_litellm.AuthenticationError("API key not valid")
        mock_litellm.acompletion.side_effect = mock_auth_error

        client = LiteLLMClient("openai/gpt-4", "invalid-key")

        with pytest.raises(LLMAuthenticationError) as exc_info:
            async for chunk in client.stream("test prompt", "test content"):
                pass

        error = exc_info.value
        assert error.model == "openai/gpt-4"
        assert error.provider == "openai"
        assert error.original_error == mock_auth_error

        error_message = str(error)
        assert "Authentication failed for model 'openai/gpt-4'" in error_message
        assert "Please set a valid API key for openai" in error_message
        assert "SPEGEL_API_KEY environment variable" in error_message
        assert "OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY" in error_message

    @pytest.mark.asyncio
    async def test_stream_authentication_error_simple_model(self, mock_litellm):
        """Test authentication error handling for simple model names."""
        # Set up the mock exception
        mock_auth_error = mock_litellm.AuthenticationError("API key not valid")
        mock_litellm.acompletion.side_effect = mock_auth_error

        client = LiteLLMClient("gpt-4", "invalid-key")

        with pytest.raises(LLMAuthenticationError) as exc_info:
            async for chunk in client.stream("test prompt", "test content"):
                pass

        error = exc_info.value
        assert error.model == "gpt-4"
        assert error.provider == "gpt-4"  # For simple model names, provider == model

        error_message = str(error)
        assert "Authentication failed for model 'gpt-4'" in error_message
        assert "Please set a valid API key for gpt-4" in error_message

    @pytest.mark.asyncio
    async def test_stream_with_empty_content(self, mock_litellm):
        """Test streaming with empty content."""

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
    async def test_stream_completion_error(self, mock_litellm):
        """Test handling of completion errors."""
        mock_litellm.acompletion.side_effect = Exception("API Error")

        client = LiteLLMClient("test-model", "test-key")

        with pytest.raises(Exception, match="API Error"):
            async for chunk in client.stream("test prompt", "test content"):
                pass

    @pytest.mark.asyncio
    async def test_stream_malformed_response(self, mock_litellm):
        """Test handling of malformed API responses."""

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
    async def test_litellm_client_stream_network_error(self, mock_litellm):
        """Test handling of network errors during streaming."""
        mock_litellm.acompletion.side_effect = Exception("Network error")

        client = LiteLLMClient("test-model", "test-key")

        # Should raise the exception
        with pytest.raises(Exception, match="Network error"):
            async for chunk in client.stream("test prompt", "test content"):
                pass

    @pytest.mark.asyncio
    async def test_litellm_client_stream_partial_failure(self, mock_litellm):
        """Test handling of partial failures during streaming."""

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
    async def test_litellm_client_stream_empty_prompt(self, mock_litellm):
        """Test streaming with empty prompt."""

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

    def test_litellm_client_logging_error_handling(self, mock_litellm):
        """Test that logging errors don't affect functionality."""
        # Mock logging to raise exception during client creation
        with patch("spegel.llm.logger.info", side_effect=Exception("Logging failed")):
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
