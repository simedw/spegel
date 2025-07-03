"""Tests specific to the OpenAI LLM provider."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from spegel.llm.openai import DEFAULT_MODEL, OpenAIClient, is_available


class TestOpenAIAvailability:
    """Test OpenAI provider availability checks."""

    def test_is_available_when_imported(self):
        """Test availability check when openai is available."""
        with patch("spegel.llm.openai.AsyncOpenAI") as mock_openai:
            mock_openai.return_value = Mock()
            # Import happens during module loading, so we need to test the function directly
            available = is_available()
            assert isinstance(available, bool)

    def test_is_available_when_not_imported(self):
        """Test availability check when openai is not available."""
        with patch("spegel.llm.openai.OPENAI_AVAILABLE", False):
            available = is_available()
            assert available is False


class TestOpenAIClient:
    """Test OpenAIClient functionality."""

    def test_init_without_openai(self):
        """Should raise RuntimeError if openai is not available."""
        with patch("spegel.llm.openai.OPENAI_AVAILABLE", False):
            with pytest.raises(RuntimeError, match="openai package not installed but OpenAIClient requested"):
                OpenAIClient("test-key")

    def test_init_with_openai(self):
        """Should initialize successfully with openai available."""
        with patch("spegel.llm.openai.AsyncOpenAI") as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client

            client = OpenAIClient("test-key", "test-model")
            assert client._client == mock_client
            assert client.model == "test-model"
            mock_openai.assert_called_once_with(api_key="test-key", base_url=None)

    def test_init_with_base_url(self):
        """Test initialization with custom base URL."""
        with patch("spegel.llm.openai.AsyncOpenAI") as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client

            client = OpenAIClient("test-key", "test-model", base_url="https://custom.openai.com")
            assert client._client == mock_client
            mock_openai.assert_called_once_with(api_key="test-key", base_url="https://custom.openai.com")

    def test_init_failure(self):
        """Test initialization failure handling."""
        with patch("spegel.llm.openai.AsyncOpenAI") as mock_openai:
            mock_openai.side_effect = Exception("OpenAI client failed")

            with pytest.raises(RuntimeError, match="Failed to initialize OpenAI client"):
                OpenAIClient("test-key")

    def test_default_model(self):
        """Test default model assignment."""
        with patch("spegel.llm.openai.AsyncOpenAI") as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client

            client = OpenAIClient("test-key")
            assert "gpt" in client.model.lower()

    def test_custom_parameters(self):
        """Test initialization with custom parameters."""
        with patch("spegel.llm.openai.AsyncOpenAI") as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client

            client = OpenAIClient(api_key="test-key", model="gpt-4", temperature=0.8, max_tokens=4096)

            assert client.model == "gpt-4"
            assert client.temperature == 0.8
            assert client.max_tokens == 4096

    @pytest.mark.asyncio
    async def test_stream_basic(self):
        """Test basic streaming functionality."""
        with patch("spegel.llm.openai.AsyncOpenAI") as mock_openai:
            # Mock the client and streaming response
            mock_client = Mock()
            mock_openai.return_value = mock_client

            # Mock the streaming response chunks
            mock_chunk1 = Mock()
            mock_chunk1.choices = [Mock()]
            mock_chunk1.choices[0].delta.content = "Hello"

            mock_chunk2 = Mock()
            mock_chunk2.choices = [Mock()]
            mock_chunk2.choices[0].delta.content = " World"

            mock_chunk3 = Mock()
            mock_chunk3.choices = [Mock()]
            mock_chunk3.choices[0].delta.content = None  # End of stream

            async def mock_stream():
                yield mock_chunk1
                yield mock_chunk2
                yield mock_chunk3

            mock_client.chat.completions.create = AsyncMock(return_value=mock_stream())

            client = OpenAIClient("test-key")

            # Collect streamed chunks
            chunks = []
            async for chunk in client.stream("test prompt", "test content"):
                chunks.append(chunk)

            assert chunks == ["Hello", " World"]

    @pytest.mark.asyncio
    async def test_stream_system_and_user_messages(self):
        """Test streaming with both system and user messages."""
        with patch("spegel.llm.openai.AsyncOpenAI") as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client

            async def mock_stream():
                return
                yield  # unreachable

            mock_client.chat.completions.create = AsyncMock(return_value=mock_stream())

            client = OpenAIClient("test-key")

            # Test with both prompt and content
            chunks = []
            async for chunk in client.stream("system prompt", "user content"):
                chunks.append(chunk)

            # Verify the messages were formatted correctly
            mock_client.chat.completions.create.assert_called_once()
            call_args = mock_client.chat.completions.create.call_args
            messages = call_args[1]["messages"]

            assert len(messages) == 2
            assert messages[0]["role"] == "system"
            assert messages[0]["content"] == "system prompt"
            assert messages[1]["role"] == "user"
            assert messages[1]["content"] == "user content"

    @pytest.mark.asyncio
    async def test_stream_prompt_only(self):
        """Test streaming with only prompt (no content)."""
        with patch("spegel.llm.openai.AsyncOpenAI") as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client

            async def mock_stream():
                return
                yield  # unreachable

            mock_client.chat.completions.create = AsyncMock(return_value=mock_stream())

            client = OpenAIClient("test-key")

            # Test with only prompt, no content
            chunks = []
            async for chunk in client.stream("user prompt", ""):
                chunks.append(chunk)

            # Verify the message was formatted as user message only
            mock_client.chat.completions.create.assert_called_once()
            call_args = mock_client.chat.completions.create.call_args
            messages = call_args[1]["messages"]

            assert len(messages) == 1
            assert messages[0]["role"] == "user"
            assert messages[0]["content"] == "user prompt"

    @pytest.mark.asyncio
    async def test_stream_with_kwargs(self):
        """Test streaming with additional keyword arguments."""
        with patch("spegel.llm.openai.AsyncOpenAI") as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client

            async def mock_stream():
                return
                yield  # unreachable

            mock_client.chat.completions.create = AsyncMock(return_value=mock_stream())

            client = OpenAIClient("test-key")

            # Test with additional kwargs
            chunks = []
            async for chunk in client.stream("prompt", "content", presence_penalty=0.5, seed=42):
                chunks.append(chunk)

            # Verify the kwargs were passed through
            mock_client.chat.completions.create.assert_called_once()
            call_args = mock_client.chat.completions.create.call_args

            assert call_args[1]["presence_penalty"] == 0.5
            assert call_args[1]["seed"] == 42

    @pytest.mark.asyncio
    async def test_stream_empty_choices(self):
        """Test handling of chunks with empty choices."""
        with patch("spegel.llm.openai.AsyncOpenAI") as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client

            # Mock chunks with empty choices
            mock_chunk1 = Mock()
            mock_chunk1.choices = []  # Empty choices

            mock_chunk2 = Mock()
            mock_chunk2.choices = [Mock()]
            mock_chunk2.choices[0].delta.content = "Valid content"

            async def mock_stream():
                yield mock_chunk1
                yield mock_chunk2

            mock_client.chat.completions.create = AsyncMock(return_value=mock_stream())

            client = OpenAIClient("test-key")

            chunks = []
            async for chunk in client.stream("test prompt", "test content"):
                chunks.append(chunk)

            # Should only get content from valid chunks
            assert chunks == ["Valid content"]

    @pytest.mark.asyncio
    async def test_stream_network_error(self):
        """Test handling of network errors during streaming."""
        with patch("spegel.llm.openai.AsyncOpenAI") as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client

            # Mock network error during streaming
            mock_client.chat.completions.create = AsyncMock(side_effect=Exception("Network error"))

            client = OpenAIClient("test-key")

            # Should propagate the error
            with pytest.raises(Exception, match="Network error"):
                async for chunk in client.stream("test prompt", "test content"):
                    pass

    @pytest.mark.asyncio
    async def test_stream_api_error(self):
        """Test handling of OpenAI API errors."""
        with patch("spegel.llm.openai.AsyncOpenAI") as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client

            # Mock API error
            mock_client.chat.completions.create = AsyncMock(side_effect=Exception("Rate limit exceeded"))

            client = OpenAIClient("test-key")

            # Should propagate the error
            with pytest.raises(Exception, match="Rate limit exceeded"):
                async for chunk in client.stream("test prompt", "test content"):
                    pass

    @pytest.mark.asyncio
    async def test_stream_parameters_passed_correctly(self):
        """Test that all streaming parameters are passed correctly."""
        with patch("spegel.llm.openai.AsyncOpenAI") as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client

            async def mock_stream():
                return
                yield  # unreachable

            mock_client.chat.completions.create = AsyncMock(return_value=mock_stream())

            client = OpenAIClient("test-key", "custom-model", 0.7, 2048)

            chunks = []
            async for chunk in client.stream("test prompt", "test content"):
                chunks.append(chunk)

            # Verify all parameters were passed correctly
            mock_client.chat.completions.create.assert_called_once()
            call_args = mock_client.chat.completions.create.call_args

            assert call_args[1]["model"] == "custom-model"
            assert call_args[1]["temperature"] == 0.7
            assert call_args[1]["max_tokens"] == 2048
            assert call_args[1]["stream"] is True

    def test_openai_client_logging_error_handling(self):
        """Test that logging errors don't affect functionality."""
        with patch("spegel.llm.openai.AsyncOpenAI") as mock_openai:
            mock_client = Mock()

            mock_openai.return_value = mock_client

            # Mock logging to raise exception during client creation
            with patch("spegel.llm.logger.info", side_effect=Exception("Logging failed")):
                # Should still create client despite logging errors
                client = OpenAIClient("test-key")
                assert client._client == mock_client
                assert client.model == DEFAULT_MODEL
