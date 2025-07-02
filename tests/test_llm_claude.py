"""Tests specific to the Anthropic Claude LLM provider."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from spegel.llm.claude import DEFAULT_MODEL, ClaudeClient, is_available


class TestClaudeAvailability:
    """Test Claude provider availability checks."""

    def test_is_available_when_imported(self):
        """Test availability check when anthropic is available."""
        with patch("spegel.llm.claude.AsyncAnthropic") as mock_anthropic:
            mock_anthropic.return_value = Mock()
            # Import happens during module loading, so we need to test the function directly
            available = is_available()
            assert isinstance(available, bool)

    def test_is_available_when_not_imported(self):
        """Test availability check when anthropic is not available."""
        with patch("spegel.llm.claude.CLAUDE_AVAILABLE", False):
            available = is_available()
            assert available is False


class TestClaudeClient:
    """Test ClaudeClient functionality."""

    def test_init_without_anthropic(self):
        """Should raise RuntimeError if anthropic is not available."""
        with patch("spegel.llm.claude.CLAUDE_AVAILABLE", False):
            with pytest.raises(RuntimeError, match="anthropic package not installed but ClaudeClient requested"):
                ClaudeClient("test-key")

    def test_init_with_anthropic(self):
        """Should initialize successfully with anthropic available."""
        with patch("spegel.llm.claude.AsyncAnthropic") as mock_anthropic:
            mock_client = Mock()
            mock_anthropic.return_value = mock_client

            client = ClaudeClient("test-key", "test-model")
            assert client._client == mock_client
            assert client.model == "test-model"
            mock_anthropic.assert_called_once_with(api_key="test-key")

    def test_init_failure(self):
        """Test initialization failure handling."""
        with patch("spegel.llm.claude.AsyncAnthropic") as mock_anthropic:
            mock_anthropic.side_effect = Exception("Anthropic client failed")

            with pytest.raises(RuntimeError, match="Failed to initialize Claude client"):
                ClaudeClient("test-key")

    def test_default_model(self):
        """Test default model assignment."""
        with patch("spegel.llm.claude.AsyncAnthropic") as mock_anthropic:
            mock_client = Mock()
            mock_anthropic.return_value = mock_client

            client = ClaudeClient("test-key")
            assert "claude" in client.model.lower()

    def test_custom_parameters(self):
        """Test initialization with custom parameters."""
        with patch("spegel.llm.claude.AsyncAnthropic") as mock_anthropic:
            mock_client = Mock()
            mock_anthropic.return_value = mock_client

            client = ClaudeClient(api_key="test-key", model="claude-3-opus-20240229", temperature=0.8, max_tokens=4096)

            assert client.model == "claude-3-opus-20240229"
            assert client.temperature == 0.8
            assert client.max_tokens == 4096

    @pytest.mark.asyncio
    async def test_stream_basic(self):
        """Test basic streaming functionality."""
        with patch("spegel.llm.claude.AsyncAnthropic") as mock_anthropic:
            # Mock the client and streaming response
            mock_client = Mock()
            mock_anthropic.return_value = mock_client

            # Mock the streaming response chunks
            mock_chunk1 = Mock()
            mock_chunk1.type = "content_block_delta"
            mock_chunk1.delta.text = "Hello"

            mock_chunk2 = Mock()
            mock_chunk2.type = "content_block_delta"
            mock_chunk2.delta.text = " World"

            mock_chunk3 = Mock()
            mock_chunk3.type = "message_stop"  # End of stream

            async def mock_stream():
                yield mock_chunk1
                yield mock_chunk2
                yield mock_chunk3

            mock_client.messages.create = AsyncMock(return_value=mock_stream())

            client = ClaudeClient("test-key")

            # Collect streamed chunks
            chunks = []
            async for chunk in client.stream("test prompt", "test content"):
                chunks.append(chunk)

            assert chunks == ["Hello", " World"]

    @pytest.mark.asyncio
    async def test_stream_system_and_user_messages(self):
        """Test streaming with both system and user messages."""
        with patch("spegel.llm.claude.AsyncAnthropic") as mock_anthropic:
            mock_client = Mock()
            mock_anthropic.return_value = mock_client

            async def mock_stream():
                return
                yield  # unreachable

            mock_client.messages.create = AsyncMock(return_value=mock_stream())

            client = ClaudeClient("test-key")

            # Test with both prompt and content
            chunks = []
            async for chunk in client.stream("system prompt", "user content"):
                chunks.append(chunk)

            # Verify the messages were formatted correctly
            mock_client.messages.create.assert_called_once()
            call_args = mock_client.messages.create.call_args

            # Claude uses system parameter separately
            assert call_args[1]["system"] == "system prompt"
            messages = call_args[1]["messages"]
            assert len(messages) == 1
            assert messages[0]["role"] == "user"
            assert messages[0]["content"] == "user content"

    @pytest.mark.asyncio
    async def test_stream_prompt_only(self):
        """Test streaming with only prompt (no content)."""
        with patch("spegel.llm.claude.AsyncAnthropic") as mock_anthropic:
            mock_client = Mock()
            mock_anthropic.return_value = mock_client

            async def mock_stream():
                return
                yield  # unreachable

            mock_client.messages.create = AsyncMock(return_value=mock_stream())

            client = ClaudeClient("test-key")

            # Test with only prompt, no content
            chunks = []
            async for chunk in client.stream("user prompt", ""):
                chunks.append(chunk)

            # Verify the message was formatted as user message only
            mock_client.messages.create.assert_called_once()
            call_args = mock_client.messages.create.call_args
            messages = call_args[1]["messages"]

            assert len(messages) == 1
            assert messages[0]["role"] == "user"
            assert messages[0]["content"] == "user prompt"
            # No system message when only prompt provided
            assert "system" not in call_args[1]

    @pytest.mark.asyncio
    async def test_stream_with_kwargs(self):
        """Test streaming with additional keyword arguments."""
        with patch("spegel.llm.claude.AsyncAnthropic") as mock_anthropic:
            mock_client = Mock()
            mock_anthropic.return_value = mock_client

            async def mock_stream():
                return
                yield  # unreachable

            mock_client.messages.create = AsyncMock(return_value=mock_stream())

            client = ClaudeClient("test-key")

            # Test with additional kwargs
            chunks = []
            async for chunk in client.stream("prompt", "content", top_p=0.9, stop_sequences=["END"]):
                chunks.append(chunk)

            # Verify the kwargs were passed through
            mock_client.messages.create.assert_called_once()
            call_args = mock_client.messages.create.call_args

            assert call_args[1]["top_p"] == 0.9
            assert call_args[1]["stop_sequences"] == ["END"]

    @pytest.mark.asyncio
    async def test_stream_different_chunk_types(self):
        """Test handling of different Claude chunk types."""
        with patch("spegel.llm.claude.AsyncAnthropic") as mock_anthropic:
            mock_client = Mock()
            mock_anthropic.return_value = mock_client

            # Mock chunks with different types
            mock_chunk1 = Mock()
            mock_chunk1.type = "message_start"  # Should be ignored

            mock_chunk2 = Mock()
            mock_chunk2.type = "content_block_delta"
            mock_chunk2.delta.text = "Valid content"

            mock_chunk3 = Mock()
            mock_chunk3.type = "content_block_stop"  # Should be ignored

            async def mock_stream():
                yield mock_chunk1
                yield mock_chunk2
                yield mock_chunk3

            mock_client.messages.create = AsyncMock(return_value=mock_stream())

            client = ClaudeClient("test-key")

            chunks = []
            async for chunk in client.stream("test prompt", "test content"):
                chunks.append(chunk)

            # Should only get content from content_block_delta chunks
            assert chunks == ["Valid content"]

    @pytest.mark.asyncio
    async def test_stream_network_error(self):
        """Test handling of network errors during streaming."""
        with patch("spegel.llm.claude.AsyncAnthropic") as mock_anthropic:
            mock_client = Mock()
            mock_anthropic.return_value = mock_client

            # Mock network error during streaming
            mock_client.messages.create = AsyncMock(side_effect=Exception("Network error"))

            client = ClaudeClient("test-key")

            # Should propagate the error
            with pytest.raises(Exception, match="Network error"):
                async for chunk in client.stream("test prompt", "test content"):
                    pass

    @pytest.mark.asyncio
    async def test_stream_api_error(self):
        """Test handling of Claude API errors."""
        with patch("spegel.llm.claude.AsyncAnthropic") as mock_anthropic:
            mock_client = Mock()
            mock_anthropic.return_value = mock_client

            # Mock API error
            mock_client.messages.create = AsyncMock(side_effect=Exception("Rate limit exceeded"))

            client = ClaudeClient("test-key")

            # Should propagate the error
            with pytest.raises(Exception, match="Rate limit exceeded"):
                async for chunk in client.stream("test prompt", "test content"):
                    pass

    @pytest.mark.asyncio
    async def test_stream_parameters_passed_correctly(self):
        """Test that all streaming parameters are passed correctly."""
        with patch("spegel.llm.claude.AsyncAnthropic") as mock_anthropic:
            mock_client = Mock()
            mock_anthropic.return_value = mock_client

            async def mock_stream():
                return
                yield  # unreachable

            mock_client.messages.create = AsyncMock(return_value=mock_stream())

            client = ClaudeClient("test-key", "claude-3-haiku-20240307", 0.7, 2048)

            chunks = []
            async for chunk in client.stream("test prompt", "test content"):
                chunks.append(chunk)

            # Verify all parameters were passed correctly
            mock_client.messages.create.assert_called_once()
            call_args = mock_client.messages.create.call_args

            assert call_args[1]["model"] == "claude-3-haiku-20240307"
            assert call_args[1]["temperature"] == 0.7
            assert call_args[1]["max_tokens"] == 2048
            assert call_args[1]["stream"] is True

    @pytest.mark.asyncio
    async def test_stream_malformed_chunks(self):
        """Test handling of malformed API response chunks."""
        with patch("spegel.llm.claude.AsyncAnthropic") as mock_anthropic:
            mock_client = Mock()
            mock_anthropic.return_value = mock_client

            # Mock chunks with missing attributes
            mock_chunk1 = Mock()
            mock_chunk1.type = "content_block_delta"
            # Missing delta attribute - should be skipped

            mock_chunk2 = Mock()
            mock_chunk2.type = "content_block_delta"
            mock_chunk2.delta.text = None  # None text - should be skipped

            mock_chunk3 = Mock()
            mock_chunk3.type = "content_block_delta"
            mock_chunk3.delta.text = "Valid"

            async def mock_stream():
                yield mock_chunk1
                yield mock_chunk2
                yield mock_chunk3

            mock_client.messages.create = AsyncMock(return_value=mock_stream())

            client = ClaudeClient("test-key")

            chunks = []
            async for chunk in client.stream("test prompt", "test content"):
                chunks.append(chunk)

            # Should only get the valid chunk
            assert chunks == ["Valid"]

    def test_claude_client_logging_error_handling(self):
        """Test that logging errors don't affect functionality."""
        with patch("spegel.llm.claude.AsyncAnthropic") as mock_anthropic:
            mock_client = Mock()
            mock_anthropic.return_value = mock_client

            # Mock logging to raise exception during client creation
            with patch("spegel.llm.logger.info", side_effect=Exception("Logging failed")):
                # Should still create client despite logging errors
                client = ClaudeClient("test-key")
                assert client._client == mock_client
                assert client.model == DEFAULT_MODEL
