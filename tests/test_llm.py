import sys
from pathlib import Path
import os
from unittest.mock import Mock, AsyncMock, patch

# Add project 'src' directory to sys.path so tests work without editable install
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pytest

from spegel.llm import get_default_client, GeminiClient, LLMClient


def test_get_default_client_no_api_key():
    """When no API key is set, should return None client."""
    with patch.dict(os.environ, {}, clear=True):
        client, available = get_default_client()
        assert client is None
        assert available is False


def test_get_default_client_with_api_key():
    """When API key is set, should return GeminiClient."""
    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
        with patch('spegel.llm.genai') as mock_genai:
            mock_genai.Client.return_value = Mock()
            client, available = get_default_client()
            assert isinstance(client, GeminiClient)
            assert available is True


def test_get_default_client_no_genai_module():
    """When genai module is not available, should return None."""
    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
        with patch('spegel.llm.genai', None):
            client, available = get_default_client()
            assert client is None
            assert available is False


class TestGeminiClient:
    """Test GeminiClient functionality."""
    
    def test_init_without_genai(self):
        """Should raise RuntimeError if genai is not available."""
        with patch('spegel.llm.genai', None):
            with pytest.raises(RuntimeError, match="google-genai not installed"):
                GeminiClient("test-key")
    
    def test_init_with_genai(self):
        """Should initialize successfully with genai available."""
        with patch('spegel.llm.genai') as mock_genai:
            mock_client = Mock()
            mock_genai.Client.return_value = mock_client
            
            client = GeminiClient("test-key", "test-model")
            assert client._client == mock_client
            assert client.model_name == "test-model"
            mock_genai.Client.assert_called_once_with(api_key="test-key")
    
    @pytest.mark.asyncio
    async def test_stream_basic(self):
        """Test basic streaming functionality."""
        with patch('spegel.llm.genai') as mock_genai:
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
        with patch('spegel.llm.genai') as mock_genai:
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
        with patch('spegel.llm.genai') as mock_genai:
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


class TestLLMClient:
    """Test the abstract LLMClient interface."""
    
    @pytest.mark.asyncio
    async def test_abstract_stream_method(self):
        """LLMClient.stream should raise NotImplementedError."""
        client = LLMClient()
        
        with pytest.raises(NotImplementedError):
            async for chunk in client.stream("test", "test"):
                pass 