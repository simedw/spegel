"""Integration tests for LLM functionality across providers.

This file contains tests that verify the integration between different LLM components
and tests that need to work with the actual provider implementations.
"""

import os
from unittest.mock import Mock, patch

import pytest

from spegel.llm import get_default_client


class TestLLMIntegration:
    """Integration tests for LLM functionality."""

    def test_get_default_client_fallback_to_openai(self):
        """When OpenAI API key is set and Gemini not available, should return OpenAIClient."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
            with patch("spegel.llm.openai.AsyncOpenAI") as mock_openai:
                mock_openai.return_value = Mock()
                # Mock Gemini as unavailable
                with patch("spegel.llm.gemini_available", return_value=False):
                    client = get_default_client()

                    from spegel.llm.openai import OpenAIClient

                    assert isinstance(client, OpenAIClient)

    def test_get_default_client_fallback_to_claude(self):
        """When Claude API key is set and other providers not available, should return ClaudeClient."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=True):
            with patch("spegel.llm.claude.AsyncAnthropic") as mock_anthropic:
                mock_anthropic.return_value = Mock()
                # Mock other providers as unavailable
                with patch("spegel.llm.gemini_available", return_value=False):
                    with patch("spegel.llm.openai_available", return_value=False):
                        client = get_default_client()

                        from spegel.llm.claude import ClaudeClient

                        assert isinstance(client, ClaudeClient)

    def test_get_default_client_no_providers_available(self):
        """When API keys are set but no providers are available, should return None."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with patch("spegel.llm.gemini_available", return_value=False):
                with patch("spegel.llm.GeminiClient", None):
                    # Should try other providers, but if none are available, return None
                    with patch("spegel.llm.openai_available", return_value=False):
                        with patch("spegel.llm.claude_available", return_value=False):
                            client = get_default_client()
                            assert client is None


class TestProviderCompatibility:
    """Test that all providers conform to the same interface."""

    def test_gemini_client_has_required_interface(self):
        """Test that GeminiClient implements the required interface."""
        with patch("spegel.llm.gemini.genai") as mock_genai:
            mock_genai.Client.return_value = Mock()

            from spegel.llm.gemini import GeminiClient

            client = GeminiClient("test-key", "test-model", 0.5, 1024)

            assert hasattr(client, "api_key")
            assert hasattr(client, "model")
            assert hasattr(client, "temperature")
            assert hasattr(client, "max_tokens")
            assert hasattr(client, "stream")

            assert client.api_key == "test-key"
            assert client.model == "test-model"
            assert client.temperature == 0.5
            assert client.max_tokens == 1024

    def test_openai_client_has_required_interface(self):
        """Test that OpenAIClient implements the required interface."""
        with patch("spegel.llm.openai.AsyncOpenAI") as mock_openai:
            mock_openai.return_value = Mock()

            from spegel.llm.openai import OpenAIClient

            client = OpenAIClient("test-key", "test-model", 0.5, 1024)

            assert hasattr(client, "api_key")
            assert hasattr(client, "model")
            assert hasattr(client, "temperature")
            assert hasattr(client, "max_tokens")
            assert hasattr(client, "stream")

            assert client.api_key == "test-key"
            assert client.model == "test-model"
            assert client.temperature == 0.5
            assert client.max_tokens == 1024

    def test_claude_client_has_required_interface(self):
        """Test that ClaudeClient implements the required interface."""
        with patch("spegel.llm.claude.AsyncAnthropic") as mock_anthropic:
            mock_anthropic.return_value = Mock()

            from spegel.llm.claude import ClaudeClient

            client = ClaudeClient("test-key", "test-model", 0.5, 1024)

            assert hasattr(client, "api_key")
            assert hasattr(client, "model")
            assert hasattr(client, "temperature")
            assert hasattr(client, "max_tokens")
            assert hasattr(client, "stream")

            assert client.api_key == "test-key"
            assert client.model == "test-model"
            assert client.temperature == 0.5
            assert client.max_tokens == 1024

    @pytest.mark.asyncio
    async def test_all_providers_have_async_stream(self):
        """Test that all providers implement async stream method."""
        # Test Gemini
        with patch("spegel.llm.gemini.genai") as mock_genai:
            mock_genai.Client.return_value = Mock()

            from spegel.llm.gemini import GeminiClient

            gemini_client = GeminiClient("test-key")
            assert hasattr(gemini_client, "stream")

        # Test OpenAI
        with patch("spegel.llm.openai.AsyncOpenAI") as mock_openai:
            mock_openai.return_value = Mock()

            from spegel.llm.openai import OpenAIClient

            openai_client = OpenAIClient("test-key")
            assert hasattr(openai_client, "stream")

        # Test Claude
        with patch("spegel.llm.claude.AsyncAnthropic") as mock_anthropic:
            mock_anthropic.return_value = Mock()

            from spegel.llm.claude import ClaudeClient

            claude_client = ClaudeClient("test-key")
            assert hasattr(claude_client, "stream")
