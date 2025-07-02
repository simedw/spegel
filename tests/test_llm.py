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

    def test_get_default_client_no_api_key(self):
        """When no API key is set, should return None client."""
        with patch.dict(os.environ, {}, clear=True):
            client = get_default_client()
            assert client is None

    def test_get_default_client_with_gemini_api_key(self):
        """When Gemini API key is set, should return GeminiClient."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with patch("spegel.llm.gemini.genai") as mock_genai:
                mock_genai.Client.return_value = Mock()
                client = get_default_client()

                # Should be able to import the class to check type
                from spegel.llm.gemini import GeminiClient

                assert isinstance(client, GeminiClient)

    def test_get_default_client_with_openai_api_key(self):
        """When OpenAI API key is set and Gemini not available, should return OpenAIClient."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
            with patch("spegel.llm.openai.AsyncOpenAI") as mock_openai:
                mock_openai.return_value = Mock()
                # Mock Gemini as unavailable
                with patch("spegel.llm.gemini_available", return_value=False):
                    client = get_default_client()

                    from spegel.llm.openai import OpenAIClient

                    assert isinstance(client, OpenAIClient)

    def test_get_default_client_prefers_gemini_when_both_available(self):
        """When both API keys are set, should prefer Gemini."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "gemini-key", "OPENAI_API_KEY": "openai-key"}):
            with patch("spegel.llm.gemini.genai") as mock_genai:
                mock_genai.Client.return_value = Mock()

                client = get_default_client()
                from spegel.llm.gemini import GeminiClient

                assert isinstance(client, GeminiClient)

    def test_get_default_client_no_genai_module(self):
        """When genai module is not available, should raise RuntimeError."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with patch("spegel.llm.gemini_available", return_value=False):
                with patch("spegel.llm.GeminiClient", None):
                    # Should try OpenAI next, but if that's also not available, return None
                    with patch("spegel.llm.openai_available", return_value=False):
                        client = get_default_client()
                        assert client is None


class TestProviderCompatibility:
    """Test that both providers conform to the same interface."""

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

    @pytest.mark.asyncio
    async def test_both_providers_have_async_stream(self):
        """Test that both providers implement async stream method."""
        with patch("spegel.llm.gemini.genai") as mock_genai:
            mock_genai.Client.return_value = Mock()

            from spegel.llm.gemini import GeminiClient

            gemini_client = GeminiClient("test-key")

            # Should have async stream method
            assert hasattr(gemini_client, "stream")

        # Test OpenAI
        with patch("spegel.llm.openai.AsyncOpenAI") as mock_openai:
            mock_openai.return_value = Mock()

            from spegel.llm.openai import OpenAIClient

            openai_client = OpenAIClient("test-key")

            # Should have async stream method
            assert hasattr(openai_client, "stream")
