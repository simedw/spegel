"""Tests for LLM factory functions and generic LLM functionality."""

import os
from unittest.mock import Mock, patch

import pytest

from spegel.llm import (
    LLMClient,
    create_client,
    get_default_api_key_env_for_provider,
    get_default_client,
    get_default_model_for_provider,
    list_available_providers,
)
from spegel.llm.claude import DEFAULT_API_KEY_ENV as CLAUDE_API_KEY_ENV
from spegel.llm.gemini import DEFAULT_API_KEY_ENV as GEMINI_API_KEY_ENV
from spegel.llm.openai import DEFAULT_API_KEY_ENV as OPENAI_API_KEY_ENV


class TestLLMFactory:
    """Test LLM factory functions."""

    def test_list_available_providers(self):
        """Test listing available providers."""
        providers = list_available_providers()

        # Should return a dict with provider names as keys and boolean availability as values
        assert isinstance(providers, dict)
        assert "gemini" in providers
        assert "openai" in providers
        assert "claude" in providers
        assert isinstance(providers["gemini"], bool)
        assert isinstance(providers["openai"], bool)
        assert isinstance(providers["claude"], bool)

    def test_get_default_model_for_provider(self):
        """Test getting default models for each provider."""
        # Test Gemini
        gemini_model = get_default_model_for_provider("gemini")
        assert isinstance(gemini_model, str)
        assert len(gemini_model) > 0
        assert "gemini" in gemini_model.lower()

        # Test OpenAI
        openai_model = get_default_model_for_provider("openai")
        assert isinstance(openai_model, str)
        assert len(openai_model) > 0
        assert "gpt" in openai_model.lower()

        # Test Claude
        claude_model = get_default_model_for_provider("claude")
        assert isinstance(claude_model, str)
        assert len(claude_model) > 0
        assert "claude" in claude_model.lower()

        # Test case insensitive
        assert get_default_model_for_provider("GEMINI") == get_default_model_for_provider("gemini")
        assert get_default_model_for_provider("OpenAI") == get_default_model_for_provider("openai")
        assert get_default_model_for_provider("CLAUDE") == get_default_model_for_provider("claude")

        # Test unknown provider
        unknown_model = get_default_model_for_provider("unknown")
        assert unknown_model == ""

    def test_get_default_api_key_env_for_provider(self):
        """Test getting default API key environment variable names."""
        # Test Gemini
        gemini_env = get_default_api_key_env_for_provider("gemini")
        assert gemini_env == GEMINI_API_KEY_ENV

        # Test OpenAI
        openai_env = get_default_api_key_env_for_provider("openai")
        assert openai_env == OPENAI_API_KEY_ENV

        # Test Claude
        claude_env = get_default_api_key_env_for_provider("claude")
        assert claude_env == CLAUDE_API_KEY_ENV

        # Test case insensitive
        assert get_default_api_key_env_for_provider("GEMINI") == GEMINI_API_KEY_ENV
        assert get_default_api_key_env_for_provider("OpenAI") == OPENAI_API_KEY_ENV
        assert get_default_api_key_env_for_provider("Claude") == CLAUDE_API_KEY_ENV

        # Test unknown provider
        unknown_env = get_default_api_key_env_for_provider("unknown")
        assert unknown_env == ""


class TestLegacyGetDefaultClient:
    """Test the legacy get_default_client function for backwards compatibility."""

    def test_get_default_client_no_api_key(self):
        """When no API key is set, should return None client."""
        with patch.dict(os.environ, {}, clear=True):
            client = get_default_client()
            assert client is None

    def test_get_default_client_prefers_gemini(self):
        """When Gemini is available and others are not, should return GeminiClient."""
        with patch.dict(
            os.environ,
            {
                GEMINI_API_KEY_ENV: "gemini-key",
            },
            clear=True,
        ):
            with patch("spegel.llm.gemini.genai") as mock_genai:
                mock_genai.Client.return_value = Mock()
                # Make sure only Gemini is available
                with patch("spegel.llm.openai.OpenAIProviderMeta.is_available", return_value=False):
                    with patch("spegel.llm.claude.ClaudeProviderMeta.is_available", return_value=False):
                        client = get_default_client()
                        from spegel.llm.gemini import GeminiClient

                        assert isinstance(client, GeminiClient)

    def test_get_default_client_falls_back_to_openai(self):
        """When only OpenAI key is set, should use OpenAI."""
        with patch.dict(os.environ, {OPENAI_API_KEY_ENV: "openai-key"}, clear=True):
            with patch("spegel.llm.openai.AsyncOpenAI") as mock_openai:
                mock_openai.return_value = Mock()

                client = get_default_client()
                from spegel.llm.openai import OpenAIClient

                assert isinstance(client, OpenAIClient)

    def test_get_default_client_falls_back_to_claude(self):
        """When only Claude key is set, should use Claude."""
        with patch.dict(os.environ, {CLAUDE_API_KEY_ENV: "claude-key"}, clear=True):
            with patch("spegel.llm.claude.AsyncAnthropic") as mock_anthropic:
                mock_anthropic.return_value = Mock()

                client = get_default_client()
                from spegel.llm.claude import ClaudeClient

                assert isinstance(client, ClaudeClient)

    def test_get_default_client_no_providers_available(self):
        """When API keys are set but no providers are available, should return None."""
        with patch.dict(os.environ, {GEMINI_API_KEY_ENV: "test-key"}):
            # Mock all providers as unavailable using dynamic system
            with patch("spegel.llm.ProviderRegistry.is_provider_available", return_value=False):
                client = get_default_client()
                assert client is None


class TestConfigBasedClientCreation:
    """Test the new config-based client creation."""

    def test_create_client_gemini_success(self):
        """Test successful Gemini client creation via config."""
        from spegel.config import AI

        ai_config = AI(
            provider="gemini",
            model="gemini-2.5-flash-lite-preview-06-17",
            api_key_env=GEMINI_API_KEY_ENV,
            temperature=0.3,
            max_tokens=4096,
        )

        with patch.dict(os.environ, {GEMINI_API_KEY_ENV: "test-key"}):
            with patch("spegel.llm.gemini.genai") as mock_genai:
                mock_genai.Client.return_value = Mock()

                client = create_client(ai_config)
                from spegel.llm.gemini import GeminiClient

                assert isinstance(client, GeminiClient)
                assert client.model == "gemini-2.5-flash-lite-preview-06-17"
                assert client.temperature == 0.3
                assert client.max_tokens == 4096

    def test_create_client_openai_success(self):
        """Test successful OpenAI client creation via config."""
        from spegel.config import AI

        ai_config = AI(
            provider="openai", model="gpt-4o-mini", api_key_env=OPENAI_API_KEY_ENV, temperature=0.1, max_tokens=2048
        )

        with patch.dict(os.environ, {OPENAI_API_KEY_ENV: "test-key"}):
            with patch("spegel.llm.openai.AsyncOpenAI") as mock_openai:
                mock_openai.return_value = Mock()

                client = create_client(ai_config)
                from spegel.llm.openai import OpenAIClient

                assert isinstance(client, OpenAIClient)
                assert client.model == "gpt-4o-mini"
                assert client.temperature == 0.1
                assert client.max_tokens == 2048

    def test_create_client_claude_success(self):
        """Test successful Claude client creation via config."""
        from spegel.config import AI

        ai_config = AI(
            provider="claude",
            model="claude-3-haiku-20240307",
            api_key_env=CLAUDE_API_KEY_ENV,
            temperature=0.4,
            max_tokens=3072,
        )

        with patch.dict(os.environ, {CLAUDE_API_KEY_ENV: "test-key"}):
            with patch("spegel.llm.claude.AsyncAnthropic") as mock_anthropic:
                mock_anthropic.return_value = Mock()

                client: LLMClient | None = create_client(ai_config)
                from spegel.llm.claude import ClaudeClient

                assert isinstance(client, ClaudeClient)
                assert client.model == "claude-3-haiku-20240307"
                assert client.temperature == 0.4
                assert client.max_tokens == 3072

    def test_create_client_missing_api_key(self):
        """Test client creation with missing API key."""
        from spegel.config import AI

        ai_config = AI(
            provider="gemini",
            model="gemini-2.5-flash-lite-preview-06-17",
            api_key_env=GEMINI_API_KEY_ENV,
            temperature=0.2,
            max_tokens=8192,
        )

        with patch.dict(os.environ, {}, clear=True):
            client: LLMClient | None = create_client(ai_config)
            assert client is None

    def test_create_client_unknown_provider(self):
        """Test client creation with unknown provider."""
        from spegel.config import AI

        with pytest.raises(ValueError, match="Invalid provider 'unknown_provider'"):
            AI(
                provider="unknown_provider",
                model="some-model",
                api_key_env="SOME_API_KEY",
                temperature=0.2,
                max_tokens=8192,
            )

    def test_create_client_provider_not_available(self):
        """Test client creation when provider package is not available."""
        from spegel.config import AI

        ai_config = AI(
            provider="gemini",
            model="gemini-2.5-flash-lite-preview-06-17",
            api_key_env=GEMINI_API_KEY_ENV,
            temperature=0.2,
            max_tokens=8192,
        )

        with patch.dict(os.environ, {GEMINI_API_KEY_ENV: "test-key"}):
            # Mock Gemini as unavailable using dynamic system
            with patch("spegel.llm.ProviderRegistry.is_provider_available") as mock_available:
                mock_available.side_effect = lambda provider: provider != "gemini"
                client = create_client(ai_config)
                assert client is None


class TestLLMClientAbstract:
    """Test the abstract LLMClient interface."""

    @pytest.mark.asyncio
    async def test_abstract_stream_method(self):
        """LLMClient.stream should raise NotImplementedError."""

        # Create a concrete class that doesn't implement stream
        class IncompleteClient(LLMClient):
            pass

        # Should not be able to instantiate without implementing stream
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteClient("test-key", "test-model")  # type: ignore[abstract]

    def test_llm_client_initialization(self):
        """Test LLMClient base class initialization."""

        # Create a minimal implementation for testing
        class TestClient(LLMClient):
            async def stream(self, prompt: str, content: str, **kwargs):
                yield "test"

        client = TestClient("test-key", "test-model", 0.5, 1024)
        assert client.api_key == "test-key"
        assert client.model == "test-model"
        assert client.temperature == 0.5
        assert client.max_tokens == 1024

    def test_llm_client_default_values(self):
        """Test LLMClient with default values."""

        class TestClient(LLMClient):
            async def stream(self, prompt: str, content: str, **kwargs):
                yield "test"

        client = TestClient("test-key", "test-model")
        assert client.api_key == "test-key"
        assert client.model == "test-model"
        assert isinstance(client.temperature, (int, float))
        assert isinstance(client.max_tokens, int)


class TestErrorScenarios:
    """Test error scenarios for LLM functionality."""

    def test_get_default_client_missing_environment(self):
        """Test client creation with missing environment variables."""
        with patch.dict(os.environ, {}, clear=True):
            client = get_default_client()
            assert client is None

    def test_get_default_client_invalid_api_key(self):
        """Test client creation with invalid API key format."""
        with patch.dict(os.environ, {}, clear=True):  # Empty key
            client = get_default_client()
            assert client is None

    def test_enable_llm_logging_error_handling(self):
        """Test error handling in logging configuration."""
        from spegel.llm import enable_llm_logging

        # Should handle gracefully
        try:
            enable_llm_logging(level=999999)  # Invalid level
        except Exception as e:
            # Should not raise unhandled exceptions
            assert False, f"Logging setup should handle errors: {e}"
