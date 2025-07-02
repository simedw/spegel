"""LLM provider factory and main interface.

This module provides a unified interface for different LLM providers while maintaining
backwards compatibility with the existing codebase.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from .base import LLMClient, enable_llm_logging, logger

if TYPE_CHECKING:
    from ..config import AI


try:
    from .gemini import GeminiClient
    from .gemini import is_available as gemini_available
except ImportError:
    GeminiClient = None

    def gemini_available() -> bool:
        return False


try:
    from .openai import OpenAIClient
    from .openai import is_available as openai_available
except ImportError:
    OpenAIClient = None

    def openai_available() -> bool:
        return False


try:
    from .claude import ClaudeClient
    from .claude import is_available as claude_available
except ImportError:
    ClaudeClient = None

    def claude_available() -> bool:
        return False


def create_client(ai_config: AI) -> LLMClient | None:
    """Create an LLM client based on configuration.

    Args:
        ai_config: AI configuration from the app config

    Returns:
        LLMClient instance or None if no suitable provider is available
    """
    api_key: str | None = os.getenv(ai_config.api_key_env)
    if not api_key:
        logger.warning(f"No API key found in environment variable: {ai_config.api_key_env}")
        return None

    provider: str = ai_config.provider.lower()

    match provider:
        case "gemini":
            if not gemini_available():
                logger.error("Gemini provider requested but google-genai package not available")
                return None
            if GeminiClient is None:
                logger.error("Gemini provider requested but GeminiClient not available")
                return None

            return GeminiClient(
                api_key=api_key,
                model=ai_config.model,
                temperature=ai_config.temperature,
                max_tokens=ai_config.max_tokens,
            )

        case "openai":
            if not openai_available():
                logger.error("OpenAI provider requested but openai package not available")
                return None
            if OpenAIClient is None:
                logger.error("OpenAI provider requested but OpenAIClient not available")
                return None

            return OpenAIClient(
                api_key=api_key,
                model=ai_config.model,
                temperature=ai_config.temperature,
                max_tokens=ai_config.max_tokens,
            )

        case "claude":
            if not claude_available():
                logger.error("Claude provider requested but anthropic package not available")
                return None
            if ClaudeClient is None:
                logger.error("Claude provider requested but ClaudeClient not available")
                return None

            return ClaudeClient(
                api_key=api_key,
                model=ai_config.model,
                temperature=ai_config.temperature,
                max_tokens=ai_config.max_tokens,
            )

        case _:
            logger.error(f"Unknown AI provider: {provider}")
            return None


def get_default_client() -> LLMClient | None:
    """Legacy function for backwards compatibility.

    This maintains compatibility with existing code while allowing
    for future migration to config-based client creation.

    Returns:
        LLMClient instance or None if no suitable provider is available
    """

    api_key: str | None = os.getenv("GEMINI_API_KEY")
    if api_key and gemini_available() and GeminiClient is not None:
        return GeminiClient(api_key=api_key)

    api_key: str | None = os.getenv("OPENAI_API_KEY")
    if api_key and openai_available() and OpenAIClient is not None:
        return OpenAIClient(api_key=api_key)

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key and claude_available() and ClaudeClient is not None:
        return ClaudeClient(api_key=api_key)

    return None


def list_available_providers() -> dict[str, bool]:
    """Return a dict of provider names and their availability status."""
    return {
        "gemini": gemini_available(),
        "openai": openai_available(),
        "claude": claude_available(),
    }


def get_default_model_for_provider(provider: str) -> str:
    """Get the default model name for a given provider."""
    defaults: dict[str, str] = {
        "gemini": "gemini-2.5-flash-lite-preview-06-17",
        "openai": "gpt-4.1-nano",
        "claude": "claude-3-haiku-20240307",
    }
    return defaults.get(provider.lower(), "")


def get_default_api_key_env_for_provider(provider: str) -> str:
    """Get the default environment variable name for a given provider's API key."""
    defaults: dict[str, str] = {
        "gemini": "GEMINI_API_KEY",
        "openai": "OPENAI_API_KEY",
        "claude": "ANTHROPIC_API_KEY",
    }
    return defaults.get(provider.lower(), "")


# Export client classes for backwards compatibility (if available)
# This allows tests to import them directly from spegel.llm
if GeminiClient is not None:
    globals()["GeminiClient"] = GeminiClient
if OpenAIClient is not None:
    globals()["OpenAIClient"] = OpenAIClient
if ClaudeClient is not None:
    globals()["ClaudeClient"] = ClaudeClient


__all__ = [
    "LLMClient",
    "GeminiClient",
    "OpenAIClient",
    "ClaudeClient",
    "create_client",
    "get_default_client",
    "enable_llm_logging",
    "list_available_providers",
    "get_default_model_for_provider",
    "get_default_api_key_env_for_provider",
]
