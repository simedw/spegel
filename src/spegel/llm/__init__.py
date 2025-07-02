"""LLM provider factory and main interface.

This module provides a unified interface for different LLM providers while maintaining
backwards compatibility with the existing codebase.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import TYPE_CHECKING

from .base import LLMClient, ProviderMeta, enable_llm_logging, logger

if TYPE_CHECKING:
    from ..config import AI


def _discover_providers() -> dict[str, ProviderMeta]:
    providers = {}
    llm_dir: Path = Path(__file__).parent

    for file in llm_dir.glob("*.py"):
        if file.stem in ["__init__", "base"]:
            continue
        try:
            module = importlib.import_module(f".{file.stem}", __package__)
            if hasattr(module, "PROVIDER_META"):
                provider_meta_class = module.PROVIDER_META
                provider_meta_instance = provider_meta_class()
                providers[provider_meta_instance.name] = provider_meta_instance
        except (ImportError, AttributeError):
            continue

    return providers


class ProviderRegistry:
    _providers: dict[str, ProviderMeta] = _discover_providers()

    @classmethod
    def get_all_providers(cls) -> dict[str, ProviderMeta]:
        return cls._providers.copy()

    @classmethod
    def get_provider_meta(cls, provider: str) -> ProviderMeta | None:
        return cls._providers.get(provider.lower())

    @classmethod
    def is_provider_available(cls, provider: str) -> bool:
        provider_meta = cls.get_provider_meta(provider)
        return provider_meta.is_available() if provider_meta else False

    @classmethod
    def get_available_providers(cls) -> dict[str, bool]:
        return {name: meta.is_available() for name, meta in cls._providers.items()}

    @classmethod
    def create_client_for_provider(cls, provider: str, api_key: str, **kwargs) -> LLMClient | None:
        provider_meta: ProviderMeta | None = cls.get_provider_meta(provider)
        if not provider_meta:
            logger.error(f"Unknown provider: {provider}")
            return None

        if not cls.is_provider_available(provider):
            logger.error(f"{provider.title()} provider requested but package not available")
            return None

        client_class = provider_meta.get_client_class()
        if client_class is None:
            logger.error(f"{provider.title()} provider requested but client class not available")
            return None
        return client_class(api_key=api_key, **kwargs)


def create_client(ai_config: AI) -> LLMClient | None:
    api_key: str | None = os.getenv(ai_config.api_key_env)
    if not api_key:
        logger.warning(f"No API key found in environment variable: {ai_config.api_key_env}")
        return None

    return ProviderRegistry.create_client_for_provider(
        provider=ai_config.provider,
        api_key=api_key,
        model=ai_config.model,
        temperature=ai_config.temperature,
        max_tokens=ai_config.max_tokens,
    )


def get_default_client() -> LLMClient | None:
    for provider_name, provider_meta in ProviderRegistry.get_all_providers().items():
        api_key: str | None = os.getenv(provider_meta.api_key_env)
        if api_key and ProviderRegistry.is_provider_available(provider=provider_name):
            return ProviderRegistry.create_client_for_provider(provider=provider_name, api_key=api_key)
    return None


def list_available_providers() -> dict[str, bool]:
    return ProviderRegistry.get_available_providers()


def get_default_model_for_provider(provider: str) -> str:
    provider_meta: ProviderMeta | None = ProviderRegistry.get_provider_meta(provider)
    return provider_meta.default_model if provider_meta else ""


def get_default_api_key_env_for_provider(provider: str) -> str:
    provider_meta: ProviderMeta | None = ProviderRegistry.get_provider_meta(provider)
    return provider_meta.api_key_env if provider_meta else ""


def get_defaults() -> ProviderMeta:
    for provider_meta in ProviderRegistry.get_all_providers().values():
        if provider_meta.is_available():
            return provider_meta
    raise RuntimeError("No available LLM providers found. Please install the required packages.")


__all__ = [
    "LLMClient",
    "ProviderMeta",
    "create_client",
    "get_default_client",
    "enable_llm_logging",
    "list_available_providers",
    "get_default_model_for_provider",
    "get_default_api_key_env_for_provider",
    "get_defaults",
]
