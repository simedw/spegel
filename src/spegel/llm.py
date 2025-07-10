from __future__ import annotations

import os
import logging
from typing import AsyncIterator, Dict, Any

"""Light abstraction layer over one or more LLM back-ends.

Uses LiteLLM to support multiple providers including OpenAI, Anthropic, 
Google Gemini, and many others through a unified interface.
"""

# Configure logger for LLM interactions (disabled by default)
logger = logging.getLogger("spegel.llm")
logger.setLevel(logging.CRITICAL + 1)  # Effectively disabled by default

# Workaround for https://github.com/BerriAI/litellm/issues/11657
os.environ["DISABLE_AIOHTTP_TRANSPORT"] = "True"


def enable_llm_logging(level: int = logging.INFO) -> None:
    """Enable LLM interaction logging at the specified level."""
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)


try:
    import litellm
except ImportError:  # pragma: no cover – dependency is optional until used
    litellm = None  # type: ignore

__all__ = ["LLMClient", "LiteLLMClient", "create_client", "LLMAuthenticationError"]


class LLMAuthenticationError(Exception):
    """Custom exception for LLM authentication failures with user-friendly messaging."""

    def __init__(self, model: str, provider: str, original_error: Exception):
        self.model = model
        self.provider = provider
        self.original_error = original_error
        super().__init__(
            f"Authentication failed for model '{model}'. "
            f"Please set a valid API key for {provider}. "
            f"You can set SPEGEL_API_KEY environment variable or the provider-specific "
            f"API key (e.g., OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY)."
        )


class LLMClient:
    """Abstract asynchronous client interface."""

    async def stream(self, prompt: str, content: str, **kwargs) -> AsyncIterator[str]:
        """Yield chunks of markdown text."""
        raise NotImplementedError
        yield  # This is unreachable, but makes this an async generator


class LiteLLMClient(LLMClient):
    """Wrapper around LiteLLM for unified LLM provider access."""

    def __init__(
        self,
        model: str = "gemini/gemini-2.5-flash-lite-preview-06-17",
        api_key: str | None = None,
        api_base: str | None = None,
        **kwargs,
    ):
        if litellm is None:
            raise RuntimeError("litellm not installed but LiteLLMClient requested")

        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.extra_kwargs = kwargs

    async def stream(
        self,
        prompt: str,
        content: str,
        temperature: float = 0.2,
        max_tokens: int = 8192,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream response from LiteLLM."""
        if litellm is None:
            raise RuntimeError("litellm not available")

        user_content = f"{prompt}\n\n{content}" if content else prompt

        # Prepare messages in the format expected by LiteLLM
        messages = [{"role": "user", "content": user_content}]

        # Set up completion parameters
        completion_params = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            **kwargs,
        }

        if self.api_key:
            completion_params["api_key"] = self.api_key

        if self.api_base:
            completion_params["api_base"] = self.api_base

        # Add any extra kwargs passed during initialization
        completion_params.update(self.extra_kwargs)

        # Log the prompt if logging is enabled
        logger.info("LLM Prompt: %s", user_content)

        collected: list[str] = []

        try:
            response = await litellm.acompletion(**completion_params)

            async for chunk in response:
                try:
                    # Extract content from the chunk
                    if chunk.choices and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta
                        if delta and hasattr(delta, "content") and delta.content:
                            text = delta.content
                            collected.append(text)
                            yield text
                except Exception as e:
                    logger.warning("Error processing chunk: %s", e)
                    continue

        except litellm.AuthenticationError as e:
            logger.error("Authentication error in LLM completion: %s", e)
            # Extract the model provider from the model name for better error message
            model_provider = (
                self.model.split("/")[0] if "/" in self.model else self.model
            )
            raise LLMAuthenticationError(self.model, model_provider, e) from e
        except Exception as e:
            logger.error("Error in LLM completion: %s", e)
            raise

        # Log the complete response if logging is enabled
        if collected:
            logger.info("LLM Response: %s", "".join(collected))


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


def create_client(model: str) -> LLMClient | None:
    """Create an LLM client with the specified model.

    Args:
        model: The model identifier (e.g., "gpt-4o-mini", "claude-3-5-haiku-20241022")

    Returns:
        LLMClient instance or None if creation failed
    """
    if litellm is None:
        return None

    # Check if a specific model is requested via environment variable (overrides everything)
    custom_model = os.getenv("SPEGEL_MODEL")
    if custom_model:
        api_key = os.getenv("SPEGEL_API_KEY")
        api_base = os.getenv("SPEGEL_API_BASE")
        try:
            return LiteLLMClient(model=custom_model, api_key=api_key, api_base=api_base)
        except Exception:
            pass

    # Create client with the specified model
    try:
        return LiteLLMClient(model=model)
    except Exception:
        return None


if __name__ == "__main__":
    import argparse
    import asyncio
    import sys

    parser = argparse.ArgumentParser(
        description="Quick CLI wrapper around the configured LLM to answer a prompt."
    )
    parser.add_argument("prompt", help="User prompt/question to send to the model")
    parser.add_argument("--model", help="Override the default model")
    args = parser.parse_args()

    # Load config to get default model
    try:
        from .config import load_config
    except ImportError:
        # Handle case when running as script directly
        from spegel.config import load_config

    config = load_config()

    model = args.model or config.ai.default_model

    client = create_client(model)
    if client is None:
        print(
            f"Error: No LLM provider configured for model '{model}'. "
            "Check your API keys and model configuration.",
            file=sys.stderr,
        )
        sys.exit(1)

    async def _main() -> None:
        try:
            async for chunk in client.stream(args.prompt, ""):
                print(chunk, end="", flush=True)
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(f"\nError: {e}", file=sys.stderr)
            sys.exit(1)

    asyncio.run(_main())
