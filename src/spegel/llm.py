from __future__ import annotations

from argparse import Namespace
import os
import logging
from typing import Any
from collections.abc import AsyncIterator

"""Light abstraction layer over one or more LLM back-ends.

Right now we only implement Google Gemini via `google-genai`, but the
interface allows us to add more providers later without touching UI code.
"""

# Configure logger for LLM interactions (disabled by default)
logger = logging.getLogger("spegel.llm")
logger.setLevel(logging.CRITICAL + 1)  # Effectively disabled by default


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
    from google import genai
    from google.genai import types
except ImportError:  # pragma: no cover – dependency is optional until used
    genai = None  # type: ignore
    types = None  # type: ignore

__all__ = [
    "LLMClient",
    "GeminiClient",
    "get_default_client",
]


class LLMClient:
    """Abstract asynchronous client interface."""

    async def stream(self, prompt: str, content: str, **kwargs) -> AsyncIterator[str]:
        """Yield chunks of markdown text."""
        raise NotImplementedError
        yield  # This is unreachable, but makes this an async generator


class GeminiClient(LLMClient):
    """Wrapper around google-genai async streaming API."""

    def __init__(
        self, api_key: str, model_name: str = "gemini-2.5-flash-lite-preview-06-17"
    ):
        if genai is None:
            raise RuntimeError("google-genai not installed but GeminiClient requested")
        self._client = genai.Client(api_key=api_key)
        self.model_name = model_name

    async def stream(
        self,
        prompt: str,
        content: str,
        generation_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        config = None
        if generation_config is None and types is not None:
            config = types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=8192,
                response_mime_type="text/plain",
                thinking_config=types.ThinkingConfig(
                    thinking_budget=0,
                ),
            )
        user_content: str = f"{prompt}\n\n{content}" if content else prompt
        stream = self._client.aio.models.generate_content_stream(
            model=self.model_name,
            contents=user_content,
            config=config,
        )

        # Log the prompt if logging is enabled
        logger.info("LLM Prompt: %s", user_content)

        collected: list[str] = []

        async for chunk in await stream:
            try:
                text = chunk.candidates[0].content.parts[0].text  # type: ignore[attr-defined]
                if text:
                    collected.append(text)
                    yield text
            except Exception:
                continue

        # Log the complete response if logging is enabled
        if collected:
            logger.info("LLM Response: %s", "".join(collected))


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


def get_default_client() -> LLMClient | None:
    """Return an LLMClient instance if credentials exist, else None."""
    api_key: str | None = os.getenv("GEMINI_API_KEY")
    if api_key and genai is not None:
        return GeminiClient(api_key=api_key)
    return None


if __name__ == "__main__":
    import argparse
    import asyncio
    import sys

    parser = argparse.ArgumentParser(
        description="Quick CLI wrapper around the configured LLM to answer a prompt."
    )
    parser.add_argument("prompt", help="User prompt/question to send to the model")
    args: Namespace = parser.parse_args()

    client: LLMClient | None = get_default_client()
    if client is None:
        print(
            "Error: GEMINI_API_KEY not set or google-genai unavailable", file=sys.stderr
        )
        sys.exit(1)

    async def _main() -> None:
        if client is None:
            print("No LLM client available", file=sys.stderr)
            return
        async for chunk in client.stream(args.prompt, ""):
            print(chunk, end="", flush=True)

    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        pass
