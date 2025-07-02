"""OpenAI LLM provider implementation."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from .base import LLMClient

try:
    from openai import AsyncOpenAI
    from openai.types.chat import ChatCompletionMessageParam

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    AsyncOpenAI = None
    if TYPE_CHECKING:
        from openai.types.chat import ChatCompletionMessageParam
    else:
        ChatCompletionMessageParam = None

DEFAULT_MODEL = "gpt-4.1-nano"


class OpenAIClient(LLMClient):
    """Wrapper around OpenAI async streaming API."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.2,
        max_tokens: int = 8192,
        base_url: str | None = None,
    ):
        if not OPENAI_AVAILABLE or AsyncOpenAI is None:
            raise RuntimeError("openai package not installed but OpenAIClient requested")

        super().__init__(api_key, model, temperature, max_tokens)

        try:
            self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI client: {e}") from e

    async def stream(
        self,
        prompt: str,
        content: str,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream response from OpenAI."""
        messages: list[ChatCompletionMessageParam] = [{"role": "system", "content": prompt}]

        if content:
            messages.append({"role": "user", "content": content})
        elif not content and prompt:
            # If no content provided, treat prompt as user message
            messages = [{"role": "user", "content": prompt}]

        self._log_request(prompt, content)

        try:
            stream = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
                **kwargs,
            )

            collected: list[str] = []

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content is not None:
                    text = chunk.choices[0].delta.content
                    collected.append(text)
                    yield text

            self._log_response(collected)

        except Exception as e:
            self._log_response([f"Error: {e}"])
            raise


def is_available() -> bool:
    """Check if OpenAI provider is available."""
    return OPENAI_AVAILABLE
