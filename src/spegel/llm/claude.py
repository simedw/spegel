"""Anthropic Claude LLM provider implementation."""

from __future__ import annotations

from collections.abc import AsyncIterator

from .base import LLMClient

try:
    from anthropic import AsyncAnthropic

    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False
    AsyncAnthropic = None

DEFAULT_MODEL = "claude-3-5-haiku-20241022"


class ClaudeClient(LLMClient):
    """Wrapper around Anthropic Claude async streaming API."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ):
        if not CLAUDE_AVAILABLE or AsyncAnthropic is None:
            raise RuntimeError("anthropic package not installed but ClaudeClient requested")

        super().__init__(api_key, model, temperature, max_tokens)

        try:
            self._client = AsyncAnthropic(api_key=api_key)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Claude client: {e}") from e

    async def stream(
        self,
        prompt: str,
        content: str,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream response from Claude."""
        # Build messages - Claude uses separate system and user messages
        messages = []
        system_message = None

        if prompt and content:
            # Both prompt and content provided - prompt is system, content is user
            system_message = prompt
            messages = [{"role": "user", "content": content}]
        elif prompt and not content:
            # Only prompt provided - treat as user message
            messages = [{"role": "user", "content": prompt}]

        self._log_request(prompt, content)

        try:
            create_args = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stream": True,
                **kwargs,
            }

            # Add system message if we have one
            if system_message:
                create_args["system"] = system_message

            # Create streaming response
            stream = await self._client.messages.create(**create_args)

            collected: list[str] = []

            async for chunk in stream:
                if chunk.type == "content_block_delta" and hasattr(chunk, "delta"):
                    if hasattr(chunk.delta, "text") and chunk.delta.text:
                        text = chunk.delta.text
                        # Ensure we have a string (not Mock object in tests)
                        if isinstance(text, str):
                            collected.append(text)
                            yield text

            self._log_response(collected)

        except Exception as e:
            self._log_response([f"Error: {e}"])
            raise


def is_available() -> bool:
    """Check if Claude provider is available."""
    return CLAUDE_AVAILABLE
