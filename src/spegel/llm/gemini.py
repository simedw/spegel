"""Google Gemini LLM provider implementation."""

from __future__ import annotations

from collections.abc import AsyncIterator

from .base import LLMClient

try:
    from google import genai
    from google.genai import types
    from google.genai.types import GenerateContentConfig, ThinkingConfig

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None
    GenerateContentConfig = None
    ThinkingConfig = None

DEFAULT_MODEL = "gemini-2.5-flash-lite-preview-06-17"
DEFAULT_API_KEY_ENV = "GOOGLE_GENAI_API_KEY"


class GeminiClient(LLMClient):
    """Wrapper around google-genai async streaming API."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.2,
        max_tokens: int = 8192,
    ):
        if not GEMINI_AVAILABLE or genai is None:
            raise RuntimeError("google-genai not installed but GeminiClient requested")

        super().__init__(api_key, model, temperature, max_tokens)

        try:
            self._client = genai.Client(api_key=api_key)
        except Exception as e:
            if "genai" in str(e) or "google" in str(e):
                raise RuntimeError("google-genai not installed but GeminiClient requested") from e
            raise

    @property
    def model_name(self) -> str:
        """Legacy property for backwards compatibility with tests."""
        return self.model

    async def stream(
        self,
        prompt: str,
        content: str,
        generation_config: types.GenerateContentConfig | None = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream response from Gemini."""
        if generation_config is None and GenerateContentConfig is not None and ThinkingConfig is not None:
            generation_config = GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                response_mime_type="text/plain",
                thinking_config=ThinkingConfig(thinking_budget=0),
            )

        user_content: str = f"{prompt}\n\n{content}" if content else prompt
        self._log_request(prompt, content)

        stream = self._client.aio.models.generate_content_stream(
            model=self.model,
            contents=user_content,
            config=generation_config,
        )

        collected: list[str] = []

        async for chunk in await stream:
            try:
                text = chunk.candidates[0].content.parts[0].text  # type: ignore[attr-defined]
                if text:
                    collected.append(text)
                    yield text
            except Exception:
                continue

        self._log_response(collected)


def is_available() -> bool:
    """Check if Gemini provider is available."""
    return GEMINI_AVAILABLE
