from __future__ import annotations

"""Light abstraction layer over one or more LLM back-ends.

Right now we only implement Google Gemini via `google-genai`, but the
interface allows us to add more providers later without touching UI code.
"""

import os
from typing import AsyncIterator, Dict, Any
from pathlib import Path
from datetime import datetime, timezone

try:
    from google import genai
    from google.genai import types
except ImportError:  # pragma: no cover – dependency is optional until used
    genai = None  # type: ignore

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


class GeminiClient(LLMClient):
    """Wrapper around google-genai async streaming API."""

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash-preview-05-20"):
        if genai is None:
            raise RuntimeError("google-genai not installed but GeminiClient requested")
        self._client = genai.Client(api_key=api_key)
        self.model_name = model_name

    async def stream(
        self,
        prompt: str,
        content: str,
        generation_config: Dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        if generation_config is None:
            generation_config = types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens =8192,
                response_mime_type="text/plain",
                thinking_config = types.ThinkingConfig(
            thinking_budget=0,
        )
            )
        user_content = f"{prompt}\n\n{content}" if content else prompt
        stream = self._client.aio.models.generate_content_stream(
            model=self.model_name,
            contents=user_content,
            config=generation_config,
        )

        log_path = Path("/tmp/spegel.log")
        # Write prompt to log with timestamp
        try:
            with log_path.open("a", encoding="utf-8") as fp:
                fp.write(f"\n[{datetime.now(timezone.utc).isoformat()}] PROMPT\n")
                fp.write(user_content + "\n")
        except Exception:
            pass  # Logging failures shouldn't break the app

        collected: list[str] = []

        async for chunk in await stream:
            try:
                text = chunk.candidates[0].content.parts[0].text  # type: ignore[attr-defined]
                if text:
                    collected.append(text)
                    yield text
            except Exception:
                continue

        # After streaming finished, write output to log
        if collected:
            try:
                with log_path.open("a", encoding="utf-8") as fp:
                    fp.write(f"[{datetime.now(timezone.utc).isoformat()}] OUTPUT\n")
                    fp.write("".join(collected) + "\n")
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


def get_default_client() -> tuple[LLMClient | None, bool]:
    """Return an LLMClient instance if credentials exist, else (None, False)."""
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key and genai is not None:
        return GeminiClient(api_key), True
    return None, False


if __name__ == "__main__":
    import argparse, asyncio, sys

    parser = argparse.ArgumentParser(
        description="Quick CLI wrapper around the configured LLM to answer a prompt."
    )
    parser.add_argument("prompt", help="User prompt/question to send to the model")
    args = parser.parse_args()

    client, ok = get_default_client()
    if not ok or client is None:
        print("Error: GEMINI_API_KEY not set or google-genai unavailable", file=sys.stderr)
        sys.exit(1)

    async def _main() -> None:
        async for chunk in client.stream(args.prompt, ""):
            print(chunk, end="", flush=True)
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        pass 