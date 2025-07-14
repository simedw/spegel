from __future__ import annotations

from typing import List

from .config import View
from .llm import LLMClient, create_client, LLMAuthenticationError
from .web import extract_clean_text, html_to_markdown


"""View processing logic for Spegel.

For now this module contains thin wrappers that will eventually host the full
HTML → markdown transformations for each view.  Currently the heavy logic is
still inside `main.py`; this file provides stubs so the new architecture is in
place without breaking runtime behaviour.
"""

__all__ = ["stream_view"]


def _get_view_llm_client(
    view: View, default_client: LLMClient | None
) -> LLMClient | None:
    """Get the appropriate LLM client for a view, considering model overrides."""
    # If view has a specific model override, create a new client
    if view.model and view.model.strip():
        view_client = create_client(model=view.model.strip())
        if view_client is not None:
            return view_client
        # Fall back to default client if view-specific model fails

    # Use the default client
    return default_client


async def stream_view(
    view: View,
    raw_html: str,
    default_llm_client: LLMClient | None,
    url: str | None,
):
    """Yield markdown chunks for a view, supporting streaming updates."""
    if view.id == "raw":
        yield html_to_markdown(raw_html, url)
        return

    clean_text = extract_clean_text(raw_html, url, max_chars=100_000)

    # Get the appropriate LLM client for this view (considering model overrides)
    view_client = _get_view_llm_client(view, default_llm_client)

    if view_client is None:
        yield "## LLM not available\n\nSet one of: OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, or SPEGEL_MODEL to enable AI processing."
        return

    full_prompt = f"{view.prompt}\n\nWebpage content:\n{clean_text}"
    buffer: str = ""

    try:
        async for chunk in view_client.stream(full_prompt, ""):
            if not chunk:
                continue
            buffer += chunk
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                yield line + "\n"

        if buffer:
            yield buffer

    except LLMAuthenticationError as e:
        # Handle authentication errors with user-friendly message
        yield f"## 🔐 Authentication Error\n\n{str(e)}\n\n**Quick Setup:**\n\n1. Create a `.env` file in your project directory\n2. Add your API key: `SPEGEL_API_KEY=your_api_key_here`\n3. Or set provider-specific keys like `OPENAI_API_KEY=your_key`\n4. Restart the application"
    except RuntimeError as e:
        # Handle other RuntimeErrors
        yield f"## ❌ Error\n\n{str(e)}"
    except Exception as e:
        # Handle other unexpected errors
        yield f"## ❌ Unexpected Error\n\n{str(e)}"
