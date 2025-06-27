from __future__ import annotations

from typing import List

from .config import View
from .llm import LLMClient
from .web import extract_clean_text, html_to_markdown


"""View processing logic for Spegel.

For now this module contains thin wrappers that will eventually host the full
HTML → markdown transformations for each view.  Currently the heavy logic is
still inside `main.py`; this file provides stubs so the new architecture is in
place without breaking runtime behaviour.
"""

__all__ = ["process_view", "stream_view"]


async def process_view(
    view: View,
    raw_html: str,
    llm_client: LLMClient | None,
    url: str | None,
) -> str:
    """Generate markdown for a single view.

    For the *raw* view we simply convert the full HTML to markdown.
    For all other views we:
      1. Extract cleaned text (limited to 8 k chars for token safety).
      2. Prepend the view.prompt.
      3. Send the combined text to the LLM and gather the streamed response.
    """

    if view.id == "raw":
        return html_to_markdown(raw_html, url)

    # Non-raw views need the LLM
    clean_text = extract_clean_text(raw_html, url, max_chars=8000)

    if llm_client is None:
        return "## LLM not available\n\nSet GEMINI_API_KEY to enable AI processing."

    full_prompt = f"{view.prompt}\n\nWebpage content:\n{clean_text}"

    parts: List[str] = []
    async for chunk in llm_client.stream(full_prompt, ""):
        if chunk:
            parts.append(chunk)

    return "\n".join(parts)


async def stream_view(
    view: View,
    raw_html: str,
    llm_client: LLMClient | None,
    url: str | None,
):
    """Yield markdown chunks for a view, supporting streaming updates."""
    if view.id == "raw":
        yield html_to_markdown(raw_html, url)
        return

    clean_text = extract_clean_text(raw_html, url, max_chars=8000)
    if llm_client is None:
        yield "## LLM not available\n\nSet GEMINI_API_KEY to enable AI processing."
        return

    full_prompt = f"{view.prompt}\n\nWebpage content:\n{clean_text}"
    buffer: str = ""
    async for chunk in llm_client.stream(full_prompt, ""):
        if not chunk:
            continue
        buffer += chunk
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            yield line + "\n"

    if buffer:
        yield buffer 