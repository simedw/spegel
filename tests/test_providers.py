import json
import os

import httpx
import pytest
from dotenv import load_dotenv

from spegel.llm import LiteLLMClient


PROVIDERS = [
    ("gpt-5.6-luna", "OPENAI_API_KEY"),
    ("anthropic/claude-haiku-5-5", "ANTHROPIC_API_KEY"),
    ("gemini/gemini-3.5-flash-lite", "GEMINI_API_KEY"),
]


def provider_events(model):
    if model.startswith("anthropic/"):
        return [
            {
                "type": "message_start",
                "message": {
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-haiku-5-5",
                    "content": [],
                    "usage": {"input_tokens": 10, "output_tokens": 0},
                },
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "SPEGEL_OK"},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 4},
            },
            {"type": "message_stop"},
        ]
    if model.startswith("gemini/"):
        return [
            {
                "candidates": [
                    {
                        "index": 0,
                        "content": {"role": "model", "parts": [{"text": "SPEGEL_OK"}]},
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 4,
                    "totalTokenCount": 14,
                },
            }
        ]
    return [
        {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": model,
            "choices": [
                {"index": 0, "delta": {"content": "SPEGEL_OK"}, "finish_reason": "stop"}
            ],
        }
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("model,key_name", PROVIDERS)
async def test_provider_stream_through_real_adapter(monkeypatch, model, key_name):
    requests = []

    async def send(client, request, **kwargs):
        requests.append(request)
        events = provider_events(model)
        body = "".join(
            (f"event: {event['type']}\n" if "type" in event else "")
            + f"data: {json.dumps(event)}\n\n"
            for event in events
        )
        if not model.startswith(("anthropic/", "gemini/")):
            body += "data: [DONE]\n\n"
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=body,
            request=request,
        )

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    client = LiteLLMClient(model, api_key="test-key", num_retries=0)
    result = "".join(
        [chunk async for chunk in client.stream("Summarize this.", "Test page.")]
    )
    assert result == "SPEGEL_OK"
    assert len(requests) == 1
    payload = json.loads(requests[0].content)
    if model.startswith("gemini/"):
        assert "gemini-3.5-flash-lite" in str(requests[0].url)
        assert payload["generationConfig"]["max_output_tokens"] == 8192
        assert payload["generationConfig"]["temperature"] == 1.0
    else:
        assert payload["model"] == model.removeprefix("anthropic/")
        assert "temperature" not in payload
        token_key = (
            "max_tokens" if model.startswith("anthropic/") else "max_completion_tokens"
        )
        assert payload[token_key] == 8192


@pytest.mark.asyncio
@pytest.mark.parametrize("model,key_name", PROVIDERS)
async def test_live_provider_stream(model, key_name):
    if os.getenv("SPEGEL_LIVE_TESTS") != "1":
        pytest.skip("Set SPEGEL_LIVE_TESTS=1 to make paid provider requests")
    load_dotenv(".env")
    api_key = os.getenv(key_name)
    if not api_key:
        pytest.skip(f"{key_name} is missing")
    client = LiteLLMClient(model, api_key=api_key, timeout=45, num_retries=0)
    result = "".join(
        [
            chunk
            async for chunk in client.stream(
                "Reply with exactly SPEGEL_OK.", "", max_tokens=256
            )
        ]
    )
    assert result.strip() == "SPEGEL_OK"
