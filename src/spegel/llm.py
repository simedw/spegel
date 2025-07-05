from __future__ import annotations

import os
import logging
from typing import AsyncIterator, Dict, Any

"""Light abstraction layer over one or more LLM back-ends.

Right now we implement Google Gemini via `google-genai` and Amazon Bedrock via boto3,
but the interface allows us to add more providers later without touching UI code.
"""

# Common Bedrock model IDs
BEDROCK_MODELS = {
    # Claude models (Anthropic)
    "claude-3-haiku": "anthropic.claude-3-haiku-20240307-v1:0",
    "claude-3-sonnet": "anthropic.claude-3-sonnet-20240229-v1:0",
    "claude-3-opus": "anthropic.claude-3-opus-20240229-v1:0",
    "claude-3-5-sonnet": "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "claude-3-7-sonnet": "anthropic.claude-3-7-sonnet-20250219-v1:0",
    "claude-opus-4": "anthropic.claude-opus-4-20250514-v1:0",
    "claude-sonnet-4": "anthropic.claude-sonnet-4-20250514-v1:0",

    # Titan models (Amazon)
    "titan-text-express": "amazon.titan-text-express-v1",
    "titan-text-lite": "amazon.titan-text-lite-v1",

    # Jurassic models (AI21)
    "j2-ultra": "ai21.j2-ultra-v1",
    "j2-mid": "ai21.j2-mid-v1",

    # Command models (Cohere)
    "command-text": "cohere.command-text-v14",
    "command-light": "cohere.command-light-text-v14",
}


def get_model_id(model_name: str) -> str:
    """Convert friendly model name to Bedrock model ID."""
    return BEDROCK_MODELS.get(model_name, model_name)


# Configure logger for LLM interactions (disabled by default)
logger = logging.getLogger("spegel.llm")
logger.setLevel(logging.CRITICAL + 1)  # Effectively disabled by default


def enable_llm_logging(level: int = logging.INFO) -> None:
    """Enable LLM interaction logging at the specified level."""
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

try:
    from google import genai
    from google.genai import types
except ImportError:  # pragma: no cover – dependency is optional until used
    genai = None  # type: ignore

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
except ImportError:  # pragma: no cover – dependency is optional until used
    boto3 = None  # type: ignore

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
        yield # This is unreachable, but makes this an async generator


class GeminiClient(LLMClient):
    """Wrapper around google-genai async streaming API."""

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash-lite-preview-06-17"):
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


class BedrockClient(LLMClient):
    """Wrapper around Amazon Bedrock API using boto3."""

    def __init__(self, region: str = "us-east-1", profile_name: str | None = None, model_name: str | None = None):
        if boto3 is None:
            raise RuntimeError("boto3 not installed but BedrockClient requested")

        if not model_name:
            raise RuntimeError("model_name is required for BedrockClient")

        self.model_name = get_model_id(model_name)  # Convert friendly name to full model ID

        try:
            # Create session with optional profile
            session = boto3.Session(profile_name=profile_name) if profile_name else boto3.Session()
            self._bedrock_client = session.client('bedrock-runtime', region_name=region)

        except (ClientError, NoCredentialsError) as e:
            raise RuntimeError(f"Failed to initialize Bedrock client: {e}")

        self.region = region
        self.profile_name = profile_name

    async def stream(
        self,
        prompt: str,
        content: str,
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream response from Amazon Bedrock."""
        user_content = f"{prompt}\n\n{content}" if content else prompt

        # Log the prompt if logging is enabled
        logger.info("Bedrock Prompt: %s", user_content)

        try:
            # Use configurable model via Bedrock
            import json

            # Use the specified model
            model_id = self.model_name

            # Handle different model formats and create appropriate request body
            if "anthropic.claude" in model_id:
                # Claude models use the Messages API format
                body = json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 4000,
                    "messages": [
                        {
                            "role": "user",
                            "content": user_content
                        }
                    ]
                })
            elif "amazon.titan" in model_id:
                # Titan models use a different format
                body = json.dumps({
                    "inputText": user_content,
                    "textGenerationConfig": {
                        "maxTokenCount": 4000,
                        "temperature": 0.7,
                        "topP": 0.9
                    }
                })
            elif "ai21.j2" in model_id:
                # AI21 Jurassic models
                body = json.dumps({
                    "prompt": user_content,
                    "maxTokens": 4000,
                    "temperature": 0.7
                })
            elif "cohere.command" in model_id:
                # Cohere Command models
                body = json.dumps({
                    "prompt": user_content,
                    "max_tokens": 4000,
                    "temperature": 0.7
                })
            else:
                # Default to Claude format for unknown models
                body = json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 4000,
                    "messages": [
                        {
                            "role": "user",
                            "content": user_content
                        }
                    ]
                })

            response = self._bedrock_client.invoke_model(
                modelId=model_id,
                body=body,
                contentType="application/json",
                accept="application/json"
            )

            response_body = json.loads(response['body'].read())

            # Extract response text based on model type
            if "anthropic.claude" in model_id:
                response_text = response_body['content'][0]['text']
            elif "amazon.titan" in model_id:
                response_text = response_body['results'][0]['outputText']
            elif "ai21.j2" in model_id:
                response_text = response_body['completions'][0]['data']['text']
            elif "cohere.command" in model_id:
                response_text = response_body['generations'][0]['text']
            else:
                # Try Claude format first, then fallback
                try:
                    response_text = response_body['content'][0]['text']
                except:
                    response_text = str(response_body)

        except Exception as e:
            logger.warning(f"Bedrock API failed: {e}")
            if "AccessDeniedException" in str(e):
                response_text = f"""# Amazon Bedrock Access Required

Bedrock is enabled but requires additional setup:

## Enable Amazon Bedrock Access
1. Go to AWS Console → Amazon Bedrock
2. Request access to Claude models
3. Ensure your AWS credentials have bedrock permissions

## Current Error
```
{str(e)}
```

*Bedrock is trying to work but needs proper AWS setup.*"""
            else:
                response_text = f"Bedrock API Error: {str(e)}\n\nPlease check your AWS credentials and permissions."

        # Log the complete response if logging is enabled
        logger.info("Bedrock Response: %s", response_text)

        # Simulate streaming by yielding chunks
        chunk_size = 50
        for i in range(0, len(response_text), chunk_size):
            chunk = response_text[i:i + chunk_size]
            yield chunk
# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


def get_default_client(config: Any = None) -> tuple[LLMClient | None, bool]:
    """Return an LLMClient instance if credentials exist, else (None, False).

    Args:
        config: Optional FullConfig object to read LLM settings from

    Priority order:
    1. Config-specified provider
    2. Bedrock (if BEDROCK_MODEL is set)
    3. Gemini (if GEMINI_API_KEY set)
    """
    # Use config if provided
    if config and hasattr(config, 'llm'):
        if config.llm.provider == "bedrock" and boto3 is not None:
            try:
                return BedrockClient(
                    region=config.llm.bedrock_region,
                    profile_name=config.llm.bedrock_profile,
                    model_name=config.llm.bedrock_model
                ), True
            except RuntimeError:
                pass  # Fall through to other options
        elif config.llm.provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key and genai is not None:
                return GeminiClient(api_key), True

    # Check for Bedrock via environment variables
    bedrock_model = os.getenv("BEDROCK_MODEL")
    if bedrock_model and boto3 is not None:
        try:
            region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
            profile = os.getenv("AWS_PROFILE")
            return BedrockClient(region=region, profile_name=profile, model_name=bedrock_model), True
        except RuntimeError:
            pass  # Fall through to other options

    # Fall back to Gemini
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key and genai is not None:
        return GeminiClient(api_key), True

    return None, False


if __name__ == "__main__":
    import argparse
    import asyncio
    import sys

    parser = argparse.ArgumentParser(
        description="Quick CLI wrapper around the configured LLM to answer a prompt."
    )
    parser.add_argument("prompt", help="User prompt/question to send to the model")
    args = parser.parse_args()

    client, ok = get_default_client()
    if not ok or client is None:
        print("Error: No LLM client available. Set GEMINI_API_KEY or configure AWS credentials with BEDROCK_MODEL", file=sys.stderr)
        sys.exit(1)

    async def _main() -> None:
        async for chunk in client.stream(args.prompt, ""):
            print(chunk, end="", flush=True)
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        pass
