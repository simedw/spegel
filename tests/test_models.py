#!/usr/bin/env python3
"""
Test which Bedrock models are available in your AWS account.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spegel.llm import BedrockClient, BEDROCK_MODELS


async def test_model(model_name: str, model_id: str):
    """Test if a specific model is available."""
    try:
        client = BedrockClient(model_name=model_name)
        
        # Try a simple request
        chunks = []
        async for chunk in client.stream("Say hello", ""):
            chunks.append(chunk)
        
        result = "".join(chunks)
        
        if "AccessDeniedException" in result or "Bedrock Access Required" in result:
            return False, "Access denied"
        else:
            return True, result[:100] + "..." if len(result) > 100 else result
            
    except Exception as e:
        return False, str(e)


async def main():
    """Test all available models."""
    print("🧪 Testing Available Bedrock Models")
    print("=" * 50)
    
    for friendly_name, model_id in BEDROCK_MODELS.items():
        print(f"\n🔍 Testing {friendly_name} ({model_id})")
        
        available, result = await test_model(friendly_name, model_id)
        
        if available:
            print(f"✅ Available - Response: {result}")
        else:
            print(f"❌ Not available - {result}")
    
    print("\n" + "=" * 50)
    print("💡 Tip: Use available models with BEDROCK_MODEL environment variable")
    print("Example: BEDROCK_MODEL=claude-3-5-sonnet")


if __name__ == "__main__":
    asyncio.run(main())
