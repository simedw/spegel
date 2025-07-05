#!/usr/bin/env python3
"""
Installation script for Amazon Bedrock integration with Spegel.
Sets up dependencies, checks credentials, and creates configuration.
"""

import os
import subprocess
import sys
from pathlib import Path


def check_command(command: str) -> bool:
    """Check if a command is available."""
    try:
        subprocess.run([command, "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def install_dependencies():
    """Install required Python packages."""
    print("📦 Installing Bedrock dependencies...")

    # Check if boto3 is installed
    try:
        import boto3
        print("✅ boto3 already installed")
    except ImportError:
        print("📥 Installing boto3...")
        subprocess.run([sys.executable, "-m", "pip", "install", "boto3"], check=True)
        print("✅ boto3 installed")

    # Check if AWS CLI is available
    if check_command("aws"):
        print("✅ AWS CLI already installed")
    else:
        print("⚠️  AWS CLI not found")
        print("   Install with: pip install awscli")
        print("   Or visit: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html")


def check_aws_credentials():
    """Check if AWS credentials are configured."""
    print("🔍 Checking AWS credentials...")

    # Check for credentials file
    aws_creds_file = Path.home() / ".aws" / "credentials"
    if aws_creds_file.exists():
        print("✅ AWS credentials file found")
        return True

    # Check for environment variables
    if os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"):
        print("✅ AWS credentials found in environment variables")
        return True

    # Check for IAM role (if on EC2)
    try:
        import boto3
        session = boto3.Session()
        credentials = session.get_credentials()
        if credentials:
            print("✅ AWS credentials available (possibly IAM role)")
            return True
    except Exception:
        pass

    print("❌ No AWS credentials found")
    print("   Run: aws configure")
    print("   Or set environment variables:")
    print("   export AWS_ACCESS_KEY_ID=your_key")
    print("   export AWS_SECRET_ACCESS_KEY=your_secret")
    return False


def setup_environment():
    """Help user set up environment variables."""
    print("\n🔧 Environment Setup")
    print("=" * 40)

    model = os.getenv("BEDROCK_MODEL")
    if model:
        print(f"✅ BEDROCK_MODEL: {model}")
    else:
        print("⚠️  BEDROCK_MODEL not set")
        print("   Add to your shell profile: export BEDROCK_MODEL=claude-3-5-sonnet")
        print("   Available models: claude-3-haiku, claude-3-sonnet, claude-3-5-sonnet, claude-3-7-sonnet")

    region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    print(f"🌍 AWS Region: {region}")

    profile = os.getenv("AWS_PROFILE")
    if profile:
        print(f"👤 AWS Profile: {profile}")
    else:
        print("👤 Using default AWS profile")


def test_bedrock_access():
    """Test if Bedrock models are accessible."""
    print("\n🧪 Testing Bedrock Access")
    print("=" * 40)

    try:
        import boto3

        # Test with default credentials
        client = boto3.client('bedrock-runtime', region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"))

        # Try to list foundation models (if available)
        try:
            bedrock_client = boto3.client('bedrock', region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
            response = bedrock_client.list_foundation_models()
            available_models = [model['modelId'] for model in response.get('modelSummaries', [])]

            claude_models = [m for m in available_models if 'claude' in m.lower()]
            if claude_models:
                print(f"✅ Found {len(claude_models)} Claude models available")
                for model in claude_models[:3]:  # Show first 3
                    print(f"   - {model}")
                if len(claude_models) > 3:
                    print(f"   ... and {len(claude_models) - 3} more")
            else:
                print("⚠️  No Claude models found - you may need to request access")

        except Exception as e:
            print(f"⚠️  Could not list models: {e}")
            print("   This is normal - you may still have access to specific models")

        print("✅ Bedrock client initialized successfully")
        return True

    except Exception as e:
        print(f"❌ Bedrock access failed: {e}")
        print("   Make sure you have:")
        print("   1. Valid AWS credentials")
        print("   2. Bedrock permissions in your IAM policy")
        print("   3. Model access requested in AWS Console → Bedrock")
        return False


def create_config():
    """Create sample configuration file."""
    config_file = Path(".spegel.toml")

    if config_file.exists():
        print(f"✅ {config_file} already exists")
        return

    print("📝 Creating sample .spegel.toml configuration...")

    config_content = """# Spegel Configuration with Amazon Bedrock
[settings]
default_view = "terminal"
app_title = "Spegel"

[llm]
provider = "bedrock"
bedrock_region = "us-east-1"
# Model to use (friendly names): claude-3-haiku, claude-3-sonnet, claude-3-5-sonnet, claude-3-7-sonnet, etc.
bedrock_model = "claude-3-5-sonnet"

[ui]
show_icons = true

# Add your custom views here
[[views]]
id = "raw"
name = "Raw View"
hotkey = "1"
order = 1
enabled = true
auto_load = true
description = "Clean HTML rendering (no LLM)"
icon = "📄"
prompt = ""

[[views]]
id = "terminal"
name = "Terminal"
hotkey = "2"
order = 2
enabled = true
auto_load = true
description = "Terminal-optimized markdown"
icon = "💻"
prompt = "Transform this webpage into clean, terminal-friendly markdown..."
"""

    with open(config_file, "w") as f:
        f.write(config_content)

    print(f"✅ Created {config_file}")
    print("   Edit this file to customize your Bedrock model and settings")


def main():
    """Main installation process."""
    print("🚀 Amazon Bedrock Integration Setup for Spegel")
    print("=" * 50)

    # Install dependencies
    install_dependencies()

    # Check AWS credentials
    has_credentials = check_aws_credentials()

    # Setup environment guidance
    setup_environment()

    # Test Bedrock access
    if has_credentials:
        test_bedrock_access()

    # Create configuration
    create_config()

    print("\n🎉 Setup Complete!")
    print("=" * 20)
    print("Next steps:")
    print("1. Request model access in AWS Console → Amazon Bedrock → Model access")
    print("2. Set environment variables or edit .spegel.toml")
    print("3. Run: spegel to start browsing with Bedrock!")

    print("\n💡 Quick start:")
    print("export BEDROCK_MODEL=claude-3-5-sonnet")
    print("spegel https://example.com")


if __name__ == "__main__":
    main()
