"""
Test script to verify credentials are loaded correctly.
Run: python test_credentials.py
"""

import os
import sys

# Try to load dotenv
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ python-dotenv loaded successfully")
except ImportError:
    print("⚠️  python-dotenv not installed. Install with: python -m pip install python-dotenv")
    print("   Will check environment variables directly...")
except Exception as e:
    print(f"⚠️  Error loading .env: {e}")

print("\n" + "="*50)
print("CREDENTIALS CHECK")
print("="*50)

# Check AWS Credentials
print("\n📦 AWS Credentials:")
aws_key = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret = os.getenv('AWS_SECRET_ACCESS_KEY')
aws_region = os.getenv('AWS_REGION', 'Not set')

if aws_key:
    print(f"  ✅ AWS_ACCESS_KEY_ID: {aws_key[:10]}...{aws_key[-4:]}")
else:
    print("  ❌ AWS_ACCESS_KEY_ID: Not found")
    print("     Note: boto3 will also check ~/.aws/credentials")

if aws_secret:
    print(f"  ✅ AWS_SECRET_ACCESS_KEY: {'*' * 20}...{aws_secret[-4:]}")
else:
    print("  ❌ AWS_SECRET_ACCESS_KEY: Not found")

print(f"  📍 AWS_REGION: {aws_region}")

# Check Hugging Face Token
print("\n🤗 Hugging Face:")
hf_token = os.getenv('HF_TOKEN')
if hf_token:
    print(f"  ✅ HF_TOKEN: {hf_token[:10]}...{hf_token[-4:]}")
else:
    print("  ❌ HF_TOKEN: Not found")
    print("     Get token from: https://huggingface.co/settings/tokens")

# Check S3 Bucket Name
print("\n🪣 S3 Bucket:")
bucket_name = os.getenv('S3_BUCKET_NAME', 'access-oap-prod-twilio-bucket')
print(f"  📍 S3_BUCKET_NAME: {bucket_name}")

# Summary
print("\n" + "="*50)
print("SUMMARY")
print("="*50)

all_set = bool(aws_key and aws_secret and hf_token)
if all_set:
    print("✅ All credentials are set!")
    print("   You can now run: python src/main.py")
else:
    print("⚠️  Some credentials are missing.")
    print("\n   To fix:")
    print("   1. Create .env file in project root")
    print("   2. Add your credentials (see SETUP_CREDENTIALS.md)")
    print("   3. Or set environment variables directly")

# Check if .env file exists
env_file = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(env_file):
    print(f"\n✅ .env file found at: {env_file}")
else:
    print(f"\n⚠️  .env file not found at: {env_file}")
    print("   Create it with your credentials (see SETUP_CREDENTIALS.md)")

