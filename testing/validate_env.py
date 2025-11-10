"""
Validate and fix .env file format.
Run: python validate_env.py
"""

import os
from pathlib import Path

def validate_env_file(env_path=".env"):
    """Validate .env file format and show errors."""
    env_file = Path(env_path)
    
    if not env_file.exists():
        print(f"❌ .env file not found at: {env_file.absolute()}")
        print("\n   Create a .env file with this format:")
        print("   AWS_ACCESS_KEY_ID=your_key")
        print("   AWS_SECRET_ACCESS_KEY=your_secret")
        print("   HF_TOKEN=your_token")
        return False
    
    print(f"✅ Found .env file at: {env_file.absolute()}\n")
    
    # Try to parse with dotenv
    try:
        from dotenv import dotenv_values
        values = dotenv_values(env_path)
        print("✅ .env file parsed successfully!")
        print(f"   Found {len(values)} variables\n")
        return True
    except Exception as e:
        print(f"❌ Error parsing .env file: {e}\n")
        
        # Read file and check line by line
        print("Checking file line by line...\n")
        with open(env_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines, 1):
            line = line.rstrip('\n\r')
            
            # Skip empty lines and comments
            if not line.strip() or line.strip().startswith('#'):
                continue
            
            # Check for common issues
            issues = []
            
            # Check for spaces around =
            if ' = ' in line or line.startswith(' ') or '= ' in line or ' =' in line:
                if ' = ' in line:
                    issues.append("❌ Spaces around = sign (use KEY=value not KEY = value)")
                if line.strip() != line:
                    issues.append("❌ Leading/trailing spaces")
            
            # Check for quotes (usually not needed)
            if line.count('"') > 0 or line.count("'") > 0:
                if not (line.startswith('"') and line.endswith('"')) and '"' in line:
                    issues.append("⚠️  Unmatched quotes")
            
            # Check for special characters that might cause issues
            if '#' in line and '=' in line:
                if line.index('#') < line.index('='):
                    pass  # Comment before = is fine
                else:
                    issues.append("⚠️  Comment on same line as value")
            
            if issues:
                print(f"Line {i}: {line}")
                for issue in issues:
                    print(f"  {issue}")
                print()
        
        print("\n💡 Fix suggestions:")
        print("   1. Remove spaces around = sign")
        print("   2. Remove quotes unless value has spaces")
        print("   3. Put comments on separate lines")
        print("   4. Ensure no trailing spaces")
        print("\n   Correct format example:")
        print("   AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE")
        print("   AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY")
        print("   HF_TOKEN=hf_your_token_here")
        
        return False

def create_example_env():
    """Create an example .env file."""
    example_content = """# AWS Credentials
AWS_ACCESS_KEY_ID=your_aws_access_key_here
AWS_SECRET_ACCESS_KEY=your_aws_secret_key_here
AWS_REGION=us-east-1

# Hugging Face Token
HF_TOKEN=your_huggingface_token_here

# S3 Bucket Name
S3_BUCKET_NAME=access-oap-prod-twilio-bucket
"""
    
    env_file = Path(".env")
    if env_file.exists():
        print("⚠️  .env file already exists. Not overwriting.")
        return
    
    with open(env_file, 'w') as f:
        f.write(example_content)
    
    print("✅ Created example .env file")
    print("   Edit it with your actual credentials")

if __name__ == "__main__":
    print("="*60)
    print("ENV FILE VALIDATOR")
    print("="*60)
    print()
    
    if validate_env_file():
        print("✅ Your .env file is valid!")
        print("\n   You can now run: python test_credentials.py")
    else:
        print("\n" + "="*60)
        response = input("\nWould you like to create an example .env file? (y/n): ")
        if response.lower() == 'y':
            create_example_env()

