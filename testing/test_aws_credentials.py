"""
Test AWS credentials and S3 bucket access.
Run: python test_aws_credentials.py
"""

import os
import sys

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded .env file")
except ImportError:
    print("⚠️  python-dotenv not installed")
except Exception as e:
    print(f"⚠️  Error loading .env: {e}")

print("\n" + "="*60)
print("AWS CREDENTIALS TEST")
print("="*60)

# Check environment variables
print("\n1. Checking Environment Variables:")
aws_key = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret = os.getenv('AWS_SECRET_ACCESS_KEY')
aws_region = os.getenv('AWS_REGION', 'us-east-1')
bucket_name = os.getenv('S3_BUCKET_NAME')

if aws_key:
    print(f"   ✅ AWS_ACCESS_KEY_ID: {aws_key[:10]}...{aws_key[-4:]}")
else:
    print("   ❌ AWS_ACCESS_KEY_ID: Not found")

if aws_secret:
    print(f"   ✅ AWS_SECRET_ACCESS_KEY: {'*' * 20}...{aws_secret[-4:]}")
else:
    print("   ❌ AWS_SECRET_ACCESS_KEY: Not found")

print(f"   📍 AWS_REGION: {aws_region}")
print(f"   🪣 S3_BUCKET_NAME: {bucket_name}")

# Test boto3 credentials
print("\n2. Testing AWS Credentials:")
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    
    # Try to get caller identity
    try:
        sts_client = boto3.client('sts', region_name=aws_region)
        identity = sts_client.get_caller_identity()
        print("   ✅ AWS credentials are valid!")
        print(f"      Account ID: {identity.get('Account', 'Unknown')}")
        print(f"      User ARN: {identity.get('Arn', 'Unknown')}")
    except NoCredentialsError:
        print("   ❌ No AWS credentials found")
        print("\n   Solutions:")
        print("   1. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in .env file")
        print("   2. Or configure AWS CLI: aws configure")
        print("   3. Or set environment variables directly")
        sys.exit(1)
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        print(f"   ❌ Credential error: {error_code}")
        print(f"      Message: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"   ❌ Error: {e}")
        sys.exit(1)
    
    # Test S3 access
    print("\n3. Testing S3 Bucket Access:")
    try:
        s3_client = boto3.client('s3', region_name=aws_region)
        
        # Check if bucket exists and is accessible
        try:
            s3_client.head_bucket(Bucket=bucket_name)
            print(f"   ✅ Bucket '{bucket_name}' is accessible!")
            
            # List some objects
            print(f"\n   Listing objects in bucket '{bucket_name}'...")
            response = s3_client.list_objects_v2(Bucket=bucket_name, MaxKeys=5)
            
            if 'Contents' in response:
                print(f"   Found {len(response['Contents'])} objects (showing first 5):")
                for obj in response['Contents'][:5]:
                    print(f"      - {obj['Key']} ({obj['Size']} bytes)")
            else:
                print("   Bucket is empty or no objects found")
                
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            if error_code == '403':
                print(f"   ❌ No permission to access bucket '{bucket_name}'")
                print("\n   Solutions:")
                print("   1. Check IAM permissions for S3 access")
                print("   2. Verify bucket name is correct")
                print("   3. Check bucket policy allows your user")
            elif error_code == '404':
                print(f"   ❌ Bucket '{bucket_name}' does not exist")
                print("\n   Solutions:")
                print("   1. Verify bucket name is correct")
                print("   2. Check AWS region is correct")
            else:
                print(f"   ❌ Error accessing bucket: {error_code}")
                print(f"      Message: {e}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            
    except Exception as e:
        print(f"   ❌ Failed to create S3 client: {e}")
        
except ImportError:
    print("   ❌ boto3 not installed")
    print("   Install with: python -m pip install boto3")
    sys.exit(1)

print("\n" + "="*60)
print("SUMMARY")
print("="*60)

if aws_key and aws_secret:
    print("✅ Environment variables are set")
    print("✅ AWS credentials are working")
    print("\n   You can now run: python src/main.py")
else:
    print("⚠️  Some credentials are missing")
    print("\n   Create .env file with:")
    print("   AWS_ACCESS_KEY_ID=your_key")
    print("   AWS_SECRET_ACCESS_KEY=your_secret")
    print("   S3_BUCKET_NAME=your_bucket_name")

