"""
Check what files are in the S3 bucket and why downloads might not be working.
Run: python check_bucket_files.py
"""

import os
import sys
from datetime import timezone

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    print("❌ boto3 not installed. Install with: python -m pip install boto3")
    sys.exit(1)

def check_bucket_files():
    """Check files in S3 bucket."""
    bucket_name = os.getenv("S3_BUCKET_NAME", "consultaion-video")
    print("="*60)
    print("S3 BUCKET FILE CHECKER")
    print("="*60)
    print(f"\nBucket: {bucket_name}\n")
    
    try:
        s3_client = boto3.client('s3')
        
        # List all objects
        print("Scanning bucket...")
        paginator = s3_client.get_paginator("list_objects_v2")
        
        total_files = 0
        webm_files = []
        other_files = []
        files_by_year = {}
        
        for page in paginator.paginate(Bucket=bucket_name):
            if "Contents" not in page:
                print("⚠️  Bucket appears to be empty or no access to list objects")
                return
            
            for obj in page.get("Contents", []):
                total_files += 1
                file_key = obj["Key"]
                file_size = obj["Size"]
                file_year = obj["LastModified"].replace(tzinfo=timezone.utc).year
                file_date = obj["LastModified"]
                
                # Track by year
                if file_year not in files_by_year:
                    files_by_year[file_year] = []
                files_by_year[file_year].append(file_key)
                
                # Check file extension
                if file_key.lower().endswith(".webm"):
                    webm_files.append({
                        "key": file_key,
                        "size": file_size,
                        "year": file_year,
                        "date": file_date
                    })
                else:
                    other_files.append({
                        "key": file_key,
                        "size": file_size,
                        "year": file_year
                    })
        
        # Print summary
        print(f"\n📊 SUMMARY")
        print(f"   Total files in bucket: {total_files}")
        print(f"   .webm files: {len(webm_files)}")
        print(f"   Other files: {len(other_files)}")
        
        # Files by year
        print(f"\n📅 FILES BY YEAR:")
        for year in sorted(files_by_year.keys(), reverse=True):
            count = len(files_by_year[year])
            print(f"   {year}: {count} files")
        
        # Webm files details
        if webm_files:
            print(f"\n🎥 .WEBM FILES FOUND ({len(webm_files)}):")
            print("-" * 60)
            
            # Group by year
            webm_by_year = {}
            for f in webm_files:
                year = f["year"]
                if year not in webm_by_year:
                    webm_by_year[year] = []
                webm_by_year[year].append(f)
            
            for year in sorted(webm_by_year.keys(), reverse=True):
                files = webm_by_year[year]
                print(f"\n   Year {year} ({len(files)} files):")
                for f in files[:5]:  # Show first 5
                    size_mb = f["size"] / (1024 * 1024)
                    print(f"      - {f['key']}")
                    print(f"        Size: {size_mb:.2f} MB | Date: {f['date']}")
                if len(files) > 5:
                    print(f"      ... and {len(files) - 5} more files")
            
            # Check 2023 filter
            webm_2023 = [f for f in webm_files if f["year"] == 2023]
            print(f"\n🔍 FILTER ANALYSIS:")
            print(f"   Files matching current filter (2023): {len(webm_2023)}")
            if len(webm_2023) == 0:
                print(f"   ⚠️  No files from 2023 found!")
                print(f"   💡 The code is currently filtering for files from 2023 only")
                if webm_files:
                    latest_year = max(f["year"] for f in webm_files)
                    print(f"   💡 Latest year with .webm files: {latest_year}")
                    print(f"   💡 To download files from {latest_year}, modify the year filter in aws_bucket.py")
        else:
            print(f"\n⚠️  NO .WEBM FILES FOUND")
            print(f"   The bucket contains {total_files} files but none are .webm")
            if other_files:
                print(f"\n   File types found:")
                extensions = {}
                for f in other_files[:20]:
                    ext = f["key"].split('.')[-1].lower() if '.' in f["key"] else "no extension"
                    extensions[ext] = extensions.get(ext, 0) + 1
                for ext, count in sorted(extensions.items(), key=lambda x: x[1], reverse=True):
                    print(f"      .{ext}: {count} files")
        
        # Check downloaded files
        downloaded_file = "logs/downloaded_files.json"
        if os.path.exists(downloaded_file):
            import json
            with open(downloaded_file, 'r') as f:
                downloaded = set(json.load(f))
            print(f"\n📥 DOWNLOADED FILES:")
            print(f"   Already downloaded: {len(downloaded)} files")
            if webm_files:
                not_downloaded = [f for f in webm_files if f["key"] not in downloaded and f["year"] == 2023]
                print(f"   Available to download (2023): {len(not_downloaded)} files")
        
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        print(f"❌ Error accessing bucket: {error_code}")
        print(f"   Message: {e}")
        if error_code == '403':
            print("\n   💡 No permission to list bucket contents")
            print("   Check IAM permissions for s3:ListBucket")
        elif error_code == '404':
            print("\n   💡 Bucket does not exist")
            print("   Check bucket name is correct")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    check_bucket_files()

