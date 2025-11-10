# Load environment variables
try:
    from dotenv import load_dotenv
    try:
        load_dotenv()
    except Exception as e:
        # If .env file has parsing errors, log but don't crash
        import logging
        logging.warning(f"Could not load .env file: {e}")
        logging.warning("Will use environment variables directly")
except ImportError:
    # dotenv not installed, will use environment variables directly
    pass

import logging
from pathlib import Path
from queue import Queue
from threading import Event
from typing import Optional
from datetime import timezone
import json
import boto3
from botocore.exceptions import ClientError



class S3ProcessingPipeline:
    def __init__(
        self, bucket_name: str, download_dir: str = "downloads", max_concurrent: int = 2
    ):
        self.bucket_name = bucket_name
        self.download_dir = Path(download_dir)
        self.max_concurrent = max_concurrent
        
        # Create directories first
        self.download_dir.mkdir(exist_ok=True)
        Path("logs").mkdir(exist_ok=True)

        # Set up logging BEFORE using loggers
        self._setup_logging()
        
        # Initialize S3 client with better error handling
        try:
            self.s3_client = boto3.client("s3")
        except Exception as e:
            self.error_logger.error(f"Failed to initialize S3 client: {e}")
            raise
            
        self.processing_queue = Queue()
        self.stop_event = Event()
        self.active_downloads = 0  # Track number of active downloads
        self.download_complete = Event()  # Signal when download is complete

        # Track downloaded files
        self.downloaded_files_path = Path("logs/downloaded_files.json")
        self.downloaded_files = self._load_downloaded_files()

    def _load_downloaded_files(self):
        """Load previously downloaded files"""
        if self.downloaded_files_path.exists():
            with open(self.downloaded_files_path, 'r') as f:
                return set(json.load(f))
        return set()

    def _save_downloaded_files(self):
        """Save list of downloaded files"""
        with open(self.downloaded_files_path, 'w') as f:
            json.dump(list(self.downloaded_files), f)

    def _setup_logging(self):
        """Configure logging for the pipeline"""
        # Main processing log
        self.logger = logging.getLogger("processing")
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler("logs/processing.log")
        fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        self.logger.addHandler(fh)

        # Error log
        self.error_logger = logging.getLogger("errors")
        self.error_logger.setLevel(logging.ERROR)
        eh = logging.FileHandler("logs/errors.log")
        eh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        self.error_logger.addHandler(eh)

    def verify_bucket_access(self) -> bool:
        """Verify access to S3 bucket"""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            return True
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code")
            if error_code == "403":
                self.error_logger.error(
                    f"No permission to access bucket '{self.bucket_name}'"
                )
            elif error_code == "404":
                self.error_logger.error(f"Bucket '{self.bucket_name}' does not exist")
            return False

    def download_file(self, s3_key: str) -> Optional[Path]:
        """Download a single file from S3 maintaining directory structure"""
        try:
            # Create full local path maintaining S3 structure
            local_path = self.download_dir / s3_key

            # Create parent directories if they don't exist
            local_path.parent.mkdir(parents=True, exist_ok=True)

            self.logger.info(f"Downloading {s3_key} to {local_path}")

            self.s3_client.download_file(self.bucket_name, s3_key, str(local_path))
            
            # Add to downloaded files set
            self.downloaded_files.add(s3_key)
            self._save_downloaded_files()
            
            return local_path

        except Exception as e:
            self.error_logger.error(f"Error downloading {s3_key}: {str(e)}")
            if local_path.exists():
                local_path.unlink()
            return None

    def process_file(self, local_path: Path):
        """Process a single file through transcription"""
        try:
            self.logger.info(f"Starting transcription for {local_path}")
            # Let the main process handle transcription
            self.active_downloads -= 1  # Decrease active downloads count
            return True

        except Exception as e:
            self.error_logger.error(f"Error processing {local_path}: {str(e)}")
            self.active_downloads -= 1  # Ensure we decrease count even on error
            return False

    def start_processing(self):
        """Start the download pipeline with strictly controlled downloads"""
        try:
            # Check credentials first
            try:
                # Try to get caller identity to verify credentials
                sts_client = boto3.client('sts')
                identity = sts_client.get_caller_identity()
                self.logger.info(f"AWS credentials verified. Account: {identity.get('Account', 'Unknown')}")
            except Exception as e:
                self.error_logger.error(f"AWS credentials not configured: {e}")
                self.error_logger.error("Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in .env file")
                return
            
            if not self.verify_bucket_access():
                self.error_logger.error("Failed to verify bucket access")
                return

            self.logger.info(f"Starting download process for bucket: {self.bucket_name}")

            # Get list of .webm files
            paginator = self.s3_client.get_paginator("list_objects_v2")
            s3_files = []
            total_files_checked = 0
            webm_files_found = 0
            already_downloaded = 0
            
            for page in paginator.paginate(Bucket=self.bucket_name):
                if "Contents" not in page:
                    self.logger.warning("No objects found in bucket")
                    break
                    
                for obj in page.get("Contents", []):
                    total_files_checked += 1
                    file_key = obj["Key"]
                    
                    # Check if it's a .webm file
                    if file_key.lower().endswith(".webm"):
                        webm_files_found += 1
                        
                        # Check if already downloaded
                        if file_key in self.downloaded_files:
                            already_downloaded += 1
                            continue
                        
                        s3_files.append(file_key)
                        self.logger.info(f"Found file to download: {file_key}")
                        # Only get 2 files at a time
                        if len(s3_files) >= 2:
                            break
                
                if len(s3_files) >= 2:
                    break
            
            # Log summary
            self.logger.info(f"Scan complete: {total_files_checked} total files, {webm_files_found} .webm files, "
                           f"{already_downloaded} already downloaded, {len(s3_files)} new files to download")
            
            if not s3_files:
                self.logger.warning("No new files found to download")
                self.logger.info("Possible reasons:")
                self.logger.info("  1. No .webm files in bucket")
                self.logger.info("  2. All matching files have already been downloaded")
                return

            # Download and process up to 2 files
            downloaded_count = 0
            for s3_key in s3_files:
                self.logger.info(f"Attempting to download: {s3_key}")
                local_path = self.download_file(s3_key)
                if local_path:
                    downloaded_count += 1
                    self.active_downloads += 1
                    self.process_file(local_path)
                    self.logger.info(f"Successfully downloaded: {local_path}")
                else:
                    self.logger.error(f"Failed to download: {s3_key}")
            
            self.logger.info(f"Download complete: {downloaded_count}/{len(s3_files)} files downloaded")

        except Exception as e:
            self.error_logger.error(f"Pipeline error: {str(e)}")
            self.stop_event.set()

if __name__ == "__main__":
    # Example usage - get bucket name from environment or use default
    import os
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    bucket_name = os.getenv("S3_BUCKET_NAME", "access-oap-prod-twilio-bucket")
    pipeline = S3ProcessingPipeline(
        bucket_name=bucket_name,
        download_dir="downloads",
        max_concurrent=2,
    )

    pipeline.start_processing()