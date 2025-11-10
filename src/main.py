import logging
import time
from pathlib import Path
import sys, os

# Ensure src/ directory is on sys.path for imports
sys.path.append(os.path.dirname(__file__))

from aws_bucket import S3ProcessingPipeline
from transcription import AudioTranscriber
from config import TranscriptionConfig
from embedding_engine import EmbeddingEngine  # <-- Added import


def setup_logging():
    """Setup logging with directory creation"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logger = logging.getLogger("main_execution")
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        main_handler = logging.FileHandler("logs/main_execution.log")
        main_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        main_handler.setFormatter(main_formatter)
        logger.addHandler(main_handler)

        download_handler = logging.FileHandler("logs/download.log")
        download_formatter = logging.Formatter("%(asctime)s - %(message)s")
        download_handler.setFormatter(download_formatter)
        download_logger = logging.getLogger("download")
        download_logger.addHandler(download_handler)
        download_logger.setLevel(logging.INFO)

        stats_handler = logging.FileHandler("logs/stats.log")
        stats_formatter = logging.Formatter("%(asctime)s - %(message)s")
        stats_handler.setFormatter(stats_formatter)
        stats_logger = logging.getLogger("stats")
        stats_logger.addHandler(stats_handler)
        stats_logger.setLevel(logging.INFO)

    return logger


def ensure_directories():
    """Ensure all required directories exist"""
    directories = ["output", "transcriptions", "logs", "data/faiss_index"]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def process_files(transcriber, local_dir, logger, pipeline):
    """Process audio files in batches."""
    stats_logger = logging.getLogger("stats")
    processed_count = 0
    failed_count = 0

    try:
        files = list(Path(local_dir).rglob("*.webm"))
        if not files:
            logger.info("No files to process")
            return

        batch_size = 4
        logger.info(f"Found {len(files)} files to process")

        current_batch = files[:batch_size]
        logger.info(f"Processing batch of {len(current_batch)} files")

        results = transcriber.process_batch(current_batch)

        for file_path, result in zip(current_batch, results):
            try:
                if result:
                    processed_count += 1
                    logger.info(f"Successfully processed: {file_path}")

                    wav_path = file_path.with_suffix('.wav')
                    if wav_path.exists():
                        logger.info(f"Deleted wav file: {wav_path}")

                    stats_logger.info(f"Processed file count: {processed_count}")
                else:
                    failed_count += 1
                    logger.error(f"Failed to process: {file_path}")

            except Exception as e:
                logger.error(f"Failed to clean up files for {file_path}: {e}")

    except Exception as e:
        logger.error(f"Error in process_files: {str(e)}")
        raise


def build_embeddings(logger):
    """Build FAISS vector DB from new text files."""
    try:
        transcripts_dir = Path("transcriptions/downloads")
        if not transcripts_dir.exists():
            logger.warning("Transcriptions directory not found, skipping embedding.")
            return

        txt_files = list(transcripts_dir.glob("*.txt"))
        if not txt_files:
            logger.info("No new transcription text files found for embedding.")
            return

        logger.info(f"Found {len(txt_files)} transcription files for embedding.")
        engine = EmbeddingEngine()
        engine.build_and_save_index_for_each_file(str(transcripts_dir))
        logger.info("✅ Embedding completed successfully.")

    except Exception as e:
        logger.error(f"Error in build_embeddings: {str(e)}")


def main():
    logger = setup_logging()
    logger.info("🚀 Starting the application")

    try:
        ensure_directories()
        bucket_name = os.getenv("S3_BUCKET_NAME", "consultaion-video")
        local_dir = "./downloads"

        config = TranscriptionConfig()
        transcriber = AudioTranscriber(config)
        pipeline = S3ProcessingPipeline(bucket_name, download_dir=local_dir)

        while True:
            try:
                # Check for files
                files_in_output = list(Path(local_dir).rglob("*.webm"))

                if not files_in_output:
                    logger.info("No .webm files to process, downloading new files...")
                    pipeline.start_processing()
                    time.sleep(30)
                    continue

                # Process audio files
                logger.info(f"Found {len(files_in_output)} files to process")
                process_files(transcriber, local_dir, logger, pipeline)

                # After transcription, build embeddings
                logger.info("Checking for new transcription files to embed...")
                build_embeddings(logger)

                # Wait a bit before next loop
                time.sleep(15)

            except Exception as e:
                logger.error(f"Error in main loop: {str(e)}")
                time.sleep(10)
                continue

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
