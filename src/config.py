from dataclasses import dataclass, field
import torch
from typing import Optional, List
from pathlib import Path
import logging
import os

# Load environment variables
try:
    from dotenv import load_dotenv
    try:
        load_dotenv()
    except Exception as e:
        # If .env file has parsing errors, log but don't crash
        logging.warning(f"Could not load .env file: {e}")
        logging.warning("Will use environment variables directly")
except ImportError:
    # dotenv not installed, will use environment variables directly
    pass

@dataclass
class TranscriptionConfig:
    model_name: str = "openai/whisper-large-v3-turbo"
    device: Optional[str] = field(default=None)
    output_dir: str = "transcriptions"
    chunk_length: int = 60
    batch_size: int = 24
    max_workers: int = 3
    use_flash_attention: bool = True
    supported_formats: List[str] = field(default_factory=lambda: ['.wav', '.mp3', '.m4a', '.webm', '.flac', '.ogg'])
    torch_dtype: Optional[torch.dtype] = field(default=None)
    hf_token: Optional[str] = field(default=None)
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Load Hugging Face token from environment if not provided
        if self.hf_token is None:
            self.hf_token = os.getenv("HF_TOKEN")
        
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = self.torch_dtype or (torch.float16 if self.device == "cuda" else torch.float32)
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)