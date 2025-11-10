import logging
import time
from pathlib import Path
from typing import Optional, Union, List, Dict, Any
import json
import os

import torch
from transformers import pipeline
from transformers.utils import is_flash_attn_2_available
import numpy as np
from pyannote.audio import Pipeline as DiarizationPipeline

try:
    from huggingface_hub import login as hf_login
except Exception:
    hf_login = None

try:
    from .config import TranscriptionConfig
except ImportError:
    from config import TranscriptionConfig

class AudioTranscriber:
    def __init__(self, config: TranscriptionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.max_workers = 2

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        model_name = self.config.model_name
        use_cuda = torch.cuda.is_available()
        device_arg = 0 if use_cuda else "cpu"
        dtype = torch.float16 if use_cuda else torch.float32

        model_kwargs = {}
        if use_cuda and is_flash_attn_2_available():
            model_kwargs["use_flash_attention_2"] = True

        pipeline_kwargs = {
            "task": "automatic-speech-recognition",
            "model": model_name,
            "torch_dtype": dtype,
            "device": device_arg,
        }
        if model_kwargs:
            pipeline_kwargs["model_kwargs"] = model_kwargs

        try:
            self.model = pipeline(**pipeline_kwargs)
        except TypeError as e:
            if model_kwargs.get("use_flash_attention_2"):
                self.logger.warning("Flash attention not supported. Falling back.")
                model_kwargs.pop("use_flash_attention_2", None)
                pipeline_kwargs.pop("model_kwargs", None)
                self.model = pipeline(**pipeline_kwargs)
            else:
                raise e
        
        # Get HF token
        auth_token = None
        
        if not auth_token:
            try:
                from huggingface_hub import HfFolder
                cli_token = HfFolder.get_token()
                if cli_token:
                    auth_token = cli_token
                    self.logger.info("Using token from HF CLI login")
            except Exception as e:
                self.logger.debug(f"Could not load CLI token: {e}")
        
        if not auth_token:
            auth_token = self.config.hf_token or os.getenv("HF_TOKEN")
            if auth_token:
                self.logger.info("Using token from .env or config")
        
        if not auth_token:
            raise ValueError("HF_TOKEN is required for speaker diarization")
        
        self.logger.info(f"Token loaded: {auth_token[:10]}... (length: {len(auth_token)})")
        
        diarization_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        diarization_pipeline_name = "pyannote/speaker-diarization-3.1"

        if hf_login is not None:
            try:
                hf_login(token=auth_token, add_to_git_credential=False)
                self.logger.info("Successfully authenticated with Hugging Face")
            except Exception as login_err:
                self.logger.warning(f"Login attempt failed: {login_err}")

        for var in ["HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "HF_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"]:
            if var in os.environ:
                del os.environ[var]
        
        os.environ["HF_TOKEN"] = auth_token
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = auth_token
        os.environ["HF_HUB_TOKEN"] = auth_token

        self.logger.info(f"Loading diarization model: {diarization_pipeline_name}")
        
        load_successful = False
        for method_name, kwargs in [
            ("use_auth_token", {"use_auth_token": auth_token}),
            ("token", {"token": auth_token}),
            ("auth_token", {"auth_token": auth_token}),
        ]:
            if load_successful:
                break
            try:
                self.logger.info(f"Trying to load with {method_name}...")
                self.diarization_pipeline = DiarizationPipeline.from_pretrained(
                    diarization_pipeline_name,
                    **kwargs
                ).to(diarization_device)
                self.logger.info(f"Diarization pipeline loaded successfully with {method_name}!")
                load_successful = True
            except TypeError as te:
                self.logger.debug(f"Method {method_name} not supported: {te}")
                continue
            except Exception as e:
                self.logger.error(f"Failed with {method_name}: {e}")
                continue
        
        if not load_successful:
            raise RuntimeError("Could not load diarization model - check licenses")
    
    def transcribe_audio(self, file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """
        Transcribe and diarize audio
        Uses Whisper pipeline's built-in audio loading - works without ffmpeg!
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            self.logger.error(f"File not found: {file_path}")
            return None

        try:
            self.logger.info(f"Starting processing: {file_path}")
            start_time = time.time()

            # Let the Whisper pipeline handle audio loading directly!
            # It can read webm, mp3, wav, etc. without ffmpeg
            self.logger.info(f"Loading audio with Whisper pipeline...")
            print(f"[INFO] Processing: {file_path.name}")
            
            # First, do transcription - the pipeline handles audio loading
            print(f"[INFO] Starting transcription (this may take a few minutes on CPU)...")
            transcription_start = time.time()
            
            transcription_result = self.model(
                str(file_path),  # Pass file path directly!
                chunk_length_s=30,
                batch_size=8,  # Reduced from 16 for faster processing
                return_timestamps=True,
                generate_kwargs={
                    "max_length": 448,
                    "num_beams": 3,  # Reduced from 5 for speed
                    "temperature": 0.2,
                    "no_repeat_ngram_size": 3,
                    "length_penalty": 1.0,
                }
            )
            transcript = transcription_result["chunks"]
            
            transcription_time = time.time() - transcription_start
            self.logger.info(f"Transcription completed with {len(transcript)} chunks in {transcription_time:.1f}s")
            print(f"[INFO] Transcription done ({transcription_time:.1f}s) - {len(transcript)} chunks")

            # Now load audio for diarization using the same method the pipeline uses
            print(f"[INFO] Loading audio for diarization...")
            from transformers.pipelines.audio_utils import ffmpeg_read
            
            with open(file_path, "rb") as f:
                audio_bytes = f.read()
            
            # This uses the transformers built-in decoder
            inputs = ffmpeg_read(audio_bytes, 16000)
            inputs_array = np.array(inputs, dtype=np.float32)
            
            # Normalize audio
            inputs_tensor = torch.from_numpy(inputs_array)
            inputs_array = torch.nn.functional.normalize(inputs_tensor, p=2, dim=0).numpy()
            diarizer_inputs = torch.from_numpy(inputs_array).float().unsqueeze(0)

            # Run diarization
            print(f"[INFO] Running speaker diarization (this may take several minutes)...")
            diarization_start = time.time()
            self.logger.info("Running speaker diarization...")
            new_segments = self._run_diarization(diarizer_inputs, min_speakers=1, max_speakers=5)
            
            diarization_time = time.time() - diarization_start
            print(f"[INFO] Diarization done ({diarization_time:.1f}s) - {len(new_segments)} speakers")
            
            # Align transcription with diarization
            end_timestamps = np.array([chunk["timestamp"][1] for chunk in transcript if chunk["timestamp"] is not None and chunk["timestamp"][1] is not None])
            segmented_preds = []

            for segment in new_segments:
                end_time = segment["segment"]["end"]
                upto_idx = np.argmin(np.abs(end_timestamps - end_time))

                text_chunks = [chunk["text"].strip() for chunk in transcript[: upto_idx + 1]]
                combined_text = " ".join(text_chunks)
                
                segmented_preds.append({
                    "speaker": segment["speaker"],
                    "text": combined_text,
                    "timestamp": (
                        transcript[0]["timestamp"][0],
                        transcript[upto_idx]["timestamp"][1]
                    ),
                    "confidence": np.mean([chunk.get("confidence", 1.0) for chunk in transcript[: upto_idx + 1]])
                })

                transcript = transcript[upto_idx + 1:]
                end_timestamps = end_timestamps[upto_idx + 1:]

                if len(end_timestamps) == 0:
                    break
            
            self._save_transcription(file_path, segmented_preds)

            total_time = time.time() - start_time
            self.logger.info(f"Total processing completed in {total_time:.2f}s")
            return segmented_preds

        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def _run_diarization(self, diarizer_inputs, num_speakers=2, min_speakers=1, max_speakers=4):
        try:
            diarization = self.diarization_pipeline(
                {"waveform": diarizer_inputs, "sample_rate": 16000},
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )

            segments = []
            # Handle pyannote's DiarizeOutput object
            try:
                # Try to iterate over the diarization object directly
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    segments.append({
                        "segment": {"start": turn.start, "end": turn.end},
                        "track": None,
                        "label": speaker,
                    })
            except AttributeError:
                # Newer API: DiarizeOutput is iterable directly
                for segment, track, label in diarization.itertracks(yield_label=True):
                    segments.append({
                        "segment": {"start": segment.start, "end": segment.end},
                        "track": track,
                        "label": label,
                    })

            if not segments:
                # Fallback if diarization fails
                self.logger.warning("No segments from diarization, using single speaker")
                return [{
                    "speaker": "SPEAKER_00",
                    "segment": {
                        "start": 0.0,
                        "end": float(len(diarizer_inputs[0])/16000)
                    }
                }]

            new_segments = []
            prev_segment = cur_segment = segments[0]

            for i in range(1, len(segments)):
                cur_segment = segments[i]

                if cur_segment["label"] != prev_segment["label"] and i < len(segments):
                    new_segments.append({
                        "segment": {
                            "start": prev_segment["segment"]["start"],
                            "end": cur_segment["segment"]["start"],
                        },
                        "speaker": prev_segment["label"],
                    })
                    prev_segment = segments[i]

            new_segments.append({
                "segment": {
                    "start": prev_segment["segment"]["start"],
                    "end": cur_segment["segment"]["end"],
                },
                "speaker": prev_segment["label"],
            })

            return new_segments

        except Exception as e:
            self.logger.error(f"Diarization failed: {str(e)}")
            # Return single speaker as fallback
            return [{
                "speaker": "SPEAKER_00",
                "segment": {
                    "start": 0.0,
                    "end": float(len(diarizer_inputs[0])/16000)
                }
            }]

    def process_batch(self, file_paths: List[Path]) -> List[Optional[Dict[str, Any]]]:
        """Process multiple audio files sequentially"""
        results = []
        
        for path in file_paths:
            try:
                result = self.transcribe_audio(path)
                results.append(result)
                
                if result:
                    self.logger.info(f"Successfully processed {path}")
                else:
                    self.logger.error(f"Failed to process {path}")
                
            except Exception as e:
                self.logger.error(f"Error processing {path}: {str(e)}")
                results.append(None)
        
        return results

    def _save_transcription(self, audio_path: Path, result: List[Dict[str, Any]]) -> None:
        """Save transcription results to JSON and text files"""
        try:
            # Get parent directories - handle cases with less than 2 parents
            parent_parts = audio_path.parent.parts
            
            if len(parent_parts) >= 2:
                parent_dirs = parent_parts[-2:]
                output_dir = Path("transcriptions").joinpath(*parent_dirs)
            else:
                # Use just the parent folder or create a default one
                output_dir = Path("transcriptions") / (audio_path.parent.name or "default")
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            json_path = output_dir / f"{audio_path.stem}.json"
            with json_path.open("w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            text_path = output_dir / f"{audio_path.stem}_dialogue.txt"
            with text_path.open("w", encoding="utf-8") as f:
                for segment in result:
                    f.write(
                        f"[{segment['timestamp'][0]:.2f}-{segment['timestamp'][1]:.2f}] "
                        f"{segment['speaker']}: {segment['text']}\n"
                    )

            self.logger.info(f"Saved transcription to {json_path}")
            self.logger.info(f"Saved dialogue to {text_path}")
            print(f"[SUCCESS] Saved to: {json_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving transcription: {str(e)}")
            raise


def main():
    """
    Standalone execution mode - process files directly
    Usage: python transcription.py [file_or_directory]
    """
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Audio Transcription with Speaker Diarization')
    parser.add_argument('path', nargs='?', default='./downloads', 
                       help='Path to audio file or directory (default: ./downloads)')
    parser.add_argument('--model', default='openai/whisper-large-v3-turbo',
                       help='Whisper model to use')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for transcription')
    
    args = parser.parse_args()
    
    # Setup logging for standalone mode
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('transcription_standalone.log'),
            logging.StreamHandler()
        ]
    )
    
    print("=" * 60)
    print("Audio Transcription System - Standalone Mode")
    print("=" * 60)
    
    # Create config
    config = TranscriptionConfig(
        model_name=args.model,
        output_dir="transcriptions",
    )
    
    # Initialize transcriber
    print(f"Initializing with model: {args.model}")
    transcriber = AudioTranscriber(config)
    
    # Process path
    path = Path(args.path)
    
    if not path.exists():
        print(f"Error: Path does not exist: {path}")
        sys.exit(1)
    
    if path.is_file():
        # Process single file
        print(f"\nProcessing single file: {path}")
        result = transcriber.transcribe_audio(path)
        if result:
            print(f"✓ Success! Results saved.")
        else:
            print(f"✗ Failed to process {path}")
    
    elif path.is_dir():
        # Process all audio files in directory
        audio_files = []
        for ext in ['.webm', '.mp3', '.wav', '.m4a', '.flac', '.ogg']:
            audio_files.extend(list(path.rglob(f"*{ext}")))
        
        if not audio_files:
            print(f"No audio files found in {path}")
            sys.exit(0)
        
        print(f"\nFound {len(audio_files)} audio file(s)")
        print("-" * 60)
        
        for i, file_path in enumerate(audio_files, 1):
            print(f"\n[{i}/{len(audio_files)}] Processing: {file_path.name}")
            result = transcriber.transcribe_audio(file_path)
            if result:
                print(f"✓ Success!")
            else:
                print(f"✗ Failed")
        
        print("\n" + "=" * 60)
        print(f"Completed processing {len(audio_files)} file(s)")
        print("=" * 60)
    
    else:
        print(f"Error: {path} is neither a file nor directory")
        sys.exit(1)


if __name__ == "__main__":
    main()