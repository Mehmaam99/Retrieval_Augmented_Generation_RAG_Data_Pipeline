from huggingface_hub import HfApi
import os
from dotenv import load_dotenv

load_dotenv()

api = HfApi()
token = os.getenv("HF_TOKEN")

models_to_check = [
    "pyannote/speaker-diarization-3.1",
    "pyannote/segmentation-3.0",
    "pyannote/wespeaker-voxceleb-resnet34-LM"
]

print("Checking model access...\n")
for model in models_to_check:
    try:
        info = api.model_info(model, token=token)
        print(f"✅ {model}: Access granted!")
    except Exception as e:
        print(f"❌ {model}: {str(e)}")
        print(f"   → Visit: https://huggingface.co/{model}\n")