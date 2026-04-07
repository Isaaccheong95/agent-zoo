# Download LLM model weights from Hugging Face Hub so that they can be loaded locally
import os

# The workspace is on a noexec mount, so hf_xet native extensions cannot be loaded here.
# Force standard HTTP downloads instead of Xet-backed downloads.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import snapshot_download
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file, including HF_TOKEN

# Add the Hugging Face repo IDs you want to download.
repo_ids = [
    "Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-GGUF",
    "unsloth/medgemma-1.5-4b-it-GGUF",
    "unsloth/gemma-4-E4B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF"
]

for repo_id in repo_ids:
    model_name = repo_id.split("/")[-1]
    local_dir = f"./models/{model_name}"
    print(f"Downloading {repo_id} to {local_dir}...")
    
    if os.path.isdir(local_dir):
        print(f"Already exists, skipping: {local_dir}")
        continue
    
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )
    
    print(f"Downloaded to {local_dir}")
