# Download LLM model weights from Hugging Face Hub so that they can be loaded locally
import os
from huggingface_hub import snapshot_download
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file, including HF_TOKEN

# Add the Hugging Face repo IDs you want to download.
repo_ids = [
    "QuantFactory/SmolLM-1.7B-Instruct-GGUF"
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
