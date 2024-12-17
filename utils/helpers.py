import os
import numpy as np
import torch
import sys
import requests
import gdown
from pathlib import Path

DETIC_PATH = os.environ.get("DETIC_PATH", Path(__file__).parent / "../Detic")
sys.path.insert(0, f"{DETIC_PATH}/third_party/CenterNet2/")
sys.path.insert(0, f"{DETIC_PATH}/")
from detic.modeling.text.text_encoder import build_text_encoder

def get_clip_embeddings(vocabulary, prompt="a "):
    text_encoder = build_text_encoder(pretrain=True)
    text_encoder.eval()
    texts = [prompt + x.replace("-", " ") for x in vocabulary]
    emb = text_encoder(texts).detach().permute(1, 0).contiguous().cpu()
    return emb

def set_seed(seed=42):
    np.random.seed(seed)
    
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 

def download_model(url: str, model_path: Path) -> None:
    try:
        # Check if the model file exists locally; if not, download it
        if not os.path.exists(model_path):
            print(f"Model file not found at {model_path}. Downloading from {model_path}...")
            gdown.download(url, str(model_path), quiet=False)
            print(f"Model downloaded and saved to {model_path}.")
        else:
            print(f"Model already exists at {model_path}. Skipping download.")
    
    except gdown.exceptions.DownloadError as e:
        # Handle errors specific to gdown download failure
        print(f"Error downloading the model from {url}: {e}")
    
    except Exception as e:
        # Handle general errors (e.g., permission, file I/O, etc.)
        print(f"An unexpected error occurred: {e}")
