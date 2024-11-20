import os
import numpy as np
import torch
import sys
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
	# 下面两个常规设置了，用来np和random的话要设置 
    np.random.seed(seed)
    
    os.environ['PYTHONHASHSEED'] = str(seed)  # 禁止hash随机化

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 