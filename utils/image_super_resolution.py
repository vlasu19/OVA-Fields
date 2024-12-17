import torch

from BSRGAN.utils import utils_image as util
from BSRGAN.models.network_rrdbnet import RRDBNet as net
import numpy as np
import requests
from PIL import Image

MODEL_URL = "https://drive.google.com/file/d/1WNULM1e8gRNvsngVscsQ8tpaOqJ4mYtv/view?usp=drive_link"

MODEL_PATH = "../checkpoints/BSRGAN.pth"  # set model path

# Check if the model file exists locally; if not, download it
if not MODEL_PATH.exists():
    print(f"Model file not found at {MODEL_PATH}. Downloading from {MODEL_URL}...")
    response = requests.get(MODEL_URL, stream=True)
    
    if response.status_code == 200:
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Model downloaded and saved to {MODEL_PATH}.")
    else:
        print(f"Failed to download model from {MODEL_URL}. HTTP Status Code: {response.status_code}")

# convert uint to 4-dimensional torch tensor
def uint2tensor4(img):
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255.).unsqueeze(0)

def super_resolution(img, sf=4):
    sf = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_path = MODEL_PATH        # set model path
    torch.cuda.empty_cache()

    model = net(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=sf)  # define network

    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    torch.cuda.empty_cache()

    img_L = uint2tensor4(img)
    img_L = img_L.to(device)

    img_E = model(img_L)
    img_E = util.tensor2uint(img_E)
    img_E = Image.fromarray(img_E, mode="RGB")
    # 将img_E转换为Image对象
    return img_E
