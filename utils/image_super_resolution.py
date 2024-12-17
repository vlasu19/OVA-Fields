import torch

from BSRGAN.utils import utils_image as util
from BSRGAN.models.network_rrdbnet import RRDBNet as net
import numpy as np
from pathlib import Path
from PIL import Image

MODEL_PATH = Path(__file__).parent /  "../checkpoints/BSRGAN.pth"  # set model path

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
    return img_E
