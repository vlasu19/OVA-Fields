import os
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mplc
from matplotlib import cm
from detectron2.utils.visualizer import Visualizer


class VisualizationHandler(Visualizer):
    def _jitter(self, color):
        color = mplc.to_rgb(color)
        vec = np.random.rand(3)
        vec = vec / np.linalg.norm(vec) * 0.01  # 1% noise
        return tuple(np.clip(vec + color, 0, 1))
    
def viz_pred_test(save_path, image, ego_pred, aff_list, aff_label, img_name, epoch=None):
    mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=image.dtype, device=image.device).view(-1, 1, 1)
    std = torch.as_tensor([0.229, 0.224, 0.225], dtype=image.dtype, device=image.device).view(-1, 1, 1)
    mean = mean.view(-1, 1, 1)
    std = std.view(-1, 1, 1)
    img = image.squeeze(0) * std + mean
    img = img.detach().cpu().numpy() * 255
    img = Image.fromarray(img.transpose(1, 2, 0).astype(np.uint8))

    aff_str = aff_list[aff_label.item()]

    ego_pred = Image.fromarray(ego_pred)
    ego_pred = overlay_mask(img, ego_pred, alpha=0.5)

    fig, ax = plt.subplots(1, 2, figsize=(10, 6))
    for axi in ax.ravel():
        axi.set_axis_off()
    ax[0].imshow(img)
    ax[0].set_title('ego')
    ax[1].imshow(ego_pred)
    ax[1].set_title(aff_str)

    os.makedirs(os.path.join(save_path, 'viz_test'), exist_ok=True)
    fig_name = os.path.join(save_path, 'viz_test', img_name+ '_' + aff_str + '.jpg')
    plt.savefig(fig_name)
    plt.close()

def overlay_mask(img: Image.Image, mask: Image.Image, colormap: str = "jet", alpha: float = 0.7) -> Image.Image:
    '''
    Overlay a mask on an image using a colormap.
    '''
    if not isinstance(img, Image.Image) or not isinstance(mask, Image.Image):
        raise TypeError("img and mask arguments need to be PIL.Image")

    if not isinstance(alpha, float) or alpha < 0 or alpha >= 1:
        raise ValueError("alpha argument is expected to be of type float between 0 and 1")

    cmap = cm.get_cmap(colormap)
    # Resize mask and apply colormap
    # overlay = mask.resize(img.size, resample=Image.BICUBIC)
    overlayed_img = (255 * cmap(np.asarray(mask) ** 2)[:, :, :3]).astype(np.uint8)
    # Overlay the image with the mask
    # overlayed_img = Image.fromarray((alpha * np.asarray(img) + (1 - alpha) * overlay).astype(np.uint8))

    return overlayed_img
