import os
import sys
import argparse
from tqdm import tqdm

import cv2
import torch
import numpy as np
import requests
from pathlib import Path

# Set LOCATE_PATH to locate the model directory, and add it to sys.path
LOCATE_PATH = os.environ.get("LOCATE_PATH", Path(__file__).parent / "../LOCATE")
sys.path.insert(0, f"{LOCATE_PATH}/")
# Set the URL for the model file
MODEL_URL = "https://drive.google.com/file/d/1XYITtc2QX9_oVH-yFOLtLHX1QFpFOMif/view?usp=drive_link"

MODEL_FOLDER = Path(__file__).parent / "../checkpoints"
MODEL_PATH = os.environ.get("MODEL_PATH", Path(__file__).parent / "../checkpoints/best_seen.pth")

# Check if the folder is exists
if not os.path.exists(MODEL_FOLDER):
    os.makedirs(MODEL_FOLDER, exist_ok=True)

# Check if the model file exists locally; if not, download it
if not os.path.exists(MODEL_PATH):
    print(f"Model file not found at {MODEL_PATH}. Downloading from {MODEL_URL}...")
    response = requests.get(MODEL_URL, stream=True)
    
    if response.status_code == 200:
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Model downloaded and saved to {MODEL_PATH}.")
    else:
        print(f"Failed to download model from {MODEL_URL}. HTTP Status Code: {response.status_code}")


from LOCATE.models.locate import Net as locate_model
from LOCATE.utils.util import set_seed
from utils.normalizers import sigmoid_normalize_map
from utils.visualizer import viz_pred_test

class AffordanceDataset:
    """
    Dataset class for generating affordance masks from input images using the LOCATE model.

    Args:
        image (torch.Tensor): Input image tensor.
        label (int): Label for the affordance.
        object (str): Name of the object in the image.
        img_size (tuple): Width and height of the input image.
        crop_size (int, optional): The size to crop the image. Defaults to 224.
        viz (bool, optional): Whether to visualize and save the affordance mask predictions. Defaults to False.
    """
    def __init__(self, image, label, object, img_size, crop_size=224, viz=False):
        self.image = image
        self.label = label
        self.object = object
        self.img_w, self.img_h = img_size
        self.aff_weights = None
        self.crop_size = crop_size
        self.viz = viz

        # Set paths and hyperparameters for the dataset
        self.save_path = './save_preds'
        self.model_file = MODEL_PATH
        self.resize_size = 256
        self.num_workers = 8
        self.test_batch_size = 1
        self.test_num_workers = 8
        self.gpu = '0'

        # Define affordance and object lists
        self.aff_list = ['beat', "boxing", "brush_with", "carry", "catch", "cut", "cut_with", "drag", 'drink_with',
                         "eat", "hit", "hold", "jump", "kick", "lie_on", "lift", "look_out", "open", "pack", "peel",
                         "pick_up", "pour", "push", "ride", "sip", "sit_on", "stick", "stir", "swing", "take_photo",
                         "talk_on", "text_on", "throw", "type_on", "wash", "write"]
        self.obj_list = ['apple', 'axe', 'badminton_racket', 'banana', 'baseball', 'baseball_bat',
                         'basketball', 'bed', 'bench', 'bicycle', 'binoculars', 'book', 'bottle',
                         'bowl', 'broccoli', 'camera', 'carrot', 'cell_phone', 'chair', 'couch',
                         'cup', 'discus', 'drum', 'fork', 'frisbee', 'golf_clubs', 'hammer', 'hot_dog',
                         'javelin', 'keyboard', 'knife', 'laptop', 'microwave', 'motorcycle', 'orange',
                         'oven', 'pen', 'punching_bag', 'refrigerator', 'rugby_ball', 'scissors',
                         'skateboard', 'skis', 'snowboard', 'soccer_ball', 'suitcase', 'surfboard',
                         'tennis_racket', 'toothbrush', 'wine_glass']

        self.num_classes = 36

        # Create save directory if visualization is enabled
        if self.viz:
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path, exist_ok=True)

        # Set random seed for reproducibility
        set_seed(seed=42)
        self.aff_mask = []
        self.get_aff_mask()
        
    def get_aff_mask(self):
        """
        Generates the affordance mask for the input image using the pre-trained LOCATE model.
        """
        # Load the pre-trained model and set to evaluation mode
        model = locate_model(aff_classes=self.num_classes).cuda()
        model.eval()

        # Ensure model file exists before loading weights
        assert os.path.exists(self.model_file), "Please provide the correct model file for testing"
        model.load_state_dict(torch.load(self.model_file))

        # Prepare input image for model inference
        self.image = self.image.unsqueeze(0)  # Add batch dimension
        self.label = torch.tensor([self.label]).long()

        # Perform forward pass to get affordance prediction
        ego_pred = model.test_forward(self.image.cuda(), self.label.cuda())
        ego_pred = np.array(ego_pred.squeeze().data.cpu())

        # Normalize the output prediction to generate affordance mask
        self.aff_mask = sigmoid_normalize_map(ego_pred, self.img_w, self.img_h)
        self.aff_weights = sigmoid_normalize_map(ego_pred, self.img_w, self.img_h, is_norm=False)

        # Visualize and save the affordance prediction if visualization is enabled
        if self.viz:
            img_name = self.object[0]
            viz_pred_test(self.save_path, self.image, ego_pred, self.aff_list, self.label, img_name)              

    def __getitem__(self):
        """
        Returns the affordance mask and weights for the input image.

        Returns:
            dict: A dictionary containing the affordance mask ('aff_mask') and affordance weights ('aff_weights').
        """
        return {
            'aff_mask': self.aff_mask,
            'aff_weights': self.aff_weights,
        }
