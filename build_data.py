import sys
import os

import torch
from dataloaders import R3DSemanticDataset, DeticDenseLabelledDataset
from dataloaders.scannet_200_classes import AFF_OBJ_LIST
DATA_PATH = './data/r3d/lab_0920.r3d'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

dataset = R3DSemanticDataset(DATA_PATH, AFF_OBJ_LIST)

os.environ['CURL_CA_BUNDLE'] = ''
labelled_dataset = DeticDenseLabelledDataset(
    dataset, 
    use_extra_classes=False, 
    exclude_gt_images=False, 
    subsample_prob=0.01, 
    visualize_results=True, 
    detic_threshold=0.6,
    visualization_path="results/detic_labelled_results",
    item_coordinates_path="results/object_coordinates",
)

torch.save(labelled_dataset, "./detic_labeled_dataset_lab0920.pt")