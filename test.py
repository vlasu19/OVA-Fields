import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain, cycle
from sentence_transformers import SentenceTransformer, util

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

import tqdm
import einops

import os
import sys

from dataloaders.real_dataset_heatmap import DeticDenseLabelledDataset
from model.grid_hash_model import GridCLIPModel

from model.misc import MLP

import pandas as pd
import pyntcloud
from pyntcloud import PyntCloud
import clip
from torch.utils.data import Dataset
from scipy.signal import find_peaks

DEVICE = "cuda"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
model, preprocess = clip.load("ViT-B/32", device=DEVICE)
sentence_model = SentenceTransformer("all-mpnet-base-v2")
scene = 'lab0920'

queries = [
    # lab
    'take out some food from the refrigerator',
    'warm up the food in the microwave',
    'help me to take the bottle',
    'give me the knife',
    'I want to eat banana',
    'I want to use the yellow pen to write something on the paper',

    # home
    # 'take out some food from the refrigerator',
    # 'help me input something on the laptop',
    # 'take the cup from the table',
    # 'where can i seat on chair'

    # scene0670
    # 'take out some food from the refrigerator',
    # 'take the bottle from the table',
    # 'give me the metal bowl',

    # scene0552
    # 'take out some food from the refrigerator',
    # 'warm up the food in the microwave',

    # scene0753
    # 'take the bottle to me from the table',
    # 'give me the book',

    # multi tasks
    # 'Put the bananas on the table in the refrigerator'
    # 'Use the knife to cut the banana',

    # disjunctive sentence
    # 'take out some food from the frige',

]

training_data = torch.load("./detic_labeled_dataset_lab0920.pt")
max_coords, _ = training_data._label_xyz.max(dim=0)
min_coords, _ = training_data._label_xyz.min(dim=0)

label_model = GridCLIPModel(
    image_rep_size=training_data[0]["clip_image_vector"].shape[-1],
    affordance_rep_size=training_data[0]["clip_affordance_vector"].shape[-1],
    mlp_depth=1,
    mlp_width=600,
    log2_hashmap_size=20,
    num_levels=18,
    level_dim=8,
    per_level_scale=2,
    max_coords=max_coords,
    min_coords=min_coords,
).to(DEVICE)

model_weights_path = "./checkpoints/implicit_scene_label_model_lab0920.pt"
model_weights = torch.load(model_weights_path, map_location=DEVICE)
label_model.load_state_dict(model_weights["model"])



class CustomDataset(Dataset):
    def __init__(self, xyz_data, affordance_values):
        self.xyz_data = xyz_data
        self.affordance_values = affordance_values

    def __len__(self):
        return len(self.xyz_data)

    def __getitem__(self, index):
        # return a tuple of (xyz, affordance)
        xyz = self.xyz_data[index]
        affordance = self.affordance_values[index]
        return xyz, affordance

xyz_data = training_data._label_xyz
affordance_values = training_data._affordance_heatmap_values

def calculate_clip_and_st_embeddings_for_queries(queries):
    all_clip_queries = clip.tokenize(queries)
    with torch.no_grad():
        all_clip_tokens = model.encode_text(all_clip_queries.to(DEVICE)).float()
        all_clip_tokens = F.normalize(all_clip_tokens, p=2, dim=-1)
        all_st_tokens = torch.from_numpy(sentence_model.encode(queries))
        all_st_tokens = F.normalize(all_st_tokens, p=2, dim=-1).to(DEVICE)
    return all_clip_tokens, all_st_tokens

def find_alignment_over_model(label_model, queries, dataloader):
    clip_text_tokens, st_text_tokens = calculate_clip_and_st_embeddings_for_queries(
        queries
    )
    vision_weight = 5.0
    affordance_weight = 10.0
    point_opacity = []
    with torch.no_grad():
        for data in tqdm.tqdm(dataloader,total=len(dataloader)):
            xyzs, affordance_values = data  # data 是一个元组，包含 (xyz, affordance)
            xyzs = xyzs.to(DEVICE)
            affordance_values = affordance_values.to(DEVICE)  # 确保 Affordance 值在相同设备上
            # Find alignmnents with the vectors
            (
                predicted_image_latents,
                predicted_affordance_latents,
            ) = label_model(xyzs, affordance_values)
            data_visual_tokens = F.normalize(predicted_image_latents, p=2, dim=-1).to(
                DEVICE
            )
            data_affordance_tokens = F.normalize(
                predicted_affordance_latents, p=2, dim=-1
            ).to(DEVICE)
            visual_alignment = data_visual_tokens @ clip_text_tokens.T
            affordance_alignment = data_affordance_tokens @ st_text_tokens.T
            total_alignment = (
                (vision_weight * visual_alignment)
                + (affordance_weight * affordance_alignment)
            )
            total_alignment /= vision_weight + affordance_weight
            point_opacity.append(total_alignment)

    point_opacity = torch.cat(point_opacity).T
    return point_opacity

merged_pcd = o3d.geometry.PointCloud()
merged_pcd.points = o3d.utility.Vector3dVector(training_data._label_xyz)
merged_pcd.colors = o3d.utility.Vector3dVector(training_data._label_rgb)
merged_downpcd = merged_pcd.voxel_down_sample(voxel_size=0.03)

print("Create pts result")
pts_result = np.concatenate((np.asarray(merged_downpcd.points), np.asarray(merged_downpcd.colors)), axis=-1)

df = pd.DataFrame(
    # same arguments that you are passing to visualize_pcl
    data=pts_result,
    columns=["x", "y", "z", "red", "green", "blue"]
)
cloud = PyntCloud(df)
print("Point cloud", cloud)

# 创建自定义数据集
custom_dataset = CustomDataset(xyz_data, affordance_values)

batch_size = 30_000
points_dataloader = DataLoader(
    custom_dataset, batch_size=batch_size, num_workers=10,
)
print("Created data loader", points_dataloader)

visual = False
alignment_q = find_alignment_over_model(label_model, queries, points_dataloader)


fig = plt.figure()
thresholds = np.zeros(len(queries))
for query_num in range(len(queries)):
    q = alignment_q[query_num].squeeze()
    print(q.shape)
    alpha = q.detach().cpu().numpy()
    counts, bins, _ = plt.hist(alpha, 100, density=True)
    # Find the bin centers
    bin_centers = (bins[:-1] + bins[1:]) / 2
    peaks, _ = find_peaks(counts)
    last_peak_value = bin_centers[peaks[-1]]
    thresholds[query_num] = last_peak_value


os.makedirs("visualized_pointcloud", exist_ok=True)

for query, q, threshold in zip(queries, alignment_q, thresholds):
    max_alpha = torch.max(q).cpu().item()
    alpha_threshold = threshold
    print(f"Max alpha: {max_alpha}, alpha threshold: {alpha_threshold}")
    alpha = q.detach().cpu().numpy()
    pts = training_data._label_xyz.detach().cpu()

    # Normalize alpha
    a_norm = (alpha - alpha.min()) / (alpha.max() - alpha.min())
    a_norm = torch.as_tensor(a_norm[..., np.newaxis])

    # Initialize colors tensor
    all_colors = training_data._label_rgb.detach().cpu()

    # Set colors based on alpha values
    all_colors = training_data._label_rgb / 255.0

    high_alpha_indices = (alpha > alpha_threshold).nonzero()[0]
    max_alpha_indices = (alpha == max_alpha).nonzero()[0]
    if len(high_alpha_indices > 0):
        all_colors[high_alpha_indices] = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)

    all_colors[max_alpha_indices] = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
    
    # 将点云和颜色进行合并
    merged_pcd = o3d.geometry.PointCloud()
    merged_pcd.points = o3d.utility.Vector3dVector(pts)
    merged_pcd.colors = o3d.utility.Vector3dVector(all_colors)
        
    # 对点云进行降采样以减少文件大小
    merged_downpcd = merged_pcd.voxel_down_sample(voxel_size=0.001)
    # 可视化点云
    o3d.visualization.draw_geometries([merged_downpcd])
        
    o3d.io.write_point_cloud(f"results/visualized_pointcloud/{scene}/{query}.ply", merged_downpcd)

    print(f"Visualized point cloud saved for query {query} with alpha thresholding.")