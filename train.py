import logging
import os
import pprint
import random
from typing import Dict, Union

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
import tqdm
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

import wandb
import sys

from model.misc import ImplicitDataparallel
from model.grid_hash_model import GridCLIPModel

from dataloaders import ClassificationExtractor

# Set up the constants for the training loop

SAVE_DIRECTORY = "YOUR_PATH_TO_SAVE_MODEL"
DATA_DIRECTORY = "./labeled_dataset.pt"
DEVICE = "cuda"
IMAGE_TO_LABEL_CLIP_LOSS_SCALE = 1.0
LABEL_TO_IMAGE_LOSS_SCALE = 1.0
EXP_DECAY_COEFF = 0.5
SAVE_EVERY = 5
IMAGE_WEIGHT = 1.0
AFFORDANCE_WEIGHT = 1.0
METRICS = {
    "accuracy": torchmetrics.Accuracy,
}

NUM_EPOCHS = 70
BATCH_SIZE = 11000
NUM_WORKERS = 10

CLIP_MODEL_NAME = "ViT-B/32"
SBERT_MODEL_NAME = "all-mpnet-base-v2"

# Load the data and create the dataloader created in the previous tutorial notebook

training_data = torch.load(DATA_DIRECTORY)
max_coords, _ = training_data._label_xyz.max(dim=0)
min_coords, _ = training_data._label_xyz.min(dim=0)

# Set up the model

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


def train(
    clip_train_loader: DataLoader,
    labelling_model: Union[GridCLIPModel, ImplicitDataparallel],
    optim: torch.optim.Optimizer,
    epoch: int,
    classifier: ClassificationExtractor,
    device: Union[str, torch.device] = DEVICE,
    exp_decay_coeff: float = EXP_DECAY_COEFF,
    disable_tqdm: bool = False,
    metric_calculators: Dict[str, Dict[str, torchmetrics.Metric]] = {},
):
    """
    Train the model for one epoch.
    """
    total_loss = 0
    image_loss = 0
    affordance_loss = 0
    total_samples = 0
    labelling_model.train()
    total = len(clip_train_loader)
    for clip_data_dict in tqdm.tqdm(
        clip_train_loader,
        total=total,
        disable=disable_tqdm,
        desc=f"Training epoch {epoch}",
    ):
        xyzs = clip_data_dict["xyz"].to(device)
        clip_image_labels = clip_data_dict["clip_image_vector"].to(device)
        clip_affordance_labels = clip_data_dict["clip_affordance_vector"].to(device)

        # Get the affordance heatmap values and confidence weights
        affordance_heatmap_values = clip_data_dict["affordance_heatmap_values"].to(device)
        affordance_weights = clip_data_dict["affordance_weight"].to(device)

        image_weights = torch.exp(-exp_decay_coeff * clip_data_dict["distance"]).to(
            device
        )

        # Convert the image and language labels to indices for computing the contrastive loss
        image_label_index: torch.Tensor = (
            clip_data_dict["img_idx"].to(device).reshape(-1, 1)
        )
        affordance_label_index: torch.Tensor = (
            clip_data_dict["affordance_label"].to(device).reshape(-1, 1)
        )

        (
            predicted_image_latents,
            predicted_affordance_latents,
        ) = labelling_model(xyzs, affordance_values=affordance_heatmap_values)

        # Compute the contrastive loss
        batch_size = len(image_label_index)
        image_label_mask: torch.Tensor = (
            image_label_index != image_label_index.t()
        ).float() + torch.eye(batch_size, device=device)
        affordance_label_mask: torch.Tensor = (
            affordance_label_index != affordance_label_index.t()
        ).float() + torch.eye(batch_size, device=device)

        image_label_mask.requires_grad = False

        contrastive_loss_images = labelling_model.compute_loss(
            predicted_image_latents,
            clip_image_labels,
            label_mask=image_label_mask,
            confidence_weights=image_weights,  # Confidence weights
        )

        affordance_contrastive_loss = labelling_model.compute_loss(
            predicted_latents=predicted_affordance_latents,
            actual_latents=clip_affordance_labels,
            label_mask=affordance_label_mask,
            confidence_weights=affordance_weights,  # Confidence weights
        )

        # clear unused variables
        del (
            image_label_mask,
            image_label_index,
        )

        # compute the total loss
        contrastive_loss = (
            IMAGE_WEIGHT * contrastive_loss_images
            + AFFORDANCE_WEIGHT * affordance_contrastive_loss
        )

        optim.zero_grad(set_to_none=True)
        contrastive_loss.backward()
        optim.step()

        # clip the temperature parameter to maintain stability
        labelling_model.temperature.data = torch.clamp(
            labelling_model.temperature.data, max=np.log(100.0)
        )
        image_loss += contrastive_loss_images.detach().cpu().item()
        affordance_loss += affordance_contrastive_loss.detach().cpu().item()
        total_loss += contrastive_loss.detach().cpu().item()
        total_samples += 1

    to_log = {
        "train_avg/contrastive_loss_images": image_loss / total_samples,
        "train_avg/contrastive_loss_affordances": affordance_loss / total_samples,
        "train_avg/loss_sum": total_loss / total_samples,
        "train_avg/labelling_temp": torch.exp(labelling_model.temperature.data.detach())
        .cpu()
        .item(),
    }
    for metric_dict in metric_calculators.values():
        for metric_name, metric in metric_dict.items():
            try:
                to_log[f"train_avg/{metric_name}"] = (
                    metric.compute().detach().cpu().item() 
                )
            except RuntimeError as e:
                to_log[f"train_avg/{metric_name}"] = 0.0
            metric.reset()
    wandb.log(to_log)
    logging.debug(pprint.pformat(to_log, indent=4, width=1))
    return total_loss


def save(
    labelling_model: Union[ImplicitDataparallel, GridCLIPModel],
    optim: torch.optim.Optimizer,
    epoch: int,
    save_directory: str = SAVE_DIRECTORY,
    saving_dataparallel: bool = False,
):
    if saving_dataparallel:
        to_save = labelling_model.module
    else:
        to_save = labelling_model
    state_dict = {
        "model": to_save.state_dict(),
        "optim": optim.state_dict(),
        "epoch": epoch,
    }
    torch.save(
        state_dict,
        f"{save_directory}/Affordance_Model.pt",
    )
    return 0


train_classifier = ClassificationExtractor(
    clip_model_name=CLIP_MODEL_NAME,
    sentence_model_name=SBERT_MODEL_NAME,
    aff_class_names=training_data._all_aff_classes,
    device=DEVICE,
)

# Set up our metrics on this dataset.
train_metric_calculators = {}
train_class_count = {"semantic": train_classifier.total_label_classes}
average_style = ["micro", "macro", "weighted"]
for classes, counts in train_class_count.items():
    train_metric_calculators[classes] = {}
    for metric_name, metric_cls in METRICS.items():
        for avg in average_style:
            if "accuracy" in metric_name:
                new_metric = metric_cls(
                    num_classes=counts, average=avg, task='multiclass'
                ).to(DEVICE)
                train_metric_calculators[classes][f"{classes}_{metric_name}_{avg}"] = (
                    new_metric
                )

# No dataparallel for now
batch_multiplier = 1

clip_train_loader = DataLoader(
    training_data,
    batch_size=batch_multiplier * BATCH_SIZE,
    shuffle=True,
    pin_memory=True,
    num_workers=NUM_WORKERS,
)
logging.debug(f"Total train dataset sizes: {len(training_data)}")

# Set up optimizer

optim = torch.optim.Adam(
    label_model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=0.003,
)

wandb.init(
    project="ovafields",
)
# Set the extra parameters.
wandb.config.web_labelled_points = len(training_data)

os.environ["TOKENIZERS_PARALLELISM"] = (
    "false"  # Just to reduce excessive logging from sbert
)

epoch = 0

while epoch <= NUM_EPOCHS:
    train(
        clip_train_loader,
        label_model,
        optim,
        epoch,
        train_classifier,
        metric_calculators=train_metric_calculators,
    )
    epoch += 1
    if epoch % SAVE_EVERY == 0:
        save(label_model, optim, epoch)
