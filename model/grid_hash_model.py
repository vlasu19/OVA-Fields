from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from gridencoder import GridEncoder
from model.misc import MLP


class GridCLIPModel(nn.Module):
    def __init__(
        self,
        max_coords: Optional[torch.Tensor] = None,
        min_coords: Optional[torch.Tensor] = None,
        mlp_depth: int = 2,
        mlp_width: int = 256,
        batchnorm: bool = False,
        num_levels: int = 16,
        level_dim: int = 8,
        log2_hashmap_size: int = 24,
        per_level_scale: float = 2.0,
        device: str = "cuda",
        image_rep_size: int = 512,
        affordance_rep_size: int = 512,
        bounds: float = 10.0,
    ):
        super().__init__()

        self._grid_model = GridEncoder(
            input_dim=3,  
            num_levels=num_levels,
            level_dim=level_dim,
            base_resolution=16,
            log2_hashmap_size=log2_hashmap_size,
            per_level_scale=per_level_scale,
            desired_resolution=None,
            gridtype="hash",
            align_corners=False,
        )
        # Affordance Embedding Layer
        self.affordance_embed = nn.Linear(1, num_levels * level_dim)

        # Now transform the output through an MLP
        self._post_grid = MLP(
            input_dim=num_levels * level_dim,
            hidden_dim=mlp_width,
            hidden_depth=mlp_depth,
            output_dim=image_rep_size + affordance_rep_size,
            batchnorm=batchnorm,
        )

        # Add attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=num_levels * level_dim,
            num_heads=8,
            batch_first=True
        )

        # Use an identity layer to allow for easy swapping of the head
        self._image_head = nn.Identity()

        # A magic value adviced from @imisra
        self.temperature = nn.Parameter(torch.log(torch.tensor(1.0 / 0.07)))

        self._image_rep_size = image_rep_size
        self._affordance_rep_size = affordance_rep_size

        if not (max_coords is not None and min_coords is not None):
            self._max_bounds, self._min_bounds = (
                torch.ones(3) * bounds,  # 改为4维
                torch.ones(3) * -bounds,
            )
        else:
            assert len(max_coords) == len(min_coords)
            self._max_bounds, self._min_bounds = max_coords, min_coords

        self._grid_model = self._grid_model.to(device)
        self._post_grid = self._post_grid.to(device)
        self.attention = self.attention.to(device)
        self.affordance_embed = self.affordance_embed.to(device)
        self._image_head = self._image_head.to(device)
        self.temperature.data = self.temperature.data.to(device)
        self._max_bounds = self._max_bounds.to(device)
        self._min_bounds = self._min_bounds.to(device)

    def forward(self, x: torch.Tensor, affordance_values: torch.Tensor, bounds: Optional[float] = None):
        if bounds is None:
            max_bounds, min_bounds = (
                self._max_bounds.to(x.device),
                self._min_bounds.to(x.device),
            )
        else:
            max_bounds, min_bounds = (
                torch.cat(torch.ones(3, device=x.device) * bounds),
                torch.cat(torch.ones(3, device=x.device) * -bounds),
            )

        # concat the affordance heatmap values with the coordinates
        bounded_x = (x - min_bounds) / (max_bounds - min_bounds)
        grid_hash = self._grid_model(bounded_x, bound=1.0)
        # result = self._post_grid(grid_hash)

        affordance_embed = self.affordance_embed(affordance_values.unsqueeze(-1))

        combined_features = grid_hash + affordance_embed

        # Apply attention mechanism
        attn_input = combined_features.unsqueeze(1)  # (N, 1, embed_dim)
        attn_output, attn_weights = self.attention(attn_input, attn_input, attn_input)
        attn_output = attn_output.squeeze(1)  # (N, embed_dim)

        result = self._post_grid(attn_output)
        image_latent, affordance_latent = (
            result[..., : self._image_rep_size],

            result[..., self._image_rep_size :],
        )

        image_latent = self._image_head(image_latent)
        return image_latent, affordance_latent

    def to(self, device):
        self._grid_model = self._grid_model.to(device)
        self._post_grid = self._post_grid.to(device)
        self.attention = self.attention.to(device)
        self._image_head = self._image_head.to(device)
        self._max_bounds = self._max_bounds.to(device)
        self._min_bounds = self._min_bounds.to(device)
        self.temperature.data = self.temperature.data.to(device)
        return self

    def compute_loss(
        self, predicted_latents, actual_latents, label_mask=None, affordance_latents=None, affordance_weight=1.0, confidence_weights=None
    ):
        normalized_predicted_latents = F.normalize(predicted_latents, p=2, dim=-1)
        normalized_actual_latents = F.normalize(actual_latents, p=2, dim=-1)
        
        # if have affordance_latents, normalize it
        if affordance_latents is not None:
            normalized_affordance_latents = F.normalize(affordance_latents, p=2, dim=-1)

        temp = torch.exp(self.temperature)
        sim = (
            torch.einsum(
                "i d, j d -> i j",
                normalized_predicted_latents,
                normalized_actual_latents,
            )
            * temp
        )

        # To prevent the model from learning the identity function
        if label_mask is not None:
            sim = sim * label_mask
            del label_mask

        labels = torch.arange(len(predicted_latents), device=predicted_latents.device)

        # Calculate contrastive loss using sample weights
        if confidence_weights is not None:
            # Make sure the shape of confidence_weights is (N,)
            confidence_weights = confidence_weights.to(predicted_latents.device)

            # Calculate contrastive loss using sample weights
            loss_fn = nn.CrossEntropyLoss(reduction='none')
            loss_i = loss_fn(sim, labels)
            loss_j = loss_fn(sim.t(), labels)
            contrastive_loss = (loss_i + loss_j) / 2
            contrastive_loss = (contrastive_loss * confidence_weights).mean()
        else:
            contrastive_loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2

        # Calculate affordance loss if affordance_latents is provided
        if affordance_latents is not None:
            affordance_sim = (
                torch.einsum(
                    "i d, j d -> i j",
                    normalized_predicted_latents,
                    normalized_affordance_latents,
                )
                * temp
            )

            # Calculate affordance loss using sample weights
            if confidence_weights is not None:
                loss_fn = nn.CrossEntropyLoss(reduction='none')
                affordance_loss_i = loss_fn(affordance_sim, labels)
                affordance_loss_j = loss_fn(affordance_sim.t(), labels)
                affordance_loss = (affordance_loss_i + affordance_loss_j) / 2
                affordance_loss = (affordance_loss * confidence_weights).mean()
            else:
                affordance_loss = (F.cross_entropy(affordance_sim, labels) + F.cross_entropy(affordance_sim.t(), labels)) / 2
        else:
            affordance_loss = 0

        # The final loss is the weighted sum of the contrastive loss and affordance loss
        total_loss = contrastive_loss + affordance_weight * affordance_loss

        return total_loss
