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
        # Affordance映射层
        self.affordance_embed = nn.Linear(1, num_levels * level_dim)
        # 现在通过MLP转换输出
        self._post_grid = MLP(
            input_dim=num_levels * level_dim,
            hidden_dim=mlp_width,
            hidden_depth=mlp_depth,
            output_dim=image_rep_size + affordance_rep_size,
            batchnorm=batchnorm,
        )
        # 添加注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=num_levels * level_dim,
            num_heads=8,
            batch_first=True
        )

        # 用于图像损失的额外存储
        self._image_head = nn.Identity()
        # 由@imisra建议的魔法值
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

        # 将Affordance热力图值与坐标拼接
        bounded_x = (x - min_bounds) / (max_bounds - min_bounds)
        grid_hash = self._grid_model(bounded_x, bound=1.0)
        # result = self._post_grid(grid_hash)

        affordance_embed = self.affordance_embed(affordance_values.unsqueeze(-1))

        combined_features = grid_hash + affordance_embed

        # 应用注意力机制
        # 需要将grid_hash的形状调整为(batch_size, seq_len, embed_dim)
        # 假设grid_hash的形状为(N, embed_dim)
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
        # 归一化处理
        normalized_predicted_latents = F.normalize(predicted_latents, p=2, dim=-1)
        normalized_actual_latents = F.normalize(actual_latents, p=2, dim=-1)
        
        # 如果有affordance_latents，则归一化处理
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

        # 对于语义标签的对比学习，零值化标签相同的元素
        if label_mask is not None:
            sim = sim * label_mask
            del label_mask

        labels = torch.arange(len(predicted_latents), device=predicted_latents.device)

        # 使用置信度权重计算损失
        if confidence_weights is not None:
            # 确保confidence_weights的形状为(N,)
            confidence_weights = confidence_weights.to(predicted_latents.device)
            # 计算对比损失，使用样本权重
            loss_fn = nn.CrossEntropyLoss(reduction='none')
            loss_i = loss_fn(sim, labels)
            loss_j = loss_fn(sim.t(), labels)
            contrastive_loss = (loss_i + loss_j) / 2
            contrastive_loss = (contrastive_loss * confidence_weights).mean()
        else:
            contrastive_loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2

        # 如果提供了affordance_latents，计算额外的affordance损失
        if affordance_latents is not None:
            affordance_sim = (
                torch.einsum(
                    "i d, j d -> i j",
                    normalized_predicted_latents,
                    normalized_affordance_latents,
                )
                * temp
            )

            # 使用置信度权重计算affordance损失
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

        # 最终损失是对比损失和affordance损失的加权和
        total_loss = contrastive_loss + affordance_weight * affordance_loss

        return total_loss
