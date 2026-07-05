# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import math
from typing import Any

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F_torch
from torchvision import models
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor

__all__ = [
    "contextual_bilateral_loss_for_rgb", "contextual_bilateral_loss_for_vgg"
]


def _extract_image_patches(x: Tensor, patch_size: int, stride: int) -> Tensor:
    """Implementation TensorFlow extract_image_patches() version"""
    batch_size, channels, height, width = x.shape
    patches = (
        x.unfold(2, patch_size, stride)
        .unfold(3, patch_size, stride)
        .reshape(batch_size, channels, -1, patch_size, patch_size)
        .permute(0, 1, 3, 4, 2)
    )

    return patches


def _compute_cosine_distance(x: Tensor, y: Tensor) -> Tensor:
    batch_size, channels, _, _ = x.size()

    # mean shifting by channel-wise mean of `y`.
    y_mean = y.mean(dim=(0, 2, 3), keepdim=True)
    x_centered = x - y_mean
    y_centered = y - y_mean

    # L2 normalization
    x_normalized = F_torch.normalize(x_centered, p=2, dim=1)
    y_normalized = F_torch.normalize(y_centered, p=2, dim=1)

    # Channel-wise vectorization
    x_normalized = x_normalized.reshape(batch_size, channels, -1)  # (N, C, H*W)
    y_normalized = y_normalized.reshape(batch_size, channels, -1)  # (N, C, H*W)

    # cosine similarity
    cosine_sim = torch.bmm(x_normalized.transpose(1, 2), y_normalized)  # (N, H*W, H*W)

    # convert to distance
    distance = 1 - cosine_sim

    return distance


def _compute_mae_distance(x: torch.Tensor, y: torch.Tensor) -> Tensor:
    batch_size, channels, height, width = x.size()

    x_vec = x.view(batch_size, channels, -1)
    y_vec = y.view(batch_size, channels, -1)

    distance = x_vec.unsqueeze(2) - y_vec.unsqueeze(3)
    distance = distance.sum(dim=1).abs()
    distance = distance.transpose(1, 2).reshape(batch_size, height * width, height * width)
    distance = distance.clamp(min=0.)

    return distance


def _compute_mse_distance(x: Tensor, y: Tensor) -> Tensor:
    batch_size, channels, height, width = x.size()

    x_vec = x.view(batch_size, channels, -1)
    y_vec = y.view(batch_size, channels, -1)
    x_s = torch.sum(x_vec ** 2, dim=1)
    y_s = torch.sum(y_vec ** 2, dim=1)
    batch_size, HW = y_s.shape

    A = y_vec.transpose(1, 2) @ x_vec  # N x(HW) x (HW)
    distance = y_s.unsqueeze(dim=2) - 2 * A + x_s.unsqueeze(dim=1)
    distance = distance.transpose(1, 2).reshape(batch_size, height * width, height * width)
    distance = distance.clamp(min=0.)

    return distance


def _compute_relative_distance(distance: Tensor) -> Tensor:
    dist_min, _ = torch.min(distance, dim=2, keepdim=True)
    relative_distance = distance / (dist_min + 1e-5)

    return relative_distance


def _compute_contextual(relative_distance: Tensor, bandwidth: float) -> Tensor:
    # This code easy OOM
    # w = torch.exp((1 - relative_distance) / bandwidth)  # Eq(3)
    # contextual = w / torch.sum(w, dim=2, keepdim=True)  # Eq(4)

    # This code is safe
    contextual = F_torch.softmax((1 - relative_distance) / bandwidth, dim=2)

    return contextual


def _compute_meshgrid(shape: Tensor) -> Tensor:
    batch_size, _, height, width = shape

    rows = torch.arange(0, height, dtype=torch.float32) / (height + 1)
    cols = torch.arange(0, width, dtype=torch.float32) / (width + 1)

    feature_grid = torch.meshgrid(rows, cols, indexing="ij")
    feature_grid = torch.stack(feature_grid).unsqueeze(0)
    feature_grid = torch.cat([feature_grid for _ in range(batch_size)], dim=0)

    return feature_grid


def _compute_contextual_bilateral_loss(x: Tensor, y: Tensor, weight_spatial: float, bandwidth: float) -> Tensor:
    device = x.device
    # Calculate two image spatial loss
    grid = _compute_meshgrid(x.shape).to(device)
    distance = _compute_mse_distance(grid, grid)
    relative_distance = _compute_relative_distance(distance)
    contextual_spatial = _compute_contextual(relative_distance, bandwidth)

    # Calculate feature loss
    # Calculate two image distance
    distance = _compute_cosine_distance(x, y)
    # Compute relative distance
    relative_distance = _compute_relative_distance(distance)
    # Calculate two image contextual loss
    contextual_feature = _compute_contextual(relative_distance, bandwidth)

    # Combine loss
    cx_combine = (1. - weight_spatial) * contextual_feature + weight_spatial * contextual_spatial
    k_max_NC, _ = torch.max(cx_combine, dim=2, keepdim=True)
    cx = k_max_NC.mean(dim=1)
    loss = torch.mean(-torch.log(cx + 1e-5))

    return loss


class _ContextualBilateralLoss(nn.Module):
    """Creates a criterion that measures the contextual loss"""

    def __init__(
            self,
            feature_model_extractor_nodes= ["features.2", "features.7", "features.12"],
            feature_model_normalize_mean = [0.485, 0.456, 0.406],
            feature_model_normalize_std = [0.229, 0.224, 0.225],
            patch_size: int = 5,
            stride: int = 1,
            weight_spatial: float = 0.5,
            bandwidth: float = 1.0,
            loss_type: str = "rgb",
    ) -> None:
        super(_ContextualBilateralLoss, self).__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.weight_spatial = weight_spatial
        self.bandwidth = bandwidth
        self.loss_type = loss_type

        if loss_type == "vgg":
            # Get the name of the specified feature extraction node
            self.feature_model_extractor_nodes = feature_model_extractor_nodes
            # Load the VGG19 model trained on the ImageNet dataset.
            model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
            # Extract the thirty-sixth layer output in the VGG19 model as the content loss.
            self.feature_extractor = create_feature_extractor(model, feature_model_extractor_nodes)
            # set to validation mode
            self.feature_extractor.eval()

            # The preprocessing method of the input data.
            # This is the VGG model preprocessing method of the ImageNet dataset.
            self.normalize = transforms.Normalize(feature_model_normalize_mean, feature_model_normalize_std)

            # Freeze model parameters.
            for model_parameters in self.feature_extractor.parameters():
                model_parameters.requires_grad = False

    def forward(self, sr_tensor: Tensor, gt_tensor: Tensor):
        assert sr_tensor.size() == gt_tensor.size(), "Two tensor must have the same size"

        device = sr_tensor.device

        losses = []
        if self.loss_type == "vgg":
            # Standardized operations
            sr_tensor = self.normalize(sr_tensor)
            gt_tensor = self.normalize(gt_tensor)

            # VGG19 conv3_4 feature extraction
            sr_feature = self.feature_extractor(sr_tensor)
            gt_feature = self.feature_extractor(gt_tensor)

            for i in range(len(self.feature_model_extractor_nodes)):
                losses.append(_compute_contextual_bilateral_loss(sr_feature[self.feature_model_extractor_nodes[i]],
                                                                 gt_feature[self.feature_model_extractor_nodes[i]],
                                                                 self.weight_spatial,
                                                                 self.bandwidth))

        elif self.loss_type == "rgb":
            sr_patches = _extract_image_patches(sr_tensor, self.patch_size, self.stride)
            gt_patches = _extract_image_patches(gt_tensor, self.patch_size, self.stride)
            _, _, h_patch, w_patch, n_patches = sr_patches.shape

            sr_tensor = sr_patches.reshape(sr_tensor.shape[0], -1, h_patch, w_patch)
            gt_tensor = gt_patches.reshape(sr_tensor.shape[0], -1, h_patch, w_patch)

            losses.append(_compute_contextual_bilateral_loss(sr_tensor,
                                                             gt_tensor,
                                                             self.weight_spatial,
                                                             self.bandwidth))

        else:
            raise ValueError("`loss_type` only supported `vgg` or `rgb`")

        losses = torch.Tensor([losses]).to(device=device)

        return losses.squeeze()


def contextual_bilateral_loss_for_rgb(**kwargs) -> _ContextualBilateralLoss:
    contextual_loss = _ContextualBilateralLoss(loss_type="rgb", **kwargs)

    return contextual_loss


def contextual_bilateral_loss_for_vgg(**kwargs) -> _ContextualBilateralLoss:
    contextual_loss = _ContextualBilateralLoss(loss_type="vgg", **kwargs)

    return contextual_loss