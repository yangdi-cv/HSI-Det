"""Content-aware upsampling for top-down multiscale fusion."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ContentAwareUpsample(nn.Module):
    """Predict per-location reassembly kernels before a 1x1 output projection."""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 3, scale: int = 2
    ) -> None:
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if scale < 1 or in_channels < 4:
            raise ValueError(
                "scale must be positive and in_channels must be at least 4"
            )
        self.kernel_size = kernel_size
        self.scale = scale
        self.reduce = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.kernel_predictor = nn.Conv2d(
            in_channels // 4,
            scale * scale * kernel_size * kernel_size,
            kernel_size,
            padding=kernel_size // 2,
        )
        self.project = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, features: Tensor) -> Tensor:
        batch, channels, height, width = features.shape
        patch_count = self.kernel_size * self.kernel_size
        scale_count = self.scale * self.scale

        kernels = self.kernel_predictor(self.reduce(features))
        kernels = F.softmax(F.pixel_shuffle(kernels, self.scale), dim=1)
        kernels = kernels.unfold(2, self.scale, self.scale)
        kernels = kernels.unfold(3, self.scale, self.scale)
        kernels = kernels.reshape(batch, patch_count, height, width, scale_count)
        kernels = kernels.permute(0, 2, 3, 1, 4)

        patches = F.unfold(features, self.kernel_size, padding=self.kernel_size // 2)
        patches = patches.reshape(batch, channels, patch_count, height, width)
        patches = patches.permute(0, 3, 4, 1, 2)
        reassembled = patches @ kernels
        reassembled = reassembled.reshape(batch, height, width, channels * scale_count)
        reassembled = reassembled.permute(0, 3, 1, 2)
        return self.project(F.pixel_shuffle(reassembled, self.scale))
