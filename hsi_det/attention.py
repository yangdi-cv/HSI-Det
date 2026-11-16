"""Regional context and spatial saliency for HSI-Det neck features."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class RoutedRegionAttention(nn.Module):
    """Attend to tokens in the four most relevant regions of a 7x7 grid."""

    def __init__(
        self, channels: int, regions: int = 7, heads: int = 8, selected_regions: int = 4
    ) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError("channels must be divisible by heads")
        if not 1 <= selected_regions <= regions * regions:
            raise ValueError("selected_regions is outside the region grid")
        self.channels = channels
        self.regions = regions
        self.heads = heads
        self.selected_regions = selected_regions
        self.scale = channels**-0.5

        self.projection = nn.Linear(channels, 3 * channels)
        self.position = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.output = nn.Linear(channels, channels)

    def forward(self, features: Tensor) -> Tensor:
        batch, channels, original_height, original_width = features.shape
        grid = self.regions
        tokens = features.permute(0, 2, 3, 1)
        pad_height = (-original_height) % grid
        pad_width = (-original_width) % grid
        if pad_height or pad_width:
            tokens = F.pad(tokens, (0, 0, 0, pad_width, 0, pad_height))
        height, width = tokens.shape[1:3]
        window_height, window_width = height // grid, width // grid
        region_count = grid * grid
        token_count = window_height * window_width

        windows = tokens.reshape(
            batch, grid, window_height, grid, window_width, channels
        )
        windows = windows.permute(0, 1, 3, 2, 4, 5)
        windows = windows.reshape(
            batch, region_count, window_height, window_width, channels
        )
        query, key_value = self.projection(windows).split(
            (channels, 2 * channels), dim=-1
        )

        region_query = query.mean(dim=(2, 3)).detach()
        region_key = key_value[..., :channels].mean(dim=(2, 3)).detach()
        scores = (region_query * self.scale) @ region_key.transpose(-1, -2)
        _, selected = scores.topk(self.selected_regions, dim=-1)

        values = key_value[..., channels:]
        positional = values.reshape(
            batch, grid, grid, window_height, window_width, channels
        )
        positional = positional.permute(0, 5, 1, 3, 2, 4).reshape(
            batch, channels, height, width
        )
        positional = self.position(positional).permute(0, 2, 3, 1)

        flat_key_value = key_value.reshape(
            batch, region_count, token_count, 2 * channels
        )
        candidates = flat_key_value[:, None].expand(-1, region_count, -1, -1, -1)
        indices = selected[..., None, None].expand(
            -1,
            -1,
            -1,
            token_count,
            2 * channels,
        )
        selected_key_value = torch.gather(candidates, dim=2, index=indices)
        selected_keys, selected_values = selected_key_value.split(channels, dim=-1)

        head_width = channels // self.heads
        key_count = self.selected_regions * token_count
        keys = (
            selected_keys.reshape(
                batch,
                region_count,
                self.selected_regions,
                token_count,
                self.heads,
                head_width,
            )
            .permute(0, 1, 4, 5, 2, 3)
            .reshape(batch * region_count, self.heads, head_width, key_count)
        )
        values = (
            selected_values.reshape(
                batch,
                region_count,
                self.selected_regions,
                token_count,
                self.heads,
                head_width,
            )
            .permute(0, 1, 4, 2, 3, 5)
            .reshape(batch * region_count, self.heads, key_count, head_width)
        )
        queries = query.reshape(
            batch, region_count, token_count, self.heads, head_width
        )
        queries = queries.permute(0, 1, 3, 2, 4).reshape(
            batch * region_count,
            self.heads,
            token_count,
            head_width,
        )

        attention = F.softmax((queries * self.scale) @ keys, dim=-1)
        attended = attention @ values
        attended = attended.reshape(
            batch, grid, grid, self.heads, window_height, window_width, head_width
        )
        attended = attended.permute(0, 1, 4, 2, 5, 3, 6).reshape(
            batch, height, width, channels
        )
        attended = self.output(attended + positional)
        attended = attended[:, :original_height, :original_width]
        return attended.permute(0, 3, 1, 2).contiguous()


class SpatialGate(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        if kernel_size not in (3, 7):
            raise ValueError("kernel_size must be 3 or 7")
        self.mask = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, features: Tensor) -> Tensor:
        summary = torch.cat(
            (features.mean(dim=1, keepdim=True), features.amax(dim=1, keepdim=True)),
            dim=1,
        )
        return features * self.mask(summary).sigmoid()


class HybridAttention(nn.Module):
    """Apply routed regional attention followed by a spatial gate."""

    def __init__(self, channels: int, kernel_size: int = 7) -> None:
        super().__init__()
        self.regional = RoutedRegionAttention(channels)
        self.spatial = SpatialGate(kernel_size)

    def forward(self, features: Tensor) -> Tensor:
        return self.spatial(self.regional(features))
