"""Cross-stage feature fusion used in the HSI-Det backbone."""

from __future__ import annotations

from torch import Tensor, nn

from ultralytics.nn.modules.block import C2f


class ConvNorm(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        padding: int = 0,
        groups: int = 1,
        zero_init: bool = False,
    ) -> None:
        super().__init__()
        self.convolution = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.normalization = nn.BatchNorm2d(out_channels)
        if zero_init:
            nn.init.zeros_(self.normalization.weight)
        nn.init.zeros_(self.normalization.bias)

    def forward(self, features: Tensor) -> Tensor:
        return self.normalization(self.convolution(features))


class SpatialMixer(nn.Module):
    """Combine depthwise 3x3, depthwise 1x1, and identity paths."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.local = ConvNorm(channels, channels, 3, padding=1, groups=channels)
        self.pointwise = nn.Conv2d(channels, channels, 1, groups=channels)
        self.normalization = nn.BatchNorm2d(channels)

    def forward(self, features: Tensor) -> Tensor:
        mixed = self.local(features) + self.pointwise(features) + features
        return self.normalization(mixed)


class ChannelMixer(nn.Module):
    """Expand and compress channels with a zero-initialized residual path."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.expand = ConvNorm(channels, 2 * channels, 1)
        self.activation = nn.GELU()
        self.compress = ConvNorm(2 * channels, channels, 1, zero_init=True)

    def forward(self, features: Tensor) -> Tensor:
        return features + self.compress(self.activation(self.expand(features)))


class FusionBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.spatial = SpatialMixer(channels)
        self.channel = ChannelMixer(channels)

    def forward(self, features: Tensor) -> Tensor:
        return self.channel(self.spatial(features))


class CrossStageFusion(C2f):
    """Preserve one projected branch while transforming the other branch."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        blocks: int = 1,
        shortcut: bool = False,
        groups: int = 1,
        expansion: float = 0.5,
    ) -> None:
        super().__init__(in_channels, out_channels, blocks, shortcut, groups, expansion)
        self.m = nn.ModuleList(FusionBlock(self.c) for _ in range(blocks))
