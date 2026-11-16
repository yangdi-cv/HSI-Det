"""Build the paper-facing HSI-Det graph with the Ultralytics detection head."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
from torch import nn
from ultralytics.nn.modules import RepConv
from ultralytics.nn.tasks import DetectionModel

from .attention import HybridAttention
from .backbone import CrossStageFusion
from .fusion import ContentAwareUpsample


def _replace_layer(model: DetectionModel, index: int, layer: nn.Module) -> None:
    previous = model.model[index]
    for attribute in ("i", "f"):
        setattr(layer, attribute, getattr(previous, attribute))
    layer.type = type(layer).__name__
    layer.np = sum(parameter.numel() for parameter in layer.parameters())
    model.model[index] = layer


def build_model(
    config: str | Path | dict[str, Any],
    *,
    channels: int = 25,
    classes: int | None = None,
    verbose: bool = False,
) -> DetectionModel:
    """Construct HSI-Det without modifying the installed Ultralytics package.

    Ultralytics parses its own stock layers first. Shape-equivalent placeholders
    are then replaced by the research modules at the same graph positions.
    This preserves all routing indices, Detect strides, and training interfaces.
    """
    if isinstance(config, dict):
        architecture = deepcopy(config)
    else:
        with Path(config).open("r", encoding="utf-8") as stream:
            architecture = yaml.safe_load(stream)
    layers = architecture["backbone"] + architecture["head"]
    parsed = deepcopy(architecture)
    parsed_layers = parsed["backbone"] + parsed["head"]

    for layer in parsed_layers:
        module = layer[2]
        if module == "CrossStageFusion":
            layer[2] = "C2f"
        elif module == "ContentAwareUpsample":
            layer[2] = "nn.ConvTranspose2d"
            layer[3] = [layer[3][0], 2, 2]
        elif module == "HybridAttention":
            layer[2] = "nn.Identity"
            layer[3] = []
        elif module == "RepConv":
            layer[2] = "Conv"

    model = DetectionModel(parsed, ch=channels, nc=classes, verbose=verbose)
    for index, (_, _, module, arguments) in enumerate(layers):
        placeholder = model.model[index]
        if module == "CrossStageFusion":
            if not isinstance(placeholder, nn.Module) or not hasattr(
                placeholder, "cv1"
            ):
                raise TypeError(f"Unexpected fusion placeholder at layer {index}")
            in_channels = placeholder.cv1.conv.in_channels
            out_channels = placeholder.cv2.conv.out_channels
            if in_channels != out_channels:
                raise ValueError(
                    "This architecture expects equal channels within each fusion stage"
                )
            repeats = len(placeholder.m)
            blocks = [
                CrossStageFusion(in_channels, out_channels) for _ in range(repeats)
            ]
            replacement = nn.Sequential(*blocks) if repeats > 1 else blocks[0]
        elif module == "ContentAwareUpsample":
            replacement = ContentAwareUpsample(
                placeholder.in_channels,
                placeholder.out_channels,
                kernel_size=arguments[1],
                scale=arguments[2],
            )
        elif module == "HybridAttention":
            previous = model.model[index - 1]
            if not hasattr(previous, "c2"):
                raise TypeError(
                    f"Attention at layer {index} must follow a convolutional layer"
                )
            replacement = HybridAttention(previous.c2)
        elif module == "RepConv":
            replacement = RepConv(
                placeholder.conv.in_channels,
                placeholder.conv.out_channels,
                k=arguments[1],
                s=arguments[2],
            )
        else:
            continue
        _replace_layer(model, index, replacement)

    model.yaml = architecture
    model.yaml["channels"] = channels
    if classes is not None:
        model.yaml["nc"] = classes
    return model
