"""Ultralytics trainer that constructs the HSI-Det architecture."""

from __future__ import annotations

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import RANK

from .model import build_model


class HSIDetTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = build_model(
            cfg,
            channels=self.data["channels"],
            classes=self.data["nc"],
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)
        return model
