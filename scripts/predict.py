"""Run inference with a checkpoint produced by this project."""

from __future__ import annotations

import argparse

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="Path to a local checkpoint")
    parser.add_argument("--source", required=True, help="Image or directory to predict")
    parser.add_argument("--device", default=None)
    arguments = parser.parse_args()

    model = YOLO(arguments.weights)
    model.predict(source=arguments.source, device=arguments.device, save=True)


if __name__ == "__main__":
    main()
