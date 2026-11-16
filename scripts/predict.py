"""Run ONNX inference on the included 25-channel TIFF examples."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights", type=Path, default=root / "weights" / "hsidet.onnx"
    )
    parser.add_argument(
        "--source", type=Path, default=root / "examples" / "tiff"
    )
    parser.add_argument("--device", default="cpu")
    arguments = parser.parse_args()

    model = YOLO(str(arguments.weights), task="detect")
    model.predict(
        source=str(arguments.source), device=arguments.device, imgsz=640, save=True
    )


if __name__ == "__main__":
    main()
