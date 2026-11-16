"""Train HSI-Det on a 25-channel Ultralytics detection dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from hsi_det.trainer import HSIDetTrainer


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, required=True, help="Ultralytics dataset YAML"
    )
    parser.add_argument(
        "--model", type=Path, default=root / "models" / "hsi_det_b.yaml"
    )
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    arguments = parser.parse_args()

    options = {
        "model": str(arguments.model),
        "data": str(arguments.data),
        "epochs": arguments.epochs,
        "batch": arguments.batch,
        "imgsz": arguments.imgsz,
        "project": str(root / "runs"),
        "name": "hsi_det_b",
    }
    if arguments.device is not None:
        options["device"] = arguments.device
    HSIDetTrainer(overrides=options).train()


if __name__ == "__main__":
    main()
