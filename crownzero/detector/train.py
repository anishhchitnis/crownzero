"""Train the unit detector: generated images, plus real labelled frames when given.

    python -m crownzero.detector.train DATASET_DIR [--model yolov8s.pt] [--epochs 60]

DATASET_DIR must hold images/{train,val} and labels/{train,val} (generate.py writes them)
and classes.json. Real frames (realdata.py) can be added to images/train with their labels
under labels/train; keep the real frames you validate on out of train.

On an Apple Silicon Mac the device is "mps"; overnight is enough for yolov8s at 640.
The best weights are exported to ONNX next to them for the bot (detector.py).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def write_yaml(ds: Path, val: str = "images/val") -> Path:
    classes = json.loads((ds / "classes.json").read_text())
    names = "\n".join(f"  {i}: '{c}'" for i, c in enumerate(classes))
    y = ds / "data.yaml"
    y.write_text(f"path: {ds.resolve()}\ntrain: images/train\nval: {val}\nnames:\n{names}\n")
    return y


def pick_device(requested: str | None) -> str:
    if requested:
        return requested
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    return "0" if torch.cuda.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--model", default="yolov8s.pt", help="yolov8n.pt for a quick run")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--val", default="images/val", help="folder of validation images")
    ap.add_argument("--name", default="crownzero_units")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(a.model)
    # No left-right flips: a unit's side is read partly from which way it faces.
    model.train(data=str(write_yaml(a.dataset, a.val)), epochs=a.epochs, imgsz=a.imgsz,
                batch=a.batch, device=pick_device(a.device), name=a.name, fliplr=0.0,
                mosaic=0.5, workers=a.workers, project=str(a.dataset / "runs"), exist_ok=True)
    best = a.dataset / "runs" / a.name / "weights" / "best.pt"
    YOLO(str(best)).export(format="onnx", imgsz=a.imgsz, nms=True)
    print(f"weights: {best}  onnx: {best.with_suffix('.onnx')}")


if __name__ == "__main__":
    main()
