"""Build the unit detector end to end, on your own machine (an Apple Silicon Mac: "mps").

    python -m crownzero.detector.build ~/Downloads/cr_detector \
        --recordings ~/Downloads/cr_recordings [--quick]

1. downloads the sprite cut-outs and some hand-labelled real frames of the KataCR detection
   dataset (MIT, github.com/wty-yy/Clash-Royale-Detection-Dataset), only the folders needed;
2. makes empty arenas from your recordings (the median of a battle's frames: the units
   average away, the towers stay; --private frames are fine, their name strips are black);
3. generates training images (generate.py) and converts the real frames (realdata.py),
   keeping one real folder out of training to measure on;
4. trains YOLOv8 (train.py) and writes models/units_cz.onnx + classes.json for the bot.

--quick (yolov8n, fewer images and epochs) takes an hour or two; the default is a night.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from .generate import Generator, SpriteBank, write_dataset
from .realdata import convert
from .space import shot_to_part2_image

REPO = "https://github.com/wty-yy/Clash-Royale-Detection-Dataset.git"
TRAIN_REAL = ["WTY_20240305", "WTY_20240306", "WTY_20240307", "WTY_20240308", "WTY_20240309",
              "WTY_20240227_miners", "WTY_20240218_episodes", "OYASSU_20230212_episodes"]
VAL_REAL = "OYASSU_20230305_episodes"  # never trained on: the measure of real frames


def download(root: Path) -> Path:
    data = root / "kata"
    if not (data / ".git").exists():
        subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", REPO,
                        str(data)], check=True)
    folders = ["images/segment"] + [f"images/part2/{f}" for f in TRAIN_REAL + [VAL_REAL]]
    subprocess.run(["git", "-C", str(data), "sparse-checkout", "set", *folders], check=True)
    subprocess.run(["git", "-C", str(data), "show",
                    "HEAD:images/part2/ClashRoyale_detection.yaml"],
                   check=True, stdout=open(root / "kata_names.yaml", "w"))
    return data


def arena_from_recording(folder: Path, n: int = 30) -> Image.Image | None:
    """The empty arena of one recorded battle, in the part2 frame."""
    frames = sorted(p for p in folder.glob("[0-9]*.jpg") if "_ann" not in p.name)
    if len(frames) < 10:
        return None
    pick = [frames[i] for i in np.linspace(0, len(frames) - 1, min(n, len(frames))).astype(int)]
    stack = np.stack([np.asarray(Image.open(p).convert("RGB").resize((368, 652)))
                      for p in pick])
    return shot_to_part2_image(Image.fromarray(np.median(stack, axis=0).astype(np.uint8)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="a working folder (a few GB)")
    ap.add_argument("--recordings", type=Path, default=None,
                    help="your cr_recordings folder, for arena backgrounds")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n", type=int, default=None, help="generated images")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-train", action="store_true", help="stop after building the dataset")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[2] / "models")
    a = ap.parse_args()
    root = a.root.expanduser()
    root.mkdir(parents=True, exist_ok=True)
    n = a.n or (2500 if a.quick else 8000)
    epochs = a.epochs or (15 if a.quick else 40)
    model = a.model or ("yolov8n.pt" if a.quick else "yolov8s.pt")

    print("1/4 downloading sprites and real frames ...", flush=True)
    data = download(root)
    sprites = data / "images" / "segment"

    print("2/4 arenas from your recordings ...", flush=True)
    arenas = []
    if a.recordings:
        for folder in sorted(a.recordings.expanduser().iterdir()):
            img = arena_from_recording(folder) if folder.is_dir() else None
            if img is not None:
                path = root / "arenas" / f"{folder.name}.png"
                path.parent.mkdir(exist_ok=True)
                img.save(path)
                arenas.append(path)
    print(f"   {len(arenas)} arenas", flush=True)

    print(f"3/4 {n} generated images + real frames ...", flush=True)
    ds = root / "dataset"
    if ds.exists():
        shutil.rmtree(ds)
    gen = Generator(SpriteBank(sprites), arenas, seed=0, p_arena=0.5 if arenas else 0.0)
    write_dataset(gen, ds, n)
    classes = gen.bank.classes
    real_boxes = 0
    for folder in TRAIN_REAL:
        tmp = root / "real_tmp"
        f, b = convert(data / "images" / "part2" / folder, root / "kata_names.yaml", classes, tmp)
        for sub in ("images", "labels"):
            for p in (tmp / sub).iterdir():
                shutil.move(str(p), ds / sub / "train" / p.name)
        shutil.rmtree(tmp)
        real_boxes += b
    f, b = convert(data / "images" / "part2" / VAL_REAL, root / "kata_names.yaml", classes,
                   root / "val_tmp")
    for sub in ("images", "labels"):
        (ds / sub / "val_real").mkdir(parents=True, exist_ok=True)
        for p in (root / "val_tmp" / sub).iterdir():
            shutil.move(str(p), ds / sub / "val_real" / p.name)
    shutil.rmtree(root / "val_tmp")
    print(f"   real: {real_boxes} training boxes, {f} held-out frames ({b} boxes)", flush=True)

    if a.no_train:
        print(f"dataset ready: {ds}")
        return
    print(f"4/4 training {model}, {epochs} epochs ...", flush=True)
    cmd = [sys.executable, "-m", "crownzero.detector.train", str(ds), "--model", model,
           "--epochs", str(epochs), "--val", "images/val_real"]
    if a.device:
        cmd += ["--device", a.device]
    subprocess.run(cmd, check=True)
    best = ds / "runs" / "crownzero_units" / "weights" / "best.onnx"
    a.out.mkdir(exist_ok=True)
    shutil.copy(best, a.out / "units_cz.onnx")
    (a.out / "classes.json").write_text(json.dumps(classes))
    print(f"done: {a.out / 'units_cz.onnx'} (use it with --detector)")


if __name__ == "__main__":
    main()
