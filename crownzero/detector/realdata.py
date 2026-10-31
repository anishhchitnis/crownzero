"""Real, hand-labelled battle frames (KataCR's part2 set) in this detector's classes.

The KataCR dataset labels each frame with `cls cx cy w h belong ...` (YOLO, plus the side:
0 ours, 1 enemy) under its own class list (ClashRoyale_detection.yaml). This converts a folder
of them to our '<unit>|<side>' classes; labels of classes we do not detect (bars, text, ...)
are dropped.

    python -m crownzero.detector.realdata PART2_FOLDER NAMES_YAML CLASSES_JSON OUT
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path


def kata_names(yaml_path: Path) -> dict[int, str]:
    names = {}
    for line in Path(yaml_path).read_text().splitlines():
        m = re.match(r"\s+(\d+):\s*'?([\w-]+)'?\s*$", line)
        if m:
            names[int(m.group(1))] = m.group(2)
    return names


def convert(folder: Path, yaml_path: Path, classes: list[str], out: Path) -> tuple[int, int]:
    names = kata_names(yaml_path)
    index = {c: i for i, c in enumerate(classes)}
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    frames = boxes = 0
    for txt in sorted(Path(folder).rglob("*.txt")):
        jpg = txt.with_suffix(".jpg")
        if not jpg.exists():
            continue
        rows = []
        for line in txt.read_text().splitlines():
            v = line.split()
            if len(v) < 6:
                continue
            c = index.get(f"{names.get(int(float(v[0])), '?')}|{int(float(v[5]))}")
            if c is not None:
                rows.append(f"{c} {v[1]} {v[2]} {v[3]} {v[4]}")
        stem = "_".join(jpg.relative_to(folder).with_suffix("").parts)
        shutil.copy(jpg, out / "images" / f"{stem}.jpg")
        (out / "labels" / f"{stem}.txt").write_text("\n".join(rows) + ("\n" if rows else ""))
        frames += 1
        boxes += len(rows)
    return frames, boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("names_yaml", type=Path)
    ap.add_argument("classes_json", type=Path)
    ap.add_argument("out", type=Path)
    a = ap.parse_args()
    f, b = convert(a.folder, a.names_yaml, json.loads(a.classes_json.read_text()), a.out)
    print(f"{f} frames, {b} boxes -> {a.out}")


if __name__ == "__main__":
    main()
