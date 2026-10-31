"""Training images for the unit detector, generated with exact labels.

Each image is an empty arena (a recorded one with the units averaged away, or one of the
dataset's backgrounds) in the part2 frame (568 x 896), with crown towers, then units cut from
real battles pasted on it where they could stand, back to front, plus things that are not
units (health bars, emotes, text, elixir drops, blood) that the detector must learn to ignore.

Every sprite file is named <unit>_<side>_<n>.png, side 0 ours (seen from behind, walking up)
and 1 the enemy's; each unit has one class per side, so the detector says which side a unit
is on as well as what it is.

    python -m crownzero.detector.generate SPRITES OUT --n 6000 [--arena arena.png ...]

SPRITES is the dataset's images/segment folder. OUT gets images/, labels/ (YOLO format)
and data.yaml.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

from .space import PART2_H, PART2_W, tile_to_part2

# Not units: never labelled, pasted only as distractors.
NOT_UNITS = {
    "bar", "bar-level", "tower-bar", "king-tower-bar", "skeleton-king-bar",
    "dagger-duchess-tower-bar", "clock", "emote", "text", "small-text", "big-text", "elixir",
    "selected", "background-items", "dirt", "evolution-symbol", "ice-spirit-evolution-symbol",
}
TOWERS = {"king-tower", "queen-tower", "cannoneer-tower", "dagger-duchess-tower"}
FOLDERS_SKIPPED = {"backgrounds"}
# Spells and effects are drawn on the ground but anywhere, flying units anywhere too
FLYING = {
    "bat", "bat-evolution", "minion", "mega-minion", "baby-dragon", "inferno-dragon",
    "electro-dragon", "skeleton-dragon", "balloon", "lava-hound", "lava-pup", "phoenix-big",
    "phoenix-small", "flying-machine", "skeleton-barrel",
}
SPELLS = {
    "arrows", "fireball", "zap", "the-log", "poison", "freeze", "rage", "tornado",
    "earthquake", "clone", "lightning", "rocket", "graveyard", "goblin-barrel",
    "barbarian-barrel", "royal-delivery", "giant-snowball", "skeleton-king-skill",
}
# Tower footprints in tiles (x0, x1, u0, u1): ground units never stand there
FOOTPRINTS = [(7, 11, 1, 5), (2, 5, 5, 8), (13, 16, 5, 8),
              (7, 11, 27, 31), (2, 5, 24, 27), (13, 16, 24, 27)]
# Where each tower sprite's bottom centre goes (tile x, u), and whose it is (0 ours, 1 enemy)
TOWER_SPOTS = [("king-tower", 9.0, 1.0, 0), ("princess", 3.5, 5.0, 0), ("princess", 14.5, 5.0, 0),
               ("king-tower", 9.0, 27.0, 1), ("princess", 3.5, 24.0, 1),
               ("princess", 14.5, 24.0, 1)]
PRINCESS_KINDS = ("queen-tower", "cannoneer-tower", "dagger-duchess-tower")


def classes_from(sprites: Path) -> list[str]:
    """Class names: '<unit>|<side>' for every unit folder, both sides; towers included (so
    the detector names them, and the bot drops them)."""
    units = sorted(p.name for p in sprites.iterdir()
                   if p.is_dir() and p.name not in NOT_UNITS | FOLDERS_SKIPPED)
    return [f"{u}|{s}" for u in units for s in (0, 1)]


@dataclass
class Sprite:
    unit: str
    side: int
    path: Path


class SpriteBank:
    def __init__(self, sprites: Path):
        self.root = Path(sprites)
        self.classes = classes_from(self.root)
        self.index = {c: i for i, c in enumerate(self.classes)}
        self.units: dict[str, list[Sprite]] = {}
        self.distractors: list[Sprite] = []
        self.backgrounds = sorted((self.root / "backgrounds").glob("*.jpg"))
        for folder in sorted(self.root.iterdir()):
            if not folder.is_dir() or folder.name in FOLDERS_SKIPPED:
                continue
            for f in sorted(folder.glob("*.png")):
                parts = f.stem.split("_")
                side = int(parts[1]) if len(parts) > 1 and parts[1] in ("0", "1") else 0
                s = Sprite(folder.name, side, f)
                if folder.name in NOT_UNITS:
                    self.distractors.append(s)
                else:
                    self.units.setdefault(folder.name, []).append(s)
        self.unit_names = sorted(u for u in self.units if u not in TOWERS)
        self._cache: dict[Path, Image.Image] = {}

    def image(self, s: Sprite) -> Image.Image:
        if s.path not in self._cache:
            self._cache[s.path] = Image.open(s.path).convert("RGBA")
        return self._cache[s.path]


def _blocked(tx: float, u: float) -> bool:
    return any(x0 <= tx < x1 and u0 <= u < u1 for x0, x1, u0, u1 in FOOTPRINTS)


class Generator:
    def __init__(self, bank: SpriteBank, arenas: list[Path] | None = None, seed: int = 0,
                 p_arena: float = 0.6):
        self.bank = bank
        self.rng = np.random.default_rng(seed)
        self.arenas = [Image.open(a).convert("RGB").resize((PART2_W, PART2_H))
                       for a in (arenas or [])]
        self.p_arena = p_arena
        self._bgs = [Image.open(b).convert("RGB").resize((PART2_W, PART2_H))
                     for b in bank.backgrounds]

    # ------------------------------------------------------------------ pieces
    def _background(self):
        """(image, has_towers): a recorded arena already has its towers."""
        if self.arenas and self.rng.random() < self.p_arena:
            return self.arenas[int(self.rng.integers(len(self.arenas)))].copy(), True
        return self._bgs[int(self.rng.integers(len(self._bgs)))].copy(), False

    def _pick(self, unit: str, side: int | None = None) -> Sprite:
        options = self.bank.units[unit]
        if side is not None:
            same = [s for s in options if s.side == side]
            options = same or options
        return options[int(self.rng.integers(len(options)))]

    def _sprite_image(self, s: Sprite, scale: float = 1.0) -> Image.Image:
        img = self.bank.image(s)
        k = scale * self.rng.uniform(0.9, 1.1)
        if abs(k - 1) > 0.02:
            img = img.resize((max(2, round(img.width * k)), max(2, round(img.height * k))))
        if self.rng.random() < 0.3:
            rgb = ImageEnhance.Brightness(img.convert("RGB")).enhance(self.rng.uniform(0.8, 1.2))
            rgb.putalpha(img.getchannel("A"))
            img = rgb
        if self.rng.random() < 0.08:  # a unit still deploying is drawn see-through
            a = np.asarray(img.getchannel("A"), np.float32) * self.rng.uniform(0.4, 0.8)
            img = img.copy()
            img.putalpha(Image.fromarray(a.astype(np.uint8)))
        return img

    def _paste(self, canvas, img, cx, bottom):
        """Paste with the image's bottom centre at (cx, bottom). Returns the visible box of
        its non-transparent pixels, clipped to the canvas, or None."""
        x0 = round(cx - img.width / 2)
        y0 = round(bottom - img.height)
        canvas.paste(img, (x0, y0), img)
        alpha = np.asarray(img.getchannel("A"))
        ys, xs = np.nonzero(alpha > 40)
        if len(xs) == 0:
            return None
        bx0, by0 = max(0, x0 + xs.min()), max(0, y0 + ys.min())
        bx1, by1 = min(PART2_W, x0 + xs.max() + 1), min(PART2_H, y0 + ys.max() + 1)
        if bx1 - bx0 < 3 or by1 - by0 < 3:
            return None
        return bx0, by0, bx1, by1

    # ------------------------------------------------------------------ one image
    def make(self, n_units: int | None = None):
        """(PIL image, labels as [(class index, x0, y0, x1, y1)] in pixels)."""
        canvas, has_towers = self._background()
        items = []  # (draw order key, kind, sprite, cx, bottom)
        if not has_towers:
            kinds = [k for k in PRINCESS_KINDS if k in self.bank.units] or ["queen-tower"]
            kind = kinds[int(self.rng.integers(len(kinds)))]
            for name, tx, u, side in TOWER_SPOTS:
                if name == "princess" and self.rng.random() < 0.1:
                    continue  # a destroyed tower
                unit = kind if name == "princess" else name
                if unit not in self.bank.units:
                    continue
                x, y = tile_to_part2(tx, u)
                items.append((y, "tower", self._pick(unit, side), x, y))
        n = int(self.rng.integers(4, 32)) if n_units is None else n_units
        for _ in range(n):
            unit = self.bank.unit_names[int(self.rng.integers(len(self.bank.unit_names)))]
            s = self._pick(unit)
            for _try in range(20):
                tx, u = self.rng.uniform(0, 18), self.rng.uniform(0, 32)
                if unit in FLYING or unit in SPELLS or not _blocked(tx, u):
                    break
            x, y = tile_to_part2(tx, u)
            if unit in FLYING:
                y -= self.rng.uniform(10, 30)  # drawn above its shadow
            items.append((y, "unit", s, x, y))
        for _ in range(int(self.rng.integers(0, 10))):
            if not self.bank.distractors:
                break
            s = self.bank.distractors[int(self.rng.integers(len(self.bank.distractors)))]
            x, y = self.rng.uniform(0, PART2_W), self.rng.uniform(40, PART2_H)
            items.append((PART2_H + 1, "noise", s, x, y))  # drawn on top of everything
        labels = []
        for _, kind, s, x, y in sorted(items, key=lambda t: t[0]):
            box = self._paste(canvas, self._sprite_image(s), x, y)
            if box is None or kind == "noise":
                continue
            labels.append((self.bank.index[f"{s.unit}|{s.side}"], *box))
        return canvas, labels


def write_dataset(gen: Generator, out: Path, n: int, val_frac: float = 0.1) -> None:
    out = Path(out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    n_val = max(1, int(n * val_frac))
    for i in range(n):
        split = "val" if i < n_val else "train"
        img, labels = gen.make()
        stem = f"gen_{i:06d}"
        img.save(out / "images" / split / f"{stem}.jpg", quality=90)
        with open(out / "labels" / split / f"{stem}.txt", "w") as f:
            for c, x0, y0, x1, y1 in labels:
                f.write(f"{c} {(x0 + x1) / 2 / PART2_W:.6f} {(y0 + y1) / 2 / PART2_H:.6f} "
                        f"{(x1 - x0) / PART2_W:.6f} {(y1 - y0) / PART2_H:.6f}\n")
    names = "\n".join(f"  {i}: '{c}'" for i, c in enumerate(gen.bank.classes))
    (out / "data.yaml").write_text(f"path: {out.resolve()}\ntrain: images/train\n"
                                   f"val: images/val\nnames:\n{names}\n")
    (out / "classes.json").write_text(json.dumps(gen.bank.classes))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sprites", type=Path, help="the dataset's images/segment folder")
    ap.add_argument("out", type=Path)
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--arena", type=Path, action="append", default=[],
                    help="an empty arena in the part2 frame (see arena_from_recording)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    gen = Generator(SpriteBank(args.sprites), args.arena, seed=args.seed)
    write_dataset(gen, args.out, args.n)
    print(f"{args.n} images, {len(gen.bank.classes)} classes -> {args.out}")


if __name__ == "__main__":
    main()
