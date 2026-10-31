"""The trained unit detector behind Build-A-Bot's interface.

``CrownZeroUnits(model).run(screenshot) -> (allies, enemies)``, lists of UnitDetection with
Build-A-Bot's unit names and tiles, so it replaces ``Detector.unit_detector`` as is.

What it does differently from Build-A-Bot's detector:
  * it reads the arena in the frame its training images were drawn in (space.py);
  * each class carries the side, so no separate side classifier;
  * towers (and their tower troops) are classes of their own, and are dropped here;
  * spells and projectiles are dropped (Build-A-Bot never reported them either);
  * an evolved unit is reported as its card's unit, with ``evolved`` on the detection.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .generate import SPELLS, TOWERS
from .space import PART2_H, PART2_W, build_a_bot_tile, part2_to_shot, shot_to_part2_image

DROPPED = TOWERS | SPELLS | {"axe", "bomb", "goblin-ball"}
# KataCR name -> Build-A-Bot name where they differ (after dropping "-evolution")
RENAME = {"goblin-brawler": "brawler", "elixir-golem-big": "elixir_golem_large",
          "elixir-golem-mid": "elixir_golem_medium", "mini-pekka": "minipekka",
          "phoenix-big": "phoenix_large"}
FLYING = {"bat", "minion", "mega_minion", "baby_dragon", "inferno_dragon", "electro_dragon",
          "skeleton_dragon", "balloon", "lava_hound", "lava_pup", "phoenix_large",
          "phoenix_small", "flying_machine", "skeleton_barrel"}
BUILDINGS = {"cannon", "tesla", "inferno_tower", "bomb_tower", "mortar", "x_bow", "tombstone",
             "goblin_hut", "barbarian_hut", "furnace", "goblin_cage", "elixir_collector",
             "goblin_drill"}
BUILDING_TARGETERS = {"giant", "hog_rider", "royal_giant", "balloon", "golem", "lava_hound",
                      "ram_rider", "battle_ram", "wall_breaker", "goblin_giant",
                      "electro_giant", "elixir_golem_large", "elixir_golem_medium",
                      "elixir_golem_small", "royal_hog", "skeleton_barrel"}


def bab_name(kata: str) -> tuple[str | None, bool]:
    """(Build-A-Bot unit name or None if not a unit we report, evolved?)"""
    evolved = kata.endswith("-evolution")
    base = kata[: -len("-evolution")] if evolved else kata
    if base in DROPPED:
        return None, evolved
    return RENAME.get(base, base.replace("-", "_")), evolved


@dataclass(frozen=True)
class _Unit:  # Build-A-Bot's Unit fields; the bot reads them by name
    name: str
    category: str
    target: str | None
    transport: str | None


@dataclass(frozen=True)
class _Position:
    bbox: tuple
    conf: float
    tile_x: int
    tile_y: int


@dataclass(frozen=True)
class Detection:
    unit: _Unit
    position: _Position
    evolved: bool = False


def _unit(name: str):
    """Build-A-Bot's own Unit when it knows the name (so everything downstream matches),
    else one with the same fields."""
    try:
        from clashroyalebuildabot.namespaces.units import Units

        for u in Units.__dict__.values() if hasattr(Units, "__dict__") else []:
            if getattr(u, "name", None) == name:
                return u
        for field in getattr(Units, "__dataclass_fields__", {}):
            u = getattr(Units, field)
            if getattr(u, "name", None) == name:
                return u
    except ImportError:
        pass
    building = name in BUILDINGS
    return _Unit(name, "building" if building else "troop",
                 "buildings" if name in BUILDING_TARGETERS else "all",
                 None if building else ("air" if name in FLYING else "ground"))


class CrownZeroUnits:
    MIN_CONF = 0.35

    def __init__(self, model_path: str | Path, classes_path: str | Path | None = None):
        import onnxruntime as ort

        model_path = Path(model_path)
        classes_path = Path(classes_path) if classes_path else model_path.with_name("classes.json")
        self.classes = json.loads(classes_path.read_text())
        self.sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        self.size = int(inp.shape[2])  # square letterbox, as exported
        self._units = {}

    def _letterbox(self, img):
        k = self.size / max(PART2_W, PART2_H)
        w, h = round(PART2_W * k), round(PART2_H * k)
        canvas = np.full((self.size, self.size, 3), 114, np.uint8)
        dx, dy = (self.size - w) // 2, (self.size - h) // 2
        canvas[dy:dy + h, dx:dx + w] = np.asarray(img.resize((w, h)))
        x = canvas.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        return x, k, dx, dy

    def detect_part2(self, part2_img):
        """[(class name, conf, (x0, y0, x1, y1) in part2 pixels)]"""
        x, k, dx, dy = self._letterbox(part2_img)
        out = self.sess.run(None, {self.input_name: x})[0][0]  # (N, 6): x0 y0 x1 y1 conf cls
        res = []
        for x0, y0, x1, y1, conf, c in out:
            if conf < self.MIN_CONF:
                continue
            box = ((x0 - dx) / k, (y0 - dy) / k, (x1 - dx) / k, (y1 - dy) / k)
            res.append((self.classes[int(c)], float(conf), box))
        return res

    def run(self, screenshot):
        allies, enemies = [], []
        for cls, conf, (x0, y0, x1, y1) in self.detect_part2(shot_to_part2_image(screenshot)):
            kata, side = cls.rsplit("|", 1)
            name, evolved = bab_name(kata)
            if name is None:
                continue
            sx0, sy0 = part2_to_shot(x0, y0)
            sx1, sy1 = part2_to_shot(x1, y1)
            # bbox in the screenshot the bot was given (Build-A-Bot's 368 x 652 frame)
            sw, sh = screenshot.size
            fx, fy = sw / 368, sh / 652
            bbox = (round(sx0 * fx), round(sy0 * fy), round(sx1 * fx), round(sy1 * fy))
            tx, ty = build_a_bot_tile((sx0 + sx1) / 2, sy1)
            if name not in self._units:
                self._units[name] = _unit(name)
            det = Detection(self._units[name], _Position(bbox, conf, tx, ty), evolved)
            (allies if side == "0" else enemies).append(det)
        return allies, enemies
