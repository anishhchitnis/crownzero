"""Three frames for the same arena.

* SCREENSHOT: Build-A-Bot's 368 x 652 screenshot of a 720 x 1280 display.
* PART2: KataCR's arena crop, 568 x 896, where the sprites were cut and the detector works.
  It spans the 18 tiles' width, and its river centre is at y 467.
* TILES: own frame, x 0-18 left to right, u 0-32 from our back row to the enemy's.

Calibrated by laying a KataCR frame over a recorded one: towers, river and bridges line up.
"""
from __future__ import annotations

DISPLAY_W, DISPLAY_H = 720, 1280
SHOT_W, SHOT_H = 368, 652
TILE_W, TILE_H = 34.0, 27.6  # display pixels (Build-A-Bot constants)
TILE_X0, TILE_Y0 = 52.0, 296.0
PART2_W, PART2_H = 568, 896
P2_TILE_W = PART2_W / 18  # 31.56 px
P2_TILE_H = 25.2
P2_RIVER_Y = 467.0  # tile u = 16


def tile_to_part2(tx: float, u: float) -> tuple[float, float]:
    return tx * P2_TILE_W, P2_RIVER_Y - (u - 16.0) * P2_TILE_H


def part2_to_tile(x: float, y: float) -> tuple[float, float]:
    return x / P2_TILE_W, 16.0 - (y - P2_RIVER_Y) / P2_TILE_H


def tile_to_shot(tx: float, u: float) -> tuple[float, float]:
    return ((TILE_X0 + TILE_W * tx) * SHOT_W / DISPLAY_W,
            (DISPLAY_H - TILE_Y0 - TILE_H * u) * SHOT_H / DISPLAY_H)


def part2_to_shot(x: float, y: float) -> tuple[float, float]:
    return tile_to_shot(*part2_to_tile(x, y))


def part2_box_in_shot() -> tuple[float, float, float, float]:
    """Where the part2 frame sits in a screenshot (x0, y0, x1, y1)."""
    x0, y0 = part2_to_shot(0, 0)
    x1, y1 = part2_to_shot(PART2_W, PART2_H)
    return x0, y0, x1, y1


def shot_to_part2_image(screenshot):
    """The part2 frame of a screenshot (PIL image of any size; scaled to 368 x 652 first)."""
    img = screenshot.convert("RGB")
    if img.size != (SHOT_W, SHOT_H):
        img = img.resize((SHOT_W, SHOT_H))
    return img.crop(tuple(round(v) for v in part2_box_in_shot())).resize((PART2_W, PART2_H))


def build_a_bot_tile(x_shot: float, y_shot_bottom: float) -> tuple[int, int]:
    """Build-A-Bot's tile for a box: its bottom centre, rounded as Build-A-Bot rounds."""
    x = x_shot * DISPLAY_W / SHOT_W
    y = y_shot_bottom * DISPLAY_H / SHOT_H
    return (round((x - TILE_X0) / TILE_W - 0.5), round((DISPLAY_H - TILE_Y0 - y) / TILE_H - 0.5))
