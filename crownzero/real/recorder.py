"""Record what the bot saw and did in a real game, so its mistakes can be studied.

    python -m crownzero.real.play --deck hog --record recordings

For every decision it writes, into recordings/<date-time>/:
  NNNNN.jpg      the frame the screen reader looked at (368 x 652, as Build-A-Bot sees it)
  NNNNN_ann.jpg  the same frame, twice the size, with the tile grid, every detected unit
                 (blue = ours, red = theirs, with its tile), the hand, elixir and the
                 bot's decision drawn on top
  steps.jsonl    one line per decision: detections, cards, elixir, tower bars, the network's
                 view, and the decision (card, tile, P(wait), win estimate)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from PIL import Image, ImageDraw

# Build-A-Bot's screen geometry (constants.py), in 720 x 1280 display pixels
DISPLAY_W, DISPLAY_H = 720, 1280
SHOT_W, SHOT_H = 368, 652
TILE_W, TILE_H = 34, 27.6
TILE_X0, TILE_Y0 = 52, 296
SCALE = 2


def _tile_box(tx, ty):
    """A tile's rectangle in screenshot pixels."""
    x0 = TILE_X0 + tx * TILE_W
    yb = DISPLAY_H - TILE_Y0 - ty * TILE_H
    sx, sy = SHOT_W / DISPLAY_W, SHOT_H / DISPLAY_H
    return (x0 * sx, (yb - TILE_H) * sy, (x0 + TILE_W) * sx, yb * sy)


def _num(n):
    return None if n is None else float(getattr(n, "number", n))


class Recorder:
    def __init__(self, root, meta: dict):
        self.dir = Path(root) / time.strftime("%Y%m%d-%H%M%S")
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "meta.json").write_text(json.dumps(meta, indent=2))
        self.log = (self.dir / "steps.jsonl").open("a")
        self.n = 0
        self.t0 = time.monotonic()

    def frame(self, image, tag: str, every: int = 3, limit: int = 40):
        """Save a raw frame (e.g. a screen the reader did not recognise), sparingly."""
        n = self.tags = getattr(self, "tags", {})
        n[tag] = n.get(tag, 0) + 1
        if image is None or n[tag] % every != 1 or n[tag] > every * limit:
            return
        image.convert("RGB").save(self.dir / f"{tag}_{n[tag]:04d}.jpg", quality=85)

    def step(self, image, state, view, decision: dict):
        self.n += 1
        stem = f"{self.n:05d}"
        if image is not None:
            image.convert("RGB").save(self.dir / f"{stem}.jpg", quality=85)
            self._annotated(image, state, view, decision).save(
                self.dir / f"{stem}_ann.jpg", quality=80)
        dets = lambda ds: [dict(name=d.unit.name, tile=[d.position.tile_x, d.position.tile_y],  # noqa: E731
                                conf=round(float(d.position.conf), 3),
                                bbox=list(d.position.bbox)) for d in ds]
        n = state.numbers
        row = dict(
            frame=stem, t=round(time.monotonic() - self.t0, 2),
            screen=getattr(state.screen, "name", str(state.screen)),
            allies=dets(state.allies), enemies=dets(state.enemies),
            cards=[c.name for c in state.cards], ready=list(state.ready),
            elixir=_num(n.elixir),
            towers={k: _num(getattr(n, k)) for k in (
                "left_ally_princess_hp", "right_ally_princess_hp",
                "left_enemy_princess_hp", "right_enemy_princess_hp")},
            view=dict(hand=view.hand, next=view.next_card, elixir=view.elixir,
                      towers=view.towers, clock=round(view.clock, 3), units=len(view.units)),
            decision=decision,
        )
        self.log.write(json.dumps(row) + "\n")
        self.log.flush()

    def _annotated(self, image, state, view, decision):
        img = image.convert("RGB").resize((SHOT_W * SCALE, SHOT_H * SCALE))
        d = ImageDraw.Draw(img, "RGBA")
        S = SCALE
        for tx in range(18):
            for ty in range(32):
                x0, y0, x1, y1 = _tile_box(tx, ty)
                d.rectangle((x0 * S, y0 * S, x1 * S, y1 * S), outline=(255, 255, 255, 28))
        for dets, colour in ((state.allies, (40, 120, 255)), (state.enemies, (255, 50, 50))):
            for det in dets:
                b = det.position.bbox
                d.rectangle([v * S for v in b], outline=colour + (255,), width=2)
                x0, y0, x1, y1 = _tile_box(det.position.tile_x, det.position.tile_y)
                d.rectangle((x0 * S, y0 * S, x1 * S, y1 * S), fill=colour + (90,))
                d.text((b[0] * S, b[1] * S - 11), f"{det.unit.name} "
                       f"({det.position.tile_x},{det.position.tile_y})", fill=colour + (255,))
        if decision.get("played"):
            tx, ty = decision["tile"]
            x0, y0, x1, y1 = _tile_box(tx, ty)
            d.rectangle((x0 * S, y0 * S, x1 * S, y1 * S), outline=(255, 230, 0, 255), width=4)
            d.text((x1 * S + 4, y0 * S), decision["card"], fill=(255, 230, 0, 255))
        hand = [c.name for c in state.cards[1:]]
        lines = [f"elixir {view.elixir:.0f}  hand {hand}  ready {list(state.ready)}",
                 f"towers {[round(t, 2) for t in view.towers]}",
                 f"P(wait) {decision.get('p_wait', float('nan')):.2f}  "
                 f"{'PLAY ' + decision['card'] + ' ' + str(decision['tile']) if decision.get('played') else 'wait'}"]
        d.rectangle((0, 0, SHOT_W * S, 14 * len(lines) + 6), fill=(0, 0, 0, 170))
        for i, line in enumerate(lines):
            d.text((4, 3 + 14 * i), line, fill=(255, 255, 255, 255))
        return img
