"""Turn a replay video into training examples: every play of both players, with the board
just before it.

    python -m crownzero.replays.replay ~/Downloads/cr_replays/pro_1_...  [--fps 4]

writes ``plays.jsonl`` into the folder (one play per line) and ``cards/`` with a picture
of each card the library could not name. The steps (see layout.py and rebuild.py):

1. Read every frame once: both hands, both elixir bars, the towers' health, and the arena
   moved to where a battle screenshot has it.
2. Plays: each player's elixir bar dropping, the hand slot that changed, and which of
   their cards that was (the same card is recognised each time it is played).
3. Where: the spot on the arena that changed most just after, where that player may
   place that card.
4. The board before each play: all plays so far replayed in the simulator.

Every example is in the playing side's own frame (x 0-17, y 0 = its back row), so the top
player's plays read like the bottom player's.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .layout import DiffPlacer, PlayReader, ReplayLayout, slot_empty

# cards the simulator calls spells that are placed on one's own half like troops
OWN_HALF_SPELLS = {"the_log", "barbarian_barrel", "heal_spirit"}
ANYWHERE_TROOPS = {"miner", "goblin_drill"}


def own_frame(side: str, x: float, y: float) -> tuple[float, float]:
    return (x, y) if side == "bottom" else (17 - x, 31 - y)


def own_towers(side: str, towers):
    """Screen [bottom L, bottom R, top L, top R] -> the side's own [my L, my R, their L,
    their R] (the top player's left is the screen's right)."""
    bl, br, tl, tr = towers
    return [bl, br, tl, tr] if side == "bottom" else [tr, tl, br, bl]


def video_frames(paths, fps: float = 4.0):
    """(seconds, 368x652 image) from mp4 parts recorded back to back."""
    import av

    from .layout import SCREEN

    offset = 0.0
    for path in sorted(paths):
        with av.open(str(path)) as box:
            stream = box.streams.video[0]
            next_t, last = 0.0, 0.0
            for frame in box.decode(stream):
                t = float(frame.pts * stream.time_base) if frame.pts is not None else last
                last = t
                if t + 1e-6 < next_t:
                    continue
                next_t = t + 1.0 / fps
                yield offset + t, frame.to_image().convert("RGB").resize(SCREEN)
        offset += last + 1.0 / fps


def how_placed(name: str | None) -> tuple[bool, bool]:
    """(thrown, anywhere) for a card name (None: unknown)."""
    from ..real import cardmap

    if name is None:
        return False, False
    if name in OWN_HALF_SPELLS:
        return False, False
    if name in ANYWHERE_TROOPS:
        return False, True
    spell = cardmap.kind(name) == "SPELL"
    return spell, spell


class ReplayRead:
    """Everything read off the frames, for both players."""

    def __init__(self, fps: float = 4.0):
        from clashroyalebuildabot.detectors.number_detector import NumberDetector

        from ..real.adapter import TowerTracker
        from ..real.towers import BadgeWatch

        self.fps = fps
        self.readers = {"bottom": PlayReader(fps), "top": PlayReader(fps)}
        self.placer = DiffPlacer()
        self.numbers = NumberDetector()
        self.badges = BadgeWatch()
        self.tracker = TowerTracker()
        self.times, self.towers = [], []
        self.crops = {"bottom": [], "top": []}
        self.t0 = None

    def add(self, t: float, img) -> bool:
        if not ReplayLayout.is_replay(img):
            return False
        if self.t0 is None:
            self.t0 = t
        warped = ReplayLayout.warp(img)
        for side, r in self.readers.items():
            crops = ReplayLayout.hand_crops(img, side)[1:]
            r.add(t, ReplayLayout.hand_descs(img, side)[1:], [slot_empty(c) for c in crops],
                  ReplayLayout.elixir(img, side))
            self.crops[side].append([c.copy() for c in crops])
        self.placer.add(t, warped)
        n = self.numbers.run(warped)
        readings = [float(getattr(n, k).number) for k in (
            "left_ally_princess_hp", "right_ally_princess_hp",
            "left_enemy_princess_hp", "right_enemy_princess_hp")]
        self.towers.append(self.tracker.update(readings, self.badges.update(warped, readings)))
        self.times.append(t)
        return True

    def at(self, t: float) -> int:
        """Index of the last frame at or before t."""
        import bisect
        return max(bisect.bisect_right(self.times, t + 1e-6) - 1, 0)


def extract_replay(folder: Path, fps: float = 4.0, log=print) -> list[dict]:
    from ..real import cardmap
    from .prior import board_features
    from .rebuild import Rebuild

    folder = Path(folder)
    parts = sorted(folder.glob("part_*.mp4"))
    if not parts:
        raise SystemExit(f"no part_*.mp4 in {folder}")
    t_start = time.time()
    rr = ReplayRead(fps)
    for t, img in video_frames(parts, fps):
        rr.add(t, img)
    if not rr.times:
        raise SystemExit(f"{folder.name}: no replay frames (is it a replay, recorded "
                         "with crownzero.real.capture?)")
    log(f"{folder.name}: read {len(rr.times)} frames in {time.time() - t_start:.0f} s")

    plays = []
    for side, r in rr.readers.items():
        for p in r.plays():
            plays.append((side, p))
    plays.sort(key=lambda sp: sp[1].t)

    # pictures of cards the library could not name, to name by hand later
    out_cards = folder / "cards"
    out_cards.mkdir(exist_ok=True)
    for old in out_cards.glob("*.png"):
        old.unlink()
    saved = set()
    for side, p in plays:
        if (side, p.card) in saved:
            continue
        saved.add((side, p.card))
        i = p.frame if p.frame >= 0 else rr.at(p.t - 0.75)
        crop = rr.crops[side][i][p.slot]
        label = p.name or "unknown"
        crop.resize((90, 110)).save(out_cards / f"{side}_{p.card}_{label}_{p.cost}.png")

    rb = Rebuild()
    records = []
    for side, p in plays:
        thrown, anywhere = how_placed(p.name)
        i = rr.at(p.t - 0.25)
        towers = rr.towers[i]
        mine = own_towers(side, towers)
        res = rr.placer.place(p.t, side, thrown=thrown, anywhere=anywhere, towers=mine)
        if p.name is None:  # unknown card: a spell if it clearly landed on their side
            spell = rr.placer.place(p.t, side, thrown=True, anywhere=True)
            if spell is not None:
                (sx, sy), peak = spell
                far = sy > 16.5 if side == "bottom" else sy < 14.5
                if far and (res is None or peak > 1.5 * res[1]):
                    res, thrown = spell, True
        sim = cardmap.sim_name(p.name) if p.name else None
        kind = (cardmap.kind(p.name) if p.name else ("SPELL" if thrown else None))
        team = 0 if side == "bottom" else 1
        rb.advance(p.t - rr.t0)
        board = board_features(rb.state(), team, rb.cards, rb.arena)
        tile = None
        if res is not None:
            (x, y), _ = res
            tile = [round(v, 2) for v in own_frame(side, x, y)]
            cid = rb.card_id(sim, p.cost, thrown)
            if cid is not None:
                rb.play(team, cid, x, y, towers)
        records.append(dict(
            replay=folder.name, t=round(p.t - rr.t0, 2), side=side, card=p.name, sim=sim,
            group=f"{side}{p.card}", cost=p.cost, kind=kind, thrown=thrown, tile=tile,
            elixir=rr.readers[side].elixir[min(rr.at(p.t - 0.25), len(rr.times) - 1)],
            towers=[round(float(v), 3) for v in mine], board=board))
    log(f"{folder.name}: {len(records)} plays ({sum(r['tile'] is not None for r in records)} "
        f"placed; simulator took {rb.played}, refused {rb.refused}) in "
        f"{time.time() - t_start:.0f} s")
    return records


def main():
    import sys

    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--fps", type=float, default=4.0)
    args = ap.parse_args()
    for f in args.folders:
        folder = Path(f).expanduser()
        recs = extract_replay(folder, args.fps)
        with open(folder / "plays.jsonl", "w") as out:
            for r in recs:
                out.write(json.dumps(r, default=float) + "\n")
        by = {}
        for r in recs:
            by.setdefault(r["side"], []).append(r["card"] or f"?{r['cost']}")
        for side, cs in by.items():
            print(f"  {side}: {len(cs)} plays: " + ", ".join(
                f"{c} x{cs.count(c)}" for c in sorted(set(cs), key=str)))
        sys.stdout.flush()


if __name__ == "__main__":
    main()
