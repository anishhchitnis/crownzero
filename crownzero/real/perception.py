"""Memory across frames, to correct what a single screenshot gets wrong.

Each of these fixes a mistake seen in a recorded real battle:

ElixirBook     After a play, the next screenshot can still show the elixir from before it
               (the emulator's video stream lags). The bot then tried a second card it could
               no longer afford ("Not enough Elixir!" on screen, recorded 2026-10-13). So
               elixir after a play is capped at what we had, minus the cost, plus regen.

SideTracker    Units of cards both decks share (Giant, Knight, Mini P.E.K.K.A...) flipped
               between ally and enemy from one frame to the next: 20 flips in one battle.
               Detections are matched to the same unit in the previous frames and take the
               side most of its recent frames gave it.

               Buildings can only be placed on their owner's half, so a building's side is
               its half (our own Goblin Hut was read as an enemy one, 2026-10-13); and a
               unit first seen beside a building that spawns it (a Hut's Spear Goblins)
               starts out on that building's side.

HiddenAttacker A troop standing at the foot of our tower is drawn behind it and the detector
               missed it: an enemy Mini P.E.K.K.A took our left tower from 100% to 21% in
               four seconds with no enemy detected near it. When a tower is losing health
               and nothing is seen near it, an attacker is assumed in front of it.
"""
from __future__ import annotations

import time
from collections import deque

ELIXIR_PER_S = 1 / 2.8  # regulation time; x2 in double elixir, x3 in overtime


def _monotonic():
    return time.monotonic()  # looked up at call time, so a replay can drive the clock


class ElixirBook:
    WINDOW_S = 3.0

    def __init__(self, now=_monotonic):
        self.now = now
        self.plays: list[tuple] = []  # (time, elixir before, cost, slot, card)

    FAILED_AFTER_S = 1.2

    def played(self, elixir_before: float, cost: int, slot: int | None = None,
               card: str | None = None):
        self.plays.append((self.now(), float(elixir_before), int(cost), slot, card))

    def effective(self, reading: float, double: bool = False, overtime: bool = False,
                  hand: list | None = None) -> float:
        """``hand``: the card names in hand now. A card still in the slot we played it from
        well after the play means the play never happened (a missed tap); forget it."""
        t = self.now()
        keep = []
        for p in self.plays:
            t0, _, _, slot, card = p
            if t - t0 >= self.WINDOW_S:
                continue
            failed = (hand is not None and slot is not None and card is not None
                      and t - t0 >= self.FAILED_AFTER_S and 0 <= slot < len(hand)
                      and hand[slot] == card)
            if not failed:
                keep.append(p)
        self.plays = keep
        rate = ELIXIR_PER_S * (3 if overtime else 2 if double else 1)
        e = float(reading)
        for t0, before, cost, _, _ in self.plays:
            e = min(e, max(0.0, before - cost + rate * (t - t0)))
        return max(0.0, min(10.0, e))


ANYWHERE_BUILDINGS = {"goblin_drill"}  # dug in anywhere on the arena
SPAWNS = {"goblin_hut": {"spear_goblin"}, "tombstone": {"skeleton"},
          "barbarian_hut": {"barbarian"}, "furnace": {"fire_spirit"},
          "goblin_cage": {"brawler", "goblin_brawler"}, "goblin_drill": {"goblin"}}


def building_side(d) -> int:
    """+1 / -1 for a building on our / the enemy's half, 0 for anything else."""
    unit = d.unit
    if getattr(unit, "category", None) != "building" or unit.name in ANYWHERE_BUILDINGS:
        return 0
    y = d.position.tile_y
    return 1 if y <= 14 else -1 if y >= 17 else 0


class SideTracker:
    """Majority vote on each unit's side over its last few frames."""

    MATCH_TILES = 2.5
    KEEP_FRAMES = 3
    VOTES = 6
    SPAWN_NEAR = 2.5  # tiles from the building that spawned it
    SPAWN_PRIOR = 3   # worth this many frames of votes

    def __init__(self):
        self.tracks: list[dict] = []
        self.frame = 0

    def reset(self):
        self.tracks, self.frame = [], 0

    def update(self, allies: list, enemies: list) -> tuple[list, list]:
        """Detections (Build-A-Bot UnitDetection) as the classifier split them -> the split
        after smoothing. A detection keeps its object; only the list it is in changes."""
        self.frame += 1
        dets = [(d, +1) for d in allies] + [(d, -1) for d in enemies]
        free = [t for t in self.tracks if self.frame - t["seen"] <= self.KEEP_FRAMES]
        out_allies, out_enemies = [], []
        # nearest-first greedy matching, one detection per track
        pairs = []
        for i, (d, _) in enumerate(dets):
            p = d.position
            for j, t in enumerate(free):
                if t["name"] != d.unit.name:
                    continue
                dist = ((t["x"] - p.tile_x) ** 2 + (t["y"] - p.tile_y) ** 2) ** 0.5
                if dist <= self.MATCH_TILES:
                    pairs.append((dist, i, j))
        pairs.sort()
        det_track, used = {}, set()
        for _, i, j in pairs:
            if i in det_track or j in used:
                continue
            det_track[i] = free[j]
            used.add(j)
        alive = [t for j, t in enumerate(free) if j in used]
        spawners = [(d.position.tile_x, d.position.tile_y, building_side(d), d.unit.name)
                    for d, _ in dets if building_side(d)]
        for i, (d, vote) in enumerate(dets):
            t = det_track.get(i)
            if t is None:
                prior = 0
                px, py = d.position.tile_x, d.position.tile_y
                for bx, by, bside, bname in spawners:
                    if (d.unit.name in SPAWNS.get(bname, ())
                            and (bx - px) ** 2 + (by - py) ** 2 <= self.SPAWN_NEAR ** 2):
                        prior = self.SPAWN_PRIOR * bside
                t = dict(name=d.unit.name, votes=deque(maxlen=self.VOTES), prior=prior)
                alive.append(t)
            t["x"], t["y"], t["seen"] = d.position.tile_x, d.position.tile_y, self.frame
            t["votes"].append(vote)
            side = building_side(d) * 100 or sum(t["votes"]) + t.get("prior", 0)
            tie = t.get("prior", 0) or vote  # a tie goes to the spawn prior, else to this frame
            (out_allies if (side > 0 or (side == 0 and tie > 0)) else out_enemies).append(d)
        self.tracks = alive + [t for t in self.tracks
                               if t not in alive and self.frame - t["seen"] <= self.KEEP_FRAMES]
        return out_allies, out_enemies


class HiddenAttacker:
    """Our princess towers' health over time -> attackers to assume where none is seen."""

    TOWERS = {1: (3, 6), 2: (14, 6)}  # View.towers index -> own-frame tile of the tower
    WINDOW_S = 2.5
    DROP = 0.06       # tower share lost within the window that means "under attack"
    HEAVY = 0.18      # ... and that means a heavy hitter

    def __init__(self, now=_monotonic):
        self.now = now
        self.hist = {k: deque() for k in self.TOWERS}

    def reset(self):
        for h in self.hist.values():
            h.clear()

    def update(self, view) -> list[tuple[str, int, int]]:
        """Feed the View (after tower smoothing). Returns [(sim card, x, y)] to add."""
        t = self.now()
        out = []
        for k, (tx, ty) in self.TOWERS.items():
            h = self.hist[k]
            h.append((t, view.towers[k]))
            while h and t - h[0][0] > self.WINDOW_S:
                h.popleft()
            drop = max(v for _, v in h) - view.towers[k]
            if drop < self.DROP or view.towers[k] <= 0:
                continue
            seen = any(not own and abs(x - tx) <= 4 and abs(y - ty) <= 5
                       for own, _, x, y, _, _ in view.units)
            if not seen:
                out.append(("MiniPekka" if drop >= self.HEAVY else "Knight", tx, ty + 2))
        return out
