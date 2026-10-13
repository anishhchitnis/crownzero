"""Turn what Build-A-Bot reads off the screen into the student's View, and back.

Build-A-Bot (github.com/Pbatch/ClashRoyaleBuildABot, MIT) gives, per screenshot:
units (type + tile, ally or enemy), the 5 card slots (slot 0 = next card), which hand
cards are ready, the elixir count, and the four princess-tower health bars as fractions.
Its tile frame is the same as ours: x 0..17 left to right, y 0 = our back row.

This module only uses plain attributes of those objects, so it can be tested without the
game, the emulator or Build-A-Bot installed (see tests/test_real_adapter.py).
"""
from __future__ import annotations

import numpy as np

from ..board import CELLS, HAND, NX, NY, NOOP, action_index
from . import cardmap
from ..features import (BUILD, CARD_TYPES, COST, FLY, TARGETS_BUILDINGS, TYPE_INDEX, View,
                        unit_value, zone)

# Build-A-Bot card names -> our card types (the 18 calibrated cards)
CARD_TO_TYPE = {
    "knight": "Knight", "archers": "Archer", "musketeer": "Musketeer", "giant": "Giant",
    "hog_rider": "HogRider", "minions": "Minions", "baby_dragon": "BabyDragon",
    "valkyrie": "Valkyrie", "skeleton_army": "SkeletonArmy", "cannon": "Cannon",
    "tesla": "Tesla", "fireball": "Fireball", "zap": "Zap", "arrows": "Arrows",
    "the_log": "Log", "prince": "Prince", "wizard": "Wizard", "goblin_barrel": "GoblinBarrel",
}
TYPE_TO_CARD = {v: k for k, v in CARD_TO_TYPE.items()}
# Build-A-Bot unit names -> the card type that makes them
UNIT_TO_TYPE = {
    "knight": "Knight", "archer": "Archer", "musketeer": "Musketeer", "giant": "Giant",
    "hog_rider": "HogRider", "minion": "Minions", "baby_dragon": "BabyDragon",
    "valkyrie": "Valkyrie", "skeleton": "SkeletonArmy", "cannon": "Cannon", "tesla": "Tesla",
    "prince": "Prince", "wizard": "Wizard", "goblin": "GoblinBarrel",
}
DEFAULT_UNKNOWN_VALUE = 3.0


def _attr(obj, name, default=None):
    v = getattr(obj, name, default)
    return getattr(v, "value", v)  # enums -> their string value


class Clock:
    """The game clock, which the screen reader does not read: started on the first
    in-game frame, reset when the battle ends."""

    REGULATION_S = 180.0
    DOUBLE_AT_S = 120.0

    def __init__(self, now):
        self.start = None
        self.now = now

    def update(self, in_game: bool):
        if in_game and self.start is None:
            self.start = self.now()
        elif not in_game:
            self.start = None

    def features(self):
        t = 0.0 if self.start is None else self.now() - self.start
        return (min(t / self.REGULATION_S, 1.0), t >= self.DOUBLE_AT_S,
                t >= self.REGULATION_S)


class TowerTracker:
    """Smooths the princess-tower health bars, and decides when a tower is gone.

    The reader returns 0 both for a destroyed tower and for a bar it cannot find (an 84-HP
    tower's bar is too short to see), and taking a living tower for a destroyed one sends
    the search badly wrong. So a 0 alone is read as "almost destroyed" (4%), and a sudden
    drop to 0 from a healthy bar is ignored for a few frames as a misread. A tower is
    destroyed when its level badge is gone too (see towers.py), for GONE_FRAMES frames in
    a row; then it stays destroyed. A one-frame drop of more than DIP waits for a second
    frame to confirm it (recorded: 0.97, 0.51, 0.92).

    Also tracks the enemy King Tower from its bar: the median of its last readings, never
    rising (a tower cannot heal)."""

    ALMOST = 0.04
    HOLD = 4  # frames a healthy bar may vanish before we believe it
    GONE_FRAMES = 3
    DIP = 0.25

    def __init__(self):
        self.reset()

    def reset(self):
        self.last = [1.0] * 4
        self.zeros = [0] * 4
        self.gone = [0] * 4
        self.dead = [False] * 4
        self.dip = [False] * 4
        self.king = 1.0
        self.king_reads = []

    def update(self, readings, badges=None):
        """``readings``: [ally L, ally R, enemy L, enemy R] from the bar reader;
        ``badges``: whether each tower's level badge is on screen (None: unknown)."""
        out = []
        for i, r in enumerate(readings):
            if self.dead[i]:
                out.append(0.0)
                continue
            if r <= 0 and badges is not None and not badges[i]:
                self.gone[i] += 1
                if self.gone[i] >= self.GONE_FRAMES:
                    self.dead[i] = True
                    out.append(0.0)
                    continue
            else:
                self.gone[i] = 0
            if r > 0:
                if r < self.last[i] - self.DIP and not self.dip[i]:
                    self.dip[i] = True  # wait for a second frame
                    out.append(self.last[i])
                    continue
                self.dip[i] = False
                self.last[i], self.zeros[i] = r, 0
                out.append(r)
                continue
            self.zeros[i] += 1
            if self.last[i] >= 0.25 and self.zeros[i] <= self.HOLD:
                out.append(self.last[i])
            else:
                out.append(min(self.last[i], self.ALMOST))
        return out

    def update_king(self, reading):
        if reading is not None:
            self.king_reads = (self.king_reads + [reading])[-3:]
            if len(self.king_reads) >= 2:
                self.king = min(self.king, sorted(self.king_reads)[len(self.king_reads) // 2])
        return self.king


def unit_row(detection, own: bool, unknown_values: dict | None = None):
    unit = detection.unit
    pos = detection.position
    name = _attr(unit, "name")
    t_name = UNIT_TO_TYPE.get(name)
    t = TYPE_INDEX[t_name] if t_name else -1
    flags = 0
    if _attr(unit, "transport") == "air":
        flags |= FLY
    if _attr(unit, "category") == "building":
        flags |= BUILD
    if _attr(unit, "target") == "buildings":
        flags |= TARGETS_BUILDINGS
    if t_name:
        value = unit_value(t_name)
    else:
        value = (unknown_values or {}).get(name, cardmap.unit_value(name, DEFAULT_UNKNOWN_VALUE))
    x = int(np.clip(pos.tile_x, 0, NX - 1))
    y = int(np.clip(pos.tile_y, 0, NY - 1))
    return (int(own), t, x, y, flags, float(value))


def in_arena(detection) -> bool:
    """Is the detection on the battlefield? The unit detector also fires on arena
    decorations outside it: in PEKKA's Playhouse two lamps beside our right tower were
    read as an enemy Executioner and Monk on every frame, and the bot spent its whole
    elixir "defending" against them. The battlefield is tiles x 0-17, y 0-31."""
    p = detection.position
    return 0 <= p.tile_x < NX and 0 <= p.tile_y < NY


def view_from_state(state, clock: Clock, unknown_values: dict | None = None) -> View:
    """A View from a Build-A-Bot State."""
    v = View()
    for own, dets in ((True, state.allies), (False, state.enemies)):
        for d in dets:
            if not in_arena(d):
                continue
            v.units.append(unit_row(d, own, unknown_values))
            v.unit_names.append(cardmap.unit_sim(_attr(d.unit, "name")))
    cards = list(state.cards)
    v.next_card = TYPE_INDEX.get(CARD_TO_TYPE.get(_attr(cards[0], "name"), ""), -1)
    v.hand = [TYPE_INDEX.get(CARD_TO_TYPE.get(_attr(c, "name"), ""), -1) for c in cards[1:5]]
    v.next_name = cardmap.sim_name(_attr(cards[0], "name"))
    v.hand_names = [cardmap.sim_name(_attr(c, "name")) for c in cards[1:5]]
    n = state.numbers
    v.elixir = float(n.elixir.number)
    # Health bars are fractions already; a bar the reader cannot find reads 0 (destroyed).
    v.towers = [1.0, float(n.left_ally_princess_hp.number), float(n.right_ally_princess_hp.number),
                1.0, float(n.left_enemy_princess_hp.number), float(n.right_enemy_princess_hp.number)]
    v.clock, v.double, v.overtime = clock.features()
    return v


def legal_mask(view: View, ready: list[int]) -> np.ndarray:
    """The moves we may attempt: a ready, affordable hand card on a tile the rules allow."""
    mask = np.zeros(1 + HAND * CELLS, bool)
    mask[NOOP] = True
    eld, erd = view.towers[4] <= 0, view.towers[5] <= 0
    for s in ready:
        t = view.hand[s] if 0 <= s < HAND else -1
        if t < 0 or COST[CARD_TYPES[t]] > view.elixir:
            continue
        z = zone(t, eld, erd) > 0
        ys, xs = np.nonzero(z)
        mask[[action_index(s, int(y), int(x)) for y, x in zip(ys, xs)]] = True
    return mask


def playable_slots(view: View, ready: list[int]) -> list[int]:
    """Hand slots we may play now, for any card: lit on screen, known, and affordable."""
    out = []
    for s in ready:
        if not 0 <= s < HAND:
            continue
        name = view.hand_names[s]
        c = cardmap.SIM_COST.get(name) if name else None
        if c is not None and c <= view.elixir:
            out.append(s)
    return out


def decode(action: int) -> tuple[int, int, int]:
    """Action id -> (hand slot, tile x, tile y). Hand slot s is Build-A-Bot card index s."""
    a = action - 1
    slot, cell = divmod(a, CELLS)
    y, x = divmod(cell, NX)
    return slot, x, y
