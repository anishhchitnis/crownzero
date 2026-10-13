"""What the learned bot sees, built the SAME way from the simulator and from the real game.

The real game is read off the screen by an object detector, which reports each unit's
type and tile but not its health, and the elixir bar, the hand and the princess-tower
health numbers. So the student network is given exactly that and nothing more:

  * every unit: side, type (one of the 18 calibrated cards, or "other"), tile, and the
    attributes the detector also knows for unknown cards (air / ground, building,
    targets buildings) plus its elixir value at full health;
  * own hand (4 slots), next card, own elixir;
  * tower health fractions (princess towers; kings only as alive / dead);
  * the clock: elapsed share of regulation, double elixir, overtime.

No unit health, no opponent elixir, no opponent hand. A ``View`` is that information;
``from_sim`` builds one from a simulator state and ``crownzero.real`` builds one from a
screenshot. ``featurize`` turns a batch of views into network tensors.

Frame: the acting player's own frame, in tiles: x 0..17 left to right, y 0..31 from the
player's own back row (y 0) to the enemy's (y 31). This is also Build-A-Bot's tile frame.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .board import HAND, NX, NY, TILE, is_tower

CARD_TYPES = [
    "Knight", "Archer", "Musketeer", "Giant", "HogRider", "Minions", "BabyDragon",
    "Valkyrie", "SkeletonArmy", "Cannon", "Tesla", "Fireball", "Zap", "Arrows", "Log",
    "Prince", "Wizard", "GoblinBarrel",
]
T = len(CARD_TYPES)
TYPE_INDEX = {n: i for i, n in enumerate(CARD_TYPES)}
SPELLS = {"Fireball", "Zap", "Arrows", "Log", "GoblinBarrel"}
BUILDINGS = {"Cannon", "Tesla"}
BUILDING_TARGETERS = {"Giant", "HogRider"}
COST = {"Knight": 3, "Archer": 3, "Musketeer": 4, "Giant": 5, "HogRider": 4, "Minions": 3,
        "BabyDragon": 4, "Valkyrie": 4, "SkeletonArmy": 3, "Cannon": 3, "Tesla": 4,
        "Fireball": 4, "Zap": 2, "Arrows": 3, "Log": 2, "Prince": 5, "Wizard": 5,
        "GoblinBarrel": 3}
UNITS_PER_CARD = {"Archer": 2, "Minions": 3, "SkeletonArmy": 15, "GoblinBarrel": 3}


def unit_value(name: str) -> float:
    return COST[name] / UNITS_PER_CARD.get(name, 1)


# ---------------------------------------------------------------- the view
FLY, BUILD, TARGETS_BUILDINGS = 1, 2, 4


@dataclass
class View:
    # units: rows of (own 0/1, type index or -1, x tile, y tile, flags, elixir value)
    units: list[tuple[int, int, int, int, int, float]] = field(default_factory=list)
    hand: list[int] = field(default_factory=lambda: [-1] * HAND)  # type index or -1
    next_card: int = -1
    elixir: float = 0.0
    # towers in OWN frame: [own king, own left, own right, enemy king, enemy left, enemy right]
    towers: list[float] = field(default_factory=lambda: [1.0] * 6)
    clock: float = 0.0  # elapsed share of regulation time, 0..1
    double: bool = False
    overtime: bool = False
    # Simulator card names, for ANY card (the type indices above only cover the 18 the
    # network knows). Used by the live search; None where unknown.
    hand_names: list = field(default_factory=lambda: [None] * HAND)
    next_name: str | None = None
    unit_names: list = field(default_factory=list)  # parallel to ``units``

    def raw(self) -> np.ndarray:
        """The non-unit part as one float row (see RAW_* offsets)."""
        return np.array([*self.hand, self.next_card, self.elixir, *self.towers,
                         self.clock, float(self.double), float(self.overtime)], np.float32)


RAW = 4 + 1 + 1 + 6 + 3  # 15


def from_sim(state, team: int, cards: dict, arena) -> View:
    """A View of a simulator BattleState for ``team``. ``cards``: card_id -> CardInfo."""
    from royalegym import EntityKind, to_own

    v = View()
    towers = [0.0] * 6
    for ent in state.entities:
        ox, oy = to_own(arena, team, ent.x, ent.y)
        x, y = ox / TILE, oy / TILE
        own = int(ent.team == team)
        if is_tower(ent):
            frac = ent.hp / max(ent.max_hp, 1)
            base = 0 if own else 3
            if ent.kind == EntityKind.KING_TOWER:
                towers[base] = frac
            else:
                towers[base + (1 if x < NX / 2 else 2)] = frac
            continue
        card = cards.get(ent.card_id)
        name = card.name if card else None
        flags = (FLY if ent.flying else 0) | (BUILD if ent.kind == EntityKind.BUILDING else 0)
        if name in BUILDING_TARGETERS:
            flags |= TARGETS_BUILDINGS
        t = TYPE_INDEX.get(name, -1)
        val = unit_value(name) if t >= 0 else ent.max_hp / 600
        v.units.append((own, t, min(int(x), NX - 1), min(int(y), NY - 1), flags, val))
        v.unit_names.append(name)
    p = state.players[team]
    v.hand = [TYPE_INDEX.get(cards[c].name, -1) if c in cards else -1 for c in p.hand]
    v.next_card = TYPE_INDEX.get(cards[p.next_card].name, -1) if p.next_card in cards else -1
    v.hand_names = [cards[c].name if c in cards else None for c in p.hand]
    v.next_name = cards[p.next_card].name if p.next_card in cards else None
    v.elixir = p.elixir_milli / 1000
    v.towers = towers
    v.clock = min(state.tick / max(state.regular_ticks, 1), 1.0)
    v.double = state.elixir_rate >= 2
    v.overtime = bool(state.overtime)
    return v


# ---------------------------------------------------------------- placement zones
def _troop_zone(enemy_left_down: bool, enemy_right_down: bool) -> np.ndarray:
    z = np.zeros((NY, NX), np.float32)
    z[1:15, :] = 1
    z[0, 6:12] = 1
    if enemy_left_down:
        z[15:17, 3] = 1
        z[17:21, 0:9] = 1
    if enemy_right_down:
        z[15:17, 14] = 1
        z[17:21, 9:18] = 1
    return z


_ZONES = {(a, b): _troop_zone(a, b) for a in (False, True) for b in (False, True)}
_ALL = np.ones((NY, NX), np.float32)
_BUILD_ZONE = _troop_zone(False, False)


def zone(card_type: int, enemy_left_down: bool, enemy_right_down: bool) -> np.ndarray:
    """Where a card may be played, by the game's rules (approximate: ignores footprints)."""
    if card_type < 0:
        return np.zeros((NY, NX), np.float32)
    name = CARD_TYPES[card_type]
    if name in SPELLS:
        return _ALL
    if name in BUILDINGS:
        return _BUILD_ZONE
    return _ZONES[(enemy_left_down, enemy_right_down)]


# ---------------------------------------------------------------- tensors
# spatial channels: per side (own, enemy): ground, air, building, value, targets-buildings,
# then one count channel per card type; then 2 coordinate planes; then 4 hand-slot zones.
PER_SIDE = 5 + T
C_COORD = 2 * PER_SIDE
C_ZONE = C_COORD + 2
C = C_ZONE + HAND  # 52
V = 1 + HAND * (T + 1) + HAND + HAND + (T + 1) + 6 + 3  # 113

_YY, _XX = np.meshgrid(np.arange(NY) / (NY - 1), np.arange(NX) / (NX - 1), indexing="ij")


def featurize(units: np.ndarray, raws: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Batch tensors from stored rows.

    ``units``: float array [M, 7] of (sample, own, type, x, y, flags, value).
    ``raws``: float array [B, RAW] of View.raw() rows.
    Returns spatial [B, C, NY, NX] and vector [B, V], both float32.
    """
    B = len(raws)
    sp = np.zeros((B, C, NY, NX), np.float32)
    if len(units):
        u = units
        b = u[:, 0].astype(np.int64)
        side = np.where(u[:, 1] > 0.5, 0, PER_SIDE).astype(np.int64)
        t = u[:, 2].astype(np.int64)
        x = np.clip(u[:, 3].astype(np.int64), 0, NX - 1)
        y = np.clip(u[:, 4].astype(np.int64), 0, NY - 1)
        flags = u[:, 5].astype(np.int64)
        val = u[:, 6]
        fly = (flags & FLY) > 0
        bld = (flags & BUILD) > 0
        tb = (flags & TARGETS_BUILDINGS) > 0
        ground = ~fly & ~bld
        np.add.at(sp, (b[ground], side[ground] + 0, y[ground], x[ground]), 1.0)
        np.add.at(sp, (b[fly], side[fly] + 1, y[fly], x[fly]), 1.0)
        np.add.at(sp, (b[bld], side[bld] + 2, y[bld], x[bld]), 1.0)
        np.add.at(sp, (b, side + 3, y, x), val / 5.0)
        np.add.at(sp, (b[tb], side[tb] + 4, y[tb], x[tb]), 1.0)
        known = t >= 0
        np.add.at(sp, (b[known], side[known] + 5 + t[known], y[known], x[known]), 1.0)
    sp[:, C_COORD] = _YY
    sp[:, C_COORD + 1] = _XX

    vec = np.zeros((B, V), np.float32)
    hand = raws[:, 0:4].astype(np.int64)
    nxt = raws[:, 4].astype(np.int64)
    elixir = np.floor(raws[:, 5] + 1e-6)  # the screen shows whole elixir only
    towers = raws[:, 6:12].copy()
    towers[:, [0, 3]] = towers[:, [0, 3]] > 0  # king health is not read off the screen
    clock = raws[:, 12:15]
    vec[:, 0] = elixir / 10.0
    o = 1
    for s in range(HAND):
        idx = np.where(hand[:, s] >= 0, hand[:, s], T)
        vec[np.arange(B), o + s * (T + 1) + idx] = 1.0
    o += HAND * (T + 1)
    costs = np.array([COST[n] for n in CARD_TYPES] + [0], np.float32)
    hcost = costs[np.where(hand >= 0, hand, T)]
    vec[:, o:o + HAND] = (hcost <= elixir[:, None]) & (hand >= 0)
    o += HAND
    vec[:, o:o + HAND] = hcost / 10.0
    o += HAND
    vec[np.arange(B), o + np.where(nxt >= 0, nxt, T)] = 1.0
    o += T + 1
    vec[:, o:o + 6] = towers
    o += 6
    vec[:, o:o + 3] = clock

    for i in range(B):
        eld, erd = towers[i, 4] <= 0, towers[i, 5] <= 0
        for s in range(HAND):
            if vec[i, 1 + HAND * (T + 1) + s] > 0:  # affordable
                sp[i, C_ZONE + s] = zone(int(hand[i, s]), eld, erd)
    return sp, vec


def view_rows(view: View, sample: int) -> np.ndarray:
    """A View's units as featurize() rows for batch position ``sample``."""
    if not view.units:
        return np.zeros((0, 7), np.float32)
    rows = np.array(view.units, np.float32)
    return np.concatenate([np.full((len(rows), 1), sample, np.float32), rows], axis=1)


def featurize_views(views: list[View]) -> tuple[np.ndarray, np.ndarray]:
    units = [view_rows(v, i) for i, v in enumerate(views)]
    units = np.concatenate(units) if units else np.zeros((0, 7), np.float32)
    return featurize(units, np.stack([v.raw() for v in views]))
