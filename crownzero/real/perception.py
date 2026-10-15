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

               Troops are deployed on their owner's half: a unit first seen deep on one
               half and first read as that side's starts with a strong vote for it (an
               enemy Mini P.E.K.K.A first seen on its side was voted ours at the bridge two
               frames later, a tie, and went unanswered: 2026-10-13).
               Enemy troops on their way over (on the bridge or beyond) that the detector
               misses for a frame or two are kept where they were last seen: in a recorded
               battle an enemy Mini P.E.K.K.A at the bridge went undetected for one frame,
               the bot spent its last elixir on the other lane, and the Mini P.E.K.K.A then
               hit our tower unanswered (2026-10-13).

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
SPAWNS = {"goblin_hut": {"spear_goblin", "goblin"}, "tombstone": {"skeleton"},
          "barbarian_hut": {"barbarian"}, "furnace": {"fire_spirit"},
          "goblin_cage": {"brawler", "goblin_brawler"}, "goblin_drill": {"goblin"}}
# (a Hut's Spear Goblins are often read as plain goblins: recorded 2026-10-13)

# Our own cards that keep making units after they are played: the simulator card name ->
# (seconds their units keep coming, unit names they make). Recorded 2026-10-13: our
# Tombstone was never detected, so nothing told the reader its Skeletons were ours; they
# were read as enemies for the whole game, and 10 of our 30 plays answered them.
OWN_SPAWNERS = {"Tombstone": (40, {"skeleton"}), "GoblinHut": (40, {"spear_goblin", "goblin"}),
                "BarbarianHut": (45, {"barbarian"}), "FirespiritHut": (35, {"fire_spirit"}),
                "GoblinCage": (30, {"brawler", "goblin_brawler"}),
                "GoblinDrill": (15, {"goblin"}), "Graveyard": (12, {"skeleton"})}


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
    AGE_PENALTY = 1.0  # tiles added per frame a track went unseen: fresh tracks match first
    VOTES = 6
    SPAWN_NEAR = 2.5  # tiles from the building that spawned it
    SPAWN_PRIOR = 3   # outweighs any run of votes (at most 1 / (1 - DECAY) = 2.5)
    DECAY = 0.6       # each older frame's vote counts this much less
    PRIOR_FRAMES = 4  # starting votes only hold for a unit's first frames: a track can be
    # taken over by a different unit of the same card where one died (recorded: enemy Archers
    # arriving where ours fell were read as ours for 4 frames)
    BIRTH_PRIOR = 3   # first seen on its owner's half, read as that side
    OUR_HALF_Y, THEIR_HALF_Y = 14, 17  # the rows either side of the river (15-16)

    OWN_SPAWN_NEAR = 6.0  # tiles from where we played a spawner (the game may move a
    # building a few tiles when the spot does not fit: recorded 3.6 tiles)

    def __init__(self, now=_monotonic):
        self.now = now
        self.tracks: list[dict] = []
        self.frame = 0
        self.own_spawners: list[tuple] = []

    def reset(self):
        self.tracks, self.frame, self.own_spawners = [], 0, []

    def own_play(self, card: str, x: float, y: float) -> None:
        """We just played ``card`` (simulator name) at own-frame tile (x, y): if it keeps
        making units, those units are ours."""
        if card in OWN_SPAWNERS:
            secs, names = OWN_SPAWNERS[card]
            self.own_spawners.append((x, y, names, self.now() + secs))

    OWN_BIRTH_Y = 13  # a unit that first shows up this deep in our half came from our side
    OWN_BIRTH_NEAR = 9.0  # ... if our spawner is this close

    def _own_spawned(self, d, new: bool) -> bool:
        """A unit our live spawner makes, near where we played it, or (a new track) first
        seen deep in our half: real enemy troops arrive over the river. The game moves a
        building that does not fit, so where we played it is only roughly where it is
        (recorded: a Tombstone played behind the King Tower stood 3.6 tiles away)."""
        p = d.position
        for x, y, names, until in self.own_spawners:
            if d.unit.name not in names or until <= self._t:
                continue
            d2 = (x - p.tile_x) ** 2 + (y - p.tile_y) ** 2
            if d2 <= self.OWN_SPAWN_NEAR ** 2:
                return True
            if new and p.tile_y <= self.OWN_BIRTH_Y and d2 <= self.OWN_BIRTH_NEAR ** 2:
                return True
        return False

    def update(self, allies: list, enemies: list) -> tuple[list, list]:
        """Detections (Build-A-Bot UnitDetection) as the classifier split them -> the split
        after smoothing. A detection keeps its object; only the list it is in changes."""
        self.frame += 1
        self._t = self.now()
        self.own_spawners = [s for s in self.own_spawners if s[3] > self._t]
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
                # an enemy Mini P.E.K.K.A was matched to a two-frame-old track of our own
                # Mini P.E.K.K.A and inherited its side (recorded 2026-10-13)
                dist += self.AGE_PENALTY * max(0, self.frame - t["seen"] - 1)
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
                if py >= self.THEIR_HALF_Y and vote < 0:
                    prior = -self.BIRTH_PRIOR
                elif py <= self.OUR_HALF_Y and vote > 0:
                    prior = self.BIRTH_PRIOR
                for bx, by, bside, bname in spawners:
                    if (d.unit.name in SPAWNS.get(bname, ())
                            and (bx - px) ** 2 + (by - py) ** 2 <= self.SPAWN_NEAR ** 2):
                        prior = self.SPAWN_PRIOR * bside
                t = dict(name=d.unit.name, votes=deque(maxlen=self.VOTES), prior=prior,
                         born=self.frame)
                alive.append(t)
            t["x"], t["y"], t["seen"] = d.position.tile_x, d.position.tile_y, self.frame
            t["det"] = d
            t["votes"].append(vote)
            if self.own_spawners and self._own_spawned(d, new=t.get("born") == self.frame):
                t["owned"] = True  # stays ours as it walks away from the spawner
            prior = self._prior(t)
            side = building_side(d) * 100 or (100 if t.get("owned") else 0) \
                or self._votes(t) + prior
            tie = prior or vote  # a tie goes to the starting vote, else to this frame
            (out_allies if (side > 0 or (side == 0 and tie > 0)) else out_enemies).append(d)
        self.tracks = alive + [t for t in self.tracks
                               if t not in alive and self.frame - t["seen"] <= self.KEEP_FRAMES]
        return out_allies, out_enemies


    def _votes(self, t) -> float:
        return sum(v * self.DECAY ** k for k, v in enumerate(reversed(t["votes"])))

    def _prior(self, t) -> float:
        return t.get("prior", 0) if self.frame - t.get("born", self.frame) < self.PRIOR_FRAMES else 0

    GHOST_FRAMES = 2
    GHOST_MIN_Y = 12  # bridge, river and their half: on its way, not dying at our tower

    def ghosts(self) -> list:
        """Enemy troops missed this frame but seen in the last GHOST_FRAMES frames, still on
        their way over: their last detections."""
        out = []
        for t in self.tracks:
            age = self.frame - t["seen"]
            if not 0 < age <= self.GHOST_FRAMES or t["y"] < self.GHOST_MIN_Y:
                continue
            d = t.get("det")
            if d is None or getattr(d.unit, "category", None) == "building":
                continue
            if not t.get("owned") and self._votes(t) + self._prior(t) < 0:
                out.append(d)
        return out


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


class EnemyElixir:
    """Counts the opponent's elixir, as good players do: they start with 5, gain one every
    2.8 s (1.4 s in double elixir), never hold more than 10, and pay for every card they
    play. A play is new enemy units appearing on their half, together; it costs what their
    card costs. Spells and units the reader misses are not seen, so the count errs high:
    it assumes the opponent can afford more, not less.

    The reader names the same unit differently from frame to frame (a Valkyrie read as a
    Mini P.E.K.K.A) and loses it for a frame now and then, so a unit is followed by where
    it is, whatever it is called: something new is something seen where nothing was.

    Recorded 2026-10-13: the bot spent its elixir as soon as it had 3 and lost both towers
    to pushes it had nothing left to answer; nothing told it the opponent had saved up."""

    START = 5.0
    CAP = 10.0
    PERIOD = 2.8          # seconds per elixir
    DOUBLE_AFTER = 120.0  # seconds of battle
    THEIR_HALF_Y = 17     # own-frame rows from here up are their half
    MATCH = 2.5           # tiles a unit moves between frames
    LOST_S, LOST_TILES = 3.0, 3.5  # seen again this soon and this close: the same unit
    GROUP_S, GROUP_TILES = 1.5, 4.0  # units of one play appear this close together

    def __init__(self):
        self.reset()

    def reset(self):
        self.value = self.START
        self.last_t = None
        self.spent = 0.0
        self.tracks = []   # [x, y, last seen]
        self.groups = []   # [first t, x, y, cost]

    @staticmethod
    def _cost(name: str) -> float:
        from . import cardmap
        sim = cardmap.unit_sim(name)
        return float(cardmap.SIM_COST.get(sim, 0)) if sim else 0.0

    def update(self, battle_t: float, enemies) -> float:
        """``battle_t``: seconds since the battle started; ``enemies``: this frame's enemy
        detections (Build-A-Bot). Returns the opponent's estimated elixir."""
        if self.last_t is not None and battle_t > self.last_t:
            rate = 2.0 if battle_t >= self.DOUBLE_AFTER else 1.0
            self.value = min(self.CAP, self.value + rate * (battle_t - self.last_t) / self.PERIOD)
        self.last_t = battle_t
        live = [tr for tr in self.tracks if battle_t - tr[2] <= self.LOST_S]
        used = set()
        for d in enemies:
            x, y = d.position.tile_x, d.position.tile_y
            best, bd = None, None
            for k, tr in enumerate(live):
                if k in used:
                    continue
                dist = ((tr[0] - x) ** 2 + (tr[1] - y) ** 2) ** 0.5
                limit = self.MATCH if battle_t - tr[2] < 0.8 else self.LOST_TILES
                if dist <= limit and (bd is None or dist < bd):
                    best, bd = k, dist
            if best is not None:
                used.add(best)
                live[best][:] = [x, y, battle_t]
                continue
            live.append([x, y, battle_t])
            used.add(len(live) - 1)
            if y < self.THEIR_HALF_Y:
                continue  # first seen on our half: a unit that came over, not a new play
            cost = self._cost(d.unit.name)
            g = next((g for g in self.groups if battle_t - g[0] <= self.GROUP_S and
                      (g[1] - x) ** 2 + (g[2] - y) ** 2 <= self.GROUP_TILES ** 2), None)
            if g is None:
                self.groups.append([battle_t, x, y, cost])
                self.value = max(0.0, self.value - cost)
                self.spent += cost
            elif cost > g[3]:  # the play is the dearest card among its units
                self.value = max(0.0, self.value - (cost - g[3]))
                self.spent += cost - g[3]
                g[3] = cost
        self.tracks = live
        self.groups = [g for g in self.groups if battle_t - g[0] <= self.GROUP_S]
        return self.value
