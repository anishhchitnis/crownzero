"""Turn one of the bot's own recorded battles into plays: who played which card, where,
when, and what the board looked like just before. (Replay videos of other players are read
by replay.py, which this hands them to.)

    python -m crownzero.replays.extract ~/Downloads/cr_replays/tv_royale_1_.../ [--fps 4]

Works on a replay video (``part_*.mp4`` from crownzero.real.capture) or on one of the bot's
own recordings (``steps.jsonl`` + frames, where the bot's own plays are known exactly, which
is how this is validated: scripts/validate_extract.py).

A play shows up on screen in two ways:

* For the player at the bottom, whose hand is drawn: a card leaves its hand slot (the next
  card takes its place). That gives the card exactly, spells included.
* For both players: new units appear on the board. Units that appear together (same side,
  within a second and a few tiles) are one play; the card is the one those units belong
  to. Units that are only spawned by something already there (a Goblin Hut's Spear
  Goblins, a Witch's skeletons, a Golem's Golemites) or that are the same unit seen again
  after the detector lost it for a moment are not plays.

Positions are tiles in the playing side's own frame (x 0-17, y 0 = its back row): the top
player's plays are turned 180 degrees, so every example reads "my side at the bottom".
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

SCREEN = (368, 652)  # what the screen reader expects

# unit -> the units that only appear because it is (or just was) there
SPAWNED_BY = {
    "goblin_hut": {"spear_goblin"}, "tombstone": {"skeleton"}, "barbarian_hut": {"barbarian"},
    "furnace": {"fire_spirit"}, "goblin_cage": {"brawler"}, "witch": {"skeleton"},
    "night_witch": {"bat"}, "golem": {"golemite"}, "lava_hound": {"lava_pup"},
    "goblin_giant": {"spear_goblin"}, "elixir_golem_large": {"elixir_golem_medium"},
    "elixir_golem_medium": {"elixir_golem_small"}, "phoenix_large": {"phoenix_egg"},
    "phoenix_egg": {"phoenix_small"}, "goblin_drill": {"goblin"}, "skeleton_king": {"skeleton"},
    "battle_ram": {"barbarian"}, "mother_witch": {"hog"},
}
# a unit seen alone that several cards make: the usual card, by how many came together
AMBIGUOUS = {
    "goblin": lambda n, others: "goblin_gang" if "spear_goblin" in others else "goblins",
    "spear_goblin": lambda n, others: "goblin_gang" if "goblin" in others else "spear_goblins",
    "skeleton": lambda n, others: "skeleton_army" if n >= 5 else "skeletons",
    "minion": lambda n, others: "minion_horde" if n >= 5 else "minions",
    "barbarian": lambda n, others: "barbarians",
    "bat": lambda n, others: "bats",
    "musketeer": lambda n, others: "three_musketeers" if n >= 2 else "musketeer",
    "royal_recruit": lambda n, others: "royal_recruits",
}


@dataclass
class Play:
    t: float
    side: str            # "bottom" or "top" of the screen
    card: str            # Build-A-Bot card name
    tile: list | None    # [x, y] in the player's own frame; None for a spell seen in hand only
    source: str          # "hand" (card left the hand) or "units" (units appeared)
    units: list = field(default_factory=list)  # unit names that appeared
    state: dict = field(default_factory=dict)  # the board just before, in the player's frame


def own_frame(side: str, x: float, y: float) -> tuple[float, float]:
    return (x, y) if side == "bottom" else (17 - x, 31 - y)


# ---------------------------------------------------------------- frame sources
def video_frames(paths, fps: float = 4.0):
    """(seconds, 368x652 image) from mp4 parts recorded back to back."""
    import av

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


def recording_frames(folder: Path):
    """(seconds, image, recorded step) from one of the bot's own recordings."""
    from PIL import Image

    for line in open(folder / "steps.jsonl"):
        r = json.loads(line)
        f = folder / f"{r['frame']}.jpg"
        if f.exists():
            yield r["t"], Image.open(f).convert("RGB"), r


# ---------------------------------------------------------------- reading
def all_cards():
    from clashroyalebuildabot.namespaces.cards import Cards
    return [getattr(Cards, c) for c in dir(Cards) if c.isupper() and c != "BLANK"]


def make_reader(deck_names=None):
    """Build-A-Bot's screen reader with its ally/enemy split left to its side model for every
    unit (it otherwise calls any unit not in our deck an enemy), and its hand reader told
    ``deck_names`` when they are known (without them, hand plays are not read)."""
    from clashroyalebuildabot.detectors.detector import Detector
    from clashroyalebuildabot.namespaces.cards import Cards

    from ..real.play import _use_current_card_art

    _use_current_card_art()
    deck = [getattr(Cards, n.upper()) for n in deck_names] if deck_names else all_cards()[:8]
    det = Detector(cards=deck)
    det.unit_detector.possible_ally_names = {u.name for c in all_cards() for u in c.units}
    det.reads_hand = bool(deck_names)
    return det


def infer_deck(images, every: int = 4) -> list[str]:
    """The 8 cards of the player at the bottom, from how often each card shows in their hand
    when the hand reader is told every card in the game."""
    from collections import Counter

    from clashroyalebuildabot.detectors.card_detector import CardDetector

    from ..real.play import _use_current_card_art

    _use_current_card_art()
    det = CardDetector(list(all_cards()))
    seen = Counter()
    for k, img in enumerate(images):
        if k % every:
            continue
        cards, _ = det._detect_cards(img)
        seen.update(c.name for c in cards if c.name != "blank")
    return [c for c, _ in seen.most_common(8)]


# ---------------------------------------------------------------- tracking
class BirthTracker:
    """Follows units frame to frame (per side and unit name) and reports the ones that
    appear: new, not seen just before nearby (the detector losing a unit for a moment), and
    not spawned by something of the same side."""

    MATCH = 2.5      # tiles a unit may move between frames and be the same unit
    KEEP_S = 1.6     # a unit unseen this long is gone
    REAPPEAR_S = 2.5  # ... but if one like it shows up within this long and ...
    REAPPEAR_TILES = 3.0  # ... this close to where it was lost, it is the same unit
    SPAWN_TILES = 3.5

    def __init__(self):
        self.tracks = []   # dict(side, name, x, y, seen)
        self.lost = []     # tracks gone recently: (side, name, x, y, when)

    def update(self, t, units):
        """``units``: [(side, name, x, y)] -> births [(side, name, x, y)]."""
        births = []
        used = set()
        pairs = []
        for i, (side, name, x, y) in enumerate(units):
            for j, tr in enumerate(self.tracks):
                if tr["side"] == side and tr["name"] == name:
                    d = ((tr["x"] - x) ** 2 + (tr["y"] - y) ** 2) ** 0.5
                    if d <= self.MATCH:
                        pairs.append((d, i, j))
        pairs.sort()
        matched_unit, matched_track = {}, set()
        for d, i, j in pairs:
            if i in matched_unit or j in matched_track:
                continue
            matched_unit[i] = j
            matched_track.add(j)
        alive_before = [tr for tr in self.tracks]
        for i, (side, name, x, y) in enumerate(units):
            if i in matched_unit:
                tr = self.tracks[matched_unit[i]]
                tr.update(x=x, y=y, seen=t)
                continue
            # the same unit seen again after the detector lost it, or read as the other
            # side for a moment: not a new unit
            again = any(n == name and t - when <= self.REAPPEAR_S
                        and ((lx - x) ** 2 + (ly - y) ** 2) ** 0.5 <= self.REAPPEAR_TILES
                        for s, n, lx, ly, when in self.lost) or any(
                tr["name"] == name and tr["side"] != side
                and ((tr["x"] - x) ** 2 + (tr["y"] - y) ** 2) ** 0.5 <= self.REAPPEAR_TILES
                for tr in alive_before)
            spawned = any(
                tr["side"] == side and name in SPAWNED_BY.get(tr["name"], ())
                and ((tr["x"] - x) ** 2 + (tr["y"] - y) ** 2) ** 0.5 <= self.SPAWN_TILES
                for tr in alive_before) or any(
                s == side and name in SPAWNED_BY.get(n, ()) and t - when <= self.REAPPEAR_S
                and ((lx - x) ** 2 + (ly - y) ** 2) ** 0.5 <= self.SPAWN_TILES
                for s, n, lx, ly, when in self.lost)
            self.tracks.append(dict(side=side, name=name, x=x, y=y, seen=t))
            used.add(id(self.tracks[-1]))
            if not again and not spawned:
                births.append((side, name, x, y))
        keep = []
        for tr in self.tracks:
            if t - tr["seen"] > self.KEEP_S:
                self.lost.append((tr["side"], tr["name"], tr["x"], tr["y"], tr["seen"]))
            else:
                keep.append(tr)
        self.tracks = keep
        self.lost = [lo for lo in self.lost if t - lo[4] <= self.REAPPEAR_S]
        return births


def card_for(names: list[str], unit_cards: dict) -> str | None:
    """The card a group of units that appeared together came from."""
    others = set(names)
    for n in names:
        if n in AMBIGUOUS:
            continue
        cands = unit_cards.get(n, [])
        if cands:
            return cands[0]
    n0 = names[0]
    if n0 in AMBIGUOUS:
        return AMBIGUOUS[n0](sum(1 for n in names if n == n0), others - {n0})
    return None


# ---------------------------------------------------------------- the extractor
class Extractor:
    GROUP_S = 1.2      # units of one play appear within this long ...
    GROUP_TILES = 3.5  # ... and this close together
    HAND_MATCH_S = (-1.0, 2.5)  # units of a hand play appear in this window around it

    def __init__(self, read_hand: bool = True):
        from ..real.perception import SideTracker

        self.read_hand = read_hand
        self.sides = SideTracker()
        self.births = BirthTracker()
        self.pending = []      # unit groups not yet closed: dict(t, side, names, xs, ys, state)
        self.plays: list[Play] = []
        self.hand = None       # last full hand read (5 names: next + 4 slots)
        self.prev_nexts = []
        self.blank_since, self.before_blank = {}, {}
        self.hand_plays = []   # (t, card) from the hand, waiting for their units
        self.last_state = {}
        self.unit_cards = {}
        self.anywhere = {c.name for c in all_cards() if c.target_anywhere} | {"miner", "royal_delivery"}
        for c in all_cards():
            for u in c.units:
                self.unit_cards.setdefault(u.name, []).append(c.name)
        # prefer single-unit cards over spells/buildings that also make that unit
        for k, v in self.unit_cards.items():
            v.sort(key=lambda c: (c != k and not c.startswith(k), len(c)))

    def _state(self, units, towers, t):
        return dict(t=t, units=[[s, n, x, y] for s, n, x, y in units], towers=towers)

    def feed(self, t, state):
        """``state``: Build-A-Bot's State for this frame."""
        allies, enemies = self.sides.update(list(state.allies), list(state.enemies))
        units = ([("bottom", d.unit.name, d.position.tile_x, d.position.tile_y) for d in allies]
                 + [("top", d.unit.name, d.position.tile_x, d.position.tile_y) for d in enemies])
        units = [u for u in units if 0 <= u[2] <= 17 and 0 <= u[3] <= 31]
        n = state.numbers
        towers = [float(getattr(n, k).number) for k in (
            "left_ally_princess_hp", "right_ally_princess_hp",
            "left_enemy_princess_hp", "right_enemy_princess_hp")]
        before = self.last_state or self._state(units, towers, t)

        # 1. hand: a card leaving its slot
        cards = [c.name for c in state.cards] if self.read_hand else []
        if cards and all(c != "blank" for c in cards):
            # A played card is always replaced by the card shown as "next" before it; a
            # slot that changes to anything else is a misread (a raised card, a flicker).
            if self.hand is not None:
                nexts = {self.hand[0]} | set(self.prev_nexts)
                for k in range(1, len(cards)):
                    old, new = self.hand[k], cards[k]
                    if new != old and new in nexts and old not in cards[1:]:
                        self.hand_plays.append(dict(t=self.blank_since.get(k, t), card=old,
                                                    state=self.before_blank.get(k, before),
                                                    done=False))
                self.prev_nexts = [self.hand[0]]
            self.hand = cards
            self.blank_since, self.before_blank = {}, {}
        elif cards and self.hand is not None:
            for k in range(1, len(cards)):  # a slot mid-animation: the play started here
                if cards[k] == "blank" and k not in self.blank_since:
                    self.blank_since[k], self.before_blank[k] = t, before

        # 2. units appearing, grouped into plays
        for side, name, x, y in self.births.update(t, units):
            g = next((g for g in self.pending if g["side"] == side and t - g["t"] <= self.GROUP_S
                      and min(((gx - x) ** 2 + (gy - y) ** 2) ** 0.5
                              for gx, gy in zip(g["xs"], g["ys"])) <= self.GROUP_TILES), None)
            if g is None:
                self.pending.append(dict(t=t, side=side, names=[name], xs=[x], ys=[y],
                                         state=before))
            else:
                g["names"].append(name)
                g["xs"].append(x)
                g["ys"].append(y)
        still = []
        for g in self.pending:
            if t - g["t"] > self.GROUP_S:
                self._close(g)
            else:
                still.append(g)
        self.pending = still
        self._flush_hand(t)
        self.last_state = self._state(units, towers, t)

    def _deployable(self, side, card, x, y, towers) -> bool:
        """Could ``side`` have placed ``card`` at (x, y) (screen tiles)? Its own half; the
        pocket in front of a destroyed enemy princess tower; anywhere for cards such as the
        Miner or the Goblin Barrel."""
        if card in self.anywhere:
            return True
        oy = y if side == "bottom" else 31 - y
        ox = x if side == "bottom" else 17 - x
        if oy <= 14.5:
            return True
        if towers:  # pocket: their princess tower on that side is down
            left, right = (towers[2], towers[3]) if side == "bottom" else (towers[1], towers[0])
            down = left <= 0 if ox < 9 else right <= 0
            return down and oy <= 25
        return False

    def _close(self, g):
        card = card_for(g["names"], self.unit_cards)
        if card is None:
            return
        x, y = float(np.mean(g["xs"])), float(np.mean(g["ys"]))
        if not self._deployable(g["side"], card, x, y, g["state"].get("towers")):
            return
        if g["side"] == "bottom":  # a hand play of the bottom player may name it exactly
            for h in self.hand_plays:
                lo, hi = self.HAND_MATCH_S
                if not h["done"] and lo <= g["t"] - h["t"] <= hi and \
                        any(u.name in g["names"] for c in all_cards() if c.name == h["card"]
                            for u in c.units):
                    h["done"] = True
                    card = h["card"]
                    self._add(Play(h["t"], "bottom", card, [round(x, 1), round(y, 1)], "hand",
                                   g["names"]), h["state"])
                    return
        if g["side"] == "bottom" and self.read_hand:
            return  # the hand names the bottom player's plays; these units are not one
        self._add(Play(g["t"], g["side"], card, [round(x, 1), round(y, 1)], "units",
                       g["names"]), g["state"])

    def _flush_hand(self, t):
        """Hand plays whose units never appeared (spells): kept with no tile."""
        keep = []
        for h in self.hand_plays:
            if h["done"]:
                continue
            if t - h["t"] > self.HAND_MATCH_S[1] + self.GROUP_S:
                self._add(Play(h["t"], "bottom", h["card"], None, "hand"), h["state"])
            else:
                keep.append(h)
        self.hand_plays = keep

    def _add(self, play: Play, state: dict):
        side = play.side
        if play.tile is not None:
            play.tile = [round(v, 1) for v in own_frame(side, *play.tile)]
        play.state = dict(
            t=state.get("t"),
            units=[[("me" if s == side else "them"), n, *own_frame(side, x, y)]
                   for s, n, x, y in state.get("units", [])],
            towers=(state.get("towers") if side == "bottom" else
                    [state["towers"][3], state["towers"][2], state["towers"][1],
                     state["towers"][0]]) if state.get("towers") else None)
        self.plays.append(play)

    def finish(self):
        for g in self.pending:
            self._close(g)
        self.pending = []
        self._flush_hand(float("inf"))
        self.plays.sort(key=lambda p: p.t)
        return self.plays


def extract(folder: Path, fps: float = 4.0, deck=None, read_hand: bool = True) -> list[Play]:
    folder = Path(folder)
    parts = sorted(folder.glob("part_*.mp4"))
    if parts:
        frames = lambda: ((t, img) for t, img in video_frames(parts, fps))  # noqa: E731
    else:
        frames = lambda: ((t, img) for t, img, _ in recording_frames(folder))  # noqa: E731
        if deck is None and (folder / "meta.json").exists():
            deck = json.loads((folder / "meta.json").read_text()).get("deck")
    if deck is None and read_hand:  # a replay: work out the bottom player's deck first
        deck = infer_deck(img for _, img in frames())
    frames = frames()
    reader = make_reader(deck)
    ex = Extractor(read_hand=reader.reads_hand and read_hand)
    for t, img in frames:
        state = reader.run(img)
        ex.feed(t, state)
    return ex.finish()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--fps", type=float, default=4.0)
    ap.add_argument("--out", default=None, help="plays.jsonl (default: inside the folder)")
    args = ap.parse_args()
    if list(Path(args.folder).expanduser().glob("part_*.mp4")):  # a replay video
        import sys

        from . import replay
        sys.argv = [sys.argv[0], args.folder, "--fps", str(args.fps)]
        return replay.main()
    plays = extract(Path(args.folder).expanduser(), args.fps)
    out = Path(args.out) if args.out else Path(args.folder).expanduser() / "plays.jsonl"
    with open(out, "w") as f:
        for p in plays:
            f.write(json.dumps(asdict(p)) + "\n")
    by = {}
    for p in plays:
        by.setdefault(p.side, []).append(p.card)
    print(f"{len(plays)} plays -> {out}")
    for side, cs in by.items():
        print(f"  {side}: {len(cs)} plays, cards {sorted(set(cs))}")


if __name__ == "__main__":
    main()
