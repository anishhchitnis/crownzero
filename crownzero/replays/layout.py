"""Reading a replay: where the arena and both hands are drawn, which cards each player
holds, and where a played card landed.

A replay (Battle Log or TV Royale) draws both players' hands and elixir, so it shows every
play of both players exactly, spells included. Its arena is drawn a little smaller and
lower than in a battle, to make room for the top player's hand; ``ReplayLayout.warp``
stretches it back to where a battle draws it, so Build-A-Bot's unit reader and tile grid
work on it unchanged.

Where did the card land? Build-A-Bot's unit reader is trained on ladder-level battles and
misses or misreads much of what pros play (evolutions, tower skins, other arenas), and it
never sees spells. So ``DiffPlacer`` finds the spot from the picture itself: the place
that changed most between just before the card left the hand and just after (a troop
dropping in, a spell's splash), minus what was already moving there before.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

SCREEN = (368, 652)  # the screenshot size Build-A-Bot reads

# Build-A-Bot's tile grid on the 720x1280 display
TILE_INIT_X, TILE_INIT_Y, TILE_W, TILE_H = 52, 296, 34, 27.6
TO_SCREEN = SCREEN[0] / 720  # 368/720 and 652/1280 are the same to 0.2%

# cards that fly to their spot (spells, the Goblin Barrel ...): they land about a second
# after leaving the hand, anywhere on the arena
THROWN = {
    "arrows", "fireball", "zap", "the_log", "rocket", "poison", "lightning", "freeze",
    "tornado", "earthquake", "rage", "giant_snowball", "barbarian_barrel", "goblin_barrel",
    "graveyard", "clone", "mirror", "royal_delivery", "goblin_curse", "void", "heal",
    "skeleton_barrel_spell", "vines",
}
# cards that may be placed anywhere on the arena (not only on their own half)
ANYWHERE = THROWN | {"miner", "goblin_drill", "mighty_miner_ability"}


def tile_to_px(x: float, y: float) -> tuple[float, float]:
    """Centre of tile (x, y) (bottom player's frame) on a 368x652 battle screenshot."""
    return ((TILE_INIT_X + (x + 0.5) * TILE_W) * TO_SCREEN,
            (1280 - TILE_INIT_Y - (y + 0.5) * TILE_H) * SCREEN[1] / 1280)


def px_to_tile(px: float, py: float) -> tuple[float, float]:
    """Inverse of tile_to_px, as fractional tiles (not rounded)."""
    return (px / TO_SCREEN - TILE_INIT_X) / TILE_W - 0.5, \
        (1280 - TILE_INIT_Y - py * 1280 / SCREEN[1]) / TILE_H - 0.5


class ReplayLayout:
    """Positions on a 368x652 replay frame."""

    # replay -> battle coordinates: battle = S * replay + B. Fitted on the princess towers'
    # level badges, at x 82 and 262, y 170 and 464.5 in a replay and at Build-A-Bot's
    # bar positions (74 and 267, 100 and 408.5) in a battle.
    SX = 193 / 180
    BX = 74 - 82 * SX
    SY = 308.5 / 294.5
    BY = 408.5 - 464.5 * SY

    # next card, then the four cards in hand (left, top, right, bottom)
    HANDS = {
        "bottom": [(11, 571, 34, 600), (50, 566, 95, 621), (97, 566, 142, 621),
                   (146, 566, 191, 621), (195, 566, 240, 621)],
        "top": [(11, 47, 34, 76), (50, 30, 95, 85), (97, 30, 142, 85),
                (146, 30, 191, 85), (195, 30, 240, 85)],
    }
    # elixir bars: the row, and the columns from empty to 10 elixir
    ELIXIR_ROW = {"bottom": 636, "top": 15}
    ELIXIR_X = (35, 353)
    # player name plates: blue for the bottom player, red for the top one
    NAME_BOX = {"bottom": (250, 575, 350, 597), "top": (255, 60, 330, 76)}

    @classmethod
    def is_replay(cls, img) -> bool:
        a = np.asarray(img.convert("RGB"), dtype=np.float32)
        x0, y0, x1, y1 = cls.NAME_BOX["bottom"]
        r, _, b = a[y0:y1, x0:x1].reshape(-1, 3).mean(axis=0)
        bottom_blue = b > r + 40
        x0, y0, x1, y1 = cls.NAME_BOX["top"]
        r, _, b = a[y0:y1, x0:x1].reshape(-1, 3).mean(axis=0)
        top_red = r > b + 40
        return bool(bottom_blue and top_red)

    @classmethod
    def warp(cls, img):
        """The replay frame with its arena moved to where a battle screenshot has it."""
        img = img.convert("RGB")
        if img.size != SCREEN:
            img = img.resize(SCREEN)
        return img.transform(SCREEN, Image.Transform.AFFINE,
                             (1 / cls.SX, 0, -cls.BX / cls.SX, 0, 1 / cls.SY, -cls.BY / cls.SY),
                             resample=Image.Resampling.BILINEAR)

    @classmethod
    def hand_crops(cls, img, side: str, dx: int = 0, dy: int = 0):
        return [img.crop((x0 + dx, y0 + dy, x1 + dx, y1 + dy))
                for x0, y0, x1, y1 in cls.HANDS[side]]

    # a card the player holds (selected, waiting for elixir or a spot) is drawn about
    # 10 pixels towards the arena with a glow: up for the bottom player, down for the top
    # one. Each slot is also read at these offsets, and the best is kept.
    SHIFTS = {
        "bottom": [(0, 0)] + [(dx, dy) for dy in (-12, -9, -6, -3, 0, 3) for dx in (-4, 0, 4)
                              if (dx, dy) != (0, 0)],
        "top": [(0, 0)] + [(dx, dy) for dy in (-3, 0, 3, 6, 9, 12) for dx in (-4, 0, 4)
                           if (dx, dy) != (0, 0)],
    }

    @classmethod
    def hand_descs(cls, img, side: str) -> np.ndarray:
        """(5 slots, shifts, 256) card descriptors; shift 0 is the slot as drawn."""
        return np.stack([np.stack([card_desc(c) for c in cls.hand_crops(img, side, dx, dy)])
                         for dx, dy in cls.SHIFTS[side]], axis=1)

    @classmethod
    def elixir(cls, img, side: str) -> float:
        """Elixir from the bar, 0-10. The replay bar fills a whole segment at a time, so
        this reads whole numbers, with in-between values while the bar animates. A held
        card of the top player can cover part of its bar, so the fill may have gaps: the
        fill ends at the last pink column with the bar mostly pink up to it."""
        a = np.asarray(img.convert("RGB"), dtype=np.int16)
        y = cls.ELIXIR_ROW[side]
        x0, x1 = cls.ELIXIR_X
        best = 0
        for row in a[y - 1:y + 2, x0:x1]:
            pink = (row[:, 0] > 150) & (row[:, 2] > 140) & (row[:, 0] - row[:, 1] > 60)
            if not pink.any():
                continue
            share = np.cumsum(pink) / np.arange(1, len(pink) + 1)
            ok = np.flatnonzero(pink & (share >= 0.6))
            if len(ok):
                best = max(best, int(ok[-1]) + 1)
        return round(10.0 * best / (x1 - x0), 1)


# ---------------------------------------------------------------- hands
CARD_INNER = (0.12, 0.08, 0.88, 0.70)  # the picture, without the frame and the cost drop
LIBRARY = Path(__file__).resolve().parent / "card_crops"  # replay crops we have named


def card_desc(img) -> np.ndarray:
    """A card picture as 256 numbers that stay the same when the game greys the card out
    (not enough elixir) or lights it up (held): the 16x16 grey picture inside the frame,
    each pixel relative to its neighbourhood. Two crops of the same card score 0.85-0.95
    (dot product / 256), different cards under 0.3."""
    from scipy.ndimage import uniform_filter

    w, h = img.size
    box = (int(CARD_INNER[0] * w), int(CARD_INNER[1] * h),
           int(CARD_INNER[2] * w), int(CARD_INNER[3] * h))
    g = np.asarray(img.convert("RGB").crop(box).resize((16, 16), Image.Resampling.BOX)
                   .convert("L"), dtype=np.float32)
    m = uniform_filter(g, 5)
    s = np.sqrt(uniform_filter((g - m) ** 2, 5)) + 4.0
    d = (g - m) / s
    d -= d.mean()
    return (d / (d.std() + 1e-6)).ravel()


def _card_images() -> dict[str, list[Path]]:
    """Every card picture we have, by card name: Build-A-Bot's (evolutions included, as the
    same card), and the in-game pictures in crownzero/real/card_art where its art is old."""
    from clashroyalebuildabot.constants import IMAGES_DIR

    from ..real.play import CARD_ART

    out: dict[str, list[Path]] = {}
    for p in sorted((Path(IMAGES_DIR) / "cards").glob("*.jpg")):
        name = p.stem
        if name.startswith("card_") or name == "blank":
            continue
        base = name.split("_ev")[0]
        out.setdefault(base, []).append(p)
    for p in sorted(Path(CARD_ART).glob("*.jpg")):
        out.setdefault(p.stem, []).insert(0, p)
    for p in sorted(LIBRARY.glob("*.png")):  # "<card>__<anything>.png"
        out.setdefault(p.stem.split("__")[0], []).insert(0, p)
    return out


class CardLibrary:
    """Names a card picture: the nearest of every picture we have of every card."""

    NAMED = 0.45   # a match this good ...
    MARGIN = 0.10  # ... and this much better than the next card's is a name

    def __init__(self):
        self.names, descs = [], []
        for name, paths in _card_images().items():
            for p in paths:
                self.names.append(name)
                descs.append(card_desc(Image.open(p)))
        self.descs = np.stack(descs)

    def scores(self, desc) -> dict[str, float]:
        s = self.descs @ desc / len(desc)
        # refine the closest pictures at the best shift (crops sit a pixel or two off)
        for i in np.argsort(-s)[:30]:
            s[i] = max(s[i], _aligned(self.descs[i], desc, 3))
        out: dict[str, float] = {}
        for n, v in zip(self.names, s):
            out[n] = max(out.get(n, -1.0), float(v))
        return out

    def name(self, desc) -> tuple[str | None, float]:
        ranked = sorted(self.scores(desc).items(), key=lambda kv: -kv[1])
        (n1, s1), (_, s2) = ranked[0], ranked[1]
        return (n1 if s1 >= self.NAMED and s1 - s2 >= self.MARGIN else None), s1


def slot_empty(crop) -> bool:
    """A slot with no card drawn (between plays, or before the battle) is nearly flat."""
    return float(np.asarray(crop.convert("RGB"), dtype=np.float32).std(axis=(0, 1)).mean()) < 5.0


def _aligned(a: np.ndarray, b: np.ndarray, r: int = 3) -> float:
    """Correlation of two card descriptors (or sums of them), at the best shift of up to
    ``r`` cells: crops of one card taken a few pixels apart still match."""
    A, B = a.reshape(16, 16), b.reshape(16, 16)
    best = -1.0
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            a_ = A[max(dy, 0):16 + min(dy, 0), max(dx, 0):16 + min(dx, 0)]
            b_ = B[max(-dy, 0):16 + min(-dy, 0), max(-dx, 0):16 + min(-dx, 0)]
            x, y = a_ - a_.mean(), b_ - b_.mean()
            best = max(best, float((x * y).sum() / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-9)))
    return best


@dataclass
class HandPlay:
    t: float              # when the elixir was spent
    slot: int             # hand slot 0-3 it left
    cost: int | None      # elixir spent, from the bar
    desc: np.ndarray      # what the card looked like in hand (card_desc)
    card: int = -1        # which of the player's cards (same number = same card)
    name: str | None = None
    frame: int = -1       # the frame whose crop the description is aligned on


class PlayReader:
    """One player's plays in a replay, from their elixir bar and their hand.

    * When: the elixir bar drops (a play is paid the moment the card is placed). The
      replay bar moves a whole segment at a time, so the drop is also the card's cost.
    * Which card: the hand slot whose card changes around then; what that card looked
      like in hand in the second before (greyed out, lit up or raised, it matches
      itself: ``card_desc``).
    * Which of the player's cards: plays whose cards look alike are the same card (a
      card is played many times), numbered by first appearance; each is named from the
      picture library where it can be, and its cost is the most common drop.
    """

    DROP = 0.6          # the bar falling this much between two frames is a play
    BEFORE_S = (-1.25, -0.25)   # the card is read in hand in this window before the drop
    AFTER_S = (0.5, 1.5)        # ... and its slot holds another card in this window after
    CHANGED = 0.45      # a slot whose before/after crops match less than this changed
    SAME_CARD = 0.7     # a played card this like a card's earlier plays (on average) is it:
                        # plays of one card match 0.8-1.0, of different cards under 0.55

    def __init__(self, fps: float = 4.0):
        self.fps = fps
        self.times, self.descs, self.empty, self.elixir = [], [], [], []

    def add(self, t: float, descs: np.ndarray, empty, elixir: float | None):
        """``descs``: (4 hand slots, shifts, 256), ReplayLayout.hand_descs(...)[1:]."""
        self.times.append(t)
        self.descs.append(descs)
        self.empty.append(list(empty))
        self.elixir.append(np.nan if elixir is None else float(elixir))

    def _clean_elixir(self) -> np.ndarray:
        e = np.array(self.elixir, dtype=float)
        # a single frame reading far below both neighbours is the bar flashing
        for i in range(1, len(e) - 1):
            if e[i] < min(e[i - 1], e[i + 1]) - 1.5 and abs(e[i - 1] - e[i + 1]) < 7:
                e[i] = np.nan
        return e

    def _frames(self, t0: float, t1: float) -> list[int]:
        return [i for i, t in enumerate(self.times) if t0 <= t <= t1]

    def _slot_desc(self, frames: list[int], k: int):
        """Mean descriptor of slot k over ``frames``, or None. The frames are aligned on
        the one the others match best at rest (offset 0), so a card is described where it
        sits when not held, the same way every time."""
        idx = [i for i in frames if not self.empty[i][k]]
        if not idx:
            return None
        ds = [self.descs[i][k] for i in idx]
        rest = np.stack([d[0] for d in ds])                       # (n, 256)
        fit = [sum(float((d @ r).max()) for d in ds) for r in rest]
        best = int(np.argmax(fit))
        ref = rest[best]
        acc = np.zeros_like(ref)
        for d in ds:
            acc += d[int(np.argmax(d @ ref))]
        self._ref_frame = idx[best]
        return acc / (np.linalg.norm(acc) + 1e-6) * 16

    def plays(self) -> list[HandPlay]:
        e = self._clean_elixir()
        t = np.array(self.times)
        out = []
        i = 1
        while i < len(e):
            prev = e[i - 1] if not np.isnan(e[i - 1]) else (e[i - 2] if i >= 2 else np.nan)
            if np.isnan(prev) or np.isnan(e[i]) or prev - e[i] < self.DROP:
                i += 1
                continue
            # the bar animates down over a frame or two: the first whole-number reading
            after = [v for v in e[i:i + 3] if not np.isnan(v) and abs(v - round(v)) < 0.15]
            cost = int(round(prev) - round(after[0])) if after else None
            when = float(t[i] - 0.5 / self.fps)
            # the slot that changed
            before = self._frames(when + self.BEFORE_S[0], when + self.BEFORE_S[1])
            later = self._frames(when + self.AFTER_S[0], when + self.AFTER_S[1])
            best = None
            for k in range(4):
                a = self._slot_desc(before, k)
                ref = getattr(self, "_ref_frame", -1)
                after_k = [self.descs[j][k] for j in later if not self.empty[j][k]]
                if a is None or not after_k:
                    continue
                # does the slot still hold that card (at any offset: held cards move)?
                s = float(np.median([(d @ a).max() / d.shape[-1] for d in after_k]))
                if s < self.CHANGED and (best is None or s < best[0]):
                    best = (s, k, a, ref)
            if best is not None:
                out.append(HandPlay(when, best[1], cost if cost and 1 <= cost <= 10 else None,
                                    best[2], frame=best[3]))
            i += 2  # the bar animates over a frame or two
        return self._identify(out)

    def _identify(self, plays: list[HandPlay]) -> list[HandPlay]:
        groups: list[list[HandPlay]] = []
        for p in plays:
            scores = [float(np.mean([_aligned(p.desc, q.desc) for q in g])) for g in groups]
            if scores and max(scores) >= self.SAME_CARD:
                groups[int(np.argmax(scores))].append(p)
            else:
                groups.append([p])
        lib = None
        for c, g in enumerate(groups):
            # the most common drop (the bar can gain a whole elixir in the same frame as a
            # play, so a drop now and then reads one short; ties go to the larger)
            costs = [p.cost for p in g if p.cost is not None]
            cost = max(set(costs), key=lambda c: (costs.count(c), c)) if costs else None
            if lib is None:
                lib = CardLibrary()
            total = sum(p.desc for p in g)
            name, _ = lib.name(total / (np.linalg.norm(total) + 1e-6) * 16)
            if name is not None:
                from ..real import cardmap
                true = cardmap.cost(name)
                # a picture match whose cost disagrees with the elixir spent is a mistake
                if true is not None and cost is not None and abs(true - cost) >= 2:
                    name = None
                elif true is not None:
                    cost = true
            for p in g:
                p.card, p.name, p.cost = c, name, cost
        return plays


# ---------------------------------------------------------------- where it landed
class DiffPlacer:
    """Where a play landed, from how the (warped) picture changed around it.

    For a play at time t: new change = |after - before|, where ``before`` is the frame
    BEFORE s ahead of t and ``after`` is AFTER s past it (AFTER_THROWN for a thrown card,
    which lands about a second after it leaves the hand); old motion = |before - earlier|
    (EARLIER s ahead), which is what was already moving there (walking troops, projectiles).
    Both are blurred to troop size; the spot is where new change minus old motion peaks,
    inside where that player may place that card."""

    BEFORE = 0.25
    AFTER = 0.5
    EARLIER = 0.75
    AFTER_THROWN = 1.25
    ARENA_ROWS = (50, 496)    # tile rows 0-31 on the battle screenshot
    THRESHOLD = 6.0           # weaker than this is no visible change
    SIGMA = 4.0               # blur, in pixels (a troop is ~10-20 px)

    def __init__(self):
        self.times: list[float] = []
        self.frames: list[np.ndarray] = []  # grey uint8 battle-layout frames

    def add(self, t: float, warped) -> None:
        self.times.append(t)
        self.frames.append(np.asarray(warped.convert("L"), dtype=np.uint8))

    def _at(self, t: float, after: bool):
        import bisect

        i = bisect.bisect_left(self.times, t - 1e-6) if after else \
            bisect.bisect_right(self.times, t + 1e-6) - 1
        if 0 <= i < len(self.frames):
            return self.frames[i].astype(np.float32)
        return None

    @staticmethod
    def _blur(a, sigma):
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(a, sigma)

    @staticmethod
    def zone(side: str, anywhere: bool, towers=None) -> np.ndarray:
        """Pixels where ``side`` may place a card (screen frame): its own half, plus the
        enemy's front pocket on a side whose enemy princess tower is down; anywhere on the
        arena for thrown cards and the Miner. ``towers``: [my L, my R, their L, their R] in
        the player's own frame (0 = destroyed)."""
        h, w = SCREEN[1], SCREEN[0]
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        tx, ty = px_to_tile(xs + 0.5, ys + 0.5)
        inside = (tx >= -0.5) & (tx < 17.5) & (ty >= -0.5) & (ty < 31.5)
        if anywhere:
            return inside
        if side == "top":
            tx, ty = 17 - tx, 31 - ty
        own = ty < 14.5
        if towers:
            left_down, right_down = towers[2] <= 0, towers[3] <= 0
            pocket = ty < 20.5
            own |= pocket & (((tx < 8.5) & left_down) | ((tx >= 8.5) & right_down))
        return inside & own

    def place(self, t: float, side: str, thrown: bool = False, anywhere: bool = False,
              towers=None):
        """(x, y) tile in the screen's frame (the bottom player's), and how strong the
        change was; None when nothing changed enough."""
        before = self._at(t - self.BEFORE, after=False)
        after = self._at(t + (self.AFTER_THROWN if thrown else self.AFTER), after=True)
        earlier = self._at(t - self.EARLIER, after=False)
        if before is None or after is None:
            return None
        new = self._blur(np.abs(after - before), self.SIGMA)
        if earlier is not None:
            new = new - self._blur(np.abs(before - earlier), self.SIGMA)
        mask = self.zone(side, anywhere or thrown, towers)
        r0, r1 = self.ARENA_ROWS
        mask[:r0] = False
        mask[r1:] = False
        score = np.where(mask, new, -np.inf)
        k = int(np.argmax(score))
        py, px = divmod(k, SCREEN[0])
        peak = float(score[py, px])
        if not np.isfinite(peak) or peak < self.THRESHOLD:
            return None
        # centre of the blob around the peak, weighted by how much it changed
        y0, y1 = max(py - 12, 0), min(py + 13, SCREEN[1])
        x0, x1 = max(px - 12, 0), min(px + 13, SCREEN[0])
        win = score[y0:y1, x0:x1] - peak / 2
        win = np.where(np.isfinite(win), np.clip(win, 0, None), 0)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        cx = float((win * xx).sum() / win.sum()) if win.sum() > 0 else float(px)
        cy = float((win * yy).sum() / win.sum()) if win.sum() > 0 else float(py)
        tx, ty = px_to_tile(cx + 0.5, cy + 0.5)
        return (float(np.clip(tx, 0, 17)), float(np.clip(ty, 0, 31))), peak
