"""Reading the towers off the screenshot, beyond Build-A-Bot's princess health bars.

Recorded 2026-10-13 (a 3-crown win): the enemy's left princess tower was destroyed at 2:23
and the bot went on believing it stood at 3% for the rest of the game, and it saw the enemy
King Tower at full health throughout, even at 780 HP. Build-A-Bot's bar reader returns 0
both for a destroyed tower and for a bar too short to read (a tower on 84 HP), so a 0 was
read as "almost destroyed", never as gone; and it does not read the king's bar at all.

A standing princess tower always shows its level badge (a small shield with its level, left
of its health bar), however low its health; rubble has none. The badge looks the same in
every arena but the floor behind it does not: a gold-colour test (v4.4) passed the orange
tiles of one arena as a badge, and the bar reader read the purple tiles of another as a
red bar at 3-21% (both recorded 2026-10-13). So BadgeWatch learns what each tower's badge
box looks like while the tower clearly stands (its bar above half), and then asks whether
the box still looks like that: a mean difference under 15 while it stands, 33-58 once it is
rubble, across four recorded battles in three arenas. A unit or a flash can hide the badge
for a frame, so a tower counts as destroyed only after several frames (see TowerTracker).

The enemy King Tower's health bar appears at the top of the screen once it has been hit.
"""
from __future__ import annotations

import numpy as np

# Build-A-Bot's princess health bars (368x652 screenshot): left x, y of each bar
PRINCESS_BARS = [(74, 404), (266, 404), (74, 95), (266, 95)]  # ally L, ally R, enemy L, enemy R
BADGE_PIXELS = 25  # gold pixels in the badge box that mean "the badge is there"

KING_BAR_X = (164, 219)  # enemy King Tower bar, columns
KING_BAR_ROWS = (23, 27)
KING_FILL = np.array([224, 35, 93])   # Build-A-Bot's enemy health-bar colours
KING_EMPTY = np.array([90, 49, 68])


def _badge_box(a, i):
    x, y = PRINCESS_BARS[i]
    return a[y - 5:y + 9, x - 14:x]


def _gold(box) -> int:
    """Saturated gold pixels (the badge), not merely orange (some arenas' floor)."""
    b = box.astype(np.int16)
    return int(((b[..., 0] > 170) & (b[..., 1] - b[..., 2] > 85)
                & (b[..., 0] - b[..., 2] > 110)).sum())


def badges(image) -> list[bool]:
    """Is each princess tower's level badge on screen, by colour alone? (Used until
    BadgeWatch has learned the badges.) [ally L, ally R, enemy L, enemy R]."""
    a = np.asarray(image.convert("RGB"), dtype=np.float32)
    return [_gold(_badge_box(a, i)) >= BADGE_PIXELS for i in range(4)]


class BadgeWatch:
    """Learns each tower's badge while the tower clearly stands, then checks it is still
    there. [ally L, ally R, enemy L, enemy R]."""

    GONE_DIFF = 25.0   # mean pixel difference from the learned badge
    LEARN_ABOVE = 0.5  # learn from frames where the tower's bar reads above this
    KEEP = 5

    def __init__(self):
        self.reset()

    def reset(self):
        self.samples = [[] for _ in range(4)]

    def update(self, image, readings) -> list[bool]:
        a = np.asarray(image.convert("RGB"), dtype=np.float32)
        out = []
        for i in range(4):
            box = _badge_box(a, i)
            s = self.samples[i]
            if len(s) >= 3:
                present = float(np.abs(box - np.median(s, axis=0)).mean()) < self.GONE_DIFF
            else:  # not learned yet: a bar above half, or a gold badge, means it stands
                present = readings[i] >= self.LEARN_ABOVE or _gold(box) >= BADGE_PIXELS
            # a bar above half means the tower stands: learn its badge (once learned, skip
            # frames where something covers it)
            if readings[i] >= self.LEARN_ABOVE and (present or len(s) < 3):
                s.append(box)
                del s[:-self.KEEP]
            out.append(present)
        return out


def enemy_king(image) -> float | None:
    """The enemy King Tower's health share from its bar, or None when no bar is drawn (it
    has not been hit yet) or the bar cannot be read (it flashes when hit)."""
    a = np.asarray(image.convert("RGB"), dtype=np.float32)
    strip = a[KING_BAR_ROWS[0]:KING_BAR_ROWS[1], KING_BAR_X[0]:KING_BAR_X[1]].mean(axis=0)
    d_fill = np.abs(strip - KING_FILL).mean(axis=1)
    d_empty = np.abs(strip - KING_EMPTY).mean(axis=1)
    fill = d_fill < d_empty
    nearest = np.where(fill, d_fill, d_empty)
    # a drawn bar: nearly every column is one of the two colours (0.96-0.98 on recorded
    # frames, 0.33-0.44 where there is no bar), and some of it is filled (an empty bar
    # would mean the king is gone and the battle over)
    if np.mean(nearest < 30) < 0.9 or not fill.any():
        return None
    # the fill runs from the left; the change point that best splits fill | empty
    score = np.cumsum(np.where(fill, -1, 1))
    cut = int(np.argmin(score)) + 1 if fill[0] else 0
    return cut / len(fill)
