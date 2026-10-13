"""Reading the towers off the screenshot, beyond Build-A-Bot's princess health bars.

Recorded 2026-10-13 (a 3-crown win): the enemy's left princess tower was destroyed at 2:23
and the bot went on believing it stood at 3% for the rest of the game, and it saw the enemy
King Tower at full health throughout, even at 780 HP. Build-A-Bot's bar reader returns 0
both for a destroyed tower and for a bar too short to read (a tower on 84 HP), so a 0 was
read as "almost destroyed", never as gone; and it does not read the king's bar at all.

A standing princess tower always shows its level badge (a small gold shield left of its
health bar), however low its health; rubble has none. Across three recorded battles: about
90-200 gold pixels in the badge box while the tower stands, 0 once it is rubble. The badge
can be hidden for a frame by a unit standing on it, so a tower counts as destroyed only when
both its badge and its bar are missing for several frames in a row (see TowerTracker).

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


def badges(image) -> list[bool]:
    """Is each princess tower's level badge on screen? [ally L, ally R, enemy L, enemy R]."""
    a = np.asarray(image.convert("RGB"), dtype=np.int16)
    out = []
    for x, y in PRINCESS_BARS:
        box = a[y - 5:y + 9, x - 14:x]
        gold = (box[..., 0] > 170) & (box[..., 0] - box[..., 1] > 10) & (box[..., 2] < 120)
        out.append(int(gold.sum()) >= BADGE_PIXELS)
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
