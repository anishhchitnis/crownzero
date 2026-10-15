"""Fair play: CrownZero does not play battles against real players.

Supercell's terms ban automating the game, and a bot on the ladder takes trophies from
people. CrownZero is for Training Camp and friendly battles. Those start without
matchmaking; every battle against a stranger (Trophy Road, ranked, 2v2 with a random
partner) starts with the "Searching for opponent" screen. When the bot sees that screen,
it sits out the battle that follows: it watches, records if asked, and plays nothing.

Recorded 2026-10-13: a test battle that turned out to be a Trophy Road match against a
931-trophy player.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent / "screens"
BANNER_BOX = (90, 52, 310, 102)   # "Searching for opponent", on a 368x652 screenshot
CANCEL_BOX = (128, 462, 248, 522)  # the red Cancel button under it
MATCH = 0.75


def _grey(img, box=None):
    if box is not None:
        img = img.crop(box)
    a = np.asarray(img.convert("L").resize((img.width // 2, img.height // 2)), np.float32)
    a = a - a.mean()
    return a / (np.linalg.norm(a) + 1e-6)


_TEMPLATES = None


def is_matchmaking(image) -> bool:
    """Is this the "Searching for opponent" screen?"""
    global _TEMPLATES
    if _TEMPLATES is None:
        _TEMPLATES = (_grey(Image.open(HERE / "searching_banner.png")),
                      _grey(Image.open(HERE / "searching_cancel.png")))
    img = image.convert("RGB")
    if img.size != (368, 652):
        img = img.resize((368, 652))
    banner = float((_grey(img, BANNER_BOX) * _TEMPLATES[0]).sum())
    cancel = float((_grey(img, CANCEL_BOX) * _TEMPLATES[1]).sum())
    return banner >= MATCH and cancel >= MATCH


class FairPlay:
    """Feeds on every screenshot; ``sit_out`` is True from a matchmaking screen until the
    battle after it is over."""

    END_FRAMES = 5  # screenshots in a row outside a battle that mean it is over

    def __init__(self):
        self.sit_out = False
        self.in_battle = False
        self.outside = 0

    def update(self, image, in_game: bool) -> bool:
        if not in_game and is_matchmaking(image):
            self.sit_out, self.in_battle, self.outside = True, False, 0
        elif in_game:
            self.in_battle, self.outside = True, 0
        elif self.sit_out and self.in_battle:
            self.outside += 1
            if self.outside >= self.END_FRAMES:
                self.sit_out = self.in_battle = False
        return self.sit_out
