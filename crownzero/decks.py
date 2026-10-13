"""Decks built from the 18 calibrated cards: four curated ones, and random legal ones."""
from __future__ import annotations

import numpy as np

from .features import CARD_TYPES

CURATED = {
    "giant": ["Giant", "Musketeer", "Minions", "Knight", "Archer", "Fireball", "Zap", "Cannon"],
    "hog": ["HogRider", "Valkyrie", "Musketeer", "SkeletonArmy", "Tesla", "Log", "Fireball",
            "BabyDragon"],
    "prince": ["Prince", "Giant", "Wizard", "Minions", "Archer", "Zap", "Arrows", "Knight"],
    "bait": ["GoblinBarrel", "Knight", "Archer", "Minions", "SkeletonArmy", "Tesla", "Log",
             "Fireball"],
}
SPELL_NAMES = {"Fireball", "Zap", "Arrows", "Log", "GoblinBarrel"}
BUILDING_NAMES = {"Cannon", "Tesla"}


def random_deck(rng: np.random.Generator) -> list[str]:
    """8 distinct calibrated cards with 1-3 spells and at most one building."""
    while True:
        deck = [str(c) for c in rng.choice(CARD_TYPES, size=8, replace=False)]
        spells = sum(c in SPELL_NAMES for c in deck)
        buildings = sum(c in BUILDING_NAMES for c in deck)
        if 1 <= spells <= 3 and buildings <= 1:
            return deck


def pick_deck(rng) -> list[str]:
    if rng.random() < 0.4:
        return list(CURATED[rng.choice(list(CURATED))])
    return random_deck(rng)
