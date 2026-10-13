"""Names: Build-A-Bot's card and unit names <-> the simulator's card names, for every card.

Build-A-Bot knows 110 cards and 96 unit types; the simulator's catalogue uses Supercell's
internal names (Bandit = "Assassin", Night Witch = "DarkWitch", ...). The tables in
cardmap_data.py are generated from both and checked against the simulator, so nothing here
needs the simulator installed.
"""
from __future__ import annotations

import re

from .cardmap_data import CARD_TO_SIM, SIM_COST, SIM_KIND, UNIT_TO_SIM, UNIT_VALUE

SIM_TO_CARD = {v: k for k, v in CARD_TO_SIM.items()}


def sim_name(card_name: str | None) -> str | None:
    """Build-A-Bot card name ("hog_rider") -> simulator card name ("HogRider")."""
    return CARD_TO_SIM.get(card_name) if card_name else None


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


_BY_NORM = {}
for _bab, _sim in CARD_TO_SIM.items():
    for _k in (_bab, _sim):
        _BY_NORM.setdefault(_norm(_k), _bab)
_BY_NORM.update({"log": "the_log", "archer": "archers", "minion": "minions",
                 "xbow": "x_bow", "pekka": "pekka", "minipekka": "minipekka"})


def resolve_deck_name(name: str) -> str | None:
    """A deck entry as the user types it ("Mini PEKKA", "the log", "hog_rider", "HogRider")
    -> Build-A-Bot card name, or None if no card has that name."""
    k = _norm(name)
    if k in _BY_NORM:
        return _BY_NORM[k]
    if k.endswith("s") and k[:-1] in _BY_NORM:
        return _BY_NORM[k[:-1]]
    return _BY_NORM.get(k + "s")


def cost(card_name: str | None) -> int | None:
    s = sim_name(card_name)
    return SIM_COST.get(s) if s else None


def kind(card_name: str | None) -> str | None:
    s = sim_name(card_name)
    return SIM_KIND.get(s) if s else None


def unit_sim(unit_name: str) -> str | None:
    return UNIT_TO_SIM.get(unit_name)


def unit_value(unit_name: str, default: float = 3.0) -> float:
    s = UNIT_TO_SIM.get(unit_name)
    return UNIT_VALUE.get(s, default) if s else default
