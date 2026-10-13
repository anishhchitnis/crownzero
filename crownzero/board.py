"""Board geometry and card lookups shared by every bot.

Coordinates: the engine works in subtiles (18,000 per tile) on an 18 x 32 tile board.
Every bot reasons in its OWN frame: its king at the bottom (low y), the enemy at the top.
The action mask planes are [hand_slot, y, x] in that same own frame.
"""
from __future__ import annotations

from dataclasses import dataclass

TILE = 18_000
KING_TOWER, PRINCESS_TOWER = 2, 3  # royalegym.EntityKind values; kept here so the
# real-game bridge can import this module without installing the simulator
NX, NY = 18, 32
HAND = 4
CELLS = NX * NY  # 576 cells per hand slot
NOOP = 0
RIVER_Y = 15  # own-frame tile rows 15-16 hold the river; < 15 is safely our half


def action_index(slot: int, y: int, x: int) -> int:
    """Flat action id: no-op first, then slot-major, row, column (the parser's layout)."""
    return 1 + slot * CELLS + y * NX + x


@dataclass(frozen=True)
class Card:
    card_id: int
    name: str
    elixir: int
    count: int
    kind: str  # TROOP, BUILDING or SPELL
    hitpoints: int = 0  # of one of the card's own units (0 for spells)

    @property
    def unit_value(self) -> float:
        """Elixir that ONE summoned unit of this card is worth at full health."""
        return self.elixir / max(self.count, 1)

    def entity_value(self, max_hp: float) -> float:
        """Elixir one unit on the board is worth at full health. Units a card summons later
        (a Goblin Hut's Spear Goblins, a Witch's skeletons) carry the card's id too, but
        are small: they are worth their share of the card's own unit, by hitpoints.
        Before this, each 133-hp Spear Goblin from a Hut counted as a whole 4-elixir Hut
        and the bot played the Hut over everything else (recorded 2026-10-13)."""
        if self.hitpoints > 0 and 0 < max_hp < SPAWN_RATIO * self.hitpoints:
            return self.unit_value * max_hp / self.hitpoints
        return self.unit_value


SPAWN_RATIO = 0.6  # a unit with less than this share of the card's hitpoints was summoned


def card_table(engine) -> dict[int, Card]:
    return {
        c.card_id: Card(c.card_id, c.name, c.elixir, c.count, c.card_kind or "TROOP",
                        int(getattr(c, "hitpoints", 0) or 0))
        for c in engine.cards()
    }


def own_tile(arena, team: int, x: int, y: int) -> tuple[float, float]:
    """An engine-frame point as (x, y) in tiles of ``team``'s own frame."""
    from royalegym import to_own

    ox, oy = to_own(arena, team, x, y)
    return ox / TILE, oy / TILE


def is_tower(ent) -> bool:
    return ent.kind in (KING_TOWER, PRINCESS_TOWER)
