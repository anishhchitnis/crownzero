"""Opponent models for the search bot's rollouts.

With no model, the search assumes the opponent stands still for the whole look-ahead. Then
any troop dropped at the bridge walks into a tower unopposed and looks like a great play,
and the bot sent lone troops to the bridge again and again in real games, where the
opponent simply defended them (recorded 2026-10-13).

``DefenderReply`` answers the way any player does: when our troops head for its towers, it
puts a troop from its hand in that lane, between them and its tower, about a second later.
It needs no knowledge of the real opponent's deck: in the rebuilt battle the opponent
holds a generic hand (Giant, Musketeer, Minions, Knight) and 5 elixir.
"""
from __future__ import annotations

import numpy as np

from .board import CELLS, HAND, NOOP, NX, NY, action_index, card_table, is_tower, own_tile

TOWER_X = (3.5, 14.5)
DEFEND_Y = 9  # own-frame row a few tiles in front of the princess towers
THREAT_Y = 21  # attackers at or below this row (own frame of the defender) get answered


class DefenderReply:
    name = "defender-reply"

    def __init__(self):
        self.cards = None
        self.arena = None

    def bind(self, engine):
        self.cards = card_table(engine)
        self.arena = engine.arena()
        return self

    def reply(self, state, team: int, mask) -> int:
        """``team``: the side answering. Returns an action id in its own frame."""
        mask = np.asarray(mask).astype(bool)
        if not mask[1:].any():
            return NOOP
        threats = []
        for ent in state.entities:
            if ent.team == team or is_tower(ent) or ent.hp <= 0:
                continue
            card = self.cards.get(ent.card_id)
            if card is None or card.kind == "SPELL":
                continue
            x, y = own_tile(self.arena, team, ent.x, ent.y)
            if y <= THREAT_Y:
                threats.append((card.entity_value(ent.max_hp) * ent.hp / max(ent.max_hp, 1), x, y))
        if not threats:
            return NOOP
        # the lane with the most attacking value
        left = sum(v for v, x, _ in threats if x < NX / 2)
        right = sum(v for v, x, _ in threats if x >= NX / 2)
        lane = [t for t in threats if (t[1] < NX / 2) == (left >= right)]
        value = sum(t[0] for t in lane)
        _, tx, ty = min(lane, key=lambda t: t[2])  # the attacker closest to our towers
        planes = mask[1:].reshape(HAND, NY, NX)
        hand = state.players[team].hand
        # a troop (or building) whose cost is closest to what is attacking; spells last
        best = None
        for slot in range(min(HAND, len(hand))):
            if not planes[slot].any():
                continue
            card = self.cards.get(hand[slot])
            if card is None:
                continue
            spell = card.kind == "SPELL"
            key = (spell, abs(card.elixir - value))
            if best is None or key < best[0]:
                best = (key, slot, card)
        if best is None:
            return NOOP
        _, slot, card = best
        if card.kind == "SPELL":
            ax, ay = tx, ty
        else:
            ax = tx
            ay = min(DEFEND_Y, max(ty - 2, 2)) if ty < DEFEND_Y + 2 else DEFEND_Y
        cells = np.argwhere(planes[slot])
        d2 = (cells[:, 0] - ay) ** 2 + (cells[:, 1] - ax) ** 2
        y, x = cells[int(np.argmin(d2))]
        return action_index(slot, int(y), int(x))


def make_opponent(spec):
    """``"defender"`` or a path to a student network, for ``SearchBot(opponent=...)``."""
    if spec is None:
        return None
    if spec == "defender":
        return DefenderReply()
    from .student_bot import StudentBot

    return StudentBot(spec)


__all__ = ["DefenderReply", "make_opponent", "CELLS"]
