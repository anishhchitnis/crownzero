"""The search bot: try a handful of plays, simulate a few seconds ahead, keep the best.

At every decision (every 0.5 s of game time) the bot:
  1. lists candidate plays: each card in hand at a few sensible spots (defend each lane,
     the bridges, behind the king, on top of enemy units for spells, ...), snapped to the
     nearest cell the rules allow;
  2. saves the engine, and for "do nothing" plus every candidate, loads the save, plays the
     candidate, runs the battle forward ``horizon_ticks`` with the opponent idle, and scores
     the board it lands on;
  3. restores the engine and returns the best play, or no-op if nothing beats waiting.

It reads only what a human sees on screen (units, health, towers, its own hand and elixir).
It never looks at the opponent's hand or elixir: in the rollouts the opponent simply plays
nothing. Its only unfair advantage is the simulator itself, which it uses as a perfect
"what happens if" machine. That is fine for a teacher; the student network in step 3 will
not get it.
"""
from __future__ import annotations

import time

import numpy as np
from royalegym import EntityKind

from .board import NOOP, NX, NY, RIVER_Y, action_index, card_table, is_tower, own_tile

# Spots for troops, own-frame tile (x, y). Princess towers sit at x 3.5 / 14.5, y 6.5;
# the king at x 9, y 3; bridges at x ~3 and ~14; the river at rows 15-16.
TROOP_SPOTS = [(3, 10), (14, 10), (8, 8), (9, 8), (3, 14), (14, 14), (8, 1), (9, 1)]
BUILDING_SPOTS = [(8, 9), (9, 9), (8, 11), (9, 11)]
SNAP_LIMIT = 3.0  # tiles: an anchor further than this from any legal cell is dropped


class Weights:
    """How the bot values a board, in elixir. Tuned by hand, then by results."""

    princess = 12.0  # a full-health princess tower
    king = 30.0
    crown = 10.0  # on top of the tower health it took
    win = 100.0
    elixir = 1.0
    material = 0.8  # elixir on the board is worth a bit less than elixir in the bank
    threat = 0.6  # extra value for a unit standing near an enemy tower
    threat_range = 10.0  # tiles over which that extra value fades to zero


class SearchBot:
    name = "search"

    def __init__(self, horizon_ticks: int = 120, margin: float = 0.05,
                 max_spell_targets: int = 4, weights: type[Weights] = Weights,
                 opponent=None, reply_ticks: int = 20,
                 defend_horizon_ticks: int | None = None) -> None:
        """``opponent``: an object with ``reply(state, team, mask) -> action``. When given,
        every rollout lets the opponent answer ``reply_ticks`` after our move (a two-ply
        search) instead of assuming it stands still."""
        self.opponent = opponent
        self.reply_ticks = reply_ticks
        self.base_horizon = horizon_ticks
        # Look further ahead when enemy troops are on our side: a Giant at the bridge needs
        # ~10 s to reach a tower, and a 6 s look-ahead sees no danger in it (recorded
        # 2026-10-13: Archers sent to the other bridge while a Giant walked into our tower).
        self.defend_horizon = defend_horizon_ticks or horizon_ticks
        self.horizon = horizon_ticks
        self.margin = margin  # a play must beat waiting by this much elixir-equivalent
        self.max_spell_targets = max_spell_targets
        self.w = weights
        self.cards = None
        self.arena = None
        self.rollouts = 0
        self.think_s = 0.0

    # ---------------------------------------------------------------- deciding
    def act(self, env, team: int, obs, mask, rng) -> int:
        planes = np.asarray(obs["mask_planes"])
        if not planes.any():
            return NOOP
        engine = env.engine
        if self.cards is None:
            self.cards = card_table(engine)
            self.arena = engine.arena()
            if self.opponent is not None and getattr(self.opponent, "cards", None) is None:
                self.opponent.bind(engine)
        state = env.battle_state
        self.horizon = self.base_horizon
        if self.defend_horizon != self.base_horizon:
            for ent in state.entities:
                if ent.team != team and not is_tower(ent):
                    if own_tile(self.arena, team, ent.x, ent.y)[1] < 16:
                        self.horizon = self.defend_horizon
                        break
        cands = self._candidates(state, team, planes)
        if not cands:
            return NOOP

        t0 = time.perf_counter()
        setup_cards = getattr(engine, "_setup_cards", None)
        blob = engine.save_state()
        # Waiting first: the engine already sits at the saved point, so no load is needed.
        best_value = self._rollout(env, engine, None, team) + self.margin
        best_action = NOOP
        for action in cands:
            cmd = env.action_parser.parse(action, state, team)
            if cmd is None:
                continue
            engine.load_state(blob)
            value = self._rollout(env, engine, cmd, team)
            if value > best_value:
                best_value, best_action = value, action
        engine.load_state(blob)
        if setup_cards is not None:
            engine._setup_cards = setup_cards
        self.rollouts += len(cands) + 1
        self.think_s += time.perf_counter() - t0
        return best_action

    def _rollout(self, env, engine, cmd, team: int) -> float:
        first = [cmd] if cmd is not None else []
        if self.opponent is None:
            engine.step(first, self.horizon)
        else:
            engine.step(first, self.reply_ticks)
            st = engine.state()
            if not st.game_over:
                enemy = 1 - team
                mask = env.action_parser.action_mask(st, enemy)
                reply = self.opponent.reply(st, enemy, mask)
                ocmd = env.action_parser.parse(reply, st, enemy) if reply != NOOP else None
                engine.step([ocmd] if ocmd is not None else [], self.horizon - self.reply_ticks)
        return self._value(engine.state(), team)

    # ---------------------------------------------------------------- candidates
    def _candidates(self, state, team: int, planes: np.ndarray) -> list[int]:
        me = state.players[team]
        enemies = []  # (x, y, value) in own-frame tiles, non-tower enemy units
        enemy_towers = []
        for ent in state.entities:
            if ent.team == team:
                continue
            x, y = own_tile(self.arena, team, ent.x, ent.y)
            if is_tower(ent):
                if ent.hp > 0:
                    enemy_towers.append((x, y))
                continue
            card = self.cards.get(ent.card_id)
            value = card.unit_value if card else ent.max_hp / 600
            enemies.append((x, y, value * ent.hp / max(ent.max_hp, 1)))

        # Spell targets: where enemy value is densest, plus the towers.
        scored = []
        for x, y, _ in enemies:
            dens = sum(v for ex, ey, v in enemies if (ex - x) ** 2 + (ey - y) ** 2 <= 2.5 ** 2)
            scored.append((dens, x, y))
        scored.sort(reverse=True)
        spell_spots = [(x, y) for _, x, y in scored[: self.max_spell_targets]]
        tower_spots = list(enemy_towers)

        # Reactive spots: meet enemy units that are on, or heading into, our half.
        reactive = []
        for x, y, _ in enemies:
            if y < RIVER_Y + 5:
                reactive += [(x, y), (x, max(y - 3, 0))]

        out: set[int] = set()
        for slot in range(len(me.hand)):
            plane = planes[slot]
            if not plane.any():
                continue
            card = self.cards.get(me.hand[slot])
            kind = card.kind if card else "TROOP"
            if kind == "SPELL":
                if card and card.name == "GoblinBarrel":
                    anchors = tower_spots
                elif card and card.name == "Log":
                    anchors = [(x, y - 2) for x, y in spell_spots] + spell_spots
                else:
                    anchors = spell_spots + tower_spots
            elif kind == "BUILDING":
                anchors = BUILDING_SPOTS + [(x, max(y - 4, 2)) for x, y in reactive[::2]]
            else:
                anchors = TROOP_SPOTS + reactive
            cells = np.argwhere(plane > 0)  # rows of (y, x)
            for ax, ay in anchors:
                d2 = (cells[:, 0] - ay) ** 2 + (cells[:, 1] - ax) ** 2
                i = int(np.argmin(d2))
                if d2[i] <= SNAP_LIMIT ** 2:
                    y, x = cells[i]
                    out.add(action_index(slot, int(y), int(x)))
        return sorted(out)

    # ---------------------------------------------------------------- scoring
    def _value(self, s, team: int) -> float:
        w = self.w
        enemy = 1 - team
        v = 0.0
        for side, sign in ((team, 1.0), (enemy, -1.0)):
            p = s.players[side]
            king = p.tower_hp[0] / max(p.tower_max_hp[0], 1)
            princesses = sum(hp / max(mx, 1) for hp, mx in zip(p.tower_hp[1:], p.tower_max_hp[1:]))
            v += sign * (w.king * king + w.princess * princesses + w.crown * p.crowns)
        v += w.elixir * s.players[team].elixir_milli / 1000  # own elixir only
        if s.game_over:
            if s.winner == team:
                v += w.win
            elif s.winner == enemy:
                v -= w.win

        towers = {team: [], enemy: []}
        units = []
        for ent in s.entities:
            x, y = own_tile(self.arena, team, ent.x, ent.y)
            if is_tower(ent):
                if ent.hp > 0:
                    towers[ent.team].append((x, y))
            elif ent.kind in (EntityKind.TROOP, EntityKind.BUILDING):
                units.append((ent, x, y))
        for ent, x, y in units:
            card = self.cards.get(ent.card_id)
            base = card.unit_value if card else ent.max_hp / 600
            val = base * ent.hp / max(ent.max_hp, 1)
            target = towers[enemy] if ent.team == team else towers[team]
            if target:
                d = min(((x - tx) ** 2 + (y - ty) ** 2) ** 0.5 for tx, ty in target)
                prox = max(0.0, 1.0 - d / w.threat_range)
            else:
                prox = 0.0
            sign = 1.0 if ent.team == team else -1.0
            v += sign * w.material * val * (1.0 + w.threat * prox)
        return v


__all__ = ["SearchBot", "Weights", "NX", "NY"]
