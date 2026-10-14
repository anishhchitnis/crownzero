"""Rebuild a replayed battle inside the simulator, play by play, to know what was on the
board when each play was made.

Build-A-Bot's unit reader sees little of a pro replay (it misses most units, takes many
for the wrong side, and knows no evolutions), but the replay reader knows every play of
both players: the card, the tile and the moment. Played into the simulator in order,
those plays give a board where every unit walks, fights and dies roughly as it did, and
it is the same kind of board the bot's search sees when it plays (its view is rebuilt in
the simulator too), which is what a model trained on it will be shown.

Each play is made by restarting the simulator from the current board (every unit where
it stands, with its health) with that card first in the player's hand and elixir to
spare, so plays the simulator would refuse for elixir or hand order still happen. The
towers are set from the replay's tower readings at each play, so they never drift.
"""
from __future__ import annotations

TICKS_PER_S = 20
START_TICK = 90  # the simulator refuses plays during the start-of-battle countdown

# a card the simulator does not have, or one we could not name: a stand-in of its cost
PROXY_TROOP = {1: "Skeletons", 2: "Goblins", 3: "Knight", 4: "MiniPekka", 5: "Prince",
               6: "RoyalGiant", 7: "Pekka", 8: "Golem", 9: "ThreeMusketeers"}
PROXY_SPELL = {1: "Zap", 2: "Zap", 3: "Arrows", 4: "Fireball", 5: "Poison", 6: "Rocket"}
FILLER = ["Knight", "Archer", "Goblins", "Giant", "Musketeer", "Minions", "Fireball", "Zap",
          "Skeletons"]
NUDGES = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (1, -1), (-1, 1), (1, 1),
          (0, -2), (0, 2), (-2, 0), (2, 0)]


class Rebuild:
    """``play`` each replay play in time order; ``state()`` is the board just before the
    next one. Tiles are screen tiles (bottom player's frame), which is the simulator's
    frame: the bottom player is team 0 (blue)."""

    def __init__(self, seed: int = 0):
        from royalegym import RustEngine

        from ..board import TILE

        self.TILE = TILE
        self.engine = RustEngine()
        self.cards = {c.card_id: c for c in self.engine.cards()}
        self.by_name = {c.name: c.card_id for c in self.cards.values()}
        self.arena = self.engine.arena()
        self.seed = seed
        self.filler = [self.by_name[n] for n in FILLER]
        self.unit_card = self._unit_cards()
        self.running = False
        self.tick = START_TICK
        self.refused = 0
        self.played = 0
        from royalegym import MatchSetup

        self.engine.reset(seed, MatchSetup(decks=[self.filler[:8], self.filler[:8]],
                                           start_tick=START_TICK))
        st = self.engine.state()
        self.king_max = st.players[0].tower_max_hp[0]
        self.princess_max = st.players[0].tower_max_hp[1]

    # ------------------------------------------------------------ cards
    def _unit_cards(self) -> dict[int, int]:
        """unit type -> the card whose own unit it is (to put a summoned unit back on the
        board as itself, not as the building or troop that summoned it)."""
        from royalegym import MatchSetup
        from royalegym.protocol import SpawnSpec

        out = {}
        for c in self.cards.values():
            if c.card_kind not in ("TROOP", "BUILDING"):
                continue
            try:
                self.engine.reset(0, MatchSetup(
                    decks=[self.filler[:8], self.filler[:8]], start_tick=START_TICK,
                    spawns=[SpawnSpec(team=0, card_id=c.card_id, x=9 * self.TILE,
                                      y=int(8.5 * self.TILE))]))
            except Exception:
                continue
            ents = [e for e in self.engine.state().entities if e.kind in (0, 1)]
            if ents:
                out.setdefault(ents[0].unit_type, c.card_id)
        return out

    def card_id(self, sim_name: str | None, cost: int | None, thrown: bool) -> int | None:
        if sim_name and sim_name in self.by_name:
            return self.by_name[sim_name]
        table = PROXY_SPELL if thrown else PROXY_TROOP
        if cost is None:
            return None
        return self.by_name.get(table.get(int(min(max(cost, 1), max(table))), ""))

    # ------------------------------------------------------------ the board
    def state(self):
        return self.engine.state()

    def advance(self, t: float) -> None:
        tick = START_TICK + int(round(t * TICKS_PER_S))
        if self.running and tick > self.tick:
            self.engine.step([], tick - self.tick)
        self.tick = max(self.tick, tick)

    def _spawns(self):
        from royalegym.protocol import SpawnSpec, spawn_violation

        out = []
        if not self.running:
            return out
        for e in self.engine.state().entities:
            if e.kind not in (0, 1) or e.hp <= 0:
                continue
            cid = self.unit_card.get(e.unit_type, e.card_id if e.card_id in self.cards else None)
            if cid is None:
                continue
            hp_max = int(getattr(self.cards[cid], "hitpoints", 0) or 0)
            hp = int(min(e.hp, hp_max)) if hp_max > 0 else -1
            spec = SpawnSpec(team=e.team, card_id=cid, x=int(e.x), y=int(e.y), hp=max(hp, 1)
                             if hp != -1 else -1)
            if spawn_violation(self.arena, list(self.cards.values()), spec) is None:
                out.append(spec)
        return out

    def _tower_hp(self, towers, state):
        """``towers``: screen [bottom L, bottom R, top L, top R] health shares (None:
        keep the simulator's)."""
        p0, p1 = state.players[0], state.players[1]
        if towers is None:
            return [list(p0.tower_hp), list(p1.tower_hp)]
        pm = self.princess_max
        bl, br, tl, tr = towers
        # the top player's own left is the screen's right
        return [[max(1, p0.tower_hp[0]), round(bl * pm), round(br * pm)],
                [max(1, p1.tower_hp[0]), round(tr * pm), round(tl * pm)]]

    def play(self, team: int, card_id: int, x: float, y: float, towers=None) -> bool:
        """Make a play at screen tile (x, y); True if the simulator took it."""
        from royalegym import MatchSetup
        from royalegym.protocol import DeployCommand

        spawns = self._spawns()
        state = self.engine.state()
        tower_hp = self._tower_hp(towers, state)
        decks = [self.filler[:8], self.filler[:8]]
        decks[team] = [card_id] + [c for c in self.filler if c != card_id][:7]
        for dx, dy in NUDGES:
            tx = min(max(int(round(x)) + dx, 0), 17)
            ty = min(max(int(round(y)) + dy, 0), 31)
            try:
                self.engine.reset(self.seed, MatchSetup(
                    decks=decks, shuffle=0, start_tick=self.tick, elixir_milli=[10000, 10000],
                    tower_hp=tower_hp, spawns=spawns))
            except ValueError:
                self.engine.reset(self.seed, MatchSetup(
                    decks=decks, shuffle=0, start_tick=self.tick, elixir_milli=[10000, 10000],
                    spawns=spawns))
            self.running = True
            res = self.engine.step([DeployCommand(team=team, hand_slot=0,
                                                  x=int((tx + 0.5) * self.TILE),
                                                  y=int((ty + 0.5) * self.TILE))], 1)
            self.tick += 1
            if res and res[0].status == 0:
                self.played += 1
                return True
            self.tick -= 1
        self.refused += 1
        return False
