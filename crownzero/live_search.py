"""Live search: rebuild what is on screen inside the simulator, then search it.

The search bot beats the trained networks 20-0, but it needs a simulator state to think
in. In the real game there is none, only what the screen reader sees. So every decision,
this module builds a fresh simulated battle from a View (the same View the network uses:
unit types and tiles, our hand, elixir, tower health, the clock), and lets the search bot
fast-forward its candidate plays in it, like a chess engine thinking about the position on
the board.

What the rebuild cannot know is filled in plainly: every unit at full health and already
deployed, the enemy's hand and elixir unknown (the search bot never reads them anyway: in
its rollouts the enemy stands still), units of cards the simulator does not model replaced
by a calibrated card with the same role.
"""
from __future__ import annotations

import numpy as np
from royalegym import ClashParallelEnv, MatchSetup, RustEngine
from royalegym.protocol import SpawnSpec, spawn_violation
from royalegym.state_mutator import StateMutator

from .board import NOOP, NX, NY, TILE
from .decks import CURATED
from .features import BUILD, CARD_TYPES, FLY, TARGETS_BUILDINGS, View
from .search_bot import SearchBot

SPAWN_AS = {"GoblinBarrel": "Goblins"}  # a spell's units are spawned as the troop card


def proxy_card(t: int, flags: int) -> str:
    """The calibrated card a unit is rebuilt as."""
    if t >= 0:
        name = CARD_TYPES[t]
        return SPAWN_AS.get(name, name)
    if flags & BUILD:
        return "Cannon"
    if flags & FLY:
        return "Minions"
    if flags & TARGETS_BUILDINGS:
        return "HogRider"
    return "Knight"


class EnemyDeck:
    """The opponent's cards, as far as the screen has shown them, for the rebuilt battle.

    A card is "seen" when a unit of it appears on the enemy side that was not there a
    moment before; the order cards were seen in is the order they were played, which is
    the cycle: the four played last are at the back of the queue, the rest are in hand.
    Spells leave no unit and stay unknown; unknown slots are filled from a generic deck.
    Before this, the rebuilt opponent always held Giant, Musketeer, Minions and Knight,
    so the look-ahead's defender answered every push with them whatever the real deck."""

    def __init__(self, known_names, generic):
        self.known_names = set(known_names)
        self.generic = list(generic)
        self.played: list[str] = []  # most recent last, each card once
        self._present: set = set()

    def update(self, view: View) -> None:
        names = view.unit_names if len(view.unit_names) == len(view.units) else []
        now = {n for (own, *_), n in zip(view.units, names) if not own and n}
        for n in sorted(now - self._present):
            n = SPAWN_AS_CARD.get(n, n)
            if n not in self.known_names:
                continue
            if n in self.played:
                self.played.remove(n)
            self.played.append(n)
        self._present = now

    def deck(self) -> list[str]:
        seen = self.played[-8:]
        recent = seen[-4:]  # just played: at the back of the queue, not in hand
        hand_side = [c for c in seen if c not in recent]
        fill = [c for c in self.generic if c not in seen]
        cards = (hand_side + fill)[: 8 - len(recent)] + recent
        return cards


SPAWN_AS_CARD = {"Goblins": "GoblinBarrel"}  # rebuilt as Goblins; ignore: not a played card


class _Fixed(StateMutator):
    setup: MatchSetup | None = None

    def build(self, rng, cards):
        return self.setup


class LiveSearch:
    def __init__(self, deck: list[str], search_kwargs: dict | None = None, seed: int = 0):
        self.engine = RustEngine()
        self.catalogue = list(self.engine.cards())
        self.by_name = {c.name: c.card_id for c in self.catalogue}
        self.arena = self.engine.arena()
        self.mutator = _Fixed()
        kw = dict(search_kwargs or {})
        # THE LIVE CLIENT'S COMMAND DELAY: a play lands 21-22 ticks (~1.1 s) after it is
        # made (RoyaleSim's measurement). With ``command_delay`` the rebuilt battle has the
        # same delay, so every rollout sees the board our play will actually meet.
        # ``lag_steps``: the screenshot is already old when we decide (capture + detection,
        # ~0.2-0.5 s), so the rebuilt board is run forward this many 0.5 s steps first.
        self.delay = int(kw.pop("command_delay", 0))
        self.lag_steps = int(kw.pop("lag_steps", 0))
        self.env = ClashParallelEnv(engine=self.engine, state_mutator=self.mutator,
                                    command_delay_ticks=self.delay or None)
        # rebuild the opponent with the elixir we counted for them (View.enemy_elixir)
        # rather than 5. Off: in the simulator, with the reserve below, it lost 15-25.
        self.use_enemy_elixir = kw.pop("count_enemy_elixir", False)
        if "reply" in kw:  # the opponent answers our play inside the rollouts
            from .opponent_model import make_opponent
            kw["opponent"] = make_opponent(kw.pop("reply"))
        if isinstance(kw.get("prior"), (str, bytes)) or hasattr(kw.get("prior"), "__fspath__"):
            from .replays.prior import ProPrior  # a path to a trained placement prior
            kw["prior"] = ProPrior(kw["prior"])
        track = kw.pop("track_enemy_deck", False)
        self.search = SearchBot(**kw)
        self.deck = list(deck)
        self.enemy_deck = [self.by_name[n] for n in CURATED["giant"]]
        # follow the opponent's real cards instead of a fixed generic hand
        self.tracker = (EnemyDeck(self.by_name, CURATED["giant"])
                        if track else None)
        self.rng = np.random.default_rng(seed)
        # full tower health at the simulator's levels
        self.mutator.setup = MatchSetup(decks=[self._ids(self.deck), self.enemy_deck])
        self.env.reset(seed=0)
        st = self.env.battle_state
        self.king_max, self.princess_max = st.players[0].tower_max_hp[0], st.players[0].tower_max_hp[1]
        self.regular_ticks = st.regular_ticks
        self.skipped = 0

    def _ids(self, names):
        return [self.by_name[n] for n in names]

    def _deck_order(self, view: View) -> list[str]:
        """Our deck with the hand first (slot order) and the next card fifth, so the
        rebuilt battle deals exactly the hand on screen."""
        rest = list(self.deck)
        order = []
        names = list(view.hand_names) + [view.next_name]
        if not any(names):  # a View without names: fall back to the 18 card types
            names = [CARD_TYPES[t] if t >= 0 else None for t in list(view.hand) + [view.next_card]]
        for name in names:
            if name in rest:
                rest.remove(name)
                order.append(name)
            else:
                order.append(None)
        filled = [n if n is not None else rest.pop(0) for n in order]
        return filled + rest

    def _spawn(self, team, name, tx, ty):
        cid = self.by_name.get(name)
        if cid is None:
            return None
        for dy in (0, 1, -1, 2, -2):  # nudge off water / walls if the tile is not standable
            y = min(max(ty + dy, 0), NY - 1)
            spec = SpawnSpec(team=team, card_id=cid, x=int((tx + 0.5) * TILE),
                             y=int((y + 0.5) * TILE))
            if spawn_violation(self.arena, self.catalogue, spec) is None:
                return spec
        return None

    def setup_for(self, view: View) -> MatchSetup:
        spawns = []
        names = view.unit_names if len(view.unit_names) == len(view.units) else [None] * len(view.units)
        for (own, t, x, y, flags, _), name in zip(view.units, names):
            team = 0 if own else 1
            spec = None
            if name is not None and name in self.by_name:
                spec = self._spawn(team, SPAWN_AS.get(name, name), int(x), int(y))
            if spec is None:  # unknown or refused: a calibrated card with the same role
                spec = self._spawn(team, proxy_card(t, flags), int(x), int(y))
            if spec is None:
                self.skipped += 1
            else:
                spawns.append(spec)
        k, p = self.king_max, self.princess_max
        tw = view.towers  # own frame: [own king, own left, own right, en king, en left, en right]
        # Red's tower slot 1 is Red's own left, which is our right.
        tower_hp = [[max(1, round(tw[0] * k)), round(tw[1] * p), round(tw[2] * p)],
                    [max(1, round(tw[3] * k)), round(tw[5] * p), round(tw[4] * p)]]
        start = int(view.clock * self.regular_ticks) + (1 if view.overtime else 0)
        # The simulator refuses plays in the first 90 ticks (the start-of-battle countdown).
        # Once we see the battle screen we can play, so never rebuild inside that window.
        start = max(start, 90)
        return MatchSetup(decks=[self._ids(self._deck_order(view)), self.enemy_deck],
                          shuffle=0, start_tick=start,
                          elixir_milli=[int(view.elixir * 1000),
                                        int((view.enemy_elixir if self.use_enemy_elixir
                                             and view.enemy_elixir is not None else 5.0)
                                            * 1000)],
                          tower_hp=tower_hp, spawns=spawns)

    def decide(self, view: View, allowed_slots=None) -> int:
        """Best action id (own frame: slot x row x column, 0 = wait) for this View.
        ``allowed_slots``: only consider these hand slots (the ones lit on screen)."""
        if self.tracker is not None:
            self.tracker.update(view)
            self.enemy_deck = self._ids(self.tracker.deck())
        self.mutator.setup = self.setup_for(view)
        obs, _ = self.env.reset(seed=int(self.rng.integers(1 << 30)))
        for _ in range(self.lag_steps):  # the world moved on while we looked
            if not self.env.agents:
                break
            obs, *_ = self.env.step({a: NOOP for a in self.env.agents})
        if not self.env.agents:
            return NOOP
        o = dict(obs["blue"])
        if allowed_slots is not None:
            planes = np.array(o["mask_planes"], copy=True)
            mask = np.array(o["action_mask"], copy=True)
            cells = planes.shape[1] * planes.shape[2]
            for s in range(planes.shape[0]):
                if s not in allowed_slots:
                    planes[s] = 0
                    mask[1 + s * cells: 1 + (s + 1) * cells] = 0
            o["mask_planes"], o["action_mask"] = planes, mask
        return int(self.search.act(self.env, 0, o, o["action_mask"], self.rng))


class LiveSearchBot:
    """For testing in the simulator: plays the real env, but thinks only about a board
    rebuilt from what the screen reader would see (no unit health, tile positions)."""

    name = "live-search"

    def __init__(self, search_kwargs=None):
        kw = dict(search_kwargs or {})
        # tell the rebuild the opponent's elixir (in the simulator we know it exactly; the
        # real game estimates it, perception.EnemyElixir)
        self.count_enemy = kw.get("count_enemy_elixir", False)
        # SIMULATING THE SCREEN'S LAG: decide on the view from this many 0.5 s steps ago,
        # as the real bot does on a screenshot that is already old (for testing lag_steps)
        self.stale = int(kw.pop("stale_steps", 0))
        # TESTING A MISREAD: enemy units (sim name, own-frame tile x, y) added to every view,
        # as the screen reader added the enemy's tower troops (recorded 2026-10-30)
        self.phantoms = list(kw.pop("phantoms", []))
        self.plays = {}  # card name -> times played, for the experiments
        self._views = []
        self.search_kwargs = kw or None
        self.live = None
        self.cards = None

    def act(self, env, team, obs, mask, rng):
        from .features import from_sim

        mask = np.asarray(mask).astype(bool)
        if not mask[1:].any():
            return NOOP
        if self.cards is None:
            self.cards = {c.card_id: c for c in env.engine.cards()}
            self.arena = env.engine.arena()
        st = env.battle_state
        deck = [self.cards[c].name for c in st.players[team].deck]
        if self.live is None or sorted(self.live.deck) != sorted(deck):
            self.live = LiveSearch(deck, self.search_kwargs)
        view = from_sim(st, team, self.cards, self.arena)
        view.elixir = float(np.floor(view.elixir + 1e-6))  # the screen shows whole elixir
        if self.count_enemy:
            view.enemy_elixir = st.players[1 - team].elixir_milli / 1000
        for name, x, y in self.phantoms:
            view.units.append((0, -1, x, y, 0, 1.0))
            view.unit_names.append(name)
        if self.stale:
            self._views.append(view)
            if len(self._views) <= self.stale:
                return NOOP
            view = self._views.pop(0)
        a = self.live.decide(view)
        if a != NOOP and mask[a] and a <= 4 * NX * NY:
            card = view.hand_names[(a - 1) // (NX * NY)]
            self.plays[card] = self.plays.get(card, 0) + 1
        return a if mask[a] else NOOP
