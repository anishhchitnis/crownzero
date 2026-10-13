"""Play bots against each other and say who is better, with an error bar.

Same rules as royalegym.evaluate (both seats, the same seeds each way round, Wilson
interval, unfinished games left out), but bots get the env and their seat, which a
search bot needs, and games run in parallel across CPU cores.
"""
from __future__ import annotations

import multiprocessing as mp
import time
from dataclasses import dataclass, field

import numpy as np
from royalegym import ClashParallelEnv, DefaultStateMutator, RustEngine, Winner, ladder
from royalegym.evaluate import wilson

AGENTS = ("blue", "red")
TEAM = {"blue": 0, "red": 1}
DEFAULT_DECK = ("Knight", "Archer", "Giant", "Minions", "Fireball", "Cannon", "Zap", "Musketeer")


class Scripted:
    """Adapter: a royalegym scripted opponent behind the (env, team, obs, mask, rng) call."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._bot = dict(ladder())[name]

    def act(self, env, team, obs, mask, rng):
        return self._bot.act(obs, mask, rng)


def make_bot(spec):
    """A bot from a picklable spec: a scripted name, or ("search", {kwargs})."""
    name, kwargs = (spec, {}) if isinstance(spec, str) else spec
    if name == "search":
        from .search_bot import SearchBot
        kw = dict(kwargs)
        if "reply_model" in kw:  # two-ply search: the opponent answers with this network
            from .student_bot import StudentBot
            kw["opponent"] = StudentBot(kw.pop("reply_model"))
        return SearchBot(**kw)
    if name == "live-search":  # search on a board rebuilt from screen-reader-level info
        from .live_search import LiveSearchBot
        return LiveSearchBot(kwargs or None)
    if name == "student":
        from .student_bot import StudentBot
        return _cached_student(kwargs.get("path"), kwargs.get("play_threshold", 0.5))
    return Scripted(name)


_STUDENTS = {}


def _cached_student(path, play_threshold):
    key = (path, play_threshold)
    if key not in _STUDENTS:
        from .student_bot import StudentBot
        _STUDENTS[key] = StudentBot(path, play_threshold=play_threshold)
    return _STUDENTS[key]


_ENV = {}


def decks_for(deck_spec, seed: int):
    """[blue deck, red deck] as names. A tuple of 8 names = mirror match with that deck;
    "random" = each side a random legal deck, drawn from the seed (so both seat halves
    of a match replay the same pairings)."""
    if deck_spec == "random":
        from .decks import pick_deck
        rng = np.random.default_rng(10_000 + seed)
        return [pick_deck(rng), pick_deck(rng)]
    return [list(deck_spec), list(deck_spec)]


def make_env(deck_names=DEFAULT_DECK) -> ClashParallelEnv:
    if "env" not in _ENV:
        _ENV["env"] = ClashParallelEnv(engine=RustEngine())
    env = _ENV["env"]
    decks = deck_names if isinstance(deck_names, list) else [list(deck_names)] * 2
    env.state_mutator = DefaultStateMutator(decks=decks)
    return env


def play_game(blue, red, seed: int, deck_names=DEFAULT_DECK):
    """One battle. Returns (winner or None if cut short, ticks, crowns, seconds per bot)."""
    env = make_env(deck_names)
    bots = {"blue": blue, "red": red}
    rngs = {a: np.random.default_rng(seed * 2 + i) for i, a in enumerate(AGENTS)}
    spent = {"blue": 0.0, "red": 0.0}
    obs, _ = env.reset(seed=seed)
    truncated_out = False
    while env.agents:
        actions = {}
        for a in env.agents:
            t0 = time.perf_counter()
            actions[a] = int(bots[a].act(env, TEAM[a], obs[a], obs[a]["action_mask"], rngs[a]))
            spent[a] += time.perf_counter() - t0
        obs, _, terminated, truncated, _ = env.step(actions)
        truncated_out = any(truncated.values()) and not any(terminated.values())
    s = env.battle_state
    winner = None if truncated_out or not s.game_over else int(s.winner)
    return winner, s.tick, [p.crowns for p in s.players], spent


def _job(args):
    spec_a, spec_b, seed, a_is_blue, deck = args
    a, b = make_bot(spec_a), make_bot(spec_b)
    blue, red = (a, b) if a_is_blue else (b, a)
    decks = decks_for(deck, seed)
    if not a_is_blue:  # A keeps its deck when it changes seat
        decks = decks[::-1]
    winner, ticks, crowns, spent = play_game(blue, red, seed, decks)
    a_seat = "blue" if a_is_blue else "red"
    if winner is None:
        outcome = "unfinished"
    elif winner == Winner.DRAW:
        outcome = "draw"
    else:
        outcome = "win" if (winner == Winner.BLUE) == a_is_blue else "loss"
    a_crowns = crowns[0] if a_is_blue else crowns[1]
    b_crowns = crowns[1] if a_is_blue else crowns[0]
    return dict(seed=seed, a_seat=a_seat, outcome=outcome, ticks=ticks,
                a_crowns=a_crowns, b_crowns=b_crowns, a_secs=spent[a_seat])


@dataclass
class Match:
    a: str
    b: str
    games: list = field(default_factory=list)

    def count(self, outcome, seat=None):
        return sum(g["outcome"] == outcome and (seat is None or g["a_seat"] == seat)
                   for g in self.games)

    @property
    def wins(self):
        return self.count("win")

    @property
    def losses(self):
        return self.count("loss")

    @property
    def draws(self):
        return self.count("draw")

    @property
    def win_rate(self):
        d = self.wins + self.losses
        return self.wins / d if d else 0.5

    @property
    def interval(self):
        return wilson(self.wins, self.wins + self.losses, 0.95)

    @property
    def verdict(self):
        lo, hi = self.interval
        return self.a if lo > 0.5 else self.b if hi < 0.5 else "too close"

    def summary(self) -> str:
        lo, hi = self.interval
        three = sum(g["a_crowns"] == 3 for g in self.games)
        secs = sum(g["a_secs"] for g in self.games) / max(len(self.games), 1)
        seats = " ".join(f"{s}:{self.count('win', s)}-{self.count('loss', s)}"
                         for s in ("blue", "red"))
        return (f"{self.a} vs {self.b}: {self.wins}-{self.losses}-{self.draws} "
                f"win {self.win_rate:.0%} [{lo:.0%}, {hi:.0%}] -> {self.verdict} "
                f"| 3-crowns {three} | seats {seats} | {secs:.1f}s thinking/game")


def match(spec_a, spec_b, games: int = 40, seed: int = 0, deck=DEFAULT_DECK,
          workers: int | None = None, name_a: str | None = None, name_b: str | None = None):
    per_seat = (games + 1) // 2
    deck = deck if deck == "random" else tuple(deck)
    jobs = [(spec_a, spec_b, seed + g, a_is_blue, deck)
            for a_is_blue in (True, False) for g in range(per_seat)]
    workers = workers or mp.cpu_count()
    with mp.get_context("fork").Pool(workers) as pool:
        results = pool.map(_job, jobs, chunksize=1)
    name = lambda s: s if isinstance(s, str) else s[0]  # noqa: E731
    return Match(name_a or name(spec_a), name_b or name(spec_b), results)
