"""Record the search bot's decisions as training data for the student network.

Each game: random or curated decks for both sides, the search bot in one or both seats,
a scripted bot otherwise. For every decision the search bot makes, store what the student
will see (a View), the legal-move mask, and the search bot's choice. A small share of the
time the move actually played is a random legal one (the label stays the search bot's),
so the data also covers the messy positions a learner's own mistakes lead to.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from royalegym import ClashParallelEnv, DefaultStateMutator, RustEngine, Winner

from .arena import Scripted
from .board import NOOP
from .decks import CURATED, pick_deck, random_deck  # noqa: F401
from .features import from_sim, view_rows
from .search_bot import SearchBot

SCRIPTED = ["random", "first-affordable", "defend", "push", "patient"]


class GameRecorder:
    def __init__(self):
        self.units, self.raws, self.masks, self.actions, self.weights, self.seats = \
            [], [], [], [], [], []

    def add(self, view, mask, action, weight, seat):
        n = len(self.raws)
        self.units.append(view_rows(view, n))
        self.raws.append(view.raw())
        self.masks.append(np.packbits(np.asarray(mask, np.uint8)))
        self.actions.append(action)
        self.weights.append(weight)
        self.seats.append(seat)

    def save(self, path: Path, outcome_by_seat: dict[int, float], meta: dict):
        if not self.raws:
            return 0
        np.savez_compressed(
            path,
            units=np.concatenate(self.units) if self.units else np.zeros((0, 7), np.float32),
            raws=np.stack(self.raws),
            masks=np.stack(self.masks),
            actions=np.array(self.actions, np.int16),
            weights=np.array(self.weights, np.float32),
            outcomes=np.array([outcome_by_seat[s] for s in self.seats], np.float32),
            seats=np.array(self.seats, np.int8),
            meta=np.array(json.dumps(meta)),
        )
        return len(self.raws)


def play_and_record(env, game_id: int, rng, out_dir: Path, *, p_selfplay=0.35,
                    noop_keep=0.25, p_explore=0.03, teacher_kwargs=None):
    cards = {c.card_id: c for c in env.engine.cards()}
    arena = env.engine.arena()
    decks = [pick_deck(rng), pick_deck(rng)]
    env.state_mutator = DefaultStateMutator(decks=decks)
    selfplay = rng.random() < p_selfplay
    teacher_seats = [0, 1] if selfplay else [int(rng.integers(2))]
    bots = {}
    for seat in (0, 1):
        if seat in teacher_seats:
            bots[seat] = SearchBot(**(teacher_kwargs or {}))
        else:
            bots[seat] = Scripted(SCRIPTED[int(rng.integers(len(SCRIPTED)))])
    rec = GameRecorder()
    agents = {0: "blue", 1: "red"}
    obs, _ = env.reset(seed=int(rng.integers(1 << 31)))
    truncated_out = False
    t0 = time.perf_counter()
    while env.agents:
        state = env.battle_state
        actions = {}
        for seat, agent in agents.items():
            o = obs[agent]
            mask = o["action_mask"]
            a = int(bots[seat].act(env, seat, o, mask, rng))
            played = a
            if seat in teacher_seats:
                legal_plays = np.flatnonzero(mask[1:]) + 1
                if len(legal_plays) and rng.random() < p_explore:
                    played = int(rng.choice(legal_plays))
                if a != NOOP or rng.random() < noop_keep:
                    w = 1.0 if a != NOOP else 1.0 / noop_keep
                    rec.add(from_sim(state, seat, cards, arena), mask, a, w, seat)
            actions[agent] = played
        obs, _, term, trunc, _ = env.step(actions)
        truncated_out = any(trunc.values()) and not any(term.values())
    s = env.battle_state
    outcome = {}
    for seat in (0, 1):
        if truncated_out or not s.game_over or s.winner == Winner.DRAW:
            outcome[seat] = 0.0
        else:
            outcome[seat] = 1.0 if s.winner == seat else -1.0
    meta = dict(game_id=game_id, decks=decks, teacher_seats=teacher_seats,
                opponents={str(k): getattr(b, "name", "?") for k, b in bots.items()},
                winner=int(s.winner), crowns=[p.crowns for p in s.players], ticks=s.tick,
                seconds=round(time.perf_counter() - t0, 1))
    n = rec.save(out_dir / f"game_{game_id:06d}.npz", outcome, meta)
    return n, meta


def worker(worker_id: int, n_workers: int, start: int, stop: int, out_dir: str, seed: int):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    env = ClashParallelEnv(engine=RustEngine())
    for gid in range(start + worker_id, stop, n_workers):
        if (out / f"game_{gid:06d}.npz").exists():
            continue
        rng = np.random.default_rng(seed * 1_000_003 + gid)
        n, meta = play_and_record(env, gid, rng, out)
        print(f"game {gid}: {n} samples, crowns {meta['crowns']}, {meta['seconds']}s, "
              f"teacher seats {meta['teacher_seats']} vs {meta['opponents']}", flush=True)
