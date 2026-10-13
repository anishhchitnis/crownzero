"""DAgger: let the student play, and have the search bot label every position it reaches.

Plain imitation only shows the student positions the teacher gets into. Once the student
plays on its own it makes small mistakes, lands in positions the teacher never would,
and has no idea what to do there. DAgger fixes that: the student drives (with the
teacher's move played a fraction ``beta`` of the time), and every position the student
reaches is labelled with the move the teacher would have made.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from royalegym import ClashParallelEnv, DefaultStateMutator, RustEngine, Winner

from .arena import Scripted
from .board import NOOP
from .datagen import SCRIPTED, GameRecorder
from .decks import pick_deck
from .features import from_sim
from .search_bot import SearchBot
from .student_bot import StudentBot


def play_and_label(env, game_id, rng, out_dir: Path, student: StudentBot, *, beta=0.15,
                   noop_keep=0.25, p_vs_student=0.35, p_vs_teacher=0.25):
    cards = {c.card_id: c for c in env.engine.cards()}
    arena = env.engine.arena()
    decks = [pick_deck(rng), pick_deck(rng)]
    env.state_mutator = DefaultStateMutator(decks=decks)
    teacher = SearchBot()
    r = rng.random()
    if r < p_vs_student:
        learner_seats, opp = [0, 1], None
    else:
        learner_seats = [int(rng.integers(2))]
        opp = teacher if r < p_vs_student + p_vs_teacher else \
            Scripted(SCRIPTED[int(rng.integers(len(SCRIPTED)))])
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
            if seat in learner_seats:
                if not np.asarray(mask)[1:].any():
                    actions[agent] = NOOP
                    continue
                label = int(teacher.act(env, seat, o, mask, rng))
                own = int(student.act(env, seat, o, mask, rng))
                actions[agent] = label if rng.random() < beta else own
                if label != NOOP or own != NOOP or rng.random() < noop_keep:
                    # keep every position where either of them wanted to play
                    w = 1.0 if (label != NOOP or own != NOOP) else 1.0 / noop_keep
                    rec.add(from_sim(state, seat, cards, arena), mask, label, w, seat)
            else:
                actions[agent] = int(opp.act(env, seat, o, mask, rng))
        obs, _, term, trunc, _ = env.step(actions)
        truncated_out = any(trunc.values()) and not any(term.values())
    s = env.battle_state
    outcome = {}
    for seat in (0, 1):
        if truncated_out or not s.game_over or s.winner == Winner.DRAW:
            outcome[seat] = 0.0
        else:
            outcome[seat] = 1.0 if s.winner == seat else -1.0
    meta = dict(game_id=game_id, decks=decks, learner_seats=learner_seats,
                opponent=getattr(opp, "name", "student"), winner=int(s.winner),
                crowns=[p.crowns for p in s.players], ticks=s.tick,
                seconds=round(time.perf_counter() - t0, 1))
    n = rec.save(out_dir / f"game_{game_id:06d}.npz", outcome, meta)
    return n, meta


def worker(worker_id, n_workers, start, stop, out_dir, student_path, seed=11, beta=0.15):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    env = ClashParallelEnv(engine=RustEngine())
    student = StudentBot(student_path)
    for gid in range(start + worker_id, stop, n_workers):
        if (out / f"game_{gid:06d}.npz").exists():
            continue
        rng = np.random.default_rng(seed * 1_000_003 + gid)
        n, meta = play_and_label(env, gid, rng, out, student, beta=beta)
        print(f"game {gid}: {n} samples, crowns {meta['crowns']}, {meta['seconds']}s, "
              f"learner seats {meta['learner_seats']} vs {meta['opponent']}", flush=True)
