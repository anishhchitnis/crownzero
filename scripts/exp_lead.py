"""Aiming spells ahead of enemy troops (lead_spells), with the live client's command delay.

    python scripts/exp_lead.py [games]
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crownzero.arena import match  # noqa: E402
from crownzero.real.play import REAL_GAME_SEARCH  # noqa: E402

GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 30
DELAY = 21
OUT = Path(__file__).resolve().parents[1] / "results" / "evals.jsonl"
base = dict(REAL_GAME_SEARCH, stale_steps=1)  # the shipped brain; the world has delay + lag
t0 = time.perf_counter()
m = match(("live-search", dict(base, lead_spells=True)), ("live-search", base),
          games=GAMES, seed=9000, deck="random", delay=DELAY,
          name_a="lead-spells", name_b="v4.7")
print(m.summary(), f"| {time.perf_counter() - t0:.0f}s", flush=True)
with OUT.open("a") as f:
    f.write(json.dumps({"a": m.a, "b": m.b, "decks": "random", "seed": 9000, "delay": DELAY,
                        "games": m.games}) + "\n")
