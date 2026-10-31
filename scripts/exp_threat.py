"""Keep more elixir back while an enemy push is coming (reserve_per_threat).

The real-game brain (live search, REAL_GAME_SEARCH) with and without the threat reserve,
in battles with the live client's command delay and a view one step old.

    python scripts/exp_threat.py [games] [per_threat]
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crownzero.arena import match  # noqa: E402
from crownzero.real.play import REAL_GAME_SEARCH  # noqa: E402

GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 30
K = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
DELAY = 21
OUT = Path(__file__).resolve().parents[1] / "results" / "evals.jsonl"
base = dict(REAL_GAME_SEARCH, stale_steps=1)
t0 = time.perf_counter()
m = match(("live-search", dict(base, reserve_per_threat=K)), ("live-search", base),
          games=GAMES, seed=10_000, deck="random", delay=DELAY,
          name_a=f"threat-reserve {K}", name_b="v5.2")
print(m.summary(), f"| {time.perf_counter() - t0:.0f}s", flush=True)
with OUT.open("a") as f:
    f.write(json.dumps({"a": m.a, "b": m.b, "decks": "random", "seed": 10_000, "delay": DELAY,
                        "games": m.games}) + "\n")
