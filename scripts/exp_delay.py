"""Does the real-game brain gain from knowing the client's command delay and screen lag?

Both bots are the real-game brain (live search on a board rebuilt from what the screen shows,
REAL_GAME_SEARCH settings), playing a battle where plays land ``DELAY`` ticks late, as on the
live client, and deciding on a view ``STALE`` steps old, as from a screenshot.

    python scripts/exp_delay.py [games] [stale_steps]
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crownzero.arena import match  # noqa: E402
from crownzero.real.play import REAL_GAME_SEARCH  # noqa: E402

GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 30
STALE = int(sys.argv[2]) if len(sys.argv) > 2 else 0
DELAY = 21
OUT = Path(__file__).resolve().parents[1] / "results" / "evals.jsonl"

aware = dict(REAL_GAME_SEARCH, command_delay=DELAY, lag_steps=STALE, stale_steps=STALE)
plain = dict(REAL_GAME_SEARCH, stale_steps=STALE)
t0 = time.perf_counter()
m = match(("live-search", aware), ("live-search", plain), games=GAMES, seed=7000,
          deck="random", delay=DELAY, name_a=f"delay-aware(lag {STALE})", name_b="v4.7")
print(m.summary(), f"| {time.perf_counter() - t0:.0f}s", flush=True)
with OUT.open("a") as f:
    f.write(json.dumps({"a": m.a, "b": m.b, "decks": "random", "seed": 7000, "delay": DELAY,
                        "stale_steps": STALE, "games": m.games}) + "\n")
