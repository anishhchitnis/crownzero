"""Does the real-game brain gain from following the opponent's real cards (EnemyDeck)?

Both bots: live search on a rebuilt board, REAL_GAME_SEARCH settings, the client's command
delay, a view one step old. One tracks the opponent's played cards for the look-ahead's
defender; the other holds the generic hand.

    python scripts/exp_track.py [games]
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
m = match(("live-search", dict(base, track_enemy_deck=True)), ("live-search", base),
          games=GAMES, seed=8000, deck="random", delay=DELAY,
          name_a="track-enemy-deck", name_b="v4.7")
print(m.summary(), f"| {time.perf_counter() - t0:.0f}s", flush=True)
with OUT.open("a") as f:
    f.write(json.dumps({"a": m.a, "b": m.b, "decks": "random", "seed": 8000, "delay": DELAY,
                        "games": m.games}) + "\n")
