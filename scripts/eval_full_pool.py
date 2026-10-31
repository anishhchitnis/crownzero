"""The search bot on the WHOLE card pool (evolutions, heroes, champions), both seats.

1. search vs each scripted bot, every game a fresh pair of decks from fulldecks.Pool
   (40% META archetypes, 60% random legal decks);
2. A/B: search with ability buttons vs the same search that never presses them, on the
   META decks that hold a hero or a champion (elsewhere the two bots are the same).

    python scripts/eval_full_pool.py [games_per_opponent] [ab_games]
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crownzero.arena import match  # noqa: E402

GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 10
AB_GAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 40
OUT = Path(__file__).resolve().parents[1] / "results" / "full_pool.jsonl"
OUT.parent.mkdir(exist_ok=True)


def log(m, **extra):
    print(m.summary(), flush=True)
    with OUT.open("a") as f:
        f.write(json.dumps({"a": m.a, "b": m.b, **extra,
                            "games": m.games}) + "\n")


for opp in ("random", "first-affordable", "defend", "push", "patient"):
    t0 = time.perf_counter()
    log(match("search", opp, games=GAMES, seed=5000, deck="full"), decks="full",
        secs=time.perf_counter() - t0)

t0 = time.perf_counter()
log(match(("search", {}), ("search", {"abilities": False}), games=AB_GAMES, seed=6000,
          deck="special", name_a="abilities", name_b="no-abilities"),
    decks="special", secs=time.perf_counter() - t0)
