"""Search bot vs every scripted bot: win rate with a 95% interval, both seats."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crownzero.arena import match  # noqa: E402

GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 20
OUT = Path(__file__).resolve().parents[1] / "results" / "search_vs_ladder.jsonl"
OUT.parent.mkdir(exist_ok=True)

for opp in ("random", "first-affordable", "defend", "push", "patient"):
    t0 = time.perf_counter()
    m = match("search", opp, games=GAMES, seed=1000)
    print(m.summary(), f"| {time.perf_counter() - t0:.0f}s", flush=True)
    with OUT.open("a") as f:
        f.write(json.dumps({"a": m.a, "b": m.b, "games": m.games}) + "\n")
