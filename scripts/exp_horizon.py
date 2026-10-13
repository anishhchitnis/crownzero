"""Does the live search defend better with a longer look-ahead? 200 vs 120 ticks."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crownzero.arena import match
m = match(("live-search", {"horizon_ticks": 200}), ("live-search", {"horizon_ticks": 120}),
          games=20, seed=6000, deck="random", name_a="live-h200", name_b="live-h120")
print(m.summary(), flush=True)
with open(Path(__file__).resolve().parents[1] / "results" / "evals.jsonl", "a") as f:
    f.write(json.dumps({"a": m.a, "b": m.b, "decks": "random", "seed": 6000, "games": m.games}) + "\n")
