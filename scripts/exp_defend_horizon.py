"""Adaptive look-ahead (10 s when enemies are on our side, else 6 s) vs fixed 6 s."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crownzero.arena import match
games = int(sys.argv[1]) if len(sys.argv) > 1 else 20
m = match(("live-search", {"defend_horizon_ticks": 200}), "live-search",
          games=games, seed=8000, deck="random", name_a="live-defend200", name_b="live-fixed120")
print(m.summary(), flush=True)
with open(Path(__file__).resolve().parents[1] / "results" / "evals.jsonl", "a") as f:
    f.write(json.dumps({"a": m.a, "b": m.b, "decks": "random", "seed": 8000, "games": m.games}) + "\n")
