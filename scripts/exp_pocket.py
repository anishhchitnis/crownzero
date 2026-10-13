"""Pocket placements (in front of the King Tower once a princess tower falls) vs none."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crownzero.arena import match
games = int(sys.argv[1]) if len(sys.argv) > 1 else 40
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 11000
m = match(("live-search", {"reply": "defender"}),
          ("live-search", {"reply": "defender", "pocket_spots": False}),
          games=games, seed=seed, deck="random", name_a="live-pockets", name_b="live-v4.3")
print(m.summary(), flush=True)
with open(Path(__file__).resolve().parents[1] / "results" / "evals.jsonl", "a") as f:
    f.write(json.dumps({"a": m.a, "b": m.b, "decks": "random", "seed": seed, "games": m.games}) + "\n")
