"""Elixir reserve (extra value on the first 3 banked elixir) vs none."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crownzero.arena import match
reserve = float(sys.argv[1]); games = int(sys.argv[2]); seed = int(sys.argv[3])
m = match(("live-search", {"reply": "defender", "reserve": reserve}),
          ("live-search", {"reply": "defender"}),
          games=games, seed=seed, deck="random", name_a=f"live-reserve{reserve}", name_b="live-v4.4")
print(m.summary(), flush=True)
g = m.games
print("3-crowns", sum(x["a_crowns"] == 3 for x in g), "vs", sum(x["b_crowns"] == 3 for x in g),
      "| crowns", sum(x["a_crowns"] for x in g), "vs", sum(x["b_crowns"] for x in g), flush=True)
with open(Path(__file__).resolve().parents[1] / "results" / "evals.jsonl", "a") as f:
    f.write(json.dumps({"a": m.a, "b": m.b, "decks": "random", "seed": seed, "games": m.games}) + "\n")
