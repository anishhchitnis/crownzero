"""Pro placement prior (replays/prior.py) vs none, both with the real-game search settings.

    python scripts/exp_prior.py models/pro_prior.pt GAMES SEED [random|user] [weight]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crownzero.arena import match  # noqa: E402
from crownzero.real.play import REAL_GAME_SEARCH  # noqa: E402

prior, games, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
decks = sys.argv[4] if len(sys.argv) > 4 else "random"
weight = float(sys.argv[5]) if len(sys.argv) > 5 else 0.3
USER = ("Giant", "MiniPekka", "Fireball", "Archer", "GoblinHut", "Minions", "Musketeer", "Knight")
deck = "random" if decks == "random" else USER
m = match(("live-search", {**REAL_GAME_SEARCH, "prior": str(Path(prior).resolve()),
                           "prior_weight": weight}),
          ("live-search", dict(REAL_GAME_SEARCH)),
          games=games, seed=seed, deck=deck, name_a=f"live-prior{weight}", name_b="live-v4.5")
print(m.summary(), flush=True)
g = m.games
print("3-crowns", sum(x["a_crowns"] == 3 for x in g), "vs", sum(x["b_crowns"] == 3 for x in g),
      "| crowns", sum(x["a_crowns"] for x in g), "vs", sum(x["b_crowns"] for x in g), flush=True)
with open(Path(__file__).resolve().parents[1] / "results" / "evals.jsonl", "a") as f:
    f.write(json.dumps({"a": m.a, "b": m.b, "decks": decks, "seed": seed,
                        "games": m.games}) + "\n")
