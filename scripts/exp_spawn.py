"""Summoned units valued by their own size vs as the whole card (the Goblin Hut bug).

Mirror match with the deck from the 2026-10-13 recording (Goblin Hut, Witch-like spawners
are where the two differ; the 18-card random decks have none)."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crownzero.arena import match
games = int(sys.argv[1]) if len(sys.argv) > 1 else 20
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
deck = ("GoblinHut", "MiniPekka", "SpearGoblins", "Archer", "GoblinCage", "Minions",
        "Goblins", "Knight")
m = match(("live-search", {"reply": "defender"}),
          ("live-search", {"reply": "defender", "spawn_scaling": False}),
          games=games, seed=seed, deck=deck, name_a="live-spawnfix", name_b="live-v4.2")
print(m.summary(), flush=True)
with open(Path(__file__).resolve().parents[1] / "results" / "evals.jsonl", "a") as f:
    f.write(json.dumps({"a": m.a, "b": m.b, "decks": list(deck), "seed": seed, "games": m.games}) + "\n")
