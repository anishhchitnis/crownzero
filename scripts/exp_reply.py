"""Opponent answers inside the rollouts vs the opponent standing still.

    python scripts/exp_reply.py A_REPLY B_SPEC GAMES SEED
A_REPLY: "defender" or a network path. B_SPEC: "live" (plain live search) or "search".
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crownzero.arena import match
reply, b, games, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
b_spec = "live-search" if b == "live" else b
name_a = f"live-reply-{Path(reply).stem}"
m = match(("live-search", {"reply": reply}), b_spec, games=games, seed=seed, deck="random",
          name_a=name_a, name_b=b if b != "live" else "live-search")
print(m.summary(), flush=True)
with open(Path(__file__).resolve().parents[1] / "results" / "evals.jsonl", "a") as f:
    f.write(json.dumps({"a": m.a, "b": m.b, "decks": "random", "seed": seed, "games": m.games}) + "\n")
