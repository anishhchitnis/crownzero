"""Evaluate one bot against several opponents.

    eval_bot.py --a student:models/student.pt --b random defend search --games 40 --decks random
A bot is a scripted name, "search", or "student:PATH".
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crownzero.arena import DEFAULT_DECK, match  # noqa: E402
from crownzero.decks import CURATED  # noqa: E402


def spec(s):
    if s.startswith("student:"):
        return ("student", {"path": s.split(":", 1)[1]})
    return s


def label(s):
    return Path(s.split(":", 1)[1]).stem if s.startswith("student:") else s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", nargs="+", required=True)
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--seed", type=int, default=2000)
    ap.add_argument("--decks", default="random", help="random, default, or a curated name")
    ap.add_argument("--out", default=str(ROOT / "results" / "evals.jsonl"))
    args = ap.parse_args()
    deck = ("random" if args.decks == "random" else
            DEFAULT_DECK if args.decks == "default" else tuple(CURATED[args.decks]))
    for b in args.b:
        t0 = time.perf_counter()
        m = match(spec(args.a), spec(b), games=args.games, seed=args.seed, deck=deck,
                  name_a=label(args.a), name_b=label(b))
        print(m.summary(), f"| decks {args.decks} | {time.perf_counter() - t0:.0f}s", flush=True)
        with open(args.out, "a") as f:
            f.write(json.dumps({"a": m.a, "b": m.b, "decks": args.decks, "seed": args.seed,
                                "games": m.games}) + "\n")
