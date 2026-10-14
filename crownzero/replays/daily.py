"""Today's replays in, a better placement prior out.

    python -m crownzero.replays.daily ~/Downloads/cr_replays

1. Reads every replay folder there that has no plays.jsonl yet (``--redo``: all of them).
2. Retrains the placement prior on the plays of every replay, scoring it first with the
   newest replay held out, then on all of them, into crownzero/models/pro_prior.pt.

The bot picks the new prior up the next time it starts (``--prior auto``).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

MODELS = Path(__file__).resolve().parents[2] / "models"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="the folder with the pro_* replay folders")
    ap.add_argument("--redo", action="store_true", help="re-read replays already read")
    ap.add_argument("--fps", type=float, default=4.0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--out", default=str(MODELS / "pro_prior.pt"))
    args = ap.parse_args()

    from .prior import train
    from .replay import extract_replay

    root = Path(args.folder).expanduser()
    folders = sorted(p for p in root.iterdir() if p.is_dir() and list(p.glob("part_*.mp4")))
    for f in folders:
        if (f / "plays.jsonl").exists() and not args.redo:
            continue
        recs = extract_replay(f, args.fps)
        with open(f / "plays.jsonl", "w") as out:
            for r in recs:
                out.write(json.dumps(r, default=float) + "\n")
    files = [str(f / "plays.jsonl") for f in folders if (f / "plays.jsonl").exists()]
    if not files:
        raise SystemExit(f"no replays in {root}")
    named = total = 0
    for f in files:
        for line in open(f):
            r = json.loads(line)
            total += 1
            named += r.get("card") is not None
    print(f"{len(files)} replays, {total} plays, {named / max(total, 1):.0%} of them with the "
          "card named (pictures of unnamed cards are in each replay's cards/ folder)")
    if len(files) > 1:
        _, rep = train(files, None, args.epochs, hold_out="last", log=lambda *a: None)
        h, b = rep["held_out"], rep["baseline_held_out"]
        print(f"newest replay held out: the pro's tile is within 1.5 tiles of the prior's "
              f"first pick {h['top1']:.0%} of the time (any legal tile: {b['top1']:.0%}); "
              f"log-likelihood {h['loglik']} (any legal tile: {b['loglik']})")
    train(files, args.out, args.epochs, hold_out=None, log=lambda *a: None)
    print(f"placement prior trained on all {total} plays -> {args.out}")


if __name__ == "__main__":
    main()
