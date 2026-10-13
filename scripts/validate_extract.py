"""How well does crownzero.replays.extract find plays? Checked on the bot's own recordings,
where its plays are known exactly (card, tile, time).

    python scripts/validate_extract.py REC_DIR [REC_DIR ...] --crbab BUILDABOT_DIR
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def truth(folder: Path):
    """The bot's plays that happened: the card left its slot within ~2.5 s."""
    from crownzero.real import cardmap

    rows = [json.loads(x) for x in open(folder / "steps.jsonl")]
    sim_to_bab = {v: k for k, v in cardmap.CARD_TO_SIM.items()}
    out = []
    for i, r in enumerate(rows):
        d = r["decision"]
        if not d.get("played"):
            continue
        card = r["cards"][d["slot"] + 1]
        later = [x for x in rows[i + 1:] if 1.0 <= x["t"] - r["t"] <= 2.5]
        if later and all(x["cards"][d["slot"] + 1] == card for x in later):
            continue  # never happened
        out.append(dict(t=r["t"], card=sim_to_bab.get(d["card"], card), tile=d["tile"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recordings", nargs="+")
    ap.add_argument("--crbab", default=None)
    ap.add_argument("--units-only", action="store_true",
                    help="ignore the hand: measure what the opponent's plays (seen only as "
                         "units) would look like")
    args = ap.parse_args()
    if args.crbab:
        sys.path.insert(0, args.crbab)
    from crownzero.replays.extract import extract

    tot = dict(true=0, found=0, hand=0, units=0, extracted=0, matched=0)
    errs = []
    for rec in args.recordings:
        folder = Path(rec)
        plays = extract(folder, read_hand=not args.units_only)
        true = truth(folder)
        mine = [p for p in plays if p.side == "bottom"]
        used = set()
        found = 0
        for tp in true:
            best = None
            for k, p in enumerate(mine):
                if k in used or p.card != tp["card"] or not -1.5 <= p.t - tp["t"] <= 3.5:
                    continue
                best = k
                break
            if best is not None:
                used.add(best)
                found += 1
                p = mine[best]
                tot[p.source] += 1
                if p.tile is not None:
                    errs.append(((p.tile[0] - tp["tile"][0]) ** 2
                                 + (p.tile[1] - tp["tile"][1]) ** 2) ** 0.5)
        missed = [tp for i, tp in enumerate(true) if not any(
            p.card == tp["card"] and -1.5 <= p.t - tp["t"] <= 3.5 for p in mine)]
        false = [p for k, p in enumerate(mine) if k not in used]
        print(f"  missed: {[(round(m['t']), m['card']) for m in missed]}")
        print(f"  false:  {[(round(p.t), p.card, p.source) for p in false]}")
        top = [p for p in plays if p.side == "top"]
        print(f"{folder.name}: bot's plays {len(true)}, found {found}; extracted for the bot "
              f"{len(mine)} ({len(used)} right); opponent plays {len(top)}: "
              f"{[(round(p.t), p.card, p.tile) for p in top[:8]]}")
        tot["true"] += len(true)
        tot["found"] += found
        tot["extracted"] += len(mine)
        tot["matched"] += len(used)
    print(f"\nrecall {tot['found']}/{tot['true']} (from the hand {tot['hand']}, from units "
          f"{tot['units']}); precision {tot['matched']}/{tot['extracted']}; tile error median "
          f"{statistics.median(errs):.1f} tiles, 90% under "
          f"{sorted(errs)[int(0.9 * len(errs))] if errs else float('nan'):.1f}")


if __name__ == "__main__":
    main()
