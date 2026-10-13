"""Post-match analysis of a recorded real-game battle (``--record`` output).

    python scripts/analyze_battle.py ~/Downloads/cr_recordings/20261013-110703

Reads steps.jsonl (one line per decision: what the screen reader saw, what the bot did)
and flags the moments worth looking at:

  * UNANSWERED   an enemy troop on our side or near our towers, the bot able to play, and
                 the bot waiting anyway, for 2+ decisions in a row
  * TOWER HIT    one of our towers lost 10%+ of its health between two decisions, with the
                 enemy units next to it
  * LEAK         elixir at 10 with a card ready
  * ODD PLAY     a play in a corner or on the back row with no enemy within 5 tiles
  * NO PLAY      a play the bot made that never happened (the card stayed in hand)
  * PERCEPTION   detections outside the battlefield, units flipping between ally and enemy,
                 blank hand slots

and prints a timeline, a list of key frames (to look at the annotated images), and the
battle's numbers. Coordinates are tiles in our own frame: x 0-17, y 0 = our back row.
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUR_TOWERS = {"left": (3.5, 6.5), "right": (14.5, 6.5), "king": (9.0, 3.0)}
BUILDINGS = {"cannon", "tesla", "goblin_cage", "goblin_hut", "tombstone", "inferno_tower",
             "bomb_tower", "x_bow", "mortar", "elixir_collector", "barbarian_hut",
             "furnace", "goblin_drill"}


def dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def in_arena(t):
    return 0 <= t[0] <= 17 and 0 <= t[1] <= 31


def threats(row):
    """Enemy troops on our half or within 5 tiles of one of our towers."""
    out = []
    for e in row["enemies"]:
        t = e["tile"]
        if not in_arena(t) or e["name"] in BUILDINGS:
            continue
        near = min(dist(t, p) for p in OUR_TOWERS.values())
        if t[1] <= 14 or near <= 5:
            out.append((e["name"], tuple(t)))
    return out


def analyze(folder: Path):
    rows = [json.loads(line) for line in open(folder / "steps.jsonl")]
    meta = json.loads((folder / "meta.json").read_text())
    flags = []

    def flag(kind, i, text):
        flags.append((rows[i]["t"], rows[i]["frame"], kind, text))

    # --- unanswered threats
    streak = 0
    for i, r in enumerate(rows):
        th = threats(r)
        can_play = bool(r["ready"]) and (r["elixir"] or 0) >= 2
        if th and can_play and not r["decision"].get("played"):
            streak += 1
            if streak == 2:
                flag("UNANSWERED", i, f"{th[:3]} with elixir {r['elixir']:.0f}, ready {r['ready']}")
        else:
            streak = 0

    # --- tower damage
    names = ["left_ally_princess_hp", "right_ally_princess_hp"]
    for i in range(1, len(rows)):
        for k, side in zip(names, ("left", "right")):
            a, b = rows[i - 1]["towers"][k], rows[i]["towers"][k]
            if a is not None and b is not None and a - b >= 0.10 and b > 0:
                near = [(e["name"], tuple(e["tile"])) for e in rows[i]["enemies"]
                        if in_arena(e["tile"]) and dist(e["tile"], OUR_TOWERS[side]) <= 7]
                flag("TOWER HIT", i, f"{side} tower {a:.2f} -> {b:.2f}; enemies near: {near}")

    # --- leaks
    leak = 0
    for i, r in enumerate(rows):
        if (r["elixir"] or 0) >= 10 and r["ready"] and not r["decision"].get("played"):
            leak += 1
            if leak == 2:
                flag("LEAK", i, "elixir 10 and waiting")
        else:
            leak = 0

    # --- odd plays
    for i, r in enumerate(rows):
        d = r["decision"]
        if not d.get("played"):
            continue
        x, y = d["tile"]
        enemies = [e for e in r["enemies"] if in_arena(e["tile"])]
        closest = min((dist((x, y), e["tile"]) for e in enemies), default=99)
        if (x in (0, 17) or y <= 1) and closest > 5:
            flag("ODD PLAY", i, f"{d['card']} at {(x, y)}, nearest enemy {closest:.1f} tiles away")

    # --- perception
    oob = Counter(e["name"] for r in rows for e in r["enemies"] + r["allies"]
                  if not in_arena(e["tile"]))
    flips = 0
    for i in range(1, len(rows)):
        prev = {(a["name"], tuple(a["tile"])) for a in rows[i - 1]["allies"]}
        for e in rows[i]["enemies"]:
            if any(n == e["name"] and dist(t, e["tile"]) <= 1.5 for n, t in prev):
                flips += 1
    blanks = sum(c == "blank" for r in rows for c in r["cards"][1:])

    # --- numbers
    plays = [r for r in rows if r["decision"].get("played")]
    think = [r["decision"]["think_ms"] for r in rows if r["decision"].get("think_ms")]
    gaps = [rows[i]["t"] - rows[i - 1]["t"] for i in range(1, len(rows))]
    last = rows[-1]["towers"]

    print(f"== {folder.name}  deck {meta.get('deck')}")
    print(f"{len(rows)} decisions over {rows[-1]['t'] - rows[0]['t']:.0f} s, {len(plays)} plays: "
          f"{dict(Counter(r['decision']['card'] for r in plays))}")
    print(f"decision every {statistics.median(gaps):.2f} s (median), search "
          f"{statistics.median(think) if think else 0:.0f} ms median")
    print(f"elixir when deciding: {dict(sorted(Counter(int(r['elixir'] or 0) for r in rows).items()))}")
    print(f"final towers: {last}")
    print(f"perception: {sum(oob.values())} out-of-arena detections {dict(oob)}, "
          f"{flips} ally->enemy flips, {blanks}/{4 * len(rows)} blank hand slots")
    # cards that sat in hand, lit, and were (almost) never played
    held = Counter()
    for r in rows:
        for slot in r["ready"]:
            if slot + 1 < len(r["cards"]) and r["cards"][slot + 1] != "blank":
                held[r["cards"][slot + 1]] += 1
    played = Counter(r["decision"]["card"] for r in plays)
    from crownzero.real import cardmap
    idle = {c: n for c, n in held.items()
            if n >= 10 and played.get(cardmap.CARD_TO_SIM.get(c, c), 0) <= 1}
    if idle:
        print(f"cards ready but (almost) never played, decisions held: {idle}")
    # did each play happen? the card must leave its slot within ~2 s
    failed = []
    for i, r in enumerate(rows):
        d = r["decision"]
        if not d.get("played"):
            continue
        card = r["cards"][d["slot"] + 1]
        later = [x for x in rows[i + 1:] if 1.0 <= x["t"] - r["t"] <= 2.5]
        if later and all(x["cards"][d["slot"] + 1] == card for x in later):
            failed.append(i)
            flag("NO PLAY", i, f"{d['card']} at {tuple(d['tile'])} never left the hand")
    print(f"plays that happened: {len(plays) - len(failed)}/{len(plays)}")
    cap = [r["decision"]["capture_ms"] for r in rows if r["decision"].get("capture_ms")]
    if cap:
        print(f"screenshot time: median {statistics.median(cap):.0f} ms, max {max(cap)} ms")
    spent = [r["elixir"] for r in plays if r["elixir"] is not None]
    if spent:
        print(f"elixir when playing: median {statistics.median(spent):.0f}; "
              f"plays at 5+ elixir: {sum(e >= 5 for e in spent)}/{len(spent)}")
    print(f"enemy cards seen: {dict(Counter(e['name'] for r in rows for e in r['enemies'] if in_arena(e['tile'])).most_common(12))}")
    print("\n-- flagged moments")
    for t, frame, kind, text in sorted(flags):
        print(f"{t:7.1f}s  {frame}  {kind:10s} {text}")
    print("\n-- plays")
    for r in plays:
        d = r["decision"]
        en = [(e["name"], tuple(e["tile"])) for e in r["enemies"] if in_arena(e["tile"])]
        print(f"{r['t']:7.1f}s  {r['frame']}  el {r['elixir']:.0f}  {d['card']:>12s} at "
              f"{tuple(d['tile'])}  | enemies {en[:4]}")
    key = sorted({f for _, f, _, _ in flags})
    print("\nkey frames:", " ".join(key))
    return rows, flags


if __name__ == "__main__":
    analyze(Path(sys.argv[1]).expanduser())
