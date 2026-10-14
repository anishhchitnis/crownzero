"""Re-run the current bot on a recorded battle, frame by frame, without the game.

    python scripts/replay_battle.py RECORDING_DIR --deck "Giant,..." [--out OUT_DIR]

Feeds the recorded screenshots, at their recorded times, through the real bot (screen
reader, frame-to-frame memory, live search) with a fake emulator that only records taps.
The battle does not react to the new decisions (it is a recording), so this checks what
the bot now SEES and would DO in each recorded situation, next to what it did then.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def explain(live, bot, recorder):
    """What the bot saw at this frame and how it scored its options."""
    from crownzero.board import CELLS, NX

    row = json.loads((recorder.dir / "steps.jsonl").read_text().splitlines()[-1])
    v = row["view"]
    print(f"    elixir {v['elixir']:.1f}  hand {[c for c in row['cards'][1:]]}  ready {row['ready']}")
    print(f"    towers {[round(t, 2) for t in v['towers']]}")
    print(f"    allies  {[(a['name'], tuple(a['tile'])) for a in row['allies']]}")
    print(f"    enemies {[(e['name'], tuple(e['tile'])) for e in row['enemies']]}")
    scores = getattr(live.search, "last_scores", [])
    hand = live.mutator.setup.decks[0][:4] if live.mutator.setup else []
    names = {c.card_id: c.name for c in live.catalogue}
    for value, a in scores[:8]:
        if a == 0:
            what = "wait"
        else:
            slot, cell = divmod(a - 1, CELLS)
            y, x = divmod(cell, NX)
            what = f"{names.get(hand[slot], slot)} at ({x}, {y})"
        print(f"      {value:7.2f}  {what}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recording")
    ap.add_argument("--deck", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--crbab", default=None, help="Build-A-Bot checkout, if not installed")
    ap.add_argument("--opponent-model", choices=["defender", "none"], default="defender")
    ap.add_argument("--prior", default="auto", help="pro placement prior: auto, none or a path")
    ap.add_argument("--explain", default="", help="frames to explain, e.g. 00119,00122: "
                                                  "what was seen and the best-scored plays")
    args = ap.parse_args()
    if args.crbab:
        sys.path.insert(0, args.crbab)

    import clashroyalebuildabot.bot.bot as botmod

    from crownzero.live_search import LiveSearch
    from crownzero.model import load
    from crownzero.real import cardmap, play
    from crownzero.real.recorder import Recorder

    rec_dir = Path(args.recording).expanduser()
    rows = [json.loads(line) for line in open(rec_dir / "steps.jsonl")]
    frames = [(r, rec_dir / f"{r['frame']}.jpg") for r in rows]
    frames = [(r, f) for r, f in frames if f.exists()]

    clock = {"t": 0.0}
    play.time.monotonic = lambda: clock["t"]  # the bot's clocks follow the recording
    play.time.sleep = lambda s: None
    import crownzero.real.perception as perception
    perception.time.monotonic = lambda: clock["t"]

    class FakeEmulator:
        width, height = 720, 1280

        def __init__(self, **kw):
            self.frame = None
            self.clicks = []

        def take_screenshot(self):
            return self.frame

        def click(self, x, y):
            self.clicks.append((x, y))

    fake = FakeEmulator()
    botmod.Emulator = lambda **kw: fake
    botmod.keyboard.wait = lambda *a, **k: __import__("threading").Event().wait()

    deck = play.parse_deck(args.deck)
    config = {"bot": {"auto_start_game": False, "load_deck": False, "enable_gui": False,
                      "log_level": "WARNING"},
              "adb": {"ip": "127.0.0.1", "device_serial": "replay"},
              "visuals": {"save_labels": False, "save_images": False, "show_images": False},
              "ingame": {"play_action": 0.0}}
    out = Path(args.out) if args.out else rec_dir.parent / f"{rec_dir.name}_replay"
    recorder = Recorder(out, dict(deck=deck, replay_of=str(rec_dir)), annotate=True)
    kw = dict(play.REAL_GAME_SEARCH)
    if args.opponent_model == "none":
        kw.pop("reply")
    if play.prior_path(args.prior):
        kw["prior"] = play.prior_path(args.prior)
    live = LiveSearch([cardmap.sim_name(c) for c in deck], kw)
    bot = play.make_bot_class()(play.make_actions(deck), config,
                                load(ROOT / "models" / "student.pt"), recorder=recorder,
                                live=live)
    from clashroyalebuildabot.namespaces import Screens

    bot.detector.screen_detector.run = lambda image: Screens.IN_GAME
    # Elixir bookkeeping must follow what was really played in the recording, not the
    # replay's own (never executed) decisions.
    book_played = bot.elixir_book.played
    bot.elixir_book.played = lambda *a, **k: None
    check_played = bot.play_check.played  # likewise for checking that plays happened
    bot.play_check.played = lambda *a, **k: None
    t0 = rows[0]["t"]
    changed = 0
    for r, f in frames:
        clock["t"] = 1000.0 + r["t"] - t0
        bot.clock.start = 1000.0 - (r["view"]["clock"] * 180 - (r["t"] - t0)) if bot.clock.start is None else bot.clock.start
        fake.frame = Image.open(f).convert("RGB")
        state = bot.compute_state()
        n_before = len(fake.clicks)
        bot._handle_game_step(state)
        new = json.loads((recorder.dir / "steps.jsonl").read_text().splitlines()[-1])["decision"]
        old = r["decision"]
        if old.get("played"):
            # with the slot and card, a tap that failed in the real game is forgotten
            book_played(r["view"]["elixir"], cardmap.SIM_COST.get(old["card"], 0),
                        old.get("slot"), old.get("card"))
            check_played(old["slot"], old["card"], "tap")
        o = f"{old.get('card')} {old.get('tile')}" if old.get("played") else "wait"
        n = f"{new.get('card')} {new.get('tile')}" if new.get("played") else "wait"
        if o != n:
            changed += 1
        print(f"{r['t']:7.1f}s {r['frame']}  was: {o:28s} now: {n}")
        if r["frame"] in args.explain.split(","):
            explain(live, bot, recorder)
        assert len(fake.clicks) - n_before in (0, 2)
    recorder.flush()
    print(f"\n{changed}/{len(frames)} decisions changed. Annotated frames in {recorder.dir}")


if __name__ == "__main__":
    main()
