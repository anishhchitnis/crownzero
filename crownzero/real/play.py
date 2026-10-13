"""Play the real game with the trained network.

    python -m crownzero.real.play --deck hog --model models/student.pt

Needs Build-A-Bot installed in the same Python environment (see REAL_GAME.md): its
emulator connection (ADB), screen reader and tap code are reused as they are; only the
decision-making is replaced by the network.

Only for Training Camp or friendly battles on a throwaway account: Supercell's fair-play
rules ban automating the game.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from loguru import logger

from ..board import NOOP
from ..decks import CURATED
from ..features import CARD_TYPES
from ..model import load
from ..student_bot import decide
from .adapter import TYPE_TO_CARD, Clock, decode, legal_mask, view_from_state

DISPLAY_W, DISPLAY_H = 720, 1280  # Build-A-Bot's coordinate space


def make_actions(deck_types):
    """Build-A-Bot wants one Action class per deck card; ours never score anything,
    because the network decides instead."""
    from clashroyalebuildabot.actions.generic.action import Action
    from clashroyalebuildabot.namespaces.cards import Cards

    actions = []
    for t in deck_types:
        card = getattr(Cards, TYPE_TO_CARD[t].upper())
        actions.append(type(f"{t}Action", (Action,),
                            {"CARD": card, "calculate_score": lambda self, s: [0]}))
    return actions


def _quiet_hotkey(botmod):
    """Build-A-Bot listens for Ctrl+P with the ``keyboard`` package, which needs root on
    macOS and Linux. Where it cannot listen, let that thread sleep instead of crashing."""
    import threading

    real_wait = botmod.keyboard.wait

    def wait(*args, **kwargs):
        try:
            return real_wait(*args, **kwargs)
        except Exception:  # no permission to read the keyboard: no pause hotkey
            threading.Event().wait()

    botmod.keyboard.wait = wait


def _connect_network_devices():
    """Emulators like BlueStacks are reached over the network (serial "127.0.0.1:5555") and
    need ``adb connect`` first. Build-A-Bot restarts the ADB server on startup, which drops
    such connections, so reconnect right after its restart."""
    import subprocess

    import clashroyalebuildabot.emulator.emulator as em

    original = em.Emulator._restart_server

    def restart(self):
        # Build-A-Bot unzips ADB with Python's zipfile, which drops the executable bit on
        # macOS and Linux ("Permission denied" on adb). Put it back.
        import os
        import stat
        if os.path.exists(em.ADB_PATH) and not os.access(em.ADB_PATH, os.X_OK):
            os.chmod(em.ADB_PATH, os.stat(em.ADB_PATH).st_mode | stat.S_IXUSR | stat.S_IXGRP)
        original(self)
        if ":" in str(self.device_serial):
            out = subprocess.run([em.ADB_PATH, "connect", self.device_serial],
                                 cwd=em.EMULATOR_DIR, capture_output=True, text=True)
            logger.info(f"adb connect {self.device_serial}: {out.stdout.strip() or out.stderr.strip()}")

    em.Emulator._restart_server = restart


def make_bot_class():
    import clashroyalebuildabot.bot.bot as botmod

    _connect_network_devices()
    from clashroyalebuildabot.bot.bot import Bot
    from clashroyalebuildabot.namespaces import Screens

    _quiet_hotkey(botmod)

    class NetworkBot(Bot):
        def __init__(self, actions, config, net, play_threshold=0.5, after_play_s=1.0,
                     recorder=None, live=None):
            super().__init__(actions=actions, config=config)
            self.net = net
            self.live = live  # a LiveSearch: think by rebuilding the board in the simulator
            self.play_threshold = play_threshold
            self.after_play_s = after_play_s
            self.recorder = recorder
            self.clock = Clock(time.monotonic)
            self.plays = 0
            self.last_image = None
            w = getattr(self.emulator, "width", DISPLAY_W)
            h = getattr(self.emulator, "height", DISPLAY_H)
            logger.info(f"device screen {w}x{h}")
            if abs(w / h - DISPLAY_W / DISPLAY_H) > 0.01:
                logger.warning(
                    f"the device screen is {w}x{h}, not 9:16 like 720x1280. The screen reader "
                    f"will misplace everything: set the emulator to 720x1280 portrait.")

        def compute_state(self):
            screenshot = self.emulator.take_screenshot()
            self.last_image = screenshot
            state = self.detector.run(screenshot)
            if state is not None:
                self.visualizer.run(screenshot, state)
            return state

        def play_action(self, action):
            """Build-A-Bot taps in 720x1280 coordinates; scale them to the real screen so
            a device at another resolution still gets the right card and tile."""
            sx = getattr(self.emulator, "width", DISPLAY_W) / DISPLAY_W
            sy = getattr(self.emulator, "height", DISPLAY_H) / DISPLAY_H
            cx, cy = self._get_card_centre(action.index)
            tx, ty = self._get_tile_centre(action.tile_x, action.tile_y)
            self.emulator.click(round(cx * sx), round(cy * sy))
            self.emulator.click(round(tx * sx), round(ty * sy))

        def step(self, state):
            new_state = super().step(state)
            in_game = new_state is not None and new_state.screen == Screens.IN_GAME
            self.clock.update(in_game)
            return new_state

        def _handle_game_step(self, state):
            if state.screen != Screens.IN_GAME:  # e.g. the lobby: never tap there
                time.sleep(0.5)
                return
            self.clock.update(True)
            view = view_from_state(state, self.clock)
            mask = legal_mask(view, list(state.ready))
            decision = dict(played=False)
            if mask[1:].any():
                if self.live is not None:
                    t0 = time.perf_counter()
                    a = self.live.decide(view)
                    a = a if mask[a] else NOOP  # never attempt a move the screen says is illegal
                    decision.update(p_wait=float("nan"), value=float("nan"),
                                    think_ms=round(1000 * (time.perf_counter() - t0)))
                else:
                    actions, p_wait, value = decide(self.net, [view], [mask],
                                                    self.play_threshold)
                    a = actions[0]
                    decision.update(p_wait=float(p_wait[0]), value=float(value[0]))
                if a != NOOP:
                    slot, x, y = decode(a)
                    decision.update(played=True, slot=slot, tile=[x, y],
                                    card=CARD_TYPES[view.hand[slot]])
            if self.recorder is not None:
                self.recorder.step(self.last_image, state, view, decision)
            if not decision["played"]:
                time.sleep(0.05)
                return

            class Play:  # what Bot.play_action expects
                index, tile_x, tile_y = decision["slot"], *decision["tile"]

            self.play_action(Play)
            self.plays += 1
            if self.live is not None:
                logger.info(f"play {self.plays}: {decision['card']} at tile "
                            f"{tuple(decision['tile'])} | searched in {decision['think_ms']} ms")
            else:
                logger.info(f"play {self.plays}: {decision['card']} at tile "
                            f"{tuple(decision['tile'])} | P(wait) {decision['p_wait']:.2f} | "
                            f"win estimate {decision['value']:+.2f}")
            time.sleep(self.after_play_s)

    return NetworkBot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck", default="hog",
                    help=f"one of {sorted(CURATED)} or 8 comma-separated card names")
    ap.add_argument("--model", default=str(Path(__file__).resolve().parents[2]
                                           / "models" / "student.pt"))
    ap.add_argument("--device-serial", default=None, help="adb serial, e.g. emulator-5554")
    ap.add_argument("--play-threshold", type=float, default=0.5)
    ap.add_argument("--brain", choices=["search", "network"], default="search",
                    help="search: rebuild the board in the simulator and search it each move "
                         "(stronger; needs the simulator installed). network: the trained "
                         "network alone (lighter).")
    ap.add_argument("--record", default=None, metavar="DIR",
                    help="save every frame, what was detected and what the bot did, to DIR")
    args = ap.parse_args()

    deck = CURATED[args.deck] if args.deck in CURATED else args.deck.split(",")
    unknown = [c for c in deck if c not in CARD_TYPES]
    if len(deck) != 8 or unknown:
        raise SystemExit(f"deck must be 8 of {CARD_TYPES}; problem with {unknown or deck}")

    from clashroyalebuildabot.gui.utils import load_config

    config = load_config()
    config["bot"]["enable_gui"] = False
    config["bot"]["load_deck"] = False
    config["bot"]["auto_start_game"] = False
    if args.device_serial:
        config["adb"]["device_serial"] = args.device_serial
    net = load(args.model)
    recorder = None
    if args.record:
        from .recorder import Recorder
        recorder = Recorder(args.record, dict(deck=deck, model=args.model,
                                              play_threshold=args.play_threshold))
        logger.info(f"recording to {recorder.dir}")
    live = None
    if args.brain == "search":
        try:
            from ..live_search import LiveSearch
        except ImportError as e:
            raise SystemExit("--brain search needs the simulator: "
                             "uv pip install -e \"../crownzero[search]\"  "
                             f"(or run with --brain network). ({e})")
        live = LiveSearch(deck)
        logger.info("brain: live search in the simulator")
    bot = make_bot_class()(make_actions(deck), config, net, args.play_threshold,
                           recorder=recorder, live=live)
    logger.info(f"CrownZero ready with deck {deck}. Start a Training Camp battle; "
                f"Ctrl+C to stop.")
    bot.run()


if __name__ == "__main__":
    np.set_printoptions(precision=2)
    main()
