"""Play the real game.

    python -m crownzero.real.play --deck "Giant,Mini PEKKA,Fireball,Archers,Goblin Hut,Minions,Musketeer,Knight"

--deck must be the 8 cards actually in your battle deck in the game (any cards).

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
from . import cardmap
from .adapter import (TYPE_TO_CARD, Clock, TowerTracker, decode, legal_mask,
                      playable_slots, view_from_state)
from .device import PlayCheck, deploy, tune_detector, use_fresh_screenshots
from .perception import ElixirBook, HiddenAttacker, SideTracker
from .towers import BadgeWatch
from .towers import enemy_king as read_enemy_king

DISPLAY_W, DISPLAY_H = 720, 1280  # Build-A-Bot's coordinate space
# The search settings for real games: the opponent answers our pushes in the look-ahead
# (v4.2), and the first 3 banked elixir are worth double (v4.5; 22-18 and 21-19 at half
# strength in simulator A/B tests, and it fixes a recorded elixir dump).
REAL_GAME_SEARCH = {"reply": "defender", "reserve": 1.0}


def parse_deck(text: str) -> list[str]:
    """--deck -> 8 Build-A-Bot card names. A curated name (hog, giant, ...) or 8 card names
    separated by commas, as the game shows them ("Mini PEKKA", "The Log", ...)."""
    names = CURATED[text] if text in CURATED else [n for n in text.split(",") if n.strip()]
    deck, bad = [], []
    for n in names:
        b = cardmap.resolve_deck_name(n.strip())
        (deck if b else bad).append(b or n.strip())
    if bad:
        raise SystemExit(f"unknown card name(s): {bad}. Use the names the game shows, e.g. "
                         f"'Mini PEKKA', 'The Log', 'Goblin Hut'.")
    if len(deck) != 8 or len(set(deck)) != 8:
        raise SystemExit(f"--deck needs 8 different cards, got {len(set(deck))}: {deck}")
    return deck


def make_actions(deck):
    """Build-A-Bot wants one Action class per deck card (Build-A-Bot names); ours never
    score anything, because our brain decides instead."""
    from clashroyalebuildabot.actions.generic.action import Action
    from clashroyalebuildabot.namespaces.cards import Cards

    actions = []
    for name in deck:
        name = TYPE_TO_CARD.get(name, name)  # also accept our 18 type names
        card = getattr(Cards, name.upper())
        actions.append(type(f"{name}Action", (Action,),
                            {"CARD": card, "calculate_score": lambda self, s: [0]}))
    return actions


def _quiet_hotkey(botmod):
    """Build-A-Bot listens for Ctrl+P with the ``keyboard`` package, which needs root on
    macOS and Linux. Where it cannot listen, let that thread sleep instead of crashing."""
    import os
    import sys
    import threading

    real_wait = botmod.keyboard.wait

    def wait(*args, **kwargs):
        # On macOS/Linux without root the keyboard listener thread raises on its own
        # (a harmless but alarming traceback), so do not start it at all there.
        if sys.platform != "win32" and os.geteuid() != 0:
            threading.Event().wait()
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


CARD_ART = Path(__file__).resolve().parent / "card_art"


def _use_current_card_art():
    """Some of Build-A-Bot's card pictures are older art than the live game shows (the
    Goblin Hut is now a wooden hut with a goblin in it), and its card reader then mistakes
    that card for another. Pictures in crownzero/real/card_art, taken from the live game,
    replace those."""
    import clashroyalebuildabot.detectors.card_detector as cd
    from PIL import Image

    if getattr(cd.CardDetector, "_crownzero_art", False):
        return

    def _calculate_card_hashes(self):
        n = len(self.cards)
        hashes = np.zeros((n, 3, self.hash_size * self.hash_size, self.HAND_SIZE), np.float32)
        for i, card in enumerate(self.cards):
            ours = CARD_ART / f"{card.name}.jpg"
            path = ours if ours.exists() else Path(cd.IMAGES_DIR) / "cards" / f"{card.name}.jpg"
            multi = self._calculate_multi_hash(Image.open(path))
            hashes[i] = np.tile(np.expand_dims(multi, axis=2), (1, 1, self.HAND_SIZE))
        return hashes

    cd.CardDetector._calculate_card_hashes = _calculate_card_hashes
    cd.CardDetector._crownzero_art = True


def make_bot_class(capture: str = "screencap", tune: bool = True):
    """``capture``: "screencap" (a fresh screenshot whenever the bot looks) or "stream"
    (Build-A-Bot's video stream, which fell seconds behind in recorded battles)."""
    import clashroyalebuildabot.bot.bot as botmod
    import clashroyalebuildabot.detectors.onnx_detector as onnx_detector
    import clashroyalebuildabot.emulator.emulator as em

    _connect_network_devices()
    _use_current_card_art()
    if capture == "screencap":
        use_fresh_screenshots(em)
    if tune:
        tune_detector(onnx_detector)
    from clashroyalebuildabot.bot.bot import Bot
    from clashroyalebuildabot.namespaces import Screens

    _quiet_hotkey(botmod)

    class NetworkBot(Bot):
        def __init__(self, actions, config, net, play_threshold=0.5, after_play_s=0.3,
                     recorder=None, live=None, input_mode="auto"):
            super().__init__(actions=actions, config=config)
            self.net = net
            self.live = live  # a LiveSearch: think by rebuilding the board in the simulator
            self.play_threshold = play_threshold
            self.after_play_s = after_play_s
            self.recorder = recorder
            self.clock = Clock(time.monotonic)
            self.plays = 0
            self.last_image = None
            self.towers = TowerTracker()
            self.badges = BadgeWatch()
            self.elixir_book = ElixirBook()
            self.sides = SideTracker()
            self.hidden = HiddenAttacker()
            self.recent_plays = []  # (time, tile x, tile y, unit names) of our last plays
            self.units_of = {cardmap.sim_name(a.CARD.name): {u.name for u in a.CARD.units}
                             for a in actions}
            # "auto": drag the card onto the tile; if drags never land, switch to taps
            self.auto_input = input_mode == "auto"
            self.input_mode = "tap" if input_mode == "tap" else "swipe"
            self.play_check = PlayCheck()
            self.blank_log = []  # recent share of the hand read as blank, mid-battle
            self.warned_deck = False
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
            t0 = time.perf_counter()
            state = self.detector.run(screenshot)
            self.detect_ms = round(1000 * (time.perf_counter() - t0))
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
            card = (round(cx * sx), round(cy * sy))
            tile = (round(tx * sx), round(ty * sy))
            if hasattr(self.emulator, "_run_command"):  # one adb command per play
                deploy(self.emulator, card, tile, self.input_mode)
            else:  # an emulator stand-in (tests, replays) that only knows taps
                self.emulator.click(*card)
                self.emulator.click(*tile)

        def step(self, state):
            new_state = super().step(state)
            in_game = new_state is not None and new_state.screen == Screens.IN_GAME
            if not in_game and self.clock.start is not None:  # the battle ended
                logger.info(f"battle over: {self.play_check.summary()}")
                self.play_check = PlayCheck()
                self.towers.reset()
                self.badges.reset()
                self.sides.reset()
                self.hidden.reset()
            self.clock.update(in_game)
            if (self.recorder is not None and new_state is not None
                    and new_state.screen == Screens.UNKNOWN):
                self.recorder.frame(self.last_image, "unknown")  # to see what it misread
            return new_state

        def _handle_game_step(self, state):
            if state.screen != Screens.IN_GAME:  # e.g. the lobby: never tap there
                time.sleep(0.5)
                return
            self.clock.update(True)
            import dataclasses

            state = self._own_recent_plays(state)
            allies, enemies = self.sides.update(list(state.allies), list(state.enemies))
            ghosts = self.sides.ghosts()  # enemies the detector missed for a frame or two
            state = dataclasses.replace(state, allies=allies, enemies=enemies + ghosts)
            view = view_from_state(state, self.clock)
            raw = [view.towers[1], view.towers[2], view.towers[4], view.towers[5]]
            marks = None
            if self.last_image is not None:
                marks = self.badges.update(self.last_image, raw)
                view.towers[3] = self.towers.update_king(read_enemy_king(self.last_image))
            p = self.towers.update(raw, marks)
            view.towers[1], view.towers[2], view.towers[4], view.towers[5] = p
            view.elixir = self.elixir_book.effective(view.elixir, view.double, view.overtime,
                                                     hand=list(view.hand_names))
            for name, x, y in self.hidden.update(view):
                view.units.append((0, -1, x, y, 0, 3.5))
                view.unit_names.append(name)
                logger.info(f"tower under attack with nothing seen near it: assuming a {name} at ({x}, {y})")
            self._check_deck(state, view)
            decision = dict(played=False)
            if ghosts:
                decision["ghosts"] = [[g.unit.name, g.position.tile_x, g.position.tile_y]
                                      for g in ghosts]
            checked = self._check_plays(view)
            if checked:
                decision["checked"] = checked
            if getattr(self.emulator, "capture_ms", None) is not None:
                decision["capture_ms"] = self.emulator.capture_ms
                self._watch_capture(self.emulator.capture_ms)
            if getattr(self, "detect_ms", None) is not None:
                decision["detect_ms"] = self.detect_ms
            if self.live is not None:
                slots = playable_slots(view, list(state.ready))
                if slots:
                    t0 = time.perf_counter()
                    a = self.live.decide(view, allowed_slots=slots)
                    decision.update(p_wait=float("nan"), value=float("nan"),
                                    think_ms=round(1000 * (time.perf_counter() - t0)))
                    if a != NOOP and decode(a)[0] in slots:
                        slot, x, y = decode(a)
                        decision.update(played=True, slot=slot, tile=[x, y],
                                        card=view.hand_names[slot])
            else:
                mask = legal_mask(view, list(state.ready))
                if mask[1:].any():
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
            self.play_check.played(decision["slot"], decision["card"], self.input_mode)
            self.elixir_book.played(view.elixir, cardmap.SIM_COST.get(decision["card"], 0),
                                    decision["slot"], decision["card"])
            self.recent_plays.append((time.monotonic(), *decision["tile"],
                                      self.units_of.get(decision["card"], set())))
            if self.live is not None:
                logger.info(f"play {self.plays}: {decision['card']} at tile "
                            f"{tuple(decision['tile'])} | searched in {decision['think_ms']} ms")
            else:
                logger.info(f"play {self.plays}: {decision['card']} at tile "
                            f"{tuple(decision['tile'])} | P(wait) {decision['p_wait']:.2f} | "
                            f"win estimate {decision['value']:+.2f}")
            time.sleep(self.after_play_s)

        def _watch_capture(self, ms):
            self.capture_log = (getattr(self, "capture_log", []) + [ms])[-20:]
            if (len(self.capture_log) == 20 and not getattr(self, "warned_capture", False)
                    and sorted(self.capture_log)[10] > 800):
                self.warned_capture = True
                logger.warning(f"screenshots take ~{sorted(self.capture_log)[10]} ms each; "
                               f"if the bot feels slow, try --capture stream")

        def _check_plays(self, view):
            """Settle earlier plays against the hand now; switch drags to taps if drags
            never land on this device."""
            out = []
            for card, landed in self.play_check.update(list(view.hand_names)):
                out.append([card, landed])
                if not landed:
                    logger.warning(f"the {card} play did not happen (still in hand)")
            pc = self.play_check
            if (self.auto_input and self.input_mode == "swipe" and pc.landed["swipe"] == 0
                    and pc.failed["swipe"] >= 3):
                self.input_mode = "tap"
                logger.warning("card drags are not landing on this device: switching to taps")
            return out

        def _own_recent_plays(self, state, window_s=4.0, dx=2, dy=3):
            """A troop we just placed is still in its deploy animation, and the reader's
            ally/enemy classifier gets those wrong: our own Giant in the corner was read as
            an enemy Giant, and the bot then threw Minions on top of it, and so on. For a
            few seconds after a play, detections of that card's units near where we put
            them are ours."""
            import dataclasses

            now = time.monotonic()
            self.recent_plays = [p for p in self.recent_plays if now - p[0] < window_s]
            if not self.recent_plays or not state.enemies:
                return state
            allies, enemies = list(state.allies), []
            for d in state.enemies:
                p = d.position
                ours = any(d.unit.name in units and abs(p.tile_x - x) <= dx
                           and abs(p.tile_y - y) <= dy for _, x, y, units in self.recent_plays)
                (allies if ours else enemies).append(d)
            return dataclasses.replace(state, allies=allies, enemies=enemies)

        def _check_deck(self, state, view):
            """If most of the hand keeps reading as blank mid-battle, the cards on screen
            are not the --deck cards: say so loudly, once."""
            if self.warned_deck or view.clock * 180 < 10:
                return
            self.blank_log.append(sum(c.name == "blank" for c in state.cards[1:]) / 4)
            self.blank_log = self.blank_log[-30:]
            if len(self.blank_log) == 30 and sum(self.blank_log) / 30 > 0.4:
                self.warned_deck = True
                logger.warning(
                    "Most of your hand doesn't match --deck. The bot can only play cards it "
                    "recognises: stop it (Ctrl+C) and pass the 8 cards that are really in your "
                    "battle deck, e.g. --deck \"Giant,Mini PEKKA,Fireball,Archers,...\"")

    return NetworkBot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck", required=True,
                    help="the 8 cards in your battle deck in the game, comma-separated, e.g. "
                         "\"Giant,Mini PEKKA,Fireball,Archers,Goblin Hut,Minions,Musketeer,Knight\"")
    ap.add_argument("--model", default=str(Path(__file__).resolve().parents[2]
                                           / "models" / "student.pt"))
    ap.add_argument("--device-serial", default=None, help="adb serial, e.g. emulator-5554")
    ap.add_argument("--play-threshold", type=float, default=0.5)
    ap.add_argument("--brain", choices=["search", "network"], default="search",
                    help="search: rebuild the board in the simulator and search it each move "
                         "(stronger; needs the simulator installed). network: the trained "
                         "network alone (lighter).")
    ap.add_argument("--opponent-model", choices=["defender", "none"], default="defender",
                    help="in the look-ahead, does the opponent defend our pushes (defender) "
                         "or stand still (none, the v4.1 behaviour)")
    ap.add_argument("--input", choices=["auto", "swipe", "tap"], default="auto",
                    help="how to play a card: drag it onto the tile (swipe), tap card then "
                         "tile (tap), or drag and fall back to taps if drags don't land (auto)")
    ap.add_argument("--capture", choices=["screencap", "stream"], default="screencap",
                    help="screencap: a fresh screenshot each time (default). stream: "
                         "Build-A-Bot's video stream (can fall seconds behind)")
    ap.add_argument("--record", default=None, metavar="DIR",
                    help="save every frame, what was detected and what the bot did, to DIR")
    args = ap.parse_args()

    deck = parse_deck(args.deck)
    if args.brain == "network":
        outside = [c for c in deck if cardmap.sim_name(c) not in CARD_TYPES]
        if outside:
            logger.warning(f"--brain network only knows 18 cards; it will never play {outside}. "
                           f"Use --brain search (the default) for any card.")

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
        recorder = Recorder(args.record, dict(deck=deck, model=args.model, brain=args.brain,
                                              opponent_model=args.opponent_model,
                                              input=args.input, capture=args.capture,
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
        kw = dict(REAL_GAME_SEARCH)
        if args.opponent_model == "none":
            kw.pop("reply")
        live = LiveSearch([cardmap.sim_name(c) for c in deck], kw)
        logger.info(f"brain: live search in the simulator (opponent model: "
                    f"{args.opponent_model})")
    bot = make_bot_class(capture=args.capture)(make_actions(deck), config, net,
                                              args.play_threshold, recorder=recorder,
                                              live=live, input_mode=args.input)
    from .. import __version__

    logger.info(f"CrownZero v{__version__} ready with deck {deck}. Start a Training Camp battle; "
                f"Ctrl+C to stop.")
    bot.run()


if __name__ == "__main__":
    np.set_printoptions(precision=2)
    main()
