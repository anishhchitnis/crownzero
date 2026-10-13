"""End-to-end smoke test of the real-game bridge, with no phone or emulator.

A fake emulator hands Build-A-Bot a synthetic screenshot with our four hand cards pasted
where the game draws them. Build-A-Bot's own card reader must recognise them, the network
must choose a play once elixir is available, and the bot must tap the card and then a tile.

Skipped unless Build-A-Bot is importable (set CRBAB_PATH to its checkout).
"""
import os
import sys
from pathlib import Path

import pytest

CRBAB = os.environ.get("CRBAB_PATH", str(Path.home() / "vendor" / "cr_buildabot"))
sys.path.insert(0, CRBAB)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
crbab = pytest.importorskip("clashroyalebuildabot")

from PIL import Image  # noqa: E402

from crownzero.decks import CURATED  # noqa: E402
from crownzero.real.adapter import TYPE_TO_CARD  # noqa: E402

MODEL = Path(__file__).resolve().parents[1] / "models" / "student.pt"


class FakeEmulator:
    def __init__(self, frame, width=720, height=1280):
        self.frame = frame
        self.width, self.height = width, height
        self.clicks = []

    def take_screenshot(self):
        return self.frame

    def click(self, x, y):
        self.clicks.append((x, y))


def synthetic_screenshot(hand_names, next_name):
    from clashroyalebuildabot.constants import (CARD_CONFIG, IMAGES_DIR, SCREENSHOT_HEIGHT,
                                                SCREENSHOT_WIDTH)
    img = Image.new("RGB", (SCREENSHOT_WIDTH, SCREENSHOT_HEIGHT), (40, 90, 40))
    for box, name in zip(CARD_CONFIG, [next_name] + hand_names):
        card = Image.open(Path(IMAGES_DIR) / "cards" / f"{name}.jpg").convert("RGB")
        img.paste(card.resize((box[2] - box[0], box[3] - box[1])), box[:2])
    return img


@pytest.mark.skipif(not MODEL.exists(), reason="train a model first")
@pytest.mark.parametrize("size,brain", [((720, 1280), "network"), ((1080, 1920), "network"),
                                        ((720, 1280), "search")])
def test_bot_reads_hand_and_plays(monkeypatch, tmp_path, size, brain):
    import clashroyalebuildabot.bot.bot as botmod
    from clashroyalebuildabot.namespaces import Screens
    from clashroyalebuildabot.namespaces.numbers import NumberDetection

    from crownzero.model import load
    from crownzero.real.play import make_actions, make_bot_class

    deck = CURATED["hog"]
    hand = [TYPE_TO_CARD[c] for c in deck[:4]]
    frame = synthetic_screenshot(hand, TYPE_TO_CARD[deck[4]])
    fake = FakeEmulator(frame, *size)
    monkeypatch.setattr(botmod, "Emulator", lambda **kw: fake)
    monkeypatch.setattr(botmod.keyboard, "wait", lambda *a, **k: __import__("threading").Event().wait())

    config = {"bot": {"auto_start_game": False, "load_deck": False, "enable_gui": False,
                      "log_level": "INFO"},
              "adb": {"ip": "127.0.0.1", "device_serial": "fake"},
              "visuals": {"save_labels": False, "save_images": False, "show_images": False},
              "ingame": {"play_action": 0.0}}
    from crownzero.real.recorder import Recorder
    rec = Recorder(tmp_path, {"test": True})
    live = None
    if brain == "search":
        from crownzero.live_search import LiveSearch
        live = LiveSearch(deck)
    bot = make_bot_class()(make_actions(deck), config, load(MODEL), after_play_s=0.0,
                           recorder=rec, live=live)

    # The game screen and the elixir bar are not drawn on the synthetic frame: say so.
    monkeypatch.setattr(bot.detector.screen_detector, "run", lambda image: Screens.IN_GAME)
    real_numbers = bot.detector.number_detector.run

    def numbers_with_full_elixir(image):
        n = real_numbers(image)
        full = {k: getattr(n, k) for k in n.__dataclass_fields__}
        full["elixir"] = NumberDetection(n.elixir.bbox, 10)
        for k in full:
            if k.endswith("_hp"):
                full[k] = NumberDetection(full[k].bbox, 1.0)
        return type(n)(**full)

    monkeypatch.setattr(bot.detector.number_detector, "run", numbers_with_full_elixir)

    state = bot.compute_state()
    assert bot.last_image is frame
    assert [c.name for c in state.cards[1:]] == hand, "Build-A-Bot misread the hand"
    for _ in range(5):
        bot._handle_game_step(state)
        if fake.clicks:
            break
    assert len(fake.clicks) == 2, "with 10 elixir and a fresh board the bot should play"
    card_xy, tile_xy = fake.clicks
    k = size[1] / 1280  # taps are scaled to the device's real resolution
    assert card_xy[1] > 1000 * k and tile_xy[1] < 1000 * k  # tapped a card, then the arena
    assert card_xy[0] < size[0] and tile_xy[0] < size[0]
    # the recording has the frame, the annotated frame and a log line per decision
    assert (rec.dir / "00001.jpg").exists() and (rec.dir / "00001_ann.jpg").exists()
    lines = (rec.dir / "steps.jsonl").read_text().splitlines()
    assert lines and '"played": true' in lines[-1]


def _bot(monkeypatch, deck_names):
    import clashroyalebuildabot.bot.bot as botmod

    from crownzero.model import load
    from crownzero.real.play import make_actions, make_bot_class, parse_deck

    deck = parse_deck(deck_names)
    fake = FakeEmulator(Image.new("RGB", (368, 652)))
    monkeypatch.setattr(botmod, "Emulator", lambda **kw: fake)
    monkeypatch.setattr(botmod.keyboard, "wait",
                        lambda *a, **k: __import__("threading").Event().wait())
    config = {"bot": {"auto_start_game": False, "load_deck": False, "enable_gui": False,
                      "log_level": "INFO"},
              "adb": {"ip": "127.0.0.1", "device_serial": "fake"},
              "visuals": {"save_labels": False, "save_images": False, "show_images": False},
              "ingame": {"play_action": 0.0}}
    return make_bot_class()(make_actions(deck), config, load(MODEL), after_play_s=0.0)


def _det(name, x, y):
    from types import SimpleNamespace as NS
    return NS(unit=NS(name=name, category="troop", target="ground", transport="ground"),
              position=NS(tile_x=x, tile_y=y, bbox=(0, 0, 1, 1), conf=0.9))


@pytest.mark.skipif(not MODEL.exists(), reason="train a model first")
def test_our_deploying_troop_is_not_an_enemy(monkeypatch):
    """Recorded 2026-10-13: the bot placed a Giant in the corner, the reader called the
    deploying Giant an enemy, and the bot played Minions on top of it."""
    import time

    from clashroyalebuildabot.namespaces.state import State

    bot = _bot(monkeypatch, "Giant,Mini PEKKA,Fireball,Archers,Goblin Cage,Minions,Musketeer,Knight")
    bot.recent_plays.append((time.monotonic(), 17, 1, bot.units_of["Giant"]))
    st = State(allies=[], enemies=[_det("giant", 17, 0), _det("giant", 3, 25)],
               numbers=None, cards=(), ready=[], screen=None)
    fixed = bot._own_recent_plays(st)
    assert [(d.unit.name, d.position.tile_x) for d in fixed.allies] == [("giant", 17)]
    assert [(d.unit.name, d.position.tile_x) for d in fixed.enemies] == [("giant", 3)]


def test_decorations_outside_the_arena_are_ignored():
    """Recorded 2026-10-13: two lamps beside our right tower read as an enemy Executioner
    and Monk at tile x 18, every frame."""
    from crownzero.real.adapter import in_arena
    assert not in_arena(_det("executioner", 18, 4))
    assert not in_arena(_det("monk", -1, 2))
    assert in_arena(_det("giant", 17, 0))
