"""--private: no player name reaches the disk."""
import json
import time
from types import SimpleNamespace

from PIL import Image

from crownzero.real import recorder as rec
from crownzero.real.recorder import NAME_BOXES, SHOT_H, SHOT_W, Recorder, blank_names


def _white():
    return Image.new("RGB", (SHOT_W, SHOT_H), (255, 255, 255))


def _state():
    n = SimpleNamespace(elixir=5, left_ally_princess_hp=1, right_ally_princess_hp=1,
                        left_enemy_princess_hp=1, right_enemy_princess_hp=1)
    return SimpleNamespace(screen="IN_GAME", allies=[], enemies=[], cards=[], ready=[],
                           numbers=n)


def _view():
    return SimpleNamespace(hand=[-1] * 4, next_card=-1, elixir=5.0, towers=[1.0] * 6,
                           clock=0.1, units=[], enemy_elixir=None)


def test_name_strips_are_black_and_the_rest_untouched():
    img = blank_names(_white())
    for x0, y0, x1, y1 in NAME_BOXES:
        assert img.getpixel(((x0 + x1) // 2, (y0 + y1) // 2)) == (0, 0, 0)
    assert img.getpixel((SHOT_W // 2, SHOT_H // 2)) == (255, 255, 255)  # mid-arena
    assert img.getpixel((SHOT_W - 20, 20)) == (255, 255, 255)  # the clock, top right


def test_private_mode_drops_menus_battle_start_and_battle_end(tmp_path, monkeypatch):
    monkeypatch.setattr(rec, "HOLD_S", 0.2)
    monkeypatch.setattr(rec, "START_SKIP_S", 0.1)
    r = Recorder(tmp_path, {}, private=True)
    r.frame(_white(), "unknown", every=1)  # a menu / VS / results screen: never saved
    r.step(_white(), _state(), _view(), {"played": False})  # first instant: dropped
    time.sleep(0.15)
    r.step(_white(), _state(), _view(), {"played": False})  # mid-battle: kept, held
    time.sleep(0.3)
    r.step(_white(), _state(), _view(), {"played": False})  # releases the one before
    r.battle_over()  # this last one is the end of the battle: dropped
    r.flush()
    saved = sorted(p.name for p in r.dir.glob("*.jpg"))
    assert saved == ["00002.jpg"]
    img = Image.open(r.dir / "00002.jpg")
    x0, y0, x1, y1 = NAME_BOXES[0]
    assert max(img.getpixel(((x0 + x1) // 2, (y0 + y1) // 2))) < 20  # black (jpeg noise)
    rows = [json.loads(line) for line in (r.dir / "steps.jsonl").read_text().splitlines()]
    assert len(rows) == 3  # the log of what the bot did is complete; it holds no screen text


def test_normal_mode_is_unchanged(tmp_path):
    r = Recorder(tmp_path, {})
    r.step(_white(), _state(), _view(), {"played": False})
    r.flush()
    img = Image.open(r.dir / "00001.jpg")
    assert min(img.getpixel((40, 20))) > 235  # not blanked
