"""Screenshots, plays and play checks (crownzero/real/device.py)."""
import struct

import pytest
from PIL import Image

from crownzero.real.device import PlayCheck, deploy, parse_screencap


def _raw(img: Image.Image, header16=True, fmt=1):
    w, h = img.size
    head = struct.pack("<III", w, h, fmt) + (struct.pack("<I", 0) if header16 else b"")
    return head + img.convert("RGBA").tobytes()


def test_raw_screencap_is_read_with_either_header():
    img = Image.new("RGB", (4, 3), (10, 200, 30))
    img.putpixel((1, 2), (255, 0, 0))
    for header16 in (True, False):
        out = parse_screencap(_raw(img, header16))
        assert out.size == (4, 3) and out.getpixel((1, 2)) == (255, 0, 0)
        assert out.getpixel((0, 0)) == (10, 200, 30)


def test_garbage_screencap_is_refused():
    with pytest.raises(ValueError):
        parse_screencap(b"\x89PNG\r\n\x1a\n" + b"\0" * 64)


class Clock:
    t = 0.0

    def __call__(self):
        return self.t


def test_play_check_tells_landed_from_failed():
    c = Clock()
    pc = PlayCheck(now=c)
    pc.played(1, "Knight", "swipe")
    pc.played(2, "Archer", "swipe")
    c.t = 0.8
    # Knight's slot already shows the next card; Archer's slot is mid-animation (unread)
    assert pc.update(["Giant", "Goblins", None, "Zap"]) == [("Knight", True)]
    c.t = 2.2
    assert pc.update(["Giant", "Goblins", "Archer", "Zap"]) == [("Archer", False)]
    assert pc.landed["swipe"] == 1 and pc.failed["swipe"] == 1


class FakeAdb:
    def __init__(self):
        self.commands = []

    def _run_command(self, cmd):
        self.commands.append(cmd)


def test_a_play_is_one_adb_command():
    e = FakeAdb()
    deploy(e, (220, 1140), (130, 900), "swipe")
    assert e.commands == [["shell", "input", "swipe", "220", "1140", "130", "900", "160"]]
    deploy(e, (220, 1140), (130, 900), "tap")
    assert len(e.commands) == 2 and e.commands[1][0] == "shell"
    assert "input tap 220 1140" in e.commands[1][1] and "input tap 130 900" in e.commands[1][1]
