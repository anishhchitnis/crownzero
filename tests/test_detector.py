"""The new unit detector's geometry, image generator and name mapping (no model needed)."""
import numpy as np
from PIL import Image

from crownzero.detector.detector import bab_name
from crownzero.detector.generate import Generator, SpriteBank, write_dataset
from crownzero.detector.space import (build_a_bot_tile, part2_box_in_shot, part2_to_shot,
                                      tile_to_part2, tile_to_shot)


def test_tiles_survive_the_round_trip_through_the_part2_frame():
    for tx, u in ((0.5, 0.5), (3.5, 10.5), (9.5, 16.5), (14.5, 25.5), (17.5, 31.5)):
        x, y = tile_to_part2(tx, u)
        assert np.allclose(part2_to_shot(x, y), tile_to_shot(tx, u))
        # a unit standing in the middle of a tile is on that tile for Build-A-Bot
        assert build_a_bot_tile(*tile_to_shot(tx, u)) == (int(tx), int(u))


def test_the_part2_frame_covers_the_arena_in_a_screenshot():
    x0, y0, x1, y1 = part2_box_in_shot()
    assert 20 < x0 < 30 and 335 < x1 < 345  # the 18 tiles' width
    assert y0 < tile_to_shot(9, 32)[1] and y1 > tile_to_shot(9, 0)[1]  # both back rows


def test_names_map_to_build_a_bot():
    assert bab_name("hog-rider") == ("hog_rider", False)
    assert bab_name("knight-evolution") == ("knight", True)
    assert bab_name("mini-pekka") == ("minipekka", False)
    assert bab_name("elixir-golem-big") == ("elixir_golem_large", False)
    assert bab_name("queen-tower")[0] is None  # the tower's own troop: never a unit
    assert bab_name("fireball")[0] is None


def _sprite(path, w, h, colour):
    a = np.zeros((h, w, 4), np.uint8)
    a[2:-2, 2:-2] = (*colour, 255)
    Image.fromarray(a).save(path)


def test_generator_labels_every_unit_with_its_side(tmp_path):
    seg = tmp_path / "segment"
    for unit in ("knight", "minion", "queen-tower", "king-tower", "bar"):
        (seg / unit).mkdir(parents=True)
        for side in (0, 1):
            _sprite(seg / unit / f"{unit}_{side}_0001.png", 20, 30, (200, 50 * side, 50))
    (seg / "backgrounds").mkdir()
    Image.new("RGB", (568, 896), (90, 160, 90)).save(seg / "backgrounds" / "bg.jpg")
    bank = SpriteBank(seg)
    assert "knight|0" in bank.classes and "knight|1" in bank.classes
    assert not any(c.startswith("bar|") for c in bank.classes)  # a distractor, not a class
    img, labels = Generator(bank, seed=1).make(n_units=10)
    assert img.size == (568, 896)
    names = [bank.classes[c] for c, *_ in labels]
    assert sum(n.startswith(("queen-tower", "king-tower")) for n in names) >= 4  # towers
    assert sum(n.startswith(("knight", "minion")) for n in names) >= 8
    for _, x0, y0, x1, y1 in labels:
        assert 0 <= x0 < x1 <= 568 and 0 <= y0 < y1 <= 896
    write_dataset(Generator(bank, seed=2), tmp_path / "ds", 5)
    assert len(list((tmp_path / "ds" / "labels" / "train").glob("*.txt"))) == 4
    assert (tmp_path / "ds" / "classes.json").exists()
