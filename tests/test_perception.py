"""Memory-across-frames fixes, each from a mistake in a recorded battle (see perception.py)."""
from types import SimpleNamespace as NS

from crownzero.features import View
from crownzero.real.perception import ElixirBook, HiddenAttacker, SideTracker


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def det(name, x, y):
    return NS(unit=NS(name=name), position=NS(tile_x=x, tile_y=y))


def test_elixir_after_a_play_is_not_the_stale_reading():
    c = Clock()
    book = ElixirBook(now=c)
    book.played(elixir_before=7, cost=4)
    c.t = 0.6
    # the screenshot still shows 7: we really have ~3.2
    assert abs(book.effective(7) - (3 + 0.6 / 2.8)) < 1e-6
    # double elixir regenerates twice as fast
    assert abs(book.effective(7, double=True) - (3 + 1.2 / 2.8)) < 1e-6
    # a reading already lower than the book is trusted
    assert book.effective(2) == 2
    c.t = 5.0  # long after: the book no longer interferes
    assert book.effective(7) == 7


def test_side_flicker_is_voted_away():
    tr = SideTracker()
    g = det("giant", 14, 12)
    for _ in range(4):  # our Giant, seen as ours for four frames
        allies, enemies = tr.update([g], [])
    allies, enemies = tr.update([], [det("giant", 14, 11)])  # one frame says enemy
    assert [a.unit.name for a in allies] == ["giant"] and enemies == []
    # an enemy Giant elsewhere is a different unit and stays an enemy
    allies, enemies = tr.update([det("giant", 14, 11)], [det("giant", 3, 20)])
    assert [e.position.tile_x for e in enemies] == [3]


def test_tower_losing_health_with_nothing_near_it_has_an_attacker():
    c = Clock()
    h = HiddenAttacker(now=c)
    v = View()
    v.towers = [1, 1.0, 1.0, 1, 1, 1]
    assert h.update(v) == []
    c.t = 1.5
    v.towers = [1, 0.74, 1.0, 1, 1, 1]  # recorded: 1.00 -> 0.74 with no enemy detected
    assert h.update(v) == [("MiniPekka", 3, 8)]
    # if the attacker is visible, nothing is invented
    v.units = [(0, 0, 3, 8, 0, 3.0)]
    assert h.update(v) == []


def test_a_play_that_never_happened_is_forgotten():
    c = Clock()
    book = ElixirBook(now=c)
    book.played(elixir_before=8, cost=4, slot=1, card="GoblinCage")
    c.t = 0.5
    assert book.effective(8, hand=[None, "GoblinCage", None, None]) < 5  # too soon to tell
    c.t = 1.5  # still in its slot 1.5 s later: the tap missed
    assert book.effective(8.5, hand=["Knight", "GoblinCage", "Giant", "Fireball"]) == 8.5


def bdet(name, x, y):
    return NS(unit=NS(name=name, category="building"), position=NS(tile_x=x, tile_y=y))


def test_a_building_is_on_its_owners_half():
    tr = SideTracker()
    hut = bdet("goblin_hut", 9, 8)
    for _ in range(3):
        tr.update([hut], [])
    allies, enemies = tr.update([], [bdet("goblin_hut", 9, 8)])  # classifier says enemy
    assert [a.unit.name for a in allies] == ["goblin_hut"] and enemies == []
    allies, enemies = tr.update([bdet("goblin_hut", 9, 23)], [])  # theirs, whatever it says
    assert [e.unit.name for e in enemies] == ["goblin_hut"]


def test_a_spear_goblin_born_at_our_hut_is_ours():
    tr = SideTracker()
    hut = bdet("goblin_hut", 9, 8)
    tr.update([hut], [])
    for _ in range(3):  # read as an enemy while it appears; the hut's prior holds
        allies, enemies = tr.update([hut], [det("spear_goblin", 9, 10)])
        assert "spear_goblin" in [a.unit.name for a in allies]
    # an enemy spear goblin coming over the bridge is still an enemy
    allies, enemies = tr.update([hut], [det("spear_goblin", 3, 14)])
    assert [e.position.tile_x for e in enemies] == [3]
