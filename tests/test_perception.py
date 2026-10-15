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


def test_an_enemy_missed_for_a_frame_on_its_way_over_is_kept():
    tr = SideTracker()
    for _ in range(3):
        tr.update([], [det("minipekka", 3, 17)])
    tr.update([], [])  # the detector misses it
    assert [(g.unit.name, g.position.tile_y) for g in tr.ghosts()] == [("minipekka", 17)]
    tr.update([], [])
    tr.update([], [])  # three frames gone: forgotten
    assert tr.ghosts() == []
    # one that vanished at our tower probably died there: not kept
    tr = SideTracker()
    for _ in range(3):
        tr.update([], [det("knight", 3, 8)])
    tr.update([], [])
    assert tr.ghosts() == []


def test_badge_watch_learns_the_badge_and_sees_rubble():
    import numpy as np
    from PIL import Image

    from crownzero.real.towers import PRINCESS_BARS, BadgeWatch

    rng = np.random.default_rng(0)
    floor = np.full((652, 368, 3), (200, 150, 100), np.uint8)  # an orange-tiled arena
    badge = rng.integers(0, 255, (14, 14, 3), dtype=np.uint8)
    standing = floor.copy()
    for x, y in PRINCESS_BARS:
        standing[y - 5:y + 9, x - 14:x] = badge
    w = BadgeWatch()
    for _ in range(4):
        assert w.update(Image.fromarray(standing), [1.0] * 4) == [True] * 4
    rubble = standing.copy()
    x, y = PRINCESS_BARS[3]
    rubble[y - 5:y + 9, x - 14:x] = floor[y - 5:y + 9, x - 14:x]
    assert w.update(Image.fromarray(rubble), [1, 1, 1, 0]) == [True, True, True, False]


def test_an_enemy_first_seen_on_its_half_stays_an_enemy_at_the_bridge():
    tr = SideTracker()
    tr.update([], [det("minipekka", 3, 17)])  # just over the river on their side, theirs
    allies, enemies = tr.update([det("minipekka", 3, 16)], [])  # one frame read as ours
    assert [e.unit.name for e in enemies] == ["minipekka"] and allies == []


def test_a_new_enemy_is_not_taken_for_our_stale_track_of_the_same_card():
    tr = SideTracker()
    for y in (14, 17, 19):  # our Mini P.E.K.K.A walking over
        tr.update([det("minipekka", 3, y)], [])
    tr.update([], [])  # missed a frame
    allies, enemies = tr.update([det("minipekka", 3, 23)], [det("minipekka", 3, 17)])
    assert [e.position.tile_y for e in enemies] == [17]
    allies, enemies = tr.update([det("minipekka", 3, 15), det("minipekka", 3, 23)], [])
    assert [e.position.tile_y for e in enemies] == [15]  # still theirs, one frame misread


def test_enemy_archers_arriving_where_ours_fell_are_enemies_quickly():
    tr = SideTracker()
    for y in (14, 16, 18, 20, 22, 23, 23, 23):  # our Archer walks over and attacks
        tr.update([det("archer", 4, y)], [])
    outs = []
    for y in (24, 23, 22, 21):  # it died; an enemy Archer walks down past that spot
        allies, enemies = tr.update([], [det("archer", 3, y)])
        outs.append(len(enemies))
    assert outs[1:] == [1, 1, 1]  # at most the first frame is wrong


def test_enemy_elixir_counts_plays_once_and_regenerates():
    from types import SimpleNamespace as NS

    from crownzero.real.perception import EnemyElixir

    def det(name, x, y):
        return NS(unit=NS(name=name), position=NS(tile_x=x, tile_y=y))

    ee = EnemyElixir()
    assert ee.update(0.0, []) == 5.0
    assert abs(ee.update(2.8, []) - 6.0) < 1e-6          # one elixir per 2.8 s
    # they play Archers (3): two units, one play
    v = ee.update(3.0, [det("archer", 5, 25), det("archer", 6, 25)])
    assert abs(v - (6.0 + 0.2 / 2.8 - 3)) < 1e-6
    # the same Archers walking, read once as something else, lost for a frame: no new play
    for k, name in enumerate(["archer", "musketeer", None, "archer"]):
        dets = [] if name is None else [det(name, 5, 24 - k), det("archer", 6, 24 - k)]
        v2 = ee.update(3.4 + 0.4 * k, dets)
    assert ee.spent == 3.0
    # never above 10, double speed after two minutes
    assert ee.update(100.0, []) == 10.0
    ee.value = 0.0
    ee.last_t = 130.0
    assert abs(ee.update(131.4, []) - 1.0) < 1e-6


def test_enemy_elixir_counts_their_plays():
    from types import SimpleNamespace as NS

    from crownzero.real.perception import EnemyElixir

    def det(name, x, y):
        return NS(unit=NS(name=name), position=NS(tile_x=x, tile_y=y))

    ee = EnemyElixir()
    assert ee.update(0.0, []) == 5.0
    assert abs(ee.update(2.8, []) - 6.0) < 1e-6          # one elixir per 2.8 s
    giant = det("giant", 3, 25)
    assert abs(ee.update(3.0, [giant]) - (6.0 + 0.2 / 2.8 - 5.0)) < 1e-6  # a Giant: 5 elixir
    ee.update(3.5, [giant])                              # the same Giant: no new play
    assert ee.value < 1.5
    archers = [det("archer", 8, 20), det("archer", 9, 20)]
    for t in (4.0, 4.5):
        v = ee.update(t, archers)
    assert v == 0.0 or v < 0.5                           # 2 x 1.5 elixir, never below 0
    assert ee.update(200.0, []) == 10.0                  # never above 10
    # our own half: units arriving there are not new plays of theirs
    ee.update(201.0, [det("knight", 5, 10)])
    assert ee.value == 10.0
