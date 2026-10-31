"""Real-game brain: the client's command delay, the screen's lag, the opponent's real cards."""
import pytest

pytest.importorskip("royalegym")

from crownzero.features import View  # noqa: E402
from crownzero.live_search import EnemyDeck, LiveSearch  # noqa: E402

GENERIC = ["Giant", "Musketeer", "Minions", "Knight", "Archer", "Fireball", "Zap", "Cannon"]
KNOWN = set(GENERIC) | {"HogRider", "Valkyrie", "SkeletonArmy"}


def _view(enemy_names):
    v = View()
    v.units = [(0, -1, 9, 25, 0, 1.0)] * len(enemy_names)
    v.unit_names = list(enemy_names)
    return v


def test_enemy_deck_puts_what_was_just_played_at_the_back():
    e = EnemyDeck(KNOWN, GENERIC)
    for frame in ([], ["HogRider"], [], ["Valkyrie"], ["Valkyrie", "SkeletonArmy"]):
        e.update(_view(frame))
    deck = e.deck()
    assert len(deck) == 8 and len(set(deck)) == 8
    assert deck[-3:] == ["HogRider", "Valkyrie", "SkeletonArmy"]  # out of hand
    assert all(c in GENERIC for c in deck[:4])  # the hand: cards not seen yet


def test_a_unit_staying_on_the_board_is_one_play():
    e = EnemyDeck(KNOWN, GENERIC)
    for frame in (["HogRider"], ["HogRider"], ["HogRider"]):
        e.update(_view(frame))
    assert e.played == ["HogRider"]


def test_live_search_with_delay_and_lag_decides():
    deck = ["HogRider", "Valkyrie", "Musketeer", "SkeletonArmy", "Tesla", "Log", "Fireball",
            "BabyDragon"]
    live = LiveSearch(deck, {"command_delay": 21, "lag_steps": 1, "track_enemy_deck": True,
                             "reply": "defender"})
    assert live.engine.command_delay_ticks == (21, 21)
    v = View()
    v.hand_names = deck[:4]
    v.next_name = deck[4]
    v.elixir = 7.0
    v.clock = 0.3
    v.units = [(0, -1, 3, 12, 0, 4.0)]  # an enemy troop on our left side
    v.unit_names = ["HogRider"]
    a = live.decide(v)
    assert isinstance(a, int) and a >= 0
    assert live.tracker.played == ["HogRider"]


def test_threat_reserve_grows_with_an_incoming_push():
    deck = ["HogRider", "Valkyrie", "Musketeer", "SkeletonArmy", "Tesla", "Log", "Fireball",
            "BabyDragon"]
    live = LiveSearch(deck, {"reserve": 1.0, "reserve_elixir": 4.0, "reserve_per_threat": 1.0})
    v = View()
    v.hand_names, v.next_name, v.elixir, v.clock = deck[:4], deck[4], 9.0, 0.3
    live.decide(v)  # an empty board: the plain reserve
    assert live.search._reserve_cap == 4.0
    v.units, v.unit_names = [(0, -1, 3, 12, 0, 5.0)], ["Giant"]  # a Giant on our half
    live.decide(v)
    assert live.search._reserve_cap > 6.0  # 4 + about 5 elixir of Giant, capped at 7
