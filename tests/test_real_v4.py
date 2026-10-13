"""v4 real-game pieces: any-card names, deck parsing, tower smoothing, live search on any deck."""
import numpy as np

from crownzero.real import cardmap
from crownzero.real.adapter import TowerTracker


def test_every_card_and_unit_maps_to_a_simulator_card():
    from royalegym import RustEngine

    cat = {c.name: c for c in RustEngine().cards()}
    assert len(cardmap.CARD_TO_SIM) == 110
    for bab, sim in cardmap.CARD_TO_SIM.items():
        assert sim in cat, bab
    for unit, sim in cardmap.UNIT_TO_SIM.items():
        assert sim in cat and cat[sim].card_kind in ("TROOP", "BUILDING"), unit


def test_deck_names_as_the_game_shows_them():
    from crownzero.real.play import parse_deck

    deck = parse_deck("Giant, Mini PEKKA, Fireball, Archers, Goblin Hut, Minions, Musketeer, Knight")
    assert deck == ["giant", "minipekka", "fireball", "archers", "goblin_hut", "minions",
                    "musketeer", "knight"]
    assert cardmap.resolve_deck_name("The Log") == "the_log"
    assert cardmap.resolve_deck_name("X-Bow") == "x_bow"
    assert cardmap.resolve_deck_name("Night Witch") == "night_witch"
    assert cardmap.resolve_deck_name("not a card") is None


def test_tower_bar_that_vanishes_is_not_a_destroyed_tower():
    t = TowerTracker()
    assert t.update([1.0, 1.0, 1.0, 1.0]) == [1.0, 1.0, 1.0, 1.0]
    # a healthy bar misread as 0 for a frame: keep the last value
    assert t.update([0.0, 1.0, 1.0, 1.0])[0] == 1.0
    # a dying tower read as 0: almost destroyed, never gone
    t = TowerTracker()
    t.update([0.1, 1, 1, 1])
    assert t.update([0.0, 1, 1, 1])[0] == TowerTracker.ALMOST


def test_live_search_plays_cards_outside_the_18():
    from crownzero.features import View
    from crownzero.live_search import LiveSearch

    deck = ["Giant", "MiniPekka", "Fireball", "Archer", "GoblinHut", "Minions", "Musketeer",
            "Knight"]
    live = LiveSearch(deck)
    v = View()
    v.hand_names = ["MiniPekka", "GoblinHut", "Minions", "Knight"]
    v.next_name = "Giant"
    v.hand = [-1, -1, 5, 0]
    v.elixir = 10.0
    v.clock = 0.3
    # an enemy Prince charging down our right lane
    v.units = [(0, 15, 14, 17, 0, 5.0)]
    v.unit_names = ["Prince"]
    a = live.decide(v, allowed_slots=[0, 1, 2, 3])
    assert a != 0, "with 10 elixir and a Prince coming the bot should play"
    setup = live.setup_for(v)
    by_id = {c.card_id: c.name for c in live.catalogue}
    assert [by_id[s.card_id] for s in setup.spawns] == ["Prince"]
    assert [by_id[c] for c in setup.decks[0][:5]] == ["MiniPekka", "GoblinHut", "Minions",
                                                        "Knight", "Giant"]
    # only the lit slots are considered
    a = live.decide(v, allowed_slots=[2])
    assert a == 0 or (a - 1) // (32 * 18) == 2
    assert np.isfinite(a)
