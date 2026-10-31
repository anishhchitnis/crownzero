"""The whole card pool: evolutions, heroes, champions, and ability buttons in the search."""
import numpy as np
import pytest

pytest.importorskip("royalegym")

from crownzero.arena import TEAM, make_bot, make_env  # noqa: E402
from crownzero.fulldecks import HERO_CARDS, META, FullDeck, Pool, Rules  # noqa: E402


@pytest.fixture(scope="module")
def pool():
    return Pool(make_env().engine)


def test_pool_has_evolutions_heroes_and_champions(pool):
    assert len(pool.evo) >= 40
    assert len(pool.champions) == 8
    assert pool.heroes == HERO_CARDS


def test_every_hero_card_loads_as_a_hero(pool):
    """HERO_CARDS is hard-coded; a new simulator release must still load each one."""
    for name in sorted(HERO_CARDS):
        other = [c for c in ("Fireball", "Zap", "Arrows", "Log", "Cannon", "Tesla", "Tornado",
                             "Rocket") if c != name][:7]
        env = make_env(FullDeck((name, *other), (2, 0, 0, 0, 0, 0, 0, 0)))
        env.reset(seed=0)


def test_meta_decks_are_legal(pool):
    for name, deck in META.items():
        assert pool.valid(deck) == [], name


def test_random_decks_follow_the_rules(pool):
    rng = np.random.default_rng(1)
    rules = Rules()
    seen_evo = seen_hero = 0
    for _ in range(200):
        d = pool.random(rng, rules)
        assert pool.valid(d, rules) == []
        seen_evo += bool(d.evos)
        seen_hero += bool(d.heroes)
    assert seen_evo > 50 and seen_hero > 5


def test_invalid_forms_are_reported(pool):
    bad = FullDeck(("Minions", "Knight", "Archer", "Giant", "Fireball", "Zap", "Cannon",
                    "Musketeer"), (2, 1, 1, 1, 0, 0, 0, 0))
    problems = " ".join(pool.valid(bad))
    assert "Minions has no hero form" in problems
    assert "Giant has no evolution" in problems
    assert "3 evolutions" in problems


def test_search_presses_a_champion_ability():
    env = make_env(META["champ_aq"])
    bot, other = make_bot("search"), make_bot("defend")
    obs, _ = env.reset(seed=3)
    rng = np.random.default_rng(0)
    n_tile = env.action_parser.n_tile_actions
    presses = 0
    while env.agents:
        acts = {}
        for a in env.agents:
            b = bot if a == "blue" else other
            acts[a] = int(b.act(env, TEAM[a], obs[a], obs[a]["action_mask"], rng))
        presses += acts.get("blue", 0) >= n_tile
        obs, *_ = env.step(acts)
    assert presses >= 1
