"""The real-game adapter must hand the network the same tensors the simulator does.

We play a simulated battle for a while, then describe that board the way Build-A-Bot's
screen reader would (unit names + tiles, card names, elixir, health-bar fractions) and
check that the adapter's View featurizes to the same unit planes and hand features as the
View built straight from the simulator.
"""
from types import SimpleNamespace as NS

import numpy as np

from crownzero.arena import Scripted, make_env
from crownzero.features import featurize_views, from_sim, PER_SIDE, C_COORD
from crownzero.real.adapter import (TYPE_TO_CARD, UNIT_TO_TYPE, Clock, decode, legal_mask,
                                    view_from_state)
from crownzero.board import action_index

TYPE_TO_UNIT = {v: k for k, v in UNIT_TO_TYPE.items()}


def _fake_state(sim_view):
    """Describe a simulator View the way Build-A-Bot would report it."""
    from crownzero.features import CARD_TYPES, BUILD, FLY, TARGETS_BUILDINGS
    allies, enemies = [], []
    for own, t, x, y, flags, _ in sim_view.units:
        unit = NS(name=TYPE_TO_UNIT[CARD_TYPES[t]],
                  transport="air" if flags & FLY else "ground",
                  category="building" if flags & BUILD else "troop",
                  target="buildings" if flags & TARGETS_BUILDINGS else "all")
        det = NS(unit=unit, position=NS(tile_x=x, tile_y=y))
        (allies if own else enemies).append(det)
    card = lambda t: NS(name=TYPE_TO_CARD[CARD_TYPES[t]]) if t >= 0 else NS(name="blank")  # noqa: E731
    num = lambda v: NS(number=v)  # noqa: E731
    tw = sim_view.towers
    numbers = NS(elixir=num(int(sim_view.elixir)),
                 left_ally_princess_hp=num(tw[1]), right_ally_princess_hp=num(tw[2]),
                 left_enemy_princess_hp=num(tw[4]), right_enemy_princess_hp=num(tw[5]))
    cards = [card(sim_view.next_card)] + [card(t) for t in sim_view.hand]
    return NS(allies=allies, enemies=enemies, cards=cards, numbers=numbers, ready=[0, 1, 2, 3])


def test_sim_and_real_views_featurize_alike():
    env = make_env(("HogRider", "Valkyrie", "Musketeer", "SkeletonArmy", "Tesla", "Log",
                    "Fireball", "BabyDragon"))
    obs, _ = env.reset(seed=4)
    rng = np.random.default_rng(0)
    bots = [Scripted("push"), Scripted("defend")]
    for _ in range(90):
        obs, *_ = env.step({a: bots[i].act(env, i, obs[a], obs[a]["action_mask"], rng)
                            for i, a in enumerate(("blue", "red"))})
    cards = {c.card_id: c for c in env.engine.cards()}
    for team in (0, 1):
        sim = from_sim(env.battle_state, team, cards, env.engine.arena())
        assert sim.units, "the board should not be empty after 45 s"
        real = view_from_state(_fake_state(sim), Clock(lambda: 0.0))
        sp_s, vec_s = featurize_views([sim])
        sp_r, vec_r = featurize_views([real])
        units = slice(0, 2 * PER_SIDE)
        assert np.array_equal(sp_s[:, units], sp_r[:, units]), "unit planes differ"
        assert np.array_equal(sp_s[:, C_COORD:C_COORD + 2], sp_r[:, C_COORD:C_COORD + 2])
        # hand, next card and affordability agree (elixir is whole numbers on screen)
        assert real.hand == sim.hand and real.next_card == sim.next_card


def test_mask_and_decode_round_trip():
    v = from_sim.__globals__["View"]()
    v.hand = [0, 11, 9, 3]  # Knight, Fireball, Cannon, Giant
    v.elixir = 4.0
    m = legal_mask(v, ready=[0, 1, 2, 3])
    assert m[0]
    assert m[action_index(0, 5, 4)]          # Knight on our half
    assert not m[action_index(0, 20, 4)]     # Knight on the enemy half: no
    assert m[action_index(1, 25, 3)]         # Fireball anywhere
    assert not m[action_index(3, 5, 4)].any()  # Giant costs 5 > 4 elixir
    assert decode(action_index(2, 7, 11)) == (2, 11, 7)
