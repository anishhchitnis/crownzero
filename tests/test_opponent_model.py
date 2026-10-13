"""The rollout opponent defends the lane we attack (see crownzero/opponent_model.py)."""
import pytest

pytest.importorskip("royalegym")

from crownzero.board import CELLS, NX  # noqa: E402
from crownzero.features import View  # noqa: E402
from crownzero.live_search import LiveSearch  # noqa: E402
from crownzero.opponent_model import DefenderReply  # noqa: E402

DECK = ["Giant", "Musketeer", "Minions", "Knight", "Archer", "Fireball", "Zap", "Cannon"]


def _state_with(units):
    live = LiveSearch(DECK)
    v = View()
    v.hand_names = ["Giant", "Musketeer", "Minions", "Knight"]
    v.next_name = "Archer"
    v.elixir = 5
    v.clock = 0.3
    v.units = [(own, -1, x, y, 0, 3.0) for own, x, y in units]
    v.unit_names = ["Knight"] * len(units)
    live.mutator.setup = live.setup_for(v)
    live.env.reset(seed=0)
    return live


def _answer(live):
    st = live.env.battle_state
    mask = live.env.action_parser.action_mask(st, 1)
    a = DefenderReply().bind(live.engine).reply(st, 1, mask)
    if a == 0:
        return None
    a -= 1
    return (a % CELLS) % NX, (a % CELLS) // NX  # (x, y) in the defender's own frame


def test_a_knight_at_our_left_bridge_is_met_in_that_lane():
    # our left bridge (x 3) is the defender's right lane: its own-frame x is 17 - 3 = 14
    xy = _answer(_state_with([(True, 3, 14)]))
    assert xy is not None
    x, y = xy
    assert x >= NX / 2 and y <= 12


def test_nothing_to_answer_when_nothing_attacks():
    assert _answer(_state_with([(True, 9, 3)])) is None
