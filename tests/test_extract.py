"""Plays from the hand: a card leaves its slot and the "next" card takes its place."""
from types import SimpleNamespace as NS

from crownzero.replays.extract import Extractor


def state(hand, allies=(), enemies=()):
    nums = {k: NS(number=1.0) for k in ("left_ally_princess_hp", "right_ally_princess_hp",
                                        "left_enemy_princess_hp", "right_enemy_princess_hp")}
    unit = lambda n, x, y: NS(unit=NS(name=n, category="troop"),  # noqa: E731
                              position=NS(tile_x=x, tile_y=y))
    return NS(cards=[NS(name=c) for c in hand], numbers=NS(**nums),
              allies=[unit(*a) for a in allies], enemies=[unit(*e) for e in enemies])


def test_a_card_replaced_by_the_next_card_is_a_play_with_its_units_tile():
    ex = Extractor(read_hand=True)
    ex.feed(0.0, state(["knight", "archers", "giant", "minions", "fireball"]))
    ex.feed(0.5, state(["knight", "archers", "giant", "minions", "fireball"]))
    # giant played: slot 2 now shows the old next card (knight); a new next card appears
    ex.feed(1.0, state(["zap", "archers", "knight", "minions", "fireball"],
                       allies=[("giant", 14, 10)]))
    ex.feed(1.5, state(["zap", "archers", "knight", "minions", "fireball"],
                       allies=[("giant", 14, 11)]))
    # a misread: slot 1 shows a card that was not next: not a play
    ex.feed(2.0, state(["zap", "musketeer", "knight", "minions", "fireball"],
                       allies=[("giant", 14, 12)]))
    plays = ex.finish()
    assert [(p.card, p.side, p.source) for p in plays] == [("giant", "bottom", "hand")]
    assert plays[0].tile == [14.0, 10.0]
