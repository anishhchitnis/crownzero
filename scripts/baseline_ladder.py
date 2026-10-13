"""Baseline: how the scripted bots rank on the real (Rust) engine with one fixed deck."""
import itertools

from royalegym import ClashParallelEnv, DefaultStateMutator, RustEngine, evaluate, ladder

DECK_NAMES = ("Knight", "Archer", "Giant", "Minions", "Fireball", "Cannon", "Zap", "Musketeer")


def make_env():
    engine = RustEngine()
    by_name = {c.name: c.card_id for c in engine.cards()}
    deck = [by_name[n] for n in DECK_NAMES]
    return ClashParallelEnv(engine=engine, state_mutator=DefaultStateMutator(decks=[deck, deck]))


bots = dict(ladder())
names = [n for n in bots if n != "noop"]
for a, b in itertools.combinations(names, 2):
    res = evaluate(bots[a], bots[b], make_env, games=40, seed=0, names=(a, b))
    print(f"{a:>16} vs {b:<16} W{res.wins:>3} L{res.losses:>3} D{res.draws:>3}  "
          f"score {res.score_rate:.0%}")
