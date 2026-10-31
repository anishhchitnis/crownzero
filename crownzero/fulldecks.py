"""Decks from the WHOLE card pool, with evolutions, heroes and champions.

The simulator (royalesim 0.1.42) plays 136 ladder cards (145 less 9 event cards). Of them,
42 have an evolution, 16 a hero form (an ability button, one charge) and 8 are champions
(an ability button on a cooldown). A deck is 8 card names plus a ``forms`` row parallel to
it: 0 the card itself, 1 its evolution, 2 its hero form (royalegym ``MatchSetup.forms``).

How many special slots a deck gets is a setting (``Rules``), because the game has changed it
more than once. The default is the shape this project plays: up to 2 evolutions, up to 1
hero, at most 1 champion. The engine itself does not cap them.

``FullDeck`` is the unit everything passes around; ``as_setup`` turns a pair of them into
the (decks, forms) the state mutator takes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Cards with a hero form on royalesim 0.1.42 (probed: a reset with form 2 loads; tests
#: re-check this list against the engine, so a new simulator release cannot drift silently).
HERO_CARDS = frozenset({
    "Knight", "Goblins", "Giant", "Balloon", "Valkyrie", "Musketeer", "Wizard", "MiniPekka",
    "DarkPrince", "Bowler", "IceGolemite", "MegaMinion", "EliteArcher", "Berserker",
    "Tombstone", "BarbLog",
})


@dataclass(frozen=True)
class Rules:
    max_evos: int = 2
    max_heroes: int = 1
    max_champions: int = 1
    min_spells: int = 1
    max_buildings: int = 2
    elixir_range: tuple[float, float] = (2.6, 4.6)  # average elixir a playable deck keeps
    p_special: float = 0.85  # chance each special slot is filled when a card fits it


@dataclass(frozen=True)
class FullDeck:
    cards: tuple[str, ...]
    forms: tuple[int, ...] = field(default=(0,) * 8)

    def __post_init__(self):
        if len(self.cards) != 8 or len(self.forms) != 8:
            raise ValueError("a deck is 8 cards with 8 forms")

    @property
    def evos(self) -> list[str]:
        return [c for c, f in zip(self.cards, self.forms) if f == 1]

    @property
    def heroes(self) -> list[str]:
        return [c for c, f in zip(self.cards, self.forms) if f == 2]

    def label(self) -> str:
        tag = {0: "", 1: "Evo ", 2: "Hero "}
        return ", ".join(tag[f] + c for c, f in zip(self.cards, self.forms))


def _deck(spec: str) -> FullDeck:
    """'Evo:Knight Hero:Giant Fireball ...' -> FullDeck."""
    cards, forms = [], []
    for tok in spec.split():
        form, _, name = tok.rpartition(":")
        cards.append(name)
        forms.append({"": 0, "Evo": 1, "Hero": 2}[form])
    return FullDeck(tuple(cards), tuple(forms))


#: Recognisable archetypes, written with the special slots a ladder player would use.
#: Names are the simulator's (Supercell internal): Assassin = Bandit, EliteArcher = Magic
#: Archer, Ghost = Royal Ghost, DarkWitch = Night Witch, AxeMan = Lumberjack,
#: IceGolemite = Ice Golem.
META = {
    "hog26": _deck("HogRider Evo:Skeletons Evo:Cannon Musketeer IceGolemite IceSpirits "
                   "Fireball Log"),
    "logbait": _deck("Evo:GoblinBarrel Evo:Princess GoblinGang Knight InfernoTower Rocket "
                     "Log IceSpirits"),
    "giant_hero": _deck("Hero:Giant Evo:Musketeer Evo:Valkyrie MiniPekka Minions Fireball "
                        "Zap Cannon"),
    "pekka_bs": _deck("Evo:Pekka BattleRam Assassin ElectroWizard Evo:Ghost EliteArcher "
                      "Poison Zap"),
    "lavaloon": _deck("LavaHound Hero:Balloon MegaMinion Minions Tombstone Fireball Zap "
                      "Evo:Bats"),
    "xbow": _deck("Xbow Evo:Tesla Evo:Knight Archer IceSpirits Skeletons Fireball Log"),
    "royalhogs": _deck("Evo:RoyalHogs RoyalRecruits Hero:Goblins Zap Arrows Fireball "
                       "Evo:Barbarians ElectroSpirit"),
    "champ_aq": _deck("ArcherQueen Evo:Knight Hero:MiniPekka HogRider Evo:Skeletons "
                      "Fireball Log Tesla"),
    "mk": _deck("MegaKnight Evo:Bats Evo:Wallbreakers Miner Assassin Zap Arrows Hero:Bowler"),
    "golem": _deck("Golem DarkWitch Evo:BabyDragon AxeMan Tornado Zap Hero:Tombstone "
                   "Evo:Bomber"),
}


#: Catalogue rows that look like special-mode or helper cards, not ladder cards. Judged from
#: the names alone (not checked against the live card list): royalegym's EVENT_CARDS misses
#: these, and random decks full of them would not look like anything a player brings.
NON_LADDER = frozenset({
    "PrinceBuff", "MinionGiant", "BarbarianLauncher", "GiantBuffer", "WarmSpell",
    "MergeMaiden_Normal", "MergeMaiden_Mounted", "SuperArcher", "SuperKnight",
    "SuperIceGolemite", "SuperHogRiderTerry",
})


def _non_ladder(name: str) -> bool:
    return name in NON_LADDER or "_Chess" in name or "_Spooky" in name


class Pool:
    """The engine's card pool, read once: kinds, costs, which cards evolve, heroes, champions."""

    def __init__(self, engine) -> None:
        from royalegym.state_mutator import EVENT_CARDS

        self.cards = {c.name: c for c in engine.cards()
                      if c.name not in EVENT_CARDS and not _non_ladder(c.name)}
        self.names = sorted(self.cards)
        self.evo = {n for n, c in self.cards.items() if c.evo_cycle}
        self.champions = {n for n, c in self.cards.items() if c.champion}
        self.heroes = {n for n in HERO_CARDS if n in self.cards}

    def kind(self, name: str) -> str:
        return self.cards[name].card_kind or "TROOP"

    def valid(self, deck: FullDeck, rules: Rules = Rules()) -> list[str]:
        """What is wrong with ``deck`` under ``rules`` (empty: nothing)."""
        bad = [c for c in deck.cards if c not in self.cards]
        if bad:
            return [f"not in the simulator: {', '.join(bad)}"]
        out = []
        if len(set(deck.cards)) != 8:
            out.append("cards repeat")
        for c, f in zip(deck.cards, deck.forms):
            if f == 1 and c not in self.evo:
                out.append(f"{c} has no evolution")
            if f == 2 and c not in self.heroes:
                out.append(f"{c} has no hero form")
        if len(deck.evos) > rules.max_evos:
            out.append(f"{len(deck.evos)} evolutions > {rules.max_evos}")
        if len(deck.heroes) > rules.max_heroes:
            out.append(f"{len(deck.heroes)} heroes > {rules.max_heroes}")
        if sum(c in self.champions for c in deck.cards) > rules.max_champions:
            out.append("too many champions")
        return out

    def random(self, rng: np.random.Generator, rules: Rules = Rules()) -> FullDeck:
        """A random deck from the whole pool that a ladder player could bring: at least
        ``min_spells`` spells, a sane average cost, the special slots filled at random."""
        names = self.names
        lo, hi = rules.elixir_range
        while True:
            pick = [names[i] for i in rng.choice(len(names), size=8, replace=False)]
            if sum(c in self.champions for c in pick) > rules.max_champions:
                continue
            kinds = [self.kind(c) for c in pick]
            if kinds.count("SPELL") < rules.min_spells:
                continue
            if kinds.count("BUILDING") > rules.max_buildings:
                continue
            avg = sum(self.cards[c].elixir for c in pick) / 8
            if not lo <= avg <= hi:
                continue
            break
        forms = [0] * 8
        order = list(rng.permutation(8))
        evos = heroes = 0
        for i in order:
            c = pick[i]
            if c in self.heroes and heroes < rules.max_heroes and rng.random() < rules.p_special:
                forms[i] = 2
                heroes += 1
            elif c in self.evo and evos < rules.max_evos and rng.random() < rules.p_special:
                forms[i] = 1
                evos += 1
        return FullDeck(tuple(pick), tuple(forms))

    def pick(self, rng: np.random.Generator, rules: Rules = Rules(), p_meta: float = 0.4):
        """A META deck (when every card is in this simulator) or a random one."""
        if rng.random() < p_meta:
            ok = [d for d in META.values() if not self.valid(d, rules)]
            if ok:
                return ok[int(rng.integers(len(ok)))]
        return self.random(rng, rules)


def as_setup(decks) -> tuple[list[list[str]], list[list[int]]]:
    """[blue, red] FullDecks (or plain name lists) -> (decks, forms) for the state mutator."""
    out_d, out_f = [], []
    for d in decks:
        if isinstance(d, FullDeck):
            out_d.append(list(d.cards))
            out_f.append(list(d.forms))
        else:
            out_d.append(list(d))
            out_f.append([0] * len(d))
    return out_d, out_f


__all__ = ["FullDeck", "Pool", "Rules", "META", "HERO_CARDS", "as_setup"]
