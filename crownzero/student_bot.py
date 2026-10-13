"""The student bot: the trained network playing in the simulator, no lookahead.

It builds the same View the real-game bridge builds (no unit health, no enemy elixir),
asks the network for move probabilities, and plays if the network puts more than
``play_threshold`` of its probability on playing something, choosing the single most
likely play. Only legal moves are considered.
"""
from __future__ import annotations

import numpy as np
import torch

from .board import NOOP
from .features import featurize_views, from_sim
from .model import load, masked_logits


def decide(net, views, masks, play_threshold=0.5, temperature=0.0, rng=None,
           force_at_full=True):
    """Batched decision. masks: bool array [B, 2305]. Returns action ids and P(wait).

    ``force_at_full``: at 10 elixir, play the best move rather than wait. The search bot
    never sits on full elixir, so the network has almost never seen it and its wait
    estimate there is a guess; a leaking bar is never right."""
    sp, vec = featurize_views(views)
    with torch.no_grad():
        logits, value = net(torch.from_numpy(sp), torch.from_numpy(vec))
        lg = masked_logits(logits, torch.from_numpy(np.asarray(masks, bool)))
        p = lg.softmax(1).numpy()
    out = []
    for i in range(len(views)):
        full = force_at_full and views[i].elixir >= 10.0 - 1e-6
        if (p[i, 0] >= play_threshold and not full) or not np.asarray(masks[i])[1:].any():
            out.append(NOOP)
            continue
        plays = p[i, 1:]
        if temperature > 0 and rng is not None:
            q = plays ** (1 / temperature)
            out.append(int(rng.choice(len(q), p=q / q.sum())) + 1)
        else:
            out.append(int(plays.argmax()) + 1)
    return out, p[:, 0], value.numpy()


class StudentBot:
    name = "student"

    def __init__(self, path=None, net=None, play_threshold=0.5, temperature=0.0):
        torch.set_num_threads(1)
        self.net = net if net is not None else load(path)
        self.net.eval()
        self.play_threshold = play_threshold
        self.temperature = temperature
        self.cards = None
        self.arena = None

    def view(self, env, team):
        if self.cards is None:
            self.cards = {c.card_id: c for c in env.engine.cards()}
            self.arena = env.engine.arena()
        return from_sim(env.battle_state, team, self.cards, self.arena)

    def reply(self, state, team, mask):
        """Opponent model for the search bot's rollouts: our move from a raw state."""
        mask = np.asarray(mask).astype(bool)
        if not mask[1:].any():
            return NOOP
        view = from_sim(state, team, self.cards, self.arena)
        return decide(self.net, [view], [mask], self.play_threshold)[0][0]

    def bind(self, engine):
        self.cards = {c.card_id: c for c in engine.cards()}
        self.arena = engine.arena()
        return self

    def act(self, env, team, obs, mask, rng):
        mask = np.asarray(mask).astype(bool)
        if not mask[1:].any():
            return NOOP
        actions, _, _ = decide(self.net, [self.view(env, team)], [mask],
                               self.play_threshold, self.temperature, rng)
        return actions[0]
