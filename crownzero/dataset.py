"""Load recorded games into memory and serve featurized training batches."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .features import featurize


class Games:
    def __init__(self, dirs, max_games: int | None = None):
        files = []
        for d in ([dirs] if isinstance(dirs, (str, Path)) else dirs):
            files += sorted(Path(d).glob("game_*.npz"))
        if max_games:
            files = files[:max_games]
        units, raws, masks, actions, weights, outcomes, game_of = [], [], [], [], [], [], []
        self.meta = []
        n = 0
        for gi, f in enumerate(files):
            d = np.load(f)
            k = len(d["raws"])
            u = d["units"].copy()
            u[:, 0] += n
            units.append(u)
            raws.append(d["raws"])
            masks.append(d["masks"])
            actions.append(d["actions"].astype(np.int64))
            weights.append(d["weights"])
            outcomes.append(d["outcomes"])
            game_of.append(np.full(k, gi, np.int32))
            self.meta.append(json.loads(str(d["meta"])))
            n += k
        self.files = files
        self.units = np.concatenate(units) if units else np.zeros((0, 7), np.float32)
        order = np.argsort(self.units[:, 0], kind="stable")
        self.units = self.units[order]
        self.raws = np.concatenate(raws)
        self.masks = np.concatenate(masks)
        self.actions = np.concatenate(actions)
        self.weights = np.concatenate(weights)
        self.outcomes = np.concatenate(outcomes)
        self.game_of = np.concatenate(game_of)
        sid = self.units[:, 0].astype(np.int64)
        self.u_start = np.searchsorted(sid, np.arange(n), "left")
        self.u_end = np.searchsorted(sid, np.arange(n), "right")
        self.n = n

    def split(self, val_frac=0.1, seed=0):
        rng = np.random.default_rng(seed)
        g = len(self.files)
        val_games = set(rng.choice(g, size=max(1, int(g * val_frac)), replace=False).tolist())
        is_val = np.array([gi in val_games for gi in self.game_of])
        return np.flatnonzero(~is_val), np.flatnonzero(is_val)

    def batch(self, idx: np.ndarray, augment: np.random.Generator | None = None):
        """Featurized batch. With ``augment`` (an rng), imitate the real game's screen
        reader: each unit is missed 8% of the time and misplaced by a tile 15% of the time."""
        rows = [self.units[self.u_start[i]:self.u_end[i]].copy() for i in idx]
        for j, r in enumerate(rows):
            r[:, 0] = j
        units = np.concatenate(rows) if rows else np.zeros((0, 7), np.float32)
        if augment is not None and len(units):
            units = units[augment.random(len(units)) >= 0.08]
            jitter = augment.random(len(units)) < 0.15
            units[jitter, 3] += augment.integers(-1, 2, jitter.sum())
            units[jitter, 4] += augment.integers(-1, 2, jitter.sum())
        sp, vec = featurize(units, self.raws[idx])
        mask = np.unpackbits(self.masks[idx], axis=1)[:, :2305].astype(bool)
        return (torch.from_numpy(sp), torch.from_numpy(vec), torch.from_numpy(mask),
                torch.from_numpy(self.actions[idx]), torch.from_numpy(self.weights[idx]),
                torch.from_numpy(self.outcomes[idx]))
