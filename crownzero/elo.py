"""Elo ratings from match records (Bradley-Terry fit, draws count half)."""
from __future__ import annotations

import json
import math
from collections import defaultdict


def fit_elo(records, anchor: str | None = None, anchor_elo: float = 1000.0, iters: int = 2000):
    """records: iterable of (a, b, score_a, games). Returns {name: elo}."""
    wins = defaultdict(float)
    games = defaultdict(float)
    names = set()
    for a, b, score, n in records:
        names |= {a, b}
        wins[(a, b)] += score
        wins[(b, a)] += n - score
        games[(a, b)] += n
        games[(b, a)] += n
    names = sorted(names)
    r = {n: 0.0 for n in names}  # natural-log strengths
    for _ in range(iters):  # minorise-maximise updates (Hunter 2004), lightly regularised
        new = {}
        for i in names:
            w = 0.5 + sum(wins[(i, j)] for j in names if j != i)
            d = 1.0 / (math.exp(r[i]) + 1.0) + sum(
                games[(i, j)] / (math.exp(r[i]) + math.exp(r[j])) for j in names if j != i)
            new[i] = math.log(w / d)
        mean = sum(new.values()) / len(new)
        r = {k: v - mean for k, v in new.items()}
    elo = {k: v * 400 / math.log(10) for k, v in r.items()}
    shift = anchor_elo - (elo[anchor] if anchor in elo else 0.0)
    return {k: round(v + shift) for k, v in sorted(elo.items(), key=lambda kv: -kv[1])}


def records_from_jsonl(path):
    out = []
    for line in open(path):
        m = json.loads(line)
        score = sum(1.0 if g["outcome"] == "win" else 0.5 if g["outcome"] == "draw" else 0.0
                    for g in m["games"] if g["outcome"] != "unfinished")
        n = sum(1 for g in m["games"] if g["outcome"] != "unfinished")
        out.append((m["a"], m["b"], score, n))
    return out
