"""Where would a pro put this card? A small network trained on pro replays.

Input, in the player's own frame (32 rows x 18 columns, row 0 = their back row):
  * the board: the value (elixir, by remaining health) of every unit on each tile, split
    into theirs / ours and ground / air / building; where this card may be placed;
  * the card: its cost, kind, unit size and count, whether it flies, and (for cards seen
    in replays) a learnt code of its own;
  * the towers' health, our elixir and the clock.
Output: a log-probability for every tile the card may be placed on.

The search bot uses it two ways (search_bot.py): the few tiles a pro would most likely
pick are tried as extra candidate plays, and plays on tiles a pro would never pick get a
small penalty. The simulator still decides; the prior only says where to look first.

    python -m crownzero.replays.prior train ~/Downloads/cr_replays/*/plays.jsonl \\
        --out models/pro_prior.pt

Training targets are soft: a pro's tile is read from the picture to within about a tile,
so the target is a small Gaussian around it (on legal tiles only). Every example is also
used mirrored left to right. One replay is held out to report how well it generalises.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np

NY, NX = 32, 18
KINDS = ("TROOP", "BUILDING", "SPELL")
BOARD_CH = 9   # 6 unit planes, legal, row, column
SIGMA = 1.0    # tiles: how sure a replay tile is
POCKET_ROWS = (15, 20)  # own-frame rows of the pocket in front of a fallen enemy tower


# ---------------------------------------------------------------- shared with the bot
def board_features(state, team: int, cards: dict, arena) -> dict:
    """The board from ``team``'s side: units [[mine, flying, building, value, x, y]]
    (own-frame tiles) and towers [my L, my R, their L, their R, my king, their king]
    (health shares). ``cards``: card_id -> the engine's CardInfo."""
    from royalegym import to_own

    from ..board import SPAWN_RATIO, TILE

    units = []
    for e in state.entities:
        if e.kind not in (0, 1) or e.hp <= 0:
            continue
        c = cards.get(e.card_id)
        if c is None:
            value = e.max_hp / 600
        else:
            value = c.elixir / max(c.count, 1)
            hp = int(getattr(c, "hitpoints", 0) or 0)
            if hp > 0 and 0 < e.max_hp < SPAWN_RATIO * hp:
                value *= e.max_hp / hp
        value *= e.hp / max(e.max_hp, 1)
        ox, oy = to_own(arena, team, e.x, e.y)
        units.append([int(e.team == team), int(bool(e.flying)), int(e.kind == 1),
                      round(float(value), 3), round(ox / TILE, 2), round(oy / TILE, 2)])
    me, en = state.players[team], state.players[1 - team]

    def share(p, i):
        return round(p.tower_hp[i] / max(p.tower_max_hp[i], 1), 3)

    towers = [share(me, 1), share(me, 2), share(en, 2), share(en, 1), share(me, 0), share(en, 0)]
    return dict(units=units, towers=towers)


def card_info(sim_name: str | None, cost: int | None, kind: str | None) -> dict:
    """What the prior knows about a card: from the simulator's catalogue when it has the
    card, else only its cost and kind (read from the replay)."""
    from .facts import SIM_CARDS

    c = SIM_CARDS.get(sim_name) if sim_name else None
    if c is not None:
        return dict(name=sim_name, cost=c["elixir"], kind=c["kind"], count=c["count"],
                    hp=c["hp"], flying=c["flying"], radius=c["radius"])
    return dict(name=None, cost=cost or 3, kind=kind or "TROOP", count=1, hp=0,
                flying=False, radius=0.5)


def legal_mask(info: dict, towers, anywhere: bool = False) -> np.ndarray:
    """Tiles this card may go on: our half (rows 0-14), the pocket on a side whose enemy
    princess tower is down, or anywhere for spells and the Miner."""
    m = np.zeros((NY, NX), bool)
    if anywhere or info["kind"] == "SPELL" or info.get("name") in ("Miner", "GoblinDrill"):
        m[:] = True
        return m
    m[:15] = True
    if towers is not None:
        if towers[2] <= 0:
            m[POCKET_ROWS[0]:POCKET_ROWS[1] + 1, :9] = True
        if towers[3] <= 0:
            m[POCKET_ROWS[0]:POCKET_ROWS[1] + 1, 9:] = True
    return m


def encode(board: dict, info: dict, elixir: float, clock: float, vocab: dict,
           anywhere: bool = False):
    """-> (planes (BOARD_CH, 32, 18), globals vector, card index, legal mask)."""
    planes = np.zeros((BOARD_CH, NY, NX), np.float32)
    for mine, fly, bld, value, x, y in board["units"]:
        ch = (0 if mine else 3) + (2 if bld else (1 if fly else 0))
        # spread over the four nearest tiles
        x, y = min(max(x - 0.5, 0), NX - 1), min(max(y - 0.5, 0), NY - 1)
        x0, y0 = int(x), int(y)
        fx, fy = x - x0, y - y0
        for dx, wx in ((0, 1 - fx), (1, fx)):
            for dy, wy in ((0, 1 - fy), (1, fy)):
                xi, yi = min(x0 + dx, NX - 1), min(y0 + dy, NY - 1)
                planes[ch, yi, xi] += value * wx * wy
    legal = legal_mask(info, board["towers"], anywhere)
    planes[6] = legal
    planes[7] = np.linspace(0, 1, NY)[:, None]
    planes[8] = np.linspace(0, 1, NX)[None, :]
    k = info["kind"] if info["kind"] in KINDS else "TROOP"
    g = np.array(list(board["towers"]) + [
        elixir / 10.0, min(clock, 300) / 180.0, info["cost"] / 10.0,
        *[float(k == kk) for kk in KINDS], math.log1p(info["hp"]) / 8.0,
        min(info["count"], 15) / 15.0, float(info["flying"]), min(info["radius"], 3) / 3.0,
        float(info["name"] is None)], np.float32)
    idx = vocab.get(info["name"], 0) if info["name"] else 0
    return planes, g, idx, legal


N_GLOBAL = 6 + 2 + 1 + 3 + 5


def soft_target(tile, legal: np.ndarray) -> np.ndarray:
    ys, xs = np.mgrid[0:NY, 0:NX]
    x, y = tile
    t = np.exp(-((xs + 0.5 - (x + 0.5)) ** 2 + (ys + 0.5 - (y + 0.5)) ** 2) / (2 * SIGMA ** 2))
    t = t * legal
    if t.sum() <= 0:  # the tile read is off the legal area: the nearest legal tile
        d = (xs - x) ** 2 + (ys - y) ** 2 + (~legal) * 1e6
        t = np.zeros((NY, NX))
        t.flat[int(np.argmin(d))] = 1.0
    return (t / t.sum()).astype(np.float32)


# ---------------------------------------------------------------- the network
def make_net(n_cards: int, width: int = 16, emb: int = 0, dropout: float = 0.3,
             mix: float = 0.2):
    """``emb``: size of a learnt code per named card (0: cards are known only by their
    traits, which is what generalises from a few hundred plays); ``mix``: share of the
    output spread evenly over the legal tiles, so a tile the replays never showed is
    unlikely, not impossible."""
    import torch
    from torch import nn

    class FiLMConv(nn.Module):
        def __init__(self, cin, cout, dil, cond):
            super().__init__()
            self.conv = nn.Conv2d(cin, cout, 3, padding=dil, dilation=dil)
            self.film = nn.Linear(cond, 2 * cout)

        def forward(self, x, c):
            h = self.conv(x)
            g, b = self.film(c).chunk(2, dim=1)
            return torch.relu(h * (1 + g[:, :, None, None]) + b[:, :, None, None])

    class PriorNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.n_cards = n_cards
            self.emb = nn.Embedding(n_cards + 1, emb) if emb else None
            self.cond = nn.Sequential(nn.Linear(N_GLOBAL + emb, 32), nn.ReLU())
            dims = [BOARD_CH] + [width] * 4
            self.layers = nn.ModuleList(
                FiLMConv(a, b, d, 32) for a, b, d in zip(dims, dims[1:], (1, 2, 4, 1)))
            self.out = nn.Conv2d(width, 1, 1)
            self.drop = nn.Dropout(dropout)

        def forward(self, planes, g, idx, legal):
            c = torch.cat([g, self.emb(idx)], dim=1) if self.emb is not None else g
            c = self.cond(c)
            h = planes
            for layer in self.layers:
                h = self.drop(layer(h, c))
            logits = self.out(h)[:, 0]
            logits = logits.masked_fill(~legal, -1e9)
            logp = torch.log_softmax(logits.flatten(1), dim=1).view_as(logits)
            if mix > 0:
                n = legal.flatten(1).sum(dim=1).clamp(min=1).float()[:, None, None]
                logp = torch.logaddexp(logp + math.log(1 - mix), torch.full_like(logp, 0.0)
                                       + math.log(mix) - torch.log(n))
                logp = logp.masked_fill(~legal, -1e9)
            return logp

    return PriorNet()


# ---------------------------------------------------------------- data
def load_examples(files) -> list[dict]:
    out = []
    for f in files:
        for line in open(f):
            r = json.loads(line)
            if r.get("tile") is None or r.get("board") is None:
                continue
            out.append(r)
    return out


def mirror(r: dict) -> dict:
    m = dict(r)
    b = r["board"]
    m["board"] = dict(units=[[u[0], u[1], u[2], u[3], 18 - u[4], u[5]] for u in b["units"]],
                      towers=[b["towers"][1], b["towers"][0], b["towers"][3], b["towers"][2],
                              b["towers"][4], b["towers"][5]])
    m["tile"] = [17 - r["tile"][0], r["tile"][1]]
    return m


def tensors(examples, vocab, augment: bool):
    import torch

    rows = []
    for r in examples:
        for ex in ([r, mirror(r)] if augment else [r]):
            info = card_info(ex.get("sim"), ex.get("cost"), ex.get("kind"))
            planes, g, idx, legal = encode(ex["board"], info, ex.get("elixir") or 5.0,
                                           ex.get("t", 0.0), vocab, anywhere=ex.get("thrown", False))
            rows.append((planes, g, idx, legal, soft_target(ex["tile"], legal)))
    P = torch.tensor(np.stack([r[0] for r in rows]))
    G = torch.tensor(np.stack([r[1] for r in rows]))
    I = torch.tensor([r[2] for r in rows], dtype=torch.long)
    L = torch.tensor(np.stack([r[3] for r in rows]))
    T = torch.tensor(np.stack([r[4] for r in rows]))
    return P, G, I, L, T


def evaluate(net, data, examples) -> dict:
    """Mean log-likelihood of the pro's tile (soft target), and how often the pro's tile
    is within 1.5 tiles of the model's best / top-5 tiles."""
    import torch

    net.eval()
    P, G, I, L, T = data
    with torch.no_grad():
        logp = net(P, G, I, L)
    ll = float((T * logp.clamp(min=-30)).sum(dim=(1, 2)).mean())
    top1 = top5 = 0
    flat = logp.flatten(1)
    for k, r in enumerate(examples):
        x, y = r["tile"]
        best = torch.topk(flat[k], 5).indices.tolist()
        d = [math.hypot(b % NX - x, b // NX - y) for b in best]
        top1 += d[0] <= 1.5
        top5 += min(d) <= 1.5
    n = max(len(examples), 1)
    return dict(n=len(examples), loglik=round(ll, 3), top1=round(top1 / n, 3),
                top5=round(top5 / n, 3))


def baseline(examples) -> dict:
    """The same scores for "any legal tile, equally likely" (what the search had)."""
    ll = top = 0.0
    for r in examples:
        info = card_info(r.get("sim"), r.get("cost"), r.get("kind"))
        legal = legal_mask(info, r["board"]["towers"], r.get("thrown", False))
        ll += -math.log(legal.sum())
        top += min(1.0, 7.0 / legal.sum())  # ~7 tiles lie within 1.5 tiles of a point
    n = max(len(examples), 1)
    return dict(loglik=round(ll / n, 3), top1=round(top / n, 3))


NET = dict(width=16, emb=0, dropout=0.3, mix=0.2)


def train(files, out: str | None = None, epochs: int = 40, seed: int = 0,
          hold_out: str | None = "last", log=print, net_kw: dict | None = None,
          weight_decay: float = 1e-2):
    """Train on the plays in ``files``. ``hold_out``: a replay name kept out to measure
    with ("last": the most recent; None: train on everything). Returns (net, report)."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    examples = load_examples(files)
    replays = sorted({r["replay"] for r in examples})
    held = replays[-1] if hold_out == "last" and len(replays) > 1 else hold_out
    tr = [r for r in examples if r["replay"] != held]
    te = [r for r in examples if r["replay"] == held]
    names = sorted({r["sim"] for r in tr if r.get("sim")})
    vocab = {n: i + 1 for i, n in enumerate(names)}
    net_kw = dict(NET, **(net_kw or {}))
    net = make_net(len(vocab), **net_kw)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=weight_decay)
    data = tensors(tr, vocab, augment=True)
    test = tensors(te, vocab, augment=False) if te else None
    n = data[0].shape[0]
    log(f"training on {len(tr)} plays from {len(replays) - (1 if te else 0)} replays "
        f"({n} with mirrors); {len(vocab)} named cards; held out: {held} ({len(te)} plays)")
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n)
        tot = 0.0
        for b in range(0, n, 64):
            ix = perm[b:b + 64]
            P, G, I, L, T = (d[ix] for d in data)
            logp = net(P, G, I, L)
            loss = -(T * logp.clamp(min=-30)).sum(dim=(1, 2)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(ix)
        if (ep + 1) % 20 == 0 or ep == epochs - 1:
            msg = f"  epoch {ep + 1}: train loss {tot / n:.3f}"
            if test is not None:
                msg += f" | held out {evaluate(net, test, te)}"
            log(msg)
    report = dict(train=evaluate(net, tensors(tr, vocab, False), tr),
                  held_out=evaluate(net, test, te) if test is not None else None,
                  baseline_held_out=baseline(te) if te else None, held_replay=held,
                  replays=len(replays), plays=len(examples))
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        kinds: dict[str, int] = {}
        for r in examples:
            k = card_info(r.get("sim"), r.get("cost"), r.get("kind"))["kind"]
            kinds[k] = kinds.get(k, 0) + 1
        torch.save(dict(state=net.state_dict(), vocab=vocab, report=report, net=net_kw,
                        kind_counts=kinds), out)
        log(f"saved {out}")
    return net, report


# ---------------------------------------------------------------- using it
class ProPrior:
    """``placement(board, sim_name, elixir, clock)`` -> (32, 18) log-probabilities, own
    frame, -inf where the card may not go."""

    def __init__(self, path):
        import torch

        torch.set_num_threads(1)  # one small net per move; the search runs alongside
        blob = torch.load(path, map_location="cpu", weights_only=False)
        self.vocab = blob["vocab"]
        self.report = blob.get("report")
        self.net = make_net(len(self.vocab), **blob.get("net", {}))
        self.net.load_state_dict(blob["state"])
        self.net.eval()
        self.cache = {}
        self.kind_counts = blob.get("kind_counts", {})

    # a kind of card the replays showed fewer plays of than this gets no advice: the first
    # prior had seen 11 buildings, and sent a Tombstone behind the King Tower to open a
    # game (recorded 2026-10-13)
    MIN_KIND_PLAYS = 30

    def knows(self, sim_name: str) -> bool:
        kind = card_info(sim_name, None, None)["kind"]
        return self.kind_counts.get(kind, 0) >= self.MIN_KIND_PLAYS

    def placement(self, board: dict, sim_name: str, elixir: float = 5.0,
                  clock: float = 60.0) -> np.ndarray | None:
        """None for a kind of card the prior has too few examples of."""
        import torch

        if not self.knows(sim_name):
            return None
        info = card_info(sim_name, None, None)
        planes, g, idx, legal = encode(board, info, elixir, clock, self.vocab)
        with torch.no_grad():
            logp = self.net(torch.tensor(planes)[None], torch.tensor(g)[None],
                            torch.tensor([idx]), torch.tensor(legal)[None])[0].numpy()
        return np.where(legal, logp, -np.inf)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("files", nargs="+")
    t.add_argument("--out", default="models/pro_prior.pt")
    t.add_argument("--epochs", type=int, default=40)
    t.add_argument("--all", action="store_true", help="hold nothing out")
    args = ap.parse_args()
    if args.cmd == "train":
        _, report = train(args.files, args.out, args.epochs,
                          hold_out=None if args.all else "last")
        print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
