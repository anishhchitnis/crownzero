"""Train the student network to copy the search bot (behaviour cloning).

python -m crownzero.train --data data/teacher --out models/student.pt
"""
from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from .board import CELLS, NX
from .dataset import Games
from .model import PolicyNet, load, masked_logits, save


def evaluate(net, games: Games, idx: np.ndarray, bs: int = 512) -> dict:
    net.eval()
    tot = dict(nll=0.0, w=0.0, dec=0, n=0, plays=0, exact=0, card=0, near=0, vmse=0.0)
    with torch.no_grad():
        for s in range(0, len(idx), bs):
            sp, vec, mask, act, w, out = games.batch(idx[s:s + bs])
            logits, value = net(sp, vec)
            lg = masked_logits(logits, mask)
            nll = F.cross_entropy(lg, act, reduction="none")
            tot["nll"] += float((nll * w).sum())
            tot["w"] += float(w.sum())
            p = lg.softmax(1)
            pred_play = p[:, 0] < 0.5
            teacher_play = act > 0
            # weights undo the no-op subsampling, so decision accuracy is weighted too
            tot["dec"] += float(((pred_play == teacher_play).float() * w).sum())
            tot["n"] += float(w.sum())
            plays = teacher_play.nonzero().squeeze(1)
            if len(plays):
                best = lg[plays, 1:].argmax(1) + 1
                a = act[plays]
                tot["plays"] += len(plays)
                tot["exact"] += int((best == a).sum())
                bs_, as_ = (best - 1) // CELLS, (a - 1) // CELLS
                tot["card"] += int((bs_ == as_).sum())
                by, bx = ((best - 1) % CELLS) // NX, ((best - 1) % CELLS) % NX
                ay, ax = ((a - 1) % CELLS) // NX, ((a - 1) % CELLS) % NX
                near = (bs_ == as_) & ((by - ay).abs() <= 2) & ((bx - ax).abs() <= 2)
                tot["near"] += int(near.sum())
            tot["vmse"] += float(((value - out) ** 2).sum())
    k = max(tot["plays"], 1)
    return dict(nll=tot["nll"] / tot["w"], decision_acc=tot["dec"] / tot["n"],
                play_exact=tot["exact"] / k, play_card=tot["card"] / k,
                play_near=tot["near"] / k, value_mse=tot["vmse"] / len(idx))


def train(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    games = Games(args.data, args.max_games)
    tr, va = games.split(0.1, seed=args.seed)
    print(f"{len(games.files)} games, {games.n} samples ({len(tr)} train / {len(va)} val), "
          f"{int((games.actions > 0).sum())} plays", flush=True)
    net = load(args.init) if args.init else PolicyNet(width=args.width, trunk=args.trunk)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = math.ceil(len(tr) / args.bs)
    total = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total,
                                                pct_start=0.1)
    rng = np.random.default_rng(args.seed)
    best = math.inf
    for ep in range(args.epochs):
        net.train()
        perm = rng.permutation(tr)
        t0, run = time.perf_counter(), 0.0
        for s in range(0, len(perm), args.bs):
            sp, vec, mask, act, w, out = games.batch(perm[s:s + args.bs],
                                                     rng if args.augment else None)
            logits, value = net(sp, vec)
            pol = (F.cross_entropy(masked_logits(logits, mask), act, reduction="none") * w).sum()
            pol = pol / w.sum()
            val = F.mse_loss(value, out)
            loss = pol + args.value_weight * val
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            run += float(pol.detach())
        m = evaluate(net, games, va)
        print(f"epoch {ep + 1}/{args.epochs}: train nll {run / steps_per_epoch:.3f} | val "
              + " ".join(f"{k} {v:.3f}" for k, v in m.items())
              + f" | {time.perf_counter() - t0:.0f}s", flush=True)
        if m["nll"] < best:
            best = m["nll"]
            save(net, args.out, metrics=m, epoch=ep + 1, data=str(args.data))
    print(f"best val nll {best:.3f} -> {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", default=None)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--trunk", type=int, default=48)
    ap.add_argument("--value-weight", type=float, default=0.25)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--max-games", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--augment", type=int, default=1)
    train(ap.parse_args())


if __name__ == "__main__":
    main()
