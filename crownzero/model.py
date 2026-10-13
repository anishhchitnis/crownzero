"""The student network: board in, a probability for every move (and a win estimate) out.

Input: spatial planes [C, 32, 18] and a vector [V] from ``features.featurize``.
Output: 2305 move logits in the simulator's action layout (index 0 = wait, then
hand slot x row x column), and a value in [-1, 1] (expected result: +1 win, -1 loss).

Shape: a small U-Net. One full-resolution layer reads the board, the trunk thinks at half
resolution (16 x 9 tiles' worth of 2x2 blocks) where it is 4x cheaper, and the result is
upsampled and joined with the full-resolution features so every tile gets its own score.
About 0.25M parameters and ~50M multiply-adds per decision: fast on a laptop CPU.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .board import HAND
from .features import C, V


class ResBlock(nn.Module):
    def __init__(self, ch: int, dilation: int = 1):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=dilation, dilation=dilation)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.n1 = nn.GroupNorm(8, ch)
        self.n2 = nn.GroupNorm(8, ch)

    def forward(self, x):
        h = F.relu(self.n1(self.c1(x)))
        h = self.n2(self.c2(h))
        return F.relu(x + h)


class PolicyNet(nn.Module):
    def __init__(self, width: int = 32, trunk: int = 48, vec_dim: int = 32,
                 dilations=(1, 2, 1, 2)):
        super().__init__()
        self.config = dict(width=width, trunk=trunk, vec_dim=vec_dim, dilations=tuple(dilations))
        self.vec = nn.Sequential(nn.Linear(V, 128), nn.ReLU(), nn.Linear(128, vec_dim))
        self.stem = nn.Conv2d(C, width, 3, padding=1)
        self.stem_vec = nn.Linear(vec_dim, width)
        self.down = nn.Conv2d(width, trunk, 3, stride=2, padding=1)
        self.trunk_vec = nn.Linear(vec_dim, trunk)
        self.blocks = nn.ModuleList([ResBlock(trunk, d) for d in dilations])
        self.glob = nn.Linear(2 * trunk + vec_dim, trunk)
        self.up = nn.Conv2d(trunk + width, width, 3, padding=1)
        self.head = nn.Conv2d(width, HAND, 1)
        pooled = 2 * trunk + vec_dim
        self.noop = nn.Sequential(nn.Linear(pooled, 64), nn.ReLU(), nn.Linear(64, 1))
        self.value = nn.Sequential(nn.Linear(pooled, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, sp: torch.Tensor, vec: torch.Tensor):
        v = self.vec(vec)
        s0 = F.relu(self.stem(sp) + self.stem_vec(v)[:, :, None, None])  # [B, w, 32, 18]
        x = F.relu(self.down(s0) + self.trunk_vec(v)[:, :, None, None])  # [B, t, 16, 9]
        half = len(self.blocks) // 2
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i == half - 1:  # mid-trunk: let every cell see the whole board
                g = torch.cat([x.mean((2, 3)), x.amax((2, 3)), v], 1)
                x = x + self.glob(g)[:, :, None, None]
        pooled = torch.cat([x.mean((2, 3)), x.amax((2, 3)), v], 1)
        u = F.interpolate(x, size=s0.shape[2:], mode="nearest")
        u = F.relu(self.up(torch.cat([u, s0], 1)))
        cells = self.head(u).flatten(1)  # [B, 4*32*18] in (slot, y, x) order
        logits = torch.cat([self.noop(pooled), cells], 1)  # [B, 2305]
        value = torch.tanh(self.value(pooled)).squeeze(1)
        return logits, value


def masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(~mask, -1e9)


def save(net: PolicyNet, path, **extra):
    torch.save({"config": net.config, "state": net.state_dict(), **extra}, path)


def load(path, device="cpu") -> PolicyNet:
    ck = torch.load(path, map_location=device, weights_only=False)
    net = PolicyNet(**ck["config"])
    net.load_state_dict(ck["state"])
    net.eval()
    return net
