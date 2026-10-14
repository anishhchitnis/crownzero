"""The pro-replay pipeline pieces that need no video: hand reading, placement, the
placement prior and its use in the search."""
import json

import numpy as np
import pytest

from crownzero.replays.layout import (DiffPlacer, PlayReader, ReplayLayout, card_desc,
                                      px_to_tile, tile_to_px)
from crownzero.replays.prior import (encode, card_info, legal_mask, mirror, soft_target,
                                     train)


def test_tile_pixel_round_trip():
    for x, y in [(0, 0), (8.5, 14), (17, 31), (3.2, 20.7)]:
        assert np.allclose(px_to_tile(*tile_to_px(x, y)), (x, y), atol=1e-6)


def test_card_desc_ignores_greying_out():
    from PIL import Image

    rng = np.random.default_rng(0)
    art = (rng.random((55, 45, 3)) * 255).astype(np.uint8)
    card = Image.fromarray(art)
    grey = Image.fromarray((art.mean(axis=2, keepdims=True).repeat(3, 2) * 0.5 + 100)
                           .astype(np.uint8))
    other = Image.fromarray((rng.random((55, 45, 3)) * 255).astype(np.uint8))
    same = card_desc(card) @ card_desc(grey) / 256
    diff = card_desc(card) @ card_desc(other) / 256
    assert same > 0.8 and abs(diff) < 0.3


def _reader(drops, slot_cards, card_descs):
    """A PlayReader fed a scripted elixir bar and hand: ``slot_cards[t]`` = 4 card keys."""
    r = PlayReader(fps=4)
    n_shift = len(ReplayLayout.SHIFTS["bottom"])
    for i, (e, cards) in enumerate(zip(drops, slot_cards)):
        d = np.stack([np.repeat(card_descs[c][None], n_shift, axis=0) for c in cards])
        r.add(i * 0.25, d, [False] * 4, e)
    return r


def test_play_reader_finds_plays_costs_and_repeats(monkeypatch):
    import crownzero.replays.layout as lay

    class NoLibrary:
        def name(self, desc):
            return None, 0.0
    monkeypatch.setattr(lay, "CardLibrary", NoLibrary)
    rng = np.random.default_rng(1)
    descs = {}
    for k in "ABCDEFGH":
        d = rng.normal(size=256).astype(np.float32)
        descs[k] = (d - d.mean()) / d.std()
    # A (cost 3) played at frame 10 from slot 0 (E comes in), then E (cost 2) at 30 from
    # slot 0 (F comes in), then B (cost 4) at 50 from slot 1, then A again at 70
    hands, elix = [], []
    e = 5.5
    for i in range(90):
        if i < 10:
            h = "ABCD"
        elif i < 30:
            h = "EBCD"
        elif i < 50:
            h = "FBCD"
        elif i < 70:
            h = "FACD" if i >= 51 else "FBCD"
        else:
            h = "FGCD"
        hands.append(h)
        if i in (10, 30, 50, 70):
            e -= {10: 3, 30: 2, 50: 4, 70: 3}[i]
        e = min(e + 0.07, 10)
        elix.append(float(int(e)))  # the replay bar shows whole elixir
    r = _reader(elix, hands, descs)
    plays = r.plays()
    got = [(round(p.t, 2), p.slot, p.cost) for p in plays]
    assert [g[1] for g in got] == [0, 0, 1, 1]
    assert [g[2] for g in got] == [3, 2, 4, 3]
    assert plays[0].card == plays[3].card != plays[1].card  # A both times


def test_diff_placer_finds_a_new_unit_on_its_own_half():
    from PIL import Image, ImageDraw

    dp = DiffPlacer()
    rng = np.random.default_rng(2)
    base = (rng.random((652, 368)) * 40 + 100).astype(np.uint8)
    X, Y = tile_to_px(5, 8)
    for i in range(12):
        img = Image.fromarray(base.copy())
        if i >= 6:  # a unit appears at tile (5, 8) at t = 1.5 s
            ImageDraw.Draw(img).ellipse([X - 8, Y - 8, X + 8, Y + 8], fill=255)
        # and something moves on the other half the whole time
        ImageDraw.Draw(img).rectangle([100 + 10 * i, 100, 115 + 10 * i, 115], fill=0)
        dp.add(i * 0.25, img.convert("RGB"))
    (x, y), _ = dp.place(1.4, "bottom")
    assert abs(x - 5) <= 1 and abs(y - 8) <= 1


def test_legal_mask_and_pockets():
    troop = card_info("Knight", None, None)
    m = legal_mask(troop, [1, 1, 1, 1, 1, 1])
    assert m[:15].all() and not m[15:].any()
    m = legal_mask(troop, [1, 1, 0, 1, 1, 1])  # their left tower is down
    assert m[18, 2] and not m[18, 15]
    assert legal_mask(card_info("Fireball", None, None), [1] * 6).all()


def test_soft_target_stays_on_legal_tiles():
    legal = np.zeros((32, 18), bool)
    legal[:15] = True
    t = soft_target([4, 14.6], legal)
    assert abs(t.sum() - 1) < 1e-5 and t[15:].sum() == 0 and t.argmax() // 18 == 14


def _examples(n=60, seed=0):
    """Synthetic replays: Knights always go 2 tiles in front of the nearest enemy unit on
    our half, Fireballs on the enemy unit."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        ex, ey = float(rng.integers(2, 16)), float(rng.integers(5, 14))
        card = "Knight" if i % 2 else "Fireball"
        tile = [ex, max(ey - 2, 0)] if card == "Knight" else [ex, ey]
        out.append(dict(replay=f"r{i % 3}", t=60.0, side="bottom", card=card.lower(),
                        sim=card, cost=None, kind=None, thrown=card == "Fireball",
                        tile=tile, elixir=6.0, towers=[1, 1, 1, 1],
                        board=dict(units=[[0, 0, 0, 5.0, ex, ey]], towers=[1, 1, 1, 1, 1, 1])))
    return out


def test_mirror_flips_left_and_right():
    r = _examples(2)[1]
    m = mirror(r)
    assert m["tile"][0] == 17 - r["tile"][0]
    assert m["board"]["units"][0][4] == 18 - r["board"]["units"][0][4]


def test_prior_learns_where_cards_go_and_helps_the_search(tmp_path):
    torch = pytest.importorskip("torch")  # noqa: F841
    files = []
    for k, chunk in enumerate((_examples(90, 0), _examples(30, 1))):
        f = tmp_path / f"r{k}.jsonl"
        for r in chunk:
            r["replay"] = f"replay{k}"
        f.write_text("\n".join(json.dumps(r) for r in chunk))
        files.append(str(f))
    out = tmp_path / "prior.pt"
    _, report = train(files, str(out), epochs=60, log=lambda *a: None)
    assert report["held_out"]["top5"] > report["baseline_held_out"]["top1"] + 0.3
    from crownzero.replays.prior import ProPrior
    pp = ProPrior(out)
    board = dict(units=[[0, 0, 0, 5.0, 12.0, 10.0]], towers=[1, 1, 1, 1, 1, 1])
    logp = pp.placement(board, "Knight")
    y, x = divmod(int(np.argmax(logp)), 18)
    assert abs(x - 12) <= 2 and abs(y - 8) <= 2
    assert np.isinf(logp[20]).all()  # a Knight never goes on their half


def test_live_search_uses_the_prior(tmp_path):
    pytest.importorskip("royalegym")
    from crownzero.features import View
    from crownzero.live_search import LiveSearch

    f = tmp_path / "r.jsonl"
    rows = _examples(40)
    for i, r in enumerate(rows):
        r["replay"] = f"replay{i % 2}"
    f.write_text("\n".join(json.dumps(r) for r in rows))
    out = tmp_path / "prior.pt"
    train([str(f)], str(out), epochs=5, log=lambda *a: None)
    deck = ["Giant", "MiniPekka", "Fireball", "Archer", "GoblinHut", "Minions", "Musketeer",
            "Knight"]
    live = LiveSearch(deck, {"prior": str(out)})
    v = View()
    v.hand_names = ["MiniPekka", "GoblinHut", "Minions", "Knight"]
    v.next_name = "Giant"
    v.hand = [-1, -1, 5, 0]
    v.elixir = 10.0
    v.clock = 0.3
    v.units = [(0, 15, 14, 17, 0, 5.0)]
    v.unit_names = ["Prince"]
    a = live.decide(v, allowed_slots=[0, 1, 2, 3])
    assert a != 0
    maps = live.search._prior_maps
    assert set(maps) == {0, 1, 2, 3} and all(np.isfinite(m).any() for m in maps.values())


def test_prior_bonus_marks_down_unlikely_tiles():
    from crownzero.search_bot import NX, NY, PRIOR_FLOOR, SearchBot, action_index

    bot = SearchBot(prior=object(), prior_weight=0.5)
    logp = np.full((NY, NX), -np.inf)
    logp[3, 4], logp[10, 10], logp[8, 2] = -1.0, -3.0, -20.0
    bot._prior_maps = {1: logp}
    assert bot._prior_bonus(action_index(1, 3, 4)) == 0.0
    assert bot._prior_bonus(action_index(1, 10, 10)) == -1.0
    assert bot._prior_bonus(action_index(1, 8, 2)) == -0.5 * PRIOR_FLOOR
    assert bot._prior_bonus(action_index(2, 3, 4)) == 0.0  # no map for that slot
