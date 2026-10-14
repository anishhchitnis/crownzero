# Learning from top-player replays

CrownZero learns where top players place their cards by watching their replays. A replay
draws **both** players' hands and elixir bars, so every play of both players can be read:
which card, when, its cost, and where it landed. Nothing is played or tapped; this only
reads what is on screen while you watch a replay.

## Which replays

* A top player's battle log: Leaderboard (trophy icon) > Top Players > tap a player >
  Battle Log > the replay button on a battle.
* TV Royale works too.

Any cards, any arena. 1v1 battles only, no 2v2 or special modes. About 5 a day is plenty.

## Record

In Terminal, inside the ClashRoyaleBuildABot folder with its environment active:

```bash
python -m crownzero.real.capture --device-serial 127.0.0.1:5555 --name pro_6
```

Start the replay, let it play to the end, then press **Ctrl+C** once and wait for "done".
One replay per run, with a new name each time (`pro_6`, `pro_7`, ...). The video lands in
`~/Downloads/cr_replays/pro_6_<date-time>/`.

## Teach the bot (once a day)

```bash
python -m crownzero.replays.daily ~/Downloads/cr_replays
```

It reads every new replay (about 45 seconds each on a laptop), then retrains the placement
prior on all of them into `crownzero/models/pro_prior.pt`, printing how well the prior
predicts the newest replay's plays before seeing it. The bot uses the new prior the next
time it starts.

## How a replay is read

1. **When**: a player's elixir bar drops. The replay bar moves a whole segment at a time,
   so the drop is also the card's cost.
2. **Which card**: the hand slot whose card changed just then. Cards are told apart by
   their pictures, learnt from the replay itself (greyed out, lit up or held, a card still
   matches itself), so the same card is recognised every time it is played. Its name comes
   from a picture library: Build-A-Bot's card art plus crops from replays
   (`crownzero/replays/card_crops/`); a match whose cost disagrees with the elixir spent is
   rejected.
3. **Where**: the arena is stretched back to where a battle draws it, and the spot that
   changed most just after the play (a troop dropping in, a spell's splash), minus what was
   already moving, is the tile, on the part of the arena that player may use.
4. **The board before it**: every play so far is replayed in the simulator, in order, with
   the towers set from the replay. (Build-A-Bot's unit reader misses most of what pros
   play.)

## Naming a card it doesn't know

Each replay folder gets a `cards/` folder with one picture per card. A file named
`top_3_unknown_4.png` is a 4-elixir card the library could not name. If you know it, copy
the picture into `crownzero/replays/card_crops/` named after the card with two
underscores, for example `mighty_miner__1.png` (Build-A-Bot's card names:
lower case, underscores). Then run the daily command with `--redo` to read the replays
again with it.
