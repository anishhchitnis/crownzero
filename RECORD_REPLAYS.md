# Recording top-player replays

CrownZero learns from top players by watching their replays: the same screen reader the
bot plays with turns a replay video into plays (which card, where, when, and the board just
before), and a network learns where and what a pro plays in each situation. This only
records what is on the screen while you watch a replay; nothing is played or tapped.

## Which replays

The best replays show **the top player at the bottom of the screen with their 4 cards in
hand visible**: the hand is how the extractor knows exactly which card was played (spells
included), so those plays are the clean training data.

* A top player's battle log: Leaderboard (trophy icon) > Top Players > tap a player >
  Battle Log > the replay button on a battle. The replay is shown from that player's side.
* TV Royale works too if it shows the bottom player's hand.

Any cards, any arena. Ladder (1v1) battles only, no 2v2 or special modes.

## Record

In Terminal, inside the ClashRoyaleBuildABot folder with its environment active:

```bash
python -m crownzero.real.capture --device-serial 127.0.0.1:5555 --name pro_1
```

Start the replay, let it play to the end, then press **Ctrl+C** once and wait for "done".
One replay per run, with a new name each time (`pro_2`, `pro_3`, ...).

The video lands in `~/Downloads/cr_replays/pro_1_<date-time>/` as `part_000.mp4`, ...

## Check one

```bash
python -m crownzero.replays.extract ~/Downloads/cr_replays/pro_1_<date-time>
```

prints how many plays it found for each side and which cards. A full 3-minute battle gives
about 25-40 plays for the bottom player; far fewer means the hand is not visible in that
kind of replay.
