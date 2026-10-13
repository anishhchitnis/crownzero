# CrownZero

An AI that learns to play Clash Royale, aiming to be the strongest bot that exists.
It trains inside [RoyaleSim](https://github.com/RoyaleGym/RoyaleSim), an open-source Rust
re-implementation of the battle engine that is calibrated against recordings of real
battles, through the [RoyaleGym](https://github.com/RoyaleGym/RoyaleGym) environment API.
Everything runs on a laptop CPU.

## Status

| Stage | What | Status |
|---|---|---|
| 1 | Simulator set up, scripted-bot baseline | done |
| 2 | Search bot (lookahead "teacher") | **done: 100-0 against every scripted bot** |
| 3 | Student neural network that imitates the teacher using only fair, human-visible information | next |
| 4 | Self-play reinforcement learning against a league of past versions | |
| 5 | Deck mastery across all 18 calibrated cards | |
| 6 | Stretch: real-game demo in Training Camp | |

## Results so far

### Baseline: the scripted bots are rock-paper-scissors

Round robin of the five scripted bots that ship with RoyaleGym, 40 games per pairing, one
mirror deck (Knight, Archer, Giant, Minions, Fireball, Cannon, Zap, Musketeer):

| Bot | Overall score | Loses to |
|---|---|---|
| defend | 74% | patient |
| first-affordable | 50% | random, defend |
| random | 46% | defend, push |
| push | 41% | first-affordable, defend |
| patient | 39% | random, first-affordable, push |

No scripted bot beats all the others. That made the first bar: beat all five, with 95%
confidence, from both seats.

### Search bot vs every scripted bot

20 games per opponent (10 per seat, the same seeds each way round). Intervals are 95%
Wilson intervals on the win rate.

| Opponent | Record | Win rate (95% CI) | 3-crown wins |
|---|---|---|---|
| random | 20-0 | 100% [84%, 100%] | 19 |
| first-affordable | 20-0 | 100% [84%, 100%] | 16 |
| defend | 20-0 | 100% [84%, 100%] | 16 |
| push | 20-0 | 100% [84%, 100%] | 20 |
| patient | 20-0 | 100% [84%, 100%] | 19 |

On a second deck it was never tuned for (Hog Rider, Valkyrie, Musketeer, Skeleton Army,
Tesla, Log, Fireball, Baby Dragon) it went 10-0 against defend and 10-0 against patient.

The scripted ladder is now used up as a measuring stick. From here on, progress is measured
against the search bot itself and against past versions of the learned bot.

## How the search bot works

Every half second of game time it:

1. Lists candidate plays: each card in hand at a few sensible spots (in front of each tower,
   the bridges, behind the king, on top of enemy units for spells), snapped to the nearest
   cell the rules allow. Typically 12 to 24 candidates (median 22).
2. Saves the engine, then for "wait" and for each candidate: reloads the save, makes the
   play, simulates 6 seconds ahead with the opponent idle, and scores the result.
3. Restores the engine exactly (checked by state hash every decision) and plays the best
   candidate, or waits if nothing beats waiting.

The score is in elixir units: tower health (a princess tower is worth 12, the king 30),
crowns, banked elixir, and the elixir value of every unit on the board, with a bonus for
units standing near an enemy tower. Leaking elixir at 10 falls out naturally: waiting gains
nothing while a play converts elixir into board value.

What it can see: the board, its own hand and elixir, which is what a human sees. It never
reads the opponent's hand or elixir. Its one unfair edge is using the simulator as a
perfect "what if" machine. That is why it is the teacher, not the final bot.

Every recorded battle re-runs identically from its commands alone
(`royalegym.verify_trace`), so the search never touches the real game state.

## Run it

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/eval_vs_ladder.py 20         # search bot vs every scripted bot
.venv/bin/python scripts/record_replay.py defend 7    # writes replays/*.html, open in a browser
```

## Layout

```
crownzero/board.py        geometry, action ids, card table
crownzero/search_bot.py   the lookahead search bot
crownzero/arena.py        parallel match runner with seat-balanced Wilson intervals
scripts/                  evaluation and replay recording
results/                  raw per-game results (JSON lines)
replays/                  self-contained HTML replays
```

## Fair play

Everything here runs in a simulator. Supercell's fair-play rules ban automating the real
game, so any real-game demo (stage 6) is limited to Training Camp on a throwaway account.
