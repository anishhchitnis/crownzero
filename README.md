# CrownZero

An AI that plays Clash Royale, built to become the strongest bot there is. It thinks inside
[RoyaleSim](https://github.com/RoyaleGym/RoyaleSim), an open-source re-implementation of the
battle engine calibrated against recordings of real battles (through the
[RoyaleGym](https://github.com/RoyaleGym/RoyaleGym) API), and plays the real game through
[Build-A-Bot](https://github.com/Pbatch/ClashRoyaleBuildABot)'s screen reader. Everything
runs on a laptop CPU.

## How it plays

Every move, in the real game:

1. **See.** Build-A-Bot reads the screenshot: every unit's type, side and tile, the four
   cards in hand and the next one, elixir, and the princess towers' health bars.
2. **Rebuild.** CrownZero rebuilds that exact board inside the simulator: the units on their
   tiles, the towers at their health, our hand dealt in the right order, the clock.
3. **Search.** For "wait" and for each candidate play (each card in hand at sensible spots:
   in front of each tower, the bridges, behind the king, on top of enemy units for spells),
   it simulates six seconds ahead and scores the result in elixir: tower health, crowns,
   banked elixir and the value of every unit on the board.
4. **Play** the best one, by tapping the card and the tile.

It thinks in about 0.1-0.4 s per move. It never sees anything a human player can't: no
unit health beyond what's drawn, no opponent hand or elixir.

## Versions

| Version | What changed | Result |
|---|---|---|
| Search bot | Lookahead search with full information, simulator only | 100-0 against all 5 scripted bots |
| v1 | Neural network trained to imitate the search bot (behaviour cloning) | 92-8 against the scripted bots; 0-20 against the search bot |
| v2 | v1 retrained on its own mistakes (DAgger correction round) | No better than v1 (30-30 head to head), not shipped |
| v3 | Live search: rebuild the board seen on screen in the simulator, search it every move | 18-2 against v1; 20-0 against defend and patient |
| v4 | Any deck, any card; current card art; tower bars smoothed; deck mismatch warning | Fixes the misread hand found in real-game recordings |
| v4.1 | Ignores arena decorations read as troops; our own deploying troops stay ours | Fixes the phantom defence found in the next recording |
| v4.2 | Memory across frames (elixir after a play, ally/enemy vote, attackers hidden behind a tower); the opponent defends inside the look-ahead | Fixes the four mistakes found in the third recording (below); in the simulator 40-40 against v4.1 and 6-14 against full-information search, same as v4.1 |

### What the real-game recordings showed (v3 → v4)

`--record` saves every frame with what the bot saw and did. Five Training Camp games
showed the bot's problems were mostly perception, not strategy:

- It was run with a deck different from the one in the game. The reader forces every card
  into the deck it is told, so it misread a third of the hand and, because its ally/enemy
  split trusts the deck, it took our own Giant, Knight and Archers for enemy troops.
- Build-A-Bot's Goblin Hut picture is older art than the live game's, so even with the
  right deck it read the Goblin Hut as another card.
- A princess tower on 84 HP has a bar too short to read and came out as "destroyed".

A second recording (v4, PEKKA's Playhouse arena) showed two more:

- Two lamps beside our right tower, outside the battlefield, were detected as an enemy
  Executioner and Monk on every frame. The bot "defended" against them with its very
  first card, in the corner.
- That card, still in its deploy animation, was then classified as an enemy, so the bot
  played another card on top of it: a chain of plays into an empty corner.

v4.1 drops detections outside the battlefield and, for four seconds after each play,
treats that card's units near where we placed them as ours.

v4 takes any 8 cards by name, maps all 110 cards and 96 unit types to the simulator, uses
in-game card art where Build-A-Bot's is outdated, smooths the tower bars and warns when
the hand on screen doesn't match the deck. Replaying the recorded battle through v4, the
hand and the ally/enemy split read correctly.

### Third recording (v4.1 → v4.2)

`scripts/analyze_battle.py` flags every unanswered threat, tower hit, elixir leak and odd
play in a recording; `scripts/replay_battle.py` runs the current bot on the recorded
frames to show what it would now do in each of those moments. The third battle had four
kinds of mistakes:

| Mistake | Cause | Fix |
|---|---|---|
| "Not enough Elixir!": cards tapped and refused | The screenshot after a play still shows the elixir from before it | Elixir after a play is capped at what we had minus the cost plus regeneration; a tap that didn't land (card still in its slot) is forgotten |
| An enemy Mini P.E.K.K.A took a tower from 100% to 21% in 4 s, unanswered | Standing at the foot of the tower it is drawn behind it, and the detector missed it | A tower losing health with nothing seen near it gets an attacker assumed in front of it |
| Our Giant and Knight read as enemies now and then (20 flips) | Both decks had them; the side classifier flickers | Each unit keeps the side most of its last six frames gave it |
| Archers sent to the left bridge while an enemy Giant walked down the right lane; troops sent alone to the bridge | In the look-ahead the opponent stood still, so a troop at the bridge always reached a tower | The opponent now answers our troops in their lane about a second later, with a generic hand |

On the recorded frames, v4.2 stops the refused taps, defends the hidden Mini P.E.K.K.A at the
tower, and meets the enemy Giant in front of the right tower. A longer look-ahead when
enemies are on our side (10 s instead of 6 s) was tried and dropped: 6-14 against 6 s.

## Baseline: the scripted bots are rock-paper-scissors

Round robin of the five scripted bots that ship with RoyaleGym (40 games per pairing):

| Bot | Overall score | Loses to |
|---|---|---|
| defend | 74% | patient |
| first-affordable | 50% | random, defend |
| random | 46% | defend, push |
| push | 41% | first-affordable, defend |
| patient | 39% | random, first-affordable, push |

## Known weaknesses (next work)

- Six-second lookahead: it undervalues buildings and slow tanks. (A longer lookahead
  did not help: 7-13 against the 6-second version.) The rollout opponent only answers once,
  with a generic hand, not the real opponent's cards.
- The screen reader misses small units such as skeletons and can't see hidden buildings.
- Card and tower levels are the simulator's, not the account's.
- Next: learn from top players' replays (recorder in `crownzero/real/capture.py`), and
  self-play training to get past the hand-written scoring.

## Run it

Real game (macOS or Windows, Android emulator with ADB): see [REAL_GAME.md](REAL_GAME.md).

Simulator experiments:

```bash
python -m venv .venv && .venv/bin/pip install -e ".[sim]"
.venv/bin/python scripts/eval_bot.py --a live-search --b defend patient --games 20
.venv/bin/python scripts/record_replay.py defend 7    # writes replays/*.html
```

## Layout

```
crownzero/search_bot.py      lookahead search (candidates, rollouts, scoring)
crownzero/opponent_model.py  how the opponent answers inside the rollouts
crownzero/live_search.py     rebuild a seen board in the simulator and search it
crownzero/features.py        what the bot sees (the same from the simulator and the screen)
crownzero/model.py, train.py the imitation network (v1/v2)
crownzero/arena.py           parallel matches with seat-balanced Wilson intervals
crownzero/real/              the real game: Build-A-Bot bridge, card names, recorder,
                             memory across frames (perception.py)
scripts/analyze_battle.py    post-match report of a recorded battle
scripts/replay_battle.py     the current bot on a recorded battle's frames
tests/                       unit and end-to-end tests (no phone needed)
results/                     raw per-game results (JSON lines)
```

## Fair play

Supercell's terms ban automating the game. Use Training Camp or friendly battles on a
throwaway account only.
