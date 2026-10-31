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
| v5.5 | A new unit detector, trained on generated images (`crownzero/detector/`, `--detector`); not yet trained or measured | Built because the eighth to tenth recordings all failed on what the screen reader saw |
| v5.4 | The battle ends only after 5 s without a battle screen; a warning when the bot joins a battle already under way | Tenth recording: a 3.5 s stretch of unrecognised frames reset the clock and the memory mid-battle; the bot's first frame showed 10 elixir and a tower at 49% |
| v5.3 | Keeps more elixir back while an enemy push is coming: one more elixir for every elixir of enemy troops on our half or within 7 rows of the river, up to 7 (`reserve_per_threat`) | Ninth recording: a Hog sent with the last 4 elixir as a Sparky and Minion Horde crossed the other bridge; the tower fell 5 s later. Simulator, the real-game brain under the client's delay: 23-7 against v5.2 [59%, 88%] |
| v5.2 | Detections standing on a crown tower that still stands are dropped (the enemy's tower troops read as Bombers, a "Baby Dragon" on our King Tower) | Eighth recording: 2 phantom Bombers on all 277 frames, Hog Rider never played. Simulator, Hog 2.6 vs `defend` under the client's delay, same 8 seeds: with the phantoms 6-2 and 17 Hog plays; without 8-0 and 39 |
| v5 | The whole card pool: random legal decks from 125 ladder cards and 10 meta archetypes, with up to 2 evolutions, 1 hero and 1 champion (royalesim 0.1.42); the search presses hero and champion ability buttons; a snapshot the simulator cannot reload is skipped instead of crashing; `--private` recording; the live client's command delay and the screen's lag can be simulated | Full-pool decks: 49-1 against the 5 scripted bots (10 games each). Three real-game ideas, each 30 games under the client's delay: delay-aware look-ahead 13-17, tracking the opponent's cards 14-16, aiming spells ahead 16-14; none clear of chance, so all three stay off |
| v4.7 | Our own spawners' units stay ours; the pro prior only advises card kinds it has data for, and no building goes on the back row; a reserve of 4 elixir; the opponent's elixir counted (recorded only); sits out matchmade battles | Replaying the seventh battle: plays answering only our own Skeletons 7 → 1, our Skeletons read as enemies 450 → 175 frames (the rest come from the Tombstone the game moved, which the building rule now prevents), no Tombstone opener. Simulator: reserve 4 vs 3, 23-17; elixir counting 15-25 (recorded, not used) |
| v4.6 | Learns from top players' replays: a replay reader (both players' plays: card, cost, tile, and the board before, rebuilt in the simulator) and a placement prior trained on them; the search tries the prior's top tiles for each card and leans toward them | Simulator A/B against v4.5, 40 games each, random decks: 26-14 (crowns 37-25), and 24-16 with the prior's tiles as candidates only |
| v4.5 | Tower check that works in any arena (learns each tower's badge); bars: median of three, never rising; a unit first seen on its owner's half keeps that side; enemies missed for a frame are remembered; the detector picks the fastest backend on the machine (CoreML on a Mac) | Fixes the four mistakes of the sixth recording (a 3-crown win in 82 s) |
| v4.4 | Sees destroyed towers (level badge gone) and the enemy King Tower's health; plays into the pocket once a tower falls (simulator: 19-21, but 5-0 in 3-crown wins); one-frame bar dips ignored; lighter recording, shorter pause after plays | Fixes the destroyed tower read as 3% for 2 minutes in the fifth recording (a 3-crown win) |
| v4.3 | Fresh screenshots, one-drag plays checked against the hand, detector on 2 cores; summoned units valued by their size; buildings keep their side | Fixes the 41% of plays that never happened in the fourth recording, and the Goblin Hut overvalued; 15-15 against v4.2 in a mirror match with that deck |
| v4.2 | Memory across frames (elixir after a play, ally/enemy vote, attackers hidden behind a tower); the opponent defends inside the look-ahead | Fixes the four mistakes found in the third recording (below); in the simulator 40-40 against v4.1 and 6-14 against full-information search, same as v4.1 |

### A new unit detector (v5.5)

Three recordings in a row lost to what Build-A-Bot's detector saw: tower troops read as
Bombers, an Elixir Golem push read as five different cards, units flipping sides. Its model
cannot be retrained (no training data ships with it), so `crownzero/detector/` builds a new one
the way KataCR did (arXiv 2504.04783):

| Step | How |
|---|---|
| Sprites | 4,654 units cut from real battles, both sides, 134 kinds incl. evolutions (KataCR's detection dataset, MIT) |
| Arenas | The median of a recorded battle's frames: the units average away, the arena stays |
| Training images | Towers, then 4-32 units pasted where they could stand, back to front, plus health bars, emotes and text it must ignore; exact boxes; one class per unit and side (276), so no separate side classifier |
| Real frames | KataCR's hand-labelled frames: 8 folders added to training, one (557 frames, 7,163 boxes) kept out to measure on |
| Model | YOLOv8, trained on the Mac's GPU (`mps`), exported to ONNX |
| In the bot | `--detector models/units_cz.onnx`: same interface as Build-A-Bot's, towers and spells dropped, evolved units reported as their card |

A one-epoch smoke run in the cloud found our King Tower in the right place on a recorded frame
and dropped it as a tower; the real training runs on your Mac (`python -m
crownzero.detector.build`, REAL_GAME.md). Not measured yet: accuracy on the held-out real
frames, and a battle with it.

### Tenth recording (v5.3 → v5.4): the bot joined late

61 decisions, 13 plays, both our princess towers lost to an Elixir Golem push.

| Mistake | Cause | Fix |
|---|---|---|
| The opening was never played: the first frame showed 10 elixir and our right tower already at 49% | The battle was started before the bot printed "ready" (loading takes 10-20 s); the seventh to ninth battles also opened at 10 elixir | A warning when the first battle frame shows 9+ elixir or a damaged tower. Start the battle only after the "ready" line |
| At 45 s the bot's clock went back to 0 and its memory of towers, sides and the opponent's elixir was wiped | 3.5 s of frames the screen reader did not recognise as the battle counted as the end of it | The battle is over only after 5 s without a battle screen (`adapter.BattleEnd`) |
| The Elixir Golem push went unanswered | The screen reader called its units Elixir Golem, Electro Giant, Royal Giant, Giant and Knight in turn | The next work: a new detector |

### Ninth recording (v5.2 → v5.3): both towers lost in 83 seconds

The phantom fix held: no tower troops read as units, only 4 units flipping sides. The loss
was the bot's own: at 73 of 125 decisions it had 1 elixir. At 19.6 s it sent its Hog Rider
down the right lane with its last 4 elixir while a Sparky and a Minion Horde crossed the
left bridge; with nothing left but Ice Spirits, the left tower went from 100% to 8% in five
seconds. The six-second look-ahead does not see a push that lands after it ends, and the
fixed reserve of 4 does not grow when one is coming.

v5.3 keeps one more elixir back for every elixir of enemy troops on our half or within 7 rows
of the river (at their health), up to 7. In the simulator, the real-game brain with it against
the same brain without, under the client's delay and a view one step old: **23-7** [59%,
88%], 8 three-crown wins (`scripts/exp_threat.py`).

### Eighth recording (v5.1 → v5.2): Hog 2.6 in a friendly, Hog Rider never played

A friendly battle with `--private` (277 decisions, 56 plays, all of them happened; our right
tower at 21%, the enemy's at 95% and 100%). The bot only defended: 15 Ice Spirits, 12
Skeletons, 9 Logs, and the Hog Rider held in hand for 46 decisions without being played once.

| Mistake | Cause | Fix |
|---|---|---|
| Hog Rider never played | The enemy's tower troops, standing on their princess towers, were read as two enemy Bombers on every frame. The rebuilt board put two Bombers in front of each tower, so every Hog push lost in the look-ahead | Nothing can stand on a crown tower: a detection inside a standing tower is the tower's own troop or art, and is dropped (`adapter.on_tower`). Reproduced in the simulator (above): with the phantoms the Hog was played 17 times in 8 games, without them 39 |
| Ice Spirits played behind the King Tower at nothing, twice | A "Baby Dragon" read on our King Tower for 21 frames | Same rule |

What is still wrong in it, for next time: 715 detections outside the arena (dropped), 23
units flipping from ours to theirs, 104 of 1,108 hand slots read blank, and the screen reader
taking 280 ms a frame. The screen reader is what needs work next.

### v5: the whole card pool, and what was tested for the real game

**Simulator.** royalesim 0.1.42 plays evolutions (42 cards), hero forms (16) and champions
(8), each hero and champion with an ability button. `crownzero/fulldecks.py` deals decks
from the whole ladder pool (`Pool.random`: at least one spell, at most two buildings, average
cost 2.6-4.6, up to 2 evolutions, 1 hero and 1 champion; the limits are `Rules`) or one of
10 named archetypes (`META`). `arena.match(..., deck="full")` plays them; `deck="special"`
plays only archetypes with a hero or champion. The search tries every ability button the
mask allows as one more candidate.

| Search bot, full-pool decks, both seats | Result |
|---|---|
| vs random | 10-0 |
| vs first-affordable | 10-0 |
| vs defend | 10-0 |
| vs push | 10-0 |
| vs patient | 9-1 |

A test game with the Archer Queen archetype: 4 ability presses, a 3-0 win. The A/B of
abilities on and off crashed on a simulator bug (below) before it finished, so the value
of the button presses is not measured yet.

**A simulator bug.** With two Hero Bowlers on the board, royalesim 0.1.42 sometimes saves
a snapshot it then refuses to load ("snapshot entity tables are inconsistent"); a refused
load leaves the battle untouched. The search now tries the load before searching and waits
that decision out when it fails: in the game that crashed, 7 of 351 decisions.

**Real-game ideas, tested in the simulator.** The live client runs a play 21-22 ticks
(~1.1 s) after it is made, and the bot decides on a screenshot already a few hundred
milliseconds old; the simulator can now do both (`arena.match(..., delay=21)`,
`live-search` with `stale_steps`). Under them, the real-game brain against itself:

| Change (off by default) | Result, 30 games | Script |
|---|---|---|
| The rebuilt battle has the same delay, and is run forward by the lag (`command_delay`, `lag_steps`) | 13-17 [27%, 61%] | scripts/exp_delay.py |
| The look-ahead's opponent holds the cards it has been seen playing, in cycle order (`track_enemy_deck`) | 14-16 [30%, 64%] | scripts/exp_track.py |
| Spells also aimed 1.5 and 3 tiles ahead of the densest enemies (`lead_spells`) | 16-14 [36%, 70%] | scripts/exp_lead.py |

None is clear of chance, so none is on. 30 games can only show a large effect; these are
small at best in the simulator, where the bot's problems are not what they are in the real
game. The recordings showed the real game's problems are mostly the screen reader's, and the
next recording (with `--private`) decides what to fix next.

### Seventh recording (v4.6 → v4.7): lost 1-2 in sudden death

The recording opens on "Searching for opponent": a matchmade battle against a 931-trophy
player, not Training Camp. All 30 plays happened; these are the mistakes, and what the
battle's frames say caused them.

| Mistake | Cause | Fix |
|---|---|---|
| 10 of 30 plays answered enemy Skeletons that were our own Tombstone's | The Tombstone was never detected, so nothing marked its Skeletons as ours; Skeletons carry no level tag, and the side classifier called them enemies all game. Skeletons it had once seen came back as "ghost" enemies too | A card we play that keeps making units (Tombstone, Huts, Goblin Cage, Graveyard, ...) is remembered for its lifetime: its units near it, or first seen deep in our half near it, are ours for good and never come back as ghosts. Replaying the battle, Skeletons near our Tombstones read as enemies: 201 frames → 3 |
| Opened with a Tombstone behind the King Tower, which the game moved 3.6 tiles to the side | The placement prior had seen 11 building plays and sent buildings to the back row; the simulator allowed a building where it does not fit | The prior only advises kinds of card it has 30+ replay plays of (troops, spells); buildings never go on the King Tower's back row. Replayed: no Tombstone opener |
| Spent its elixir the moment it reached 3 (307 of 426 decisions at 2 or less); Goblin Cage, Goblin Hut and Fireball sat in hand for 47, 38 and 24 decisions; both towers fell to pushes it had nothing left to answer | The reserve was a fixed 3 elixir, and the look-ahead rebuilt the opponent with 5 elixir every time, whatever they had saved up | The reserve kept back is 4 elixir, enough for any 4-elixir answer (simulator A/B against 3: 23-17). Elixir counting was tried too: the opponent starts with 5, gains 1 every 2.8 s (1.4 s in double elixir) and pays for every new unit on their side; the rebuild used that and the reserve grew with it. It lost 15-25 against the plain reserve of 4 in the simulator, so the count is only recorded (`enemy_elixir` in steps.jsonl), not used to decide |

### Learning from top players (v4.6)

Watching a replay (Battle Log or TV Royale) shows both players' hands and elixir bars, so
every play of both players can be read. `crownzero/replays/` turns a replay video into
training examples (see [RECORD_REPLAYS.md](RECORD_REPLAYS.md)):

| Step | How | Why not the obvious way |
|---|---|---|
| When a card was played, and its cost | A player's elixir bar drops; the replay bar moves a whole segment at a time, so the drop is the cost | |
| Which card | The hand slot that changed then; cards are told apart by their pictures, learnt from the replay itself, and named from a library (Build-A-Bot's art plus replay crops). A name whose cost disagrees with the elixir spent is dropped | Build-A-Bot's 8x8 picture hash, compared against its art, picked the right card barely more often than chance on replay crops: pros play cards it has no art for (Monk, evolutions), and a held card is drawn 10 pixels off its slot |
| Where | The replay arena is stretched to the battle layout; the spot that changed most just after the play, minus what was already moving, on the part of the arena that player may use | The unit reader misses most of what pros play and takes many units for the wrong side |
| The board before it | Every play so far replayed in the simulator in order, towers set from the replay | Same; and it is the same kind of board the bot's search sees |

On the first 5 replays: 225 plays, every one placed, about two thirds with the card named once the
library had crops of the pros' cards; looking at every play of one replay on the frame just
after it, most marks sit on the troop's deploy clock, a few are off.

The **placement prior** (`replays/prior.py`) is a small convolutional network: the board
(units by side, ground/air/building, their value), where the card may go, and the card's
traits (cost, kind, size, count, flying) modulating every layer, give a probability for
every tile. Trained on 225 plays it is still weak: on a replay it has not seen, its first
pick is within 1.5 tiles of the pro's tile 9% of the time (2% for a random legal tile), and
its likelihood is no better than a per-kind heat map. So it only advises: the search tries
its 3 best tiles for each card as extra candidates and marks plays down by 0.3 elixir per
unit of log-probability below the card's best tile (at most 1.5); the simulator still
picks. That was worth 26-14 against v4.5 in the simulator. It is retrained daily as new
replays come in (`python -m crownzero.replays.daily ~/Downloads/cr_replays`).

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

### Fourth recording (v4.2 → v4.3): the bot won, but 12 of 29 plays never happened

`analyze_battle.py` now checks every play against the hand afterwards. In the fourth
battle (a win, 2 crowns) 12 of 29 plays never happened, and 13 of 29 in the third. The game
clock drawn on the screenshots showed why: right before most failed plays, the frame the bot
was deciding on was 4-6 s old ("2:40 left", then 2.1 s later "2:33").

| Mistake | Cause | Fix |
|---|---|---|
| Plays that never happened; a Goblin Hut tap that brought out Spear Goblins at the wrong tower; our left tower lost 85% to a Mini P.E.K.K.A while two defending plays failed | Build-A-Bot reads the screen as a video stream in a background thread that falls seconds behind when the detector and search are busy; the detector runs on every core and starves the emulator; a play was two separate tap commands, and a lost second tap left the card raised | A screenshot is taken when the bot looks (`screencap`); the detector gets 2 threads; a play is one drag from card to tile; every play is checked against the hand, and drags fall back to taps if they never land |
| Goblin Hut played first again and again | Units a card summons carry its id: each 133-hp Spear Goblin from a Hut was scored as a whole 4-elixir Hut (a Witch's skeletons likewise) | A summoned unit is worth its share of the card's own unit, by hitpoints. In the recorded Mini P.E.K.K.A defence the Hut's score fell from 32.9 to 24.4, next to the Mini P.E.K.K.A's 23.3 |
| Our Goblin Hut and its Spear Goblins read as enemies (30 flips) | Both decks were nearly the same cards | A building is on its owner's half; a unit first seen beside a building that spawns it starts on that building's side (20 flips on the same frames) |

### Fifth recording (v4.3 → v4.4): a 3-crown win, all 31 plays happened

With v4.3 every play happened (31 of 31, from 17 of 29) and the bot won 3-0 with its towers
at 82% and 79%. What was still wrong:

| Mistake | Cause | Fix |
|---|---|---|
| The enemy's left tower, destroyed at 2:23, was believed to stand at 3% for the rest of the game; troops kept going to "finish" it, and none went into the pocket in front of the King Tower | The bar reader returns 0 for a destroyed tower and for a bar too short to read, so 0 was read as "almost destroyed". In the simulator a tower at 3% still stands and blocks the pocket | A standing tower always shows its level badge (gold shield) however low its health; a tower is destroyed when badge and bar are both gone for 3 frames. The search also tries pocket spots once a tower is down |
| The enemy King Tower was seen at full health throughout, even at 780 HP | The reader does not read the king's bar | Its bar at the top of the screen is read (median of the last readings, never rising) |
| One frame read our right tower at 51% between 97% and 92% | Bar misread | A drop of more than 25% in one frame waits for a second frame |
| A decision only every 1.2 s | A fresh screenshot costs ~0.3 s, the recorder drew and saved two images per decision, and the bot paused 0.6 s after each play | Images are saved in the background and annotated frames are left to replay_battle.py; the pause is 0.3 s; the screen reader's time is now logged |

Replaying the recording, the bot now sees the left tower fall at 0:38 into the game, follows
the king from 98% down to 7%, and in the last 30 seconds plays into the pocket.

### Sixth recording (v4.4 → v4.5): a 3-crown win in 82 seconds

All 13 plays happened. What was still wrong:

| Mistake | Cause | Fix |
|---|---|---|
| Both destroyed enemy towers read as standing at 4% | A new arena with an orange-tiled floor: the gold-colour badge test passed the floor as a badge. (Replaying the third recording showed a purple floor read by the bar reader as a red bar at 3-21%.) | Each tower's badge is learned while its bar reads above half, then compared frame by frame (difference under 15 standing, 33-58 rubble, in three arenas); destroyed = badge gone and bar under 30% for 3 frames. All 8 destroyed towers in four recordings caught within a frame or two, no standing tower called destroyed |
| A one-frame misread (77%, 54%, 77%) slipped under the 25% threshold | | A reading is the median of the last three and never rises |
| An enemy Mini P.E.K.K.A crossed the left bridge unanswered while the bot spent its last elixir on the other lane; our tower lost 23% | Both decks had Mini P.E.K.K.A: the new enemy one was matched to a stale track of ours and inherited its "ally" votes | Tracks unseen for a frame match less readily; a unit first seen on its owner's half and read as that side starts with a strong vote; enemies on their way over missed for a frame or two are remembered. On the recorded frames the bot now answers with Minions on the bridge |
| Four plays to the bridges in 7 s took the bot to 0 elixir, so the Mini P.E.K.K.A above met only a 3-elixir answer | The score counts banked elixir at face value, so dumping it on offence cost nothing extra | The first 3 banked elixir are worth double. Simulator A/B against v4.4: 22-18 (crowns 35-27), and 21-19 at half strength |
| The screen reader took 673 ms a frame on the MacBook Air (the slowest part of each decision) | Build-A-Bot's float16 model on the CPU, never on the Mac's GPU / Neural Engine | When a model loads, CoreML and several CPU thread counts are timed on the machine and the fastest that gives the same output is kept |

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
- Cards: the screen reader knows 110 of the game's ~127 cards (not the newest: Goblin
  Machine, Goblinstein, Boss Bandit, Rune Giant, ...). In the simulator the bot plays the
  whole pool with evolutions, heroes and champion abilities (v5); in the real game the
  screen reader cannot yet see evolved or hero cards, so evolutions are played as the normal
  card and abilities are never pressed there.
- The Hero Bowler snapshot bug (see v5) still needs reporting to RoyaleSim.
- The placement prior has seen 225 pro plays; it needs thousands to say more than "where
  this kind of card usually goes". Some cards are still unnamed and a few tiles are read a
  tile or two off.
- Next: more replays daily; a card-choice model from the same data (which card a pro plays
  in this position); self-play training to get past the hand-written scoring.

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
crownzero/fulldecks.py       decks from the whole pool: evolutions, heroes, champions
crownzero/detector/          the new unit detector: generated training images, YOLOv8
scripts/eval_full_pool.py    search vs the scripted bots on full-pool decks
scripts/exp_delay.py, exp_track.py, exp_lead.py   the v5 real-game A/Bs
crownzero/features.py        what the bot sees (the same from the simulator and the screen)
crownzero/model.py, train.py the imitation network (v1/v2)
crownzero/arena.py           parallel matches with seat-balanced Wilson intervals
crownzero/real/              the real game: Build-A-Bot bridge, card names, recorder,
                             memory across frames (perception.py), screenshots and
                             plays (device.py), tower badges and king health (towers.py)
crownzero/replays/           learning from replays: replay.py (video -> plays), layout.py
                             (hands, elixir, where a play landed), rebuild.py (the board in
                             the simulator), prior.py (the placement prior), daily.py
scripts/exp_prior.py         simulator A/B of the prior
scripts/analyze_battle.py    post-match report of a recorded battle
scripts/replay_battle.py     the current bot on a recorded battle's frames
tests/                       unit and end-to-end tests (no phone needed)
results/                     raw per-game results (JSON lines)
```

## Fair play

Supercell's terms ban automating the game. Use Training Camp or friendly battles on a
throwaway account only. The bot enforces the ladder part: every battle against a stranger
(Trophy Road, ranked, 2v2 with a random team) starts with the "Searching for opponent"
screen, and when the bot sees it, it sits that battle out (`crownzero/real/fairplay.py`).
Training Camp and friendly battles never show that screen.
